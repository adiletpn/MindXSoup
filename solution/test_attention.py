"""fp16 attention path against the original Qwen2Attention.forward, forward and backward.

The layer is built like in training: real Qwen2.5-1.5B sizes, NF4 bitsandbytes projections with
fp16 compute and double quantization, peft LoRA (r=16, alpha=32, dropout 0.05) on q_proj and
v_proj, fp32 input, fp16 autocast. Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM = 1536, 12, 2, 128


def qwen2_config(**sizes):
    from transformers import Qwen2Config

    sizes = sizes or dict(hidden_size=HIDDEN, num_attention_heads=Q_HEADS,
                          num_key_value_heads=KV_HEADS, intermediate_size=8960)
    config = Qwen2Config(**sizes)
    config._attn_implementation = "sdpa"
    return config


def to_nf4(linear):
    """bitsandbytes NF4 copy of an nn.Linear (fp16 compute, double-quantized absmax), on CUDA."""
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(linear.in_features, linear.out_features,
                              bias=linear.bias is not None, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(linear.weight.data.clone(), requires_grad=False,
                                     compress_statistics=True, quant_type="nf4", module=layer)
    if linear.bias is not None:
        layer.bias = torch.nn.Parameter(linear.bias.data.clone(), requires_grad=False)
    return layer.cuda()


def with_lora(base, seed):
    """peft LoRA around a 4-bit layer, like Soup's adapter; random B so the branch matters."""
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    torch.manual_seed(seed)
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, init_lora_weights=False)
    return LoraLinear4bit(base, "default", config=config, r=16, lora_alpha=32).cuda()


def quantize_attention(attn, seed=0):
    """Qwen2Attention as Soup trains it: NF4 projections, LoRA on q_proj and v_proj."""
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attn, name, to_nf4(getattr(attn, name)))
    attn.q_proj = with_lora(attn.q_proj, seed + 1)
    attn.v_proj = with_lora(attn.v_proj, seed + 2)
    return attn


def build_attention(seed=0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    torch.manual_seed(seed)
    return quantize_attention(Qwen2Attention(qwen2_config(), layer_idx=0), seed)


def padding_mask(batch, n_tokens, lengths):
    """[B, 1, T, T] bool mask (True = attend): causal, keys past each row's length hidden."""
    causal = torch.ones(n_tokens, n_tokens, dtype=torch.bool, device="cuda").tril()
    keys = torch.arange(n_tokens, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    return (causal[None] & keys[:, None, :])[:, None]


def lora_params(attn):
    return [p for name, p in attn.named_parameters() if "lora_" in name]


def run(attn, forward, hidden, cos, sin, mask, seed):
    """fwd under fp16 autocast + bwd; returns output, grad of the input, grads of LoRA weights."""
    for p in lora_params(attn):
        p.grad = None
    x = hidden.detach().clone().requires_grad_(True)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        out, _ = forward(attn, x, (cos, sin), mask)
    grad = torch.linspace(-1, 1, out.numel(), device="cuda").view_as(out).float()
    (out.float() * grad).sum().backward()
    return [out.detach().float(), x.grad.detach().clone()] + \
        [p.grad.detach().clone() for p in lora_params(attn)]


def max_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


@pytest.fixture
def forwards(monkeypatch):
    """(original forward, patched forward, call counter of the fused kernel path)."""
    from transformers.models.qwen2 import modeling_qwen2

    from solution import attention

    cls = modeling_qwen2.Qwen2Attention
    # transformers' forward, even if another test already ran solution.apply(); restored after
    original = getattr(cls.forward, "_mindx_original", cls.forward)
    monkeypatch.setattr(cls, "forward", original)
    monkeypatch.setattr(cls, "_mindx_fp16", False, raising=False)
    attention.patch_qwen2_attention_fp16()
    attention.patch_qwen2_attention_fp16()  # idempotent
    calls = []
    real = attention.qkv_rope
    monkeypatch.setattr(attention, "qkv_rope", lambda *a: calls.append(1) or real(*a))
    return original, cls.forward, calls


def inputs(batch, n_tokens, seed=0):
    from solution.test_rope import model_like_cos_sin

    torch.manual_seed(seed)
    hidden = torch.randn(batch, n_tokens, HIDDEN, device="cuda")
    cos, sin = model_like_cos_sin(1, n_tokens, torch.float32)
    return hidden, cos, sin


@pytest.mark.parametrize("batch,n_tokens,masked", [(2, 64, False), (2, 77, True), (4, 512, True)])
def test_matches_original_without_dropout(forwards, batch, n_tokens, masked):
    original, patched, calls = forwards
    attn = build_attention().eval()  # eval: LoRA dropout off, compared separately below
    hidden, cos, sin = inputs(batch, n_tokens)
    mask = padding_mask(batch, n_tokens, [n_tokens - 9 * i for i in range(batch)]) \
        if masked else None
    ref = run(attn, original, hidden, cos, sin, mask, seed=0)
    ours = run(attn, patched, hidden, cos, sin, mask, seed=0)
    assert calls, "the fused path was not taken"
    assert ours[0].dtype == ref[0].dtype and ours[0].shape == ref[0].shape
    names = ["output", "grad input"] + [n for n, _ in attn.named_parameters() if "lora_" in n]
    for name, a, b in zip(names, ours, ref):
        assert max_rel(a, b) < 1e-2, f"{name}: {max_rel(a, b):.2e}"


def test_lora_dropout_masks_unchanged(forwards):
    """Training mode: same seed -> same LoRA dropout masks as the original (q first, then v)."""
    original, patched, calls = forwards
    attn = build_attention().train()
    hidden, cos, sin = inputs(2, 96)
    ref = run(attn, original, hidden, cos, sin, None, seed=123)
    ours = run(attn, patched, hidden, cos, sin, None, seed=123)
    other_seed = run(attn, original, hidden, cos, sin, None, seed=7)
    assert calls
    for a, b, c in zip(ours, ref, other_seed):
        assert max_rel(a, b) < 1e-2
    # the check above would notice different masks: another seed moves the LoRA grads a lot
    assert max(max_rel(c, b) for c, b in zip(other_seed[2:], ref[2:])) > 3e-2
    # and the RNG stream advanced by the same amount
    torch.manual_seed(5)
    run(attn, original, hidden, cos, sin, None, seed=5)
    after_ref = torch.rand(1, device="cuda")
    run(attn, patched, hidden, cos, sin, None, seed=5)
    after_ours = torch.rand(1, device="cuda")
    assert torch.equal(after_ref, after_ours)


def test_without_autocast_takes_original(forwards):
    original, patched, calls = forwards
    attn = build_attention().eval()
    hidden, cos, sin = inputs(1, 32)
    with torch.no_grad():
        a, _ = patched(attn, hidden, (cos, sin), None)
        b, _ = original(attn, hidden, (cos, sin), None)
    assert not calls
    torch.testing.assert_close(a, b)


def test_cpu_module_takes_original(forwards):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

    original, patched, calls = forwards
    config = qwen2_config(hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
                          intermediate_size=128)
    attn = Qwen2Attention(config, layer_idx=0).eval()
    torch.manual_seed(0)
    hidden = torch.randn(2, 10, 64)
    cos, sin = torch.ones(1, 10, 16), torch.zeros(1, 10, 16)
    with torch.no_grad():
        a, _ = patched(attn, hidden, (cos, sin), None)
        b, _ = original(attn, hidden, (cos, sin), None)
    assert not calls
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("n_tokens", [16, 300])
def test_qkv_rope_kernel_against_pytorch(n_tokens):
    """The fused kernel alone: (base + LoRA) rotated -> fp16, against fp32 PyTorch math."""
    from solution.attention import qkv_rope
    from solution.test_rope import model_like_cos_sin, reference

    torch.manual_seed(0)
    batch = 3

    def proj(heads):
        return (torch.randn(batch, n_tokens, heads * HEAD_DIM, device="cuda") * 2).half() \
            .requires_grad_(True)

    q_base, q_lora, k, v_base, v_lora = proj(Q_HEADS), proj(Q_HEADS), proj(KV_HEADS), \
        proj(KV_HEADS), proj(KV_HEADS)
    cos, sin = model_like_cos_sin(batch, n_tokens, torch.float32)

    def heads(x, n):
        return x.view(batch, n_tokens, n, HEAD_DIM).transpose(1, 2)

    inputs_ = (q_base, q_lora, k, v_base, v_lora)
    ref_q, ref_k = reference(heads(q_base.float() + q_lora.float(), Q_HEADS),
                             heads(k.float(), KV_HEADS), cos, sin)
    ref = (ref_q.half(), ref_k.half(), heads(v_base.float() + v_lora.float(), KV_HEADS).half())
    grads = [torch.randn_like(r) for r in ref]
    ref_grads = torch.autograd.grad(ref, inputs_, grads)
    ours = qkv_rope(q_base, q_lora, k, v_base, v_lora, cos, sin, Q_HEADS, KV_HEADS)
    our_grads = torch.autograd.grad(ours, inputs_, grads)
    for a, b in zip(ours, ref):
        assert a.dtype == torch.float16 and a.shape == b.shape
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-3, atol=2e-3)
    for a, b in zip(our_grads, ref_grads):
        assert a.dtype == b.dtype and a.shape == b.shape
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-3, atol=2e-3)


def test_power_of_two_lora_scale_in_the_kernel_is_bit_exact():
    from solution.attention import qkv_rope
    from solution.test_rope import model_like_cos_sin

    torch.manual_seed(1)
    batch, n_tokens = 2, 37

    def proj(heads):
        return (torch.randn(batch, n_tokens, heads * HEAD_DIM, device="cuda") * 2).half() \
            .requires_grad_(True)

    q_base, q_lora, k, v_base, v_lora = proj(Q_HEADS), proj(Q_HEADS), proj(KV_HEADS), \
        proj(KV_HEADS), proj(KV_HEADS)
    cos, sin = model_like_cos_sin(batch, n_tokens, torch.float32)
    inputs_ = (q_base, q_lora, k, v_base, v_lora)
    # peft's way: the LoRA output times scaling (2.0 here) first, then our kernel adds it
    want = qkv_rope(q_base, q_lora * 2.0, k, v_base, v_lora * 2.0, cos, sin, Q_HEADS, KV_HEADS)
    got = qkv_rope(q_base, q_lora, k, v_base, v_lora, cos, sin, Q_HEADS, KV_HEADS, 2.0, 2.0)
    grads = [torch.randn_like(r) for r in want]
    want_grads = torch.autograd.grad(want, inputs_, grads)
    got_grads = torch.autograd.grad(got, inputs_, grads)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    for a, b in zip(got_grads, want_grads):
        assert torch.equal(a, b)
