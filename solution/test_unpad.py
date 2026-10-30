"""Training on the real tokens only gives the step of the padded batch: same loss, LoRA gradients,
per-token values and dropout masks.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENGTHS = [37, 20, 5, 30]


class LoraLinear(torch.nn.Module):
    """base(x) + B(A(dropout(x))) * scale, the structure of peft's LoRA layer (peft itself is
    not used here: on Kaggle's image it trips over an old torchao for plain nn.Linear)."""

    def __init__(self, base, r, dropout):
        super().__init__()
        self.base = base
        self.dropout = torch.nn.Dropout(dropout)
        self.lora_A = torch.nn.Linear(base.in_features, r, bias=False)
        self.lora_B = torch.nn.Linear(r, base.out_features, bias=False)
        torch.nn.init.normal_(self.lora_B.weight, std=0.02)  # non-zero, so the LoRA path matters
        self.scale = 2.0

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale


def tiny_lora_qwen2(dropout):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.fused_ce import patch_qwen2_causal_lm
    from solution.unpad import patch_unpad

    patch_qwen2_causal_lm()
    patch_unpad()
    config = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attn_implementation="sdpa")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    for param in model.parameters():
        param.requires_grad_(False)
    for layer in model.model.layers:
        layer.self_attn.q_proj = LoraLinear(layer.self_attn.q_proj, 4, dropout)
        layer.self_attn.v_proj = LoraLinear(layer.self_attn.v_proj, 4, dropout)
    return model.cuda().train()


def batch(left_padded=False):
    torch.manual_seed(1)
    length = max(LENGTHS)
    ids = torch.randint(0, 1000, (len(LENGTHS), length), device="cuda")
    mask = torch.zeros_like(ids)
    for row, n in enumerate(LENGTHS):
        if left_padded:
            mask[row, length - n:] = 1
        else:
            mask[row, :n] = 1
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :3] = -100  # a prompt part, as in SFT
    return ids, mask, labels


def step(model, unpadded, ids, mask, labels, recompute=False):
    from solution import fused_ce, recompute as layer_recompute, unpad

    layer_recompute.patch_qwen2_layer_recompute()
    layer_recompute.enabled = recompute
    layer_recompute.BUDGET_TOKENS = 0  # rerun every layer of the tiny batch
    unpad.enabled = unpadded
    torch.manual_seed(123)  # the same dropout draws in both runs
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    entropy, token_loss, correct, labelled = getattr(model, fused_ce.STEP_ATTR)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    unpad.enabled = True
    layer_recompute.enabled = False
    layer_recompute.BUDGET_TOKENS = 1024
    return out.loss.detach(), entropy, token_loss, correct, labelled, grads


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_unpadded_step_matches_padded(dropout, recompute):
    model = tiny_lora_qwen2(dropout)
    ids, mask, labels = batch()
    ref = step(model, False, ids, mask, labels)
    got = step(model, True, ids, mask, labels, recompute)
    real = mask.bool()
    # the masked (padded) and per-row causal SDPA kernels round differently in the last fp32 bits,
    # which the fp16 head matmul can carry one fp16 step further
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[1][real], ref[1][real], rtol=1e-3, atol=1e-3)  # entropy
    torch.testing.assert_close(got[2], ref[2], rtol=1e-3, atol=1e-3)  # loss per token
    assert torch.equal(got[4], ref[4])  # labelled tokens
    assert abs(int(got[3]) - int(ref[3])) <= 1  # correct argmax: a near-tie may flip
    for name, grad in ref[5].items():
        torch.testing.assert_close(got[5][name], grad, rtol=2e-3, atol=1e-5, msg=name)


def test_recompute_alone_matches():
    model = tiny_lora_qwen2(0.05)
    ids, mask, labels = batch()
    ref = step(model, False, ids, mask, labels)
    got = step(model, False, ids, mask, labels, recompute=True)
    torch.testing.assert_close(got[0], ref[0], rtol=1e-6, atol=1e-6)
    for name, grad in ref[5].items():
        torch.testing.assert_close(got[5][name], grad, rtol=1e-5, atol=1e-7, msg=name)


def test_dropout_keeps_the_base_mask():
    from solution import unpad

    unpad.patch_dropout_full_mask()
    drop = torch.nn.Dropout(0.05).train()
    x = torch.randn(4, 37, 128, device="cuda")
    _, mask, _ = batch()
    lengths, index = unpad.layout(mask)
    torch.manual_seed(5)
    ref = drop(x)
    torch.manual_seed(5)
    with unpad.packed(lengths, 4, 37, index):
        got = drop(unpad.pack(x, index))
    assert torch.equal(got, unpad.pack(ref, index))


def test_left_padding_and_full_rows_keep_the_padded_path():
    from solution import unpad

    _, mask, _ = batch(left_padded=True)
    assert unpad.layout(mask) is None
    lengths, index = unpad.layout(torch.ones(2, 9, dtype=torch.long, device="cuda"))
    assert lengths == [9, 9] and index.numel() == 18
    assert unpad.layout(mask.cpu()) is None


def test_layers_to_rerun():
    from solution.recompute import layers_to_rerun

    assert layers_to_rerun(800, 28, 1024) == 0
    assert layers_to_rerun(1024, 28, 1024) == 0
    assert layers_to_rerun(2048, 28, 1024) == 14
    assert layers_to_rerun(1500, 28, 1024) == 9  # 19 layers x 1500 <= 28 x 1024
    for n in range(1, 2049):
        k = layers_to_rerun(n, 28, 1024)
        assert (28 - k) * n <= 28 * 1024


def test_per_row_sdpa_runs_the_fused_fp16_kernel_with_gqa():
    """Packed rows with 12 q heads and 2 k/v heads in fp16: the per-row path must not need
    enable_gqa (no fused kernel on a T4, fp32 math fallback); it matches the padded, masked
    attention of the base."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from solution import unpad

    unpad.patch_sdpa_per_row()
    lengths, length = [50, 33, 7, 64], 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(len(lengths), heads, length, 128, device="cuda", dtype=torch.float16)
               for heads in (12, 2, 2))
    module = torch.nn.Module()
    module.num_key_value_groups, module.is_causal, module.training = 6, True, True
    keep = torch.arange(length, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    index = keep.reshape(-1).nonzero().squeeze(1)

    def packed(x):  # [B, H, T, D] -> [1, H, N, D]
        return x.transpose(1, 2).reshape(-1, x.shape[1], 128)[index].transpose(0, 1)[None]

    with unpad.packed(lengths, len(lengths), length, index), \
            sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):  # errors if it needed the math path
        out, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](module, packed(q), packed(k), packed(v), None,
                                                 dropout=0.0, scaling=128 ** -0.5)
    causal = torch.ones(length, length, dtype=torch.bool, device="cuda").tril()
    mask = (causal[None] & keep[:, None, :])[:, None]
    ref, _ = sdpa_attention_forward(module, q, k, v, mask, dropout=0.0, scaling=128 ** -0.5)
    ref = ref.reshape(-1, 12, 128)[index][None]  # [1, N, heads, head_dim]
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)


def test_packed_dropout_kernel_matches_torch_dropout():
    from solution import dropout, unpad

    batch_size, length, hidden, p = len(LENGTHS), max(LENGTHS), 1536, 0.05
    _, mask, _ = batch()
    _, index = unpad.layout(mask)
    torch.manual_seed(5)
    x = torch.randn(batch_size, length, hidden, device="cuda", requires_grad=True)
    grad_rows = torch.randn(1, index.numel(), hidden, device="cuda")
    torch.manual_seed(7)
    ref = unpad.pack(torch.nn.functional.dropout(x, p, True), index)  # the base, then our rows
    ref.backward(grad_rows)
    ref_grad = unpad.pack(x.grad, index)

    for out_dtype in (torch.float32, torch.float16):
        rows = unpad.pack(x.detach(), index).requires_grad_(True)
        torch.manual_seed(7)
        got = dropout.packed_dropout(rows, p, batch_size, length, index, out_dtype)
        assert got.dtype == out_dtype
        assert torch.equal(got, ref.detach().to(out_dtype))  # fp16: the cast autocast does
        got.backward(grad_rows.to(out_dtype))
        expected = ref_grad if out_dtype == torch.float32 else \
            torch.where(ref.detach() != 0, grad_rows.half().float() * (1.0 / (1.0 - p)), 0.0)
        assert rows.grad.dtype == torch.float32 and torch.equal(rows.grad, expected)
