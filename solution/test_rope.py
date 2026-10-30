"""RoPE kernel against transformers' apply_rotary_pos_emb math, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-5}
HEAD_DIM = 128


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reference(q, k, cos, sin, unsqueeze_dim=1):
    """apply_rotary_pos_emb, transformers 5.16.1 (modeling_qwen2.py)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def model_like_qk(batch, n_q, n_k, n_tokens, dtype):
    """q and k as Qwen2Attention makes them: a [B, T, heads, D] buffer seen as [B, heads, T, D]."""
    q = torch.randn(batch, n_tokens, n_q * HEAD_DIM, device="cuda").to(dtype)
    k = torch.randn(batch, n_tokens, n_k * HEAD_DIM, device="cuda").to(dtype)
    q = q.view(batch, n_tokens, n_q, HEAD_DIM).transpose(1, 2).detach().requires_grad_(True)
    k = k.view(batch, n_tokens, n_k, HEAD_DIM).transpose(1, 2).detach().requires_grad_(True)
    assert not q.is_contiguous() and not k.is_contiguous()
    return q, k


def model_like_cos_sin(batch, n_tokens, dtype, theta=1e6):
    """Qwen2RotaryEmbedding: freqs = pos * inv_freq, emb = cat(freqs, freqs)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device="cuda").float() / HEAD_DIM))
    pos = torch.arange(n_tokens, device="cuda").float()[None, :].expand(batch, -1)
    pos = pos + torch.arange(batch, device="cuda")[:, None] * 3  # rows differ, like left padding
    freqs = pos[:, :, None] * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def run_both(q, k, cos, sin):
    from solution.rope import apply_rope

    ref_q, ref_k = reference(q, k, cos, sin)
    grad_q, grad_k = torch.randn_like(ref_q), torch.randn_like(ref_k)
    ref_dq, ref_dk = torch.autograd.grad((ref_q, ref_k), (q, k), (grad_q, grad_k))
    out_q, out_k = apply_rope(q, k, cos, sin)
    dq, dk = torch.autograd.grad((out_q, out_k), (q, k), (grad_q, grad_k))
    return (out_q, out_k, dq, dk), (ref_q, ref_k, ref_dq, ref_dk)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 12, 2, 512), (3, 12, 2, 77)])
def test_rope_matches_reference(dtype, shape):
    torch.manual_seed(0)
    batch, n_q, n_k, n_tokens = shape
    q, k = model_like_qk(batch, n_q, n_k, n_tokens, dtype)
    cos, sin = model_like_cos_sin(batch, n_tokens, dtype)

    ours, ref = run_both(q, k, cos, sin)
    tol = TOLERANCE[dtype]
    for name, a, b in zip(("q", "k", "dq", "dk"), ours, ref):
        assert a.dtype == b.dtype and a.shape == b.shape, name
        torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol, msg=name)
    # no hidden copies: outputs and grads keep the [B, T, heads, D] layout of q and k
    assert ours[0].stride() == q.stride() and ours[1].stride() == k.stride()


def test_cos_sin_with_batch_one():
    """position_ids of shape [1, T] give cos/sin [1, T, D], broadcast over the batch."""
    torch.manual_seed(0)
    q, k = model_like_qk(4, 12, 2, 100, torch.float32)
    cos, sin = model_like_cos_sin(1, 100, torch.float32)
    ours, ref = run_both(q, k, cos, sin)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_mixed_dtypes_promote_like_pytorch():
    """fp16 q/k with fp32 cos/sin: outputs fp32 as in the original, grads in the input dtype."""
    torch.manual_seed(0)
    q, k = model_like_qk(2, 12, 2, 33, torch.float16)
    cos, sin = model_like_cos_sin(2, 33, torch.float32)
    ours, ref = run_both(q, k, cos, sin)
    for a, b in zip(ours, ref):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-3, atol=2e-3)


def test_contiguous_inputs_and_grads():
    torch.manual_seed(0)
    q = torch.randn(2, 12, 50, HEAD_DIM, device="cuda", requires_grad=True)
    k = torch.randn(2, 2, 50, HEAD_DIM, device="cuda", requires_grad=True)
    cos, sin = model_like_cos_sin(2, 50, torch.float32)
    ours, ref = run_both(q, k, cos, sin)
    for a, b in zip(ours, ref):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_patch_routes_cuda_and_falls_back(monkeypatch):
    modeling_qwen2 = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution.rope import patch_rope

    monkeypatch.setattr(modeling_qwen2, "apply_rotary_pos_emb",
                        modeling_qwen2.apply_rotary_pos_emb)  # restored after the test
    patch_rope()
    patch_rope()  # idempotent
    patched = modeling_qwen2.apply_rotary_pos_emb
    assert getattr(patched, "_mindx_patched", False)

    torch.manual_seed(0)
    q, k = model_like_qk(2, 12, 2, 40, torch.float32)
    cos, sin = model_like_cos_sin(2, 40, torch.float32)
    out_q, out_k = patched(q, k, cos, sin)
    assert out_q.grad_fn is not None and "RoPEFunction" in type(out_q.grad_fn).__name__
    ref_q, ref_k = reference(q, k, cos, sin)
    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-5)

    # CPU tensors take the original code
    cpu = [x.detach().cpu() for x in (q, k, cos, sin)]
    cpu_q, cpu_k = patched(*cpu)
    ref_q, ref_k = reference(*cpu)
    torch.testing.assert_close(cpu_q, ref_q)
    torch.testing.assert_close(cpu_k, ref_k)


def test_out_dtype_writes_fp16_directly():
    """fp32 q/k with out_dtype=fp16: the fp32 rotation rounded once; grads come back in fp32."""
    from solution.rope import apply_rope

    torch.manual_seed(0)
    q, k = model_like_qk(2, 12, 2, 64, torch.float32)
    cos, sin = model_like_cos_sin(2, 64, torch.float32)
    ref_q, ref_k = (x.half() for x in reference(q, k, cos, sin))
    grads = (torch.randn_like(ref_q), torch.randn_like(ref_k))
    ref_dq, ref_dk = torch.autograd.grad((ref_q, ref_k), (q, k), grads)
    out_q, out_k = apply_rope(q, k, cos, sin, out_dtype=torch.float16)
    dq, dk = torch.autograd.grad((out_q, out_k), (q, k), grads)
    for a, b in ((out_q, ref_q), (out_k, ref_k), (dq, ref_dq), (dk, ref_dk)):
        assert a.dtype == b.dtype
        torch.testing.assert_close(a.float(), b.float(), rtol=2e-3, atol=2e-3)
