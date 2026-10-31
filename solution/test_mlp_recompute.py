"""gate/up redone in backward give the same MLP output and gradients, and the forward keeps less.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def build_mlp():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution.test_attention import qwen2_config, to_nf4

    torch.manual_seed(0)
    mlp = Qwen2MLP(qwen2_config())
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, to_nf4(getattr(mlp, name)))
    return mlp.cuda().train()


def run(mlp, forward, x, grad):
    x = x.detach().requires_grad_(True)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    out = forward(mlp, x)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - base - out.numel() * out.element_size()
    out.backward(grad)
    return out.detach(), x.grad, kept


@pytest.mark.parametrize("n_tokens", [37, 1430])
def test_gateup_recompute_matches(n_tokens):
    from solution import mlp_recompute
    from solution.mlp import patch_qwen2_mlp_fp16
    from solution.swiglu import patch_qwen2_mlp

    patch_qwen2_mlp()
    patch_qwen2_mlp_fp16()
    mlp_recompute.patch_qwen2_mlp_gateup_recompute()
    # other tests may wrap Qwen2MLP.forward again: take our function and the one it wraps
    patched, previous = mlp_recompute.patched_forward, mlp_recompute.previous_forward
    mlp = build_mlp()
    torch.manual_seed(1)
    x = torch.randn(1, n_tokens, 1536, device="cuda")
    grad = torch.randn(1, n_tokens, 1536, device="cuda")
    out0, gx0, kept0 = run(mlp, previous, x, grad)
    out1, gx1, kept1 = run(mlp, patched, x, grad)
    assert out1.dtype == out0.dtype == torch.float32
    assert torch.equal(out0, out1)
    assert torch.equal(gx0, gx1)
    if n_tokens >= 1024:  # gate + up (2 x N x 8960 fp16) are no longer kept
        assert kept1 < kept0 - n_tokens * 8960 * 2


def test_not_applied_without_grad_or_in_eval():
    from solution import mlp_recompute

    mlp_recompute.patch_qwen2_mlp_gateup_recompute()
    forward = mlp_recompute.patched_forward
    mlp = build_mlp()
    x = torch.randn(1, 16, 1536, device="cuda")
    with torch.no_grad():
        a = forward(mlp, x)
    mlp.eval()
    b = forward(mlp, x)
    assert torch.equal(a, b)
