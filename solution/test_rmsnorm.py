"""Fused RMSNorm against transformers' Qwen2RMSNorm math, forward and backward.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-5}


def reference(x, weight, eps):
    """Qwen2RMSNorm.forward, transformers 5.16.1."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(4, 512, 1536), (3, 77, 1536), (2, 5, 1000)])
def test_rmsnorm_matches_reference(x_dtype, shape):
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    eps = 1e-6
    x = (torch.randn(shape, device="cuda") * 3).to(x_dtype).requires_grad_(True)
    weight = (1 + 0.1 * torch.randn(shape[-1], device="cuda")).float()

    ref = reference(x, weight, eps)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    ref_dx = x.grad.clone()
    x.grad = None

    out = rmsnorm(x, weight, eps)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    out.backward(grad)
    tol = TOLERANCE[x_dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(x.grad.float(), ref_dx.float(), rtol=tol, atol=tol)


def test_weight_grad_when_trainable():
    from solution.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(4, 33, 1536, device="cuda", requires_grad=True)
    weight = torch.randn(1536, device="cuda", requires_grad=True)
    reference(x, weight, 1e-6).sum().backward()
    ref_dw = weight.grad.clone()
    weight.grad = None
    rmsnorm(x, weight, 1e-6).sum().backward()
    torch.testing.assert_close(weight.grad, ref_dw, rtol=1e-4, atol=1e-4)


def test_patch_falls_back_on_cpu():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    from solution.rmsnorm import patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    norm = Qwen2RMSNorm(64)
    x = torch.randn(2, 3, 64)
    torch.testing.assert_close(norm(x), reference(x, norm.weight, norm.variance_epsilon))
