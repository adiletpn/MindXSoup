"""The fused pack and restore of a rerun layer's input give recompute.py's PyTorch formulas.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("tokens", [1, 37, 1430])
def test_matches_pytorch(tokens):
    from solution.layer_input import pack, restore

    torch.manual_seed(0)
    x = torch.randn(1, tokens, 1536, device="cuda") * torch.logspace(-3, 4, tokens, device="cuda")[None, :, None]
    scale_ref = x.abs().amax(dim=-1, keepdim=True).float().clamp_min(1.0) / 32768.0
    half_ref = (x / scale_ref).to(torch.float16)
    half, scale = pack(x)
    assert torch.equal(scale, scale_ref)
    assert torch.isfinite(half).all()
    # Triton's fp32 division can differ from PyTorch's in the last bit, so a value may round to
    # the neighbouring fp16 number: at most one fp16 step (2^-10 of the value; 2^-14 near zero)
    steps = (half.float() - half_ref.float()).abs() / half_ref.float().abs().clamp_min(2.0 ** -4)
    assert steps.max().item() <= 2.0 ** -10
    torch.testing.assert_close(restore(half, scale), x, atol=scale.max().item() * 2.0 ** -9,
                               rtol=2.0 ** -9)
