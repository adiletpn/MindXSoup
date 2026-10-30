"""NF4 unpacking against bitsandbytes' own dequantization, plain and double-quantized.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def bnb_reference(packed, state):
    """bitsandbytes' two-kernel path, called through its ops directly (not our patch)."""
    import bitsandbytes.functional as F

    absmax = state.absmax
    if state.nested:
        absmax = F.dequantize_blockwise(state.absmax, state.state2) + state.offset
    return torch.ops.bitsandbytes.dequantize_4bit.default(
        packed, absmax.float(), state.blocksize, state.quant_type, state.shape, state.dtype)


@pytest.mark.parametrize("shape", [(1536, 1536), (8960, 1536), (256, 1536), (1536, 8960)])
@pytest.mark.parametrize("nested", [True, False])
def test_unpack_matches_bitsandbytes(shape, nested):
    import bitsandbytes.functional as F

    from solution.nf4 import unpack_nf4

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02
    packed, state = F.quantize_4bit(weight, blocksize=64, quant_type="nf4",
                                    compress_statistics=nested)
    ref = bnb_reference(packed, state)
    if nested:
        ours = unpack_nf4(packed, state.shape, state.state2.absmax, 64, state.dtype,
                          absmax_8bit=state.absmax, code=state.state2.code, offset=state.offset)
    else:
        ours = unpack_nf4(packed, state.shape, state.absmax.float(), 64, state.dtype)
    assert ours.shape == ref.shape and ours.dtype == ref.dtype
    # Same formula; at most one fp16 ulp apart where the compiler fuses the absmax multiply-add.
    torch.testing.assert_close(ours, ref, rtol=1e-3, atol=1e-6)


def test_patched_matmul_matches_unpatched():
    import bitsandbytes as bnb
    import bitsandbytes.functional as F

    from solution.nf4 import patch_bnb_nf4

    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(1536, 256, bias=False, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4").cuda()
    x = torch.randn(4, 512, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
    ref = layer(x)
    ref.sum().backward()
    ref_grad = x.grad.clone()
    x.grad = None

    patch_bnb_nf4()
    assert F.dequantize_4bit.__module__ == "solution.nf4"
    out = layer(x)
    out.sum().backward()
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(x.grad, ref_grad, rtol=2e-3, atol=2e-3)
