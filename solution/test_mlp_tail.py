"""The rerun MLP without its dead values gives the same input gradient, bit for bit.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("n_tokens", [37, 1430])
def test_tail_gradient_is_bit_identical(n_tokens):
    from solution import linear4bit, stream
    from solution.mlp_tail import MlpTailRerun
    from solution.nf4 import patch_bnb_nf4
    from solution.silu_mul import silu_mul
    from solution.test_mlp_recompute import build_mlp

    patch_bnb_nf4()
    linear4bit.patch_bnb_linear4bit()
    mlp = build_mlp()
    for layer in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        layer.compute_type_is_set = True
    torch.manual_seed(2)
    x = torch.randn(1, n_tokens, mlp.gate_proj.weight.quant_state.shape[1], device="cuda")
    grad = torch.randn(x.shape, device="cuda")

    def full(leaf):  # mlp.py's fp16 forward, written out: other tests patch the class differently
        x16 = leaf.to(torch.float16)
        return mlp.down_proj(silu_mul(mlp.gate_proj(x16), mlp.up_proj(x16))).to(leaf.dtype)

    def gradient(forward):
        leaf = x.clone().requires_grad_(True)
        forward(leaf).backward(grad)
        return leaf.grad

    ref = gradient(full)
    try:
        stream._streamer = stream.LayerStreamer([[mlp.gate_proj, mlp.up_proj, mlp.down_proj]], slots=1)
        streamed = gradient(full)
        got = gradient(lambda leaf: MlpTailRerun.apply(leaf, mlp))
    finally:
        stream._streamer = None
    assert torch.equal(got, streamed)
    if n_tokens > 1024:  # with few tokens the GPU-resident path takes bitsandbytes' own kernel
        assert torch.equal(got, ref)


def test_tail_not_taken_outside_a_rerun():
    from solution.mlp_tail import _tail_ok
    from solution.test_mlp_recompute import build_mlp

    assert not _tail_ok(build_mlp(), torch.zeros(1, 4, 8, device="cuda"))
