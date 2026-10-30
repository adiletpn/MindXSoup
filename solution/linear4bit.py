"""bitsandbytes Linear4bit on fp16 inputs without the dispatch layers around the same two ops.

On a T4, for every token count this run sees, bitsandbytes computes a 4-bit linear layer as
"unpack the NF4 weight to fp16, then F.linear" (its own fused 4-bit kernels never win the
heuristic there; a profile of our run shows none of them), and backward as "unpack again, then
grad @ W". Between the module and those ops sit Linear4bit.forward, matmul_4bit, the MatMul4Bit
autograd function and the torch.library custom op `bitsandbytes::gemm_4bit` with its
torch._dynamo-disable wrappers: a few hundred calls per training step whose Python cost leaves the
GPU waiting. This patch runs the same unpack (our `nf4_unpack_kernel`) and the same F.linear /
matmul directly, so the results are bit-identical; anything else (other dtypes or quantization
settings, CPU, the first call that sets the compute dtype) takes the original forward.
"""

import contextlib

import torch

from .nf4 import _supported, unpack_nf4

_enabled = True


@contextlib.contextmanager
def disabled():
    """Run bitsandbytes' own path inside the block (tests compare against it)."""
    global _enabled
    previous, _enabled = _enabled, False
    try:
        yield
    finally:
        _enabled = previous


def _unpack(packed, state, dtype):
    """The fp16 [N, K] weight, as bitsandbytes' fallback and dequantize_4bit produce it."""
    return unpack_nf4(packed, state.shape, state.state2.absmax, state.blocksize, dtype,
                      absmax_8bit=state.absmax, code=state.state2.code, offset=state.offset)


class NF4LinearFunction(torch.autograd.Function):
    """F.linear(x, unpack(W), bias); backward grad_x = grad @ unpack(W). W and bias are frozen."""

    @staticmethod
    def forward(ctx, x, packed, state, bias):
        ctx.packed, ctx.state = packed, state
        return torch.nn.functional.linear(x, _unpack(packed, state, x.dtype), bias)

    @staticmethod
    def backward(ctx, grad):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad, _unpack(ctx.packed, ctx.state, ctx.state.dtype)
                                  .to(grad.dtype))
        return grad_x, None, None, None


def _direct_ok(layer, x):
    if not _enabled:
        return False
    weight = layer.weight
    state = getattr(weight, "quant_state", None)
    return (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float16
            and getattr(layer, "compute_type_is_set", False)
            and layer.compute_dtype == torch.float16
            and state is not None and getattr(state, "nested", False)
            and getattr(state, "state2", None) is not None
            and state.state2.blocksize == 256 and state.dtype == torch.float16
            and not getattr(state, "packing_format_for_cpu", False)
            and len(state.shape) == 2 and x.shape[-1] == state.shape[1] and x.numel() > 0
            and not weight.requires_grad
            and (layer.bias is None or not layer.bias.requires_grad)
            and _supported(weight.data, state.shape, state.blocksize, state.quant_type,
                           x.dtype, state.state2.blocksize))


def patch_bnb_linear4bit():
    import bitsandbytes as bnb

    cls = bnb.nn.Linear4bit
    if getattr(cls, "_mindx_direct", False):
        return
    original = cls.forward

    def forward(self, x):
        if not _direct_ok(self, x):
            return original(self, x)
        bias = self.bias
        if bias is not None:
            if bias.dtype != x.dtype:  # what Linear4bit.forward does, once
                bias.data = bias.data.to(x.dtype)
            bias = bias.to(self.compute_dtype)
        packed = self.weight.data.view(-1, 1)  # canonical [(N*K+1)//2, 1], as MatMul4Bit uses
        return NF4LinearFunction.apply(x, packed, self.weight.quant_state, bias)

    cls.forward = forward
    cls._mindx_direct = True
