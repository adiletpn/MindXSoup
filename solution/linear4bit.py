"""bitsandbytes Linear4bit on fp16 inputs without the dispatch layers around the same two ops.

On a T4, past a few hundred tokens, bitsandbytes computes a 4-bit linear layer as "unpack the NF4
weight to fp16, then F.linear" (its fused 4-bit kernel only wins its heuristic for few tokens), and
backward always as "unpack again, then grad @ W". Between the module and those ops sit Linear4bit.forward, matmul_4bit, the MatMul4Bit
autograd function and the torch.library custom op `bitsandbytes::gemm_4bit` with its
torch._dynamo-disable wrappers: a few hundred calls per training step whose Python cost leaves the
GPU waiting. This patch runs the same unpack (our `nf4_unpack_kernel`) and the same F.linear /
matmul directly, so the results are bit-identical; anything else (other dtypes or quantization
settings, CPU, the first call that sets the compute dtype, token counts where bitsandbytes picks
its fused kernel) takes the original forward.
"""

import contextlib

import torch

from . import direct, stream
from .nf4 import _supported, unpack_nf4

_enabled = True
cache_fp16 = False  # set by patch_bnb_linear4bit(cache_fp16=True): each weight unpacked once, kept in fp16


@contextlib.contextmanager
def disabled():
    """Run bitsandbytes' own path inside the block (tests compare against it)."""
    global _enabled
    previous, _enabled = _enabled, False
    try:
        yield
    finally:
        _enabled = previous


def _unpack(packed, state, dtype, absmax_8bit=None):
    """The fp16 [N, K] weight, as bitsandbytes' fallback and dequantize_4bit produce it."""
    return unpack_nf4(packed, state.shape, state.state2.absmax, state.blocksize, dtype,
                      absmax_8bit=state.absmax if absmax_8bit is None else absmax_8bit,
                      code=state.state2.code, offset=state.offset)


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


def streamed_weight(module, state):
    """The module's fp16 weight: from the running layer's buffer, or unpacked from its bytes."""
    weight = stream.weight_for(module)
    if weight is not None:
        return weight
    packed, absmax = stream.bytes_for(module)
    return _unpack(packed, state, state.dtype, absmax)


class NF4StreamFunction(torch.autograd.Function):
    """NF4LinearFunction for a module whose bytes live on the host (stream.py): the bytes are
    asked for right before each use, in forward and again in backward."""

    @staticmethod
    def forward(ctx, x, module, state, bias):
        ctx.module, ctx.state = module, state
        return torch.nn.functional.linear(x, streamed_weight(module, state).to(x.dtype), bias)

    @staticmethod
    def backward(ctx, grad):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad, streamed_weight(ctx.module, ctx.state).to(grad.dtype))
        return grad_x, None, None, None


def _streamed_forward(layer, x):
    """Linear4bit.forward's own steps (input to the compute dtype, bias cast, output back to the
    input dtype) around the streamed matmul."""
    inp_dtype = x.dtype
    if layer.compute_dtype is not None:
        x = x.to(layer.compute_dtype)
    bias = layer.bias
    if bias is not None:
        if bias.dtype != x.dtype:
            bias.data = bias.data.to(x.dtype)
        bias = bias.to(layer.compute_dtype)
    return direct.run(NF4StreamFunction, x, layer, layer.weight.quant_state, bias).to(inp_dtype)


def _bnb_picks_fused_kernel(x, state):
    """bitsandbytes' own choice (backends/cuda/ops.py, gemm_4bit): its fused 4-bit kernel for few
    tokens, dequantize + F.linear past that. We only take over the second case."""
    from bitsandbytes.backends.cuda import ops

    k = x.shape[-1]
    m = x.numel() // k
    if m > ops._gemm_4bit_custom_max_m or k % state.blocksize != 0:
        return False
    return bool(ops._gemm_4bit_use_custom_fn(x.device.index, x.dtype, m, state.shape[0], k))


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
                           x.dtype, state.state2.blocksize)
            and not _bnb_picks_fused_kernel(x, state))


def _cached_weight(layer, x):
    """The module's weight unpacked once to fp16 and kept, while the packed bytes wait on the host
    (same values as every per-use unpack, without the kernel, for 2 B per value of GPU memory
    instead of 0.5 B). None unless this call would take the direct path."""
    weight16 = layer.__dict__.get("_mindx_weight16")
    if weight16 is None:
        if not _direct_ok(layer, x):
            return None
        packed = layer.weight.data.view(-1, 1)
        weight16 = _unpack(packed, layer.weight.quant_state, torch.float16)
        layer.__dict__["_mindx_weight16"] = weight16
        stream.offload(layer.weight, pin=False)  # only a call that needs bnb's kernel reads them now
        return weight16
    if not (_enabled and isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float16
            and x.numel() > 0 and x.shape[-1] == weight16.shape[1]):
        return None
    return weight16


def patch_bnb_linear4bit(cache_weights=False):
    import bitsandbytes as bnb

    global cache_fp16
    cache_fp16 = cache_weights
    cls = bnb.nn.Linear4bit
    if getattr(cls, "_mindx_direct", False):
        return
    original = cls.forward

    def forward(self, x):
        if cache_fp16:
            weight16 = _cached_weight(self, x)
            if weight16 is not None:
                bias = self.bias
                if bias is not None and bias.dtype != x.dtype:  # what Linear4bit.forward does, once
                    bias.data = bias.data.to(x.dtype)
                return torch.nn.functional.linear(x, weight16, bias)
        if isinstance(x, torch.Tensor) and x.is_cuda and stream.offloaded(self):
            return _streamed_forward(self, x)
        if not _direct_ok(self, x):
            return original(self, x)
        bias = self.bias
        if bias is not None:
            if bias.dtype != x.dtype:  # what Linear4bit.forward does, once
                bias.data = bias.data.to(x.dtype)
            bias = bias.to(self.compute_dtype)
        packed = self.weight.data.view(-1, 1)  # canonical [(N*K+1)//2, 1], as MatMul4Bit uses
        return direct.run(NF4LinearFunction, x, packed, self.weight.quant_state, bias)

    cls.forward = forward
    cls._mindx_direct = True
