"""silu(gate) * up for the fp16 MLP, with a backward that writes its gradients over gate and up.

The training peak (tools/peak_memory.py, final run without the table on the GPU: 343 MiB of
tensors) is reached in the MLP backward of a recomputed layer: gate and up of that layer
(2 x [N, 8960] fp16), the gradient of down_proj's input, and two more [N, 8960] buffers for the
gradients of gate and up, 48.8 MiB at N = 1 430 tokens. gate and up are read only by this
backward and are dead right after it, so here the backward stores d(gate) into gate's memory and
d(up) into up's: the two extra buffers are never allocated.

One kernel for both directions; each program walks one row ([rows, 8960] contiguous) in blocks
of columns. The number of rows changes from batch to batch but only enters the launch grid, so a
new length never recompiles. Values are computed in fp32 from the fp16 inputs and rounded to
fp16 once, as before.
"""

import torch
import triton
import triton.language as tl

from . import direct

COLS_PER_PROGRAM = 1024
WARPS = 4


@triton.jit
def silu_mul_kernel(grad_ptr, gate_ptr, up_ptr, out_ptr, n_cols, COLS: tl.constexpr,
                    BACKWARD: tl.constexpr):
    """One row per program, walking its columns in blocks. Forward: out = silu(gate) * up.
    Backward: gate and up are overwritten with d(gate) and d(up) for the incoming d(out)."""
    base = tl.program_id(0).to(tl.int64) * n_cols
    for col0 in range(0, n_cols, COLS):
        cols = col0 + tl.arange(0, COLS)
        inside = cols < n_cols
        g = tl.load(gate_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
        s = tl.sigmoid(g)
        silu = g * s
        if BACKWARD:
            dh = tl.load(grad_ptr + base + cols, mask=inside, other=0.0).to(tl.float32)
            # d silu(g) / dg = s + silu * (1 - s)
            tl.store(gate_ptr + base + cols, ((dh * u) * (s + silu * (1.0 - s))).to(tl.float16),
                     mask=inside)
            tl.store(up_ptr + base + cols, (dh * silu).to(tl.float16), mask=inside)
        else:
            tl.store(out_ptr + base + cols, (u * silu).to(tl.float16), mask=inside)


def _grid(t):
    n_cols = t.shape[-1]
    return (t.numel() // n_cols,), n_cols


class SiluMulInPlace(torch.autograd.Function):
    """silu(gate) * up in fp16; backward returns gate's and up's own memory holding their
    gradients. Only for gate/up that nothing else reads after this backward (the MLP's
    projections, whose 4-bit weights are frozen and keep no input)."""

    @staticmethod
    def forward(ctx, gate, up):
        gate = gate.contiguous()
        up = up.to(gate.dtype).contiguous()
        out = torch.empty_like(gate)
        grid, n_cols = _grid(gate)
        silu_mul_kernel[grid](gate, gate, up, out, n_cols, COLS=COLS_PER_PROGRAM,
                              BACKWARD=False, num_warps=WARPS)
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        grad = grad.to(gate.dtype).contiguous()
        grid, n_cols = _grid(gate)
        silu_mul_kernel[grid](grad, gate, up, gate, n_cols, COLS=COLS_PER_PROGRAM,
                              BACKWARD=True, num_warps=WARPS)
        return gate, up


def silu_mul(gate, up):
    if gate.dtype != torch.float16 or not gate.is_cuda or gate.shape != up.shape \
            or gate.numel() == 0:
        return torch.nn.functional.silu(gate) * up
    return direct.run(SiluMulInPlace, gate, up)
