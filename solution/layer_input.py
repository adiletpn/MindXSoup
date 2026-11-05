"""A rerun layer's fp32 input as scaled fp16 rows, packed and restored in one launch each way.

recompute.py keeps a layer's input for its rerun as fp16 rows with one fp32 scale per row, so that
no value overflows fp16: scale = max(|x|_max, 1) / 32768, half = fp16(x / scale). In PyTorch ops
that is abs, amax, clamp, division and a cast to pack and a cast and a product to restore, each a
launch and a pass over the row. Here one program per row does each direction in registers.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def layer_input_pack_kernel(x_ptr, half_ptr, scale_ptr, COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < COLS
    x = tl.load(x_ptr + row * COLS + cols, mask=mask, other=0.0)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0), 1.0) / 32768.0
    tl.store(scale_ptr + row, scale)
    tl.store(half_ptr + row * COLS + cols, (x / scale).to(tl.float16), mask=mask)


@triton.jit
def layer_input_restore_kernel(half_ptr, scale_ptr, x_ptr, COLS: tl.constexpr,
                               BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < COLS
    half = tl.load(half_ptr + row * COLS + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_ptr + row * COLS + cols, half * tl.load(scale_ptr + row), mask=mask)


def _config(cols):
    block = triton.next_power_of_2(cols)
    return block, (8 if block >= 2048 else 4)


def pack(hidden):
    """hidden: [1, N, cols] fp32 -> (half [1, N, cols] fp16, scale [1, N, 1] fp32)."""
    x = hidden.detach().contiguous()
    half = torch.empty_like(x, dtype=torch.float16)
    scale = torch.empty((1, x.shape[1], 1), dtype=torch.float32, device=x.device)
    block, warps = _config(x.shape[2])
    layer_input_pack_kernel[(x.shape[1],)](x, half, scale, COLS=x.shape[2], BLOCK=block,
                                           num_warps=warps)
    return half, scale


def restore(half, scale):
    """half on the GPU, contiguous: the fp32 input back."""
    x = torch.empty_like(half, dtype=torch.float32)
    block, warps = _config(half.shape[2])
    layer_input_restore_kernel[(half.shape[1],)](half, scale, x, COLS=half.shape[2], BLOCK=block,
                                                 num_warps=warps)
    return x


enabled = True  # set by apply()


def fits(hidden):
    return enabled and hidden.dim() == 3 and hidden.shape[0] == 1 and hidden.dtype == torch.float32
