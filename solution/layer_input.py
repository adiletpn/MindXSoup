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
