"""LoRA dropout on the packed (unpadded) tokens in one launch each way.

To keep the base run's random numbers, unpad.py draws the dropout mask exactly as the base does:
`F.dropout` on a padded [B, T, hidden] tensor (the draw depends only on its shape, never on its
values, so a persistent buffer of ones serves). This kernel then does in one pass what took five
PyTorch ops (gather of the real-token rows of the mask, compare, scale, masked fill, and the
fp16 cast that autocast applies right after for lora_A): out = x * scale where the mask kept the
element, else 0, written in the dtype lora_A consumes. Backward is the same kernel (BACKWARD):
dx = grad * scale where kept, else 0, in the input's dtype. Same values as torch's dropout.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def lora_dropout_kernel(src_ptr, drawn_ptr, index_ptr, keep_ptr, out_ptr, scale,
                        COLS: tl.constexpr, BLOCK: tl.constexpr, BACKWARD: tl.constexpr):
    # One program per packed token row.
    # forward:  out = x * scale where the base's draw kept the element; the row of the draw is
    #           the token's row in the padded [B * T, hidden] tensor (index); keep is stored
    # backward: dx = grad * scale where kept (keep read back), in the input's dtype
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < COLS
    here = row * COLS + cols
    if BACKWARD:
        keep = tl.load(keep_ptr + here, mask=valid, other=0) != 0
    else:
        source = tl.load(index_ptr + row)
        keep = tl.load(drawn_ptr + source * COLS + cols, mask=valid, other=0.0) != 0.0
        tl.store(keep_ptr + here, keep.to(tl.int8), mask=valid)
    value = tl.load(src_ptr + here, mask=valid, other=0.0).to(tl.float32)
    result = tl.where(keep, value * scale, 0.0)
    tl.store(out_ptr + here, result.to(out_ptr.dtype.element_ty), mask=valid)


def _launch(src, drawn, index, keep, out, scale, backward):
    rows, cols = src.shape[-2], src.shape[-1]
    block = triton.next_power_of_2(cols)
    lora_dropout_kernel[(rows,)](src, drawn, index, keep, out, scale, COLS=cols, BLOCK=block,
                                 BACKWARD=backward, num_warps=8 if block >= 2048 else 4)


class PackedDropout(torch.autograd.Function):
    """x: [1, N, hidden] packed tokens; drawn: the base's [B, T, hidden] dropout of ones;
    index: [N] rows of the real tokens in the flattened [B * T]."""

    @staticmethod
    def forward(ctx, x, drawn, index, scale, out_dtype):
        x = x.contiguous()
        rows, cols = x.shape[1], x.shape[2]
        out = torch.empty((1, rows, cols), dtype=out_dtype, device=x.device)
        keep = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        _launch(x, drawn, index, keep, out, scale, backward=False)
        ctx.save_for_backward(keep)
        ctx.scale, ctx.x_dtype = scale, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        (keep,) = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty(grad.shape, dtype=ctx.x_dtype, device=grad.device)
        # the draw and the index are not read in backward; keep stands in for their pointers
        _launch(grad, keep, keep, keep, dx, ctx.scale, backward=True)
        return dx, None, None, None, None


_ones = None


def ones_like_draw(batch, length, cols, dtype, device):
    """A [B, T, hidden] view of a persistent buffer of ones (the draw ignores the values)."""
    global _ones
    n = batch * length * cols
    if _ones is None or _ones.numel() < n or _ones.dtype != dtype or _ones.device != device:
        _ones = torch.ones(max(n, 4 * 512 * cols), dtype=dtype, device=device)
    return _ones[:n].view(batch, length, cols)


def packed_dropout(x, p, batch, length, index, out_dtype):
    drawn = torch.nn.functional.dropout(
        ones_like_draw(batch, length, x.shape[2], x.dtype, x.device), p, True)
    return PackedDropout.apply(x, drawn, index, 1.0 / (1.0 - p), out_dtype)
