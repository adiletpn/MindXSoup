"""LoRA dropout on the packed (unpadded) tokens in one launch each way.

To keep the base run's random numbers, unpad.py draws the dropout mask exactly as the base does:
`F.dropout` on a padded [B, T, hidden] tensor (the draw depends only on its shape, never on its
values, so a persistent buffer of ones serves). This kernel then does in one pass what took five
PyTorch ops (gather of the real-token rows of the mask, compare, scale, masked fill, and the
fp16 cast that autocast applies right after for lora_A): out = x * scale where the mask kept the
element, else 0, written in the dtype lora_A consumes. Backward is one launch too:
dx = grad * scale where kept, else 0, in the input's dtype. Same values as torch's dropout.
"""

import torch
import triton
import triton.language as tl

BACKWARD_BLOCK = 1024


@triton.jit
def lora_dropout_fwd_kernel(x_ptr, drawn_ptr, index_ptr, out_ptr, keep_ptr, scale,
                            COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source = tl.load(index_ptr + row)  # this token's row in the padded [B * T, hidden] draw
    cols = tl.arange(0, BLOCK)
    valid = cols < COLS
    keep = tl.load(drawn_ptr + source * COLS + cols, mask=valid, other=0.0) != 0.0
    x = tl.load(x_ptr + row * COLS + cols, mask=valid, other=0.0)
    out = tl.where(keep, x * scale, 0.0)
    tl.store(out_ptr + row * COLS + cols, out.to(out_ptr.dtype.element_ty), mask=valid)
    tl.store(keep_ptr + row * COLS + cols, keep.to(tl.int8), mask=valid)


# n_elements changes with every batch; not specialized, so it never recompiles
@triton.jit(do_not_specialize=["n_elements"])
def lora_dropout_bwd_kernel(grad_ptr, keep_ptr, dx_ptr, n_elements, scale, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    grad = tl.load(grad_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + offsets, mask=valid, other=0)
    dx = tl.where(keep != 0, grad * scale, 0.0)
    tl.store(dx_ptr + offsets, dx.to(dx_ptr.dtype.element_ty), mask=valid)


class PackedDropout(torch.autograd.Function):
    """x: [1, N, hidden] packed tokens; drawn: the base's [B, T, hidden] dropout of ones;
    index: [N] rows of the real tokens in the flattened [B * T]."""

    @staticmethod
    def forward(ctx, x, drawn, index, scale, out_dtype):
        x = x.contiguous()
        rows, cols = x.shape[1], x.shape[2]
        out = torch.empty((1, rows, cols), dtype=out_dtype, device=x.device)
        keep = torch.empty((rows, cols), dtype=torch.int8, device=x.device)
        block = triton.next_power_of_2(cols)
        lora_dropout_fwd_kernel[(rows,)](x, drawn, index, out, keep, scale, COLS=cols,
                                         BLOCK=block, num_warps=8 if block >= 2048 else 4)
        ctx.save_for_backward(keep)
        ctx.scale, ctx.x_dtype = scale, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        (keep,) = ctx.saved_tensors
        grad = grad.contiguous()
        dx = torch.empty(grad.shape, dtype=ctx.x_dtype, device=grad.device)
        n = grad.numel()
        lora_dropout_bwd_kernel[(triton.cdiv(n, BACKWARD_BLOCK),)](
            grad, keep, dx, n, ctx.scale, BLOCK=BACKWARD_BLOCK, num_warps=4)
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
