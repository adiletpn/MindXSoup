"""Rotary position embedding (RoPE) for Qwen2 q and k, forward and backward in Triton.

transformers' apply_rotary_pos_emb runs about a dozen PyTorch ops per call (two slices, a negation
and a cat for rotate_half, four muls and two adds for q and k) and autograd repeats them in
backward. In this task q and k are fp32, [B, heads, T, 128] views of the projection output with
the memory laid out as [B, T, heads, 128]. Here one program takes one position (b, t), reads its
cos/sin row once and rotates all 12 q heads and both k heads in registers, reading every element
of q and k once and writing it once. Inputs are read through their strides (no .contiguous()
copy), and the outputs keep the same layout.

Backward is the same rotation by the opposite angle, so it needs only cos and sin, which the model
keeps anyway; q and k are not saved.

    forward:   y1 = x1 * cos1 - x2 * sin1        y2 = x2 * cos2 + x1 * sin2
    backward:  dx1 = dy1 * cos1 + dy2 * sin2     dx2 = dy2 * cos2 - dy1 * sin1

(x1, x2 are the two halves of a head, cos1/cos2 and sin1/sin2 the halves of the cos/sin row.)
Idea: RoPE from RoFormer, Su et al. 2021 (https://arxiv.org/abs/2104.09864); the code is ours.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_rotate_heads(x_ptr, y_ptr, b, t, sxb, sxh, sxt, syb, syh, syt, c1, c2, p, r,
                       N_HEADS: tl.constexpr, BLOCK_H: tl.constexpr, HALF: tl.constexpr):
    # y1 = x1 * c1 + x2 * p,  y2 = x2 * c2 + x1 * r  for every head of one position
    heads = tl.arange(0, BLOCK_H)[:, None]
    cols = tl.arange(0, HALF)[None, :]
    mask = heads < N_HEADS
    x = x_ptr + b * sxb + t * sxt + heads * sxh + cols
    x1 = tl.load(x, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x + HALF, mask=mask, other=0.0).to(tl.float32)
    y = y_ptr + b * syb + t * syt + heads * syh + cols
    tl.store(y, (x1 * c1 + x2 * p).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(y + HALF, (x2 * c2 + x1 * r).to(y_ptr.dtype.element_ty), mask=mask)


# Forward and backward are one kernel, so the run pays for one Triton compile instead of two
# (about 2.5 s each on Kaggle's CPU, more than the kernel saves over 64 steps). Both directions are
#     y1 = x1 * cos1 - sign * x2 * sin_a        y2 = x2 * cos2 + sign * x1 * sin_b
# forward:  sign = +1, sin_a = sin1, sin_b = sin2
# backward: sign = -1, sin_a = sin2, sin_b = sin1   (dx1 = dy1*cos1 + dy2*sin2, dx2 = dy2*cos2 - dy1*sin1)
# n_tokens changes from batch to batch; without do_not_specialize Triton would compile a new
# variant whenever its divisibility by 16 changes. The batch strides change too, but they are
# always multiples of 16 (T * heads * 128 and T * 128), so they never trigger a recompile, and
# keeping them specialized lets Triton vectorize the loads.
@triton.jit(do_not_specialize=["n_tokens", "sin_a_off", "sin_b_off", "sign"])
def rope_kernel(q_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr, n_tokens, sin_a_off, sin_b_off,
                sign, sqb, sqh, sqt, skb, skh, skt, sqob, sqoh, sqot, skob, skoh, skot, scb, sct,
                N_Q: tl.constexpr, N_K: tl.constexpr, BLOCK_Q: tl.constexpr,
                BLOCK_K: tl.constexpr, HALF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // n_tokens
    t = pid % n_tokens
    cols = tl.arange(0, HALF)[None, :]
    row = b * scb + t * sct + cols
    c1 = tl.load(cos_ptr + row).to(tl.float32)
    c2 = tl.load(cos_ptr + row + HALF).to(tl.float32)
    p = -sign * tl.load(sin_ptr + row + sin_a_off).to(tl.float32)
    r = sign * tl.load(sin_ptr + row + sin_b_off).to(tl.float32)
    _rope_rotate_heads(q_ptr, qo_ptr, b, t, sqb, sqh, sqt, sqob, sqoh, sqot, c1, c2, p, r,
                       N_Q, BLOCK_Q, HALF)
    _rope_rotate_heads(k_ptr, ko_ptr, b, t, skb, skh, skt, skob, skoh, skot, c1, c2, p, r,
                       N_K, BLOCK_K, HALF)


def _last_dim_dense(x):
    return x if x.stride(-1) == 1 else x.contiguous()


def _rope_launch(backward, q, k, cos, sin, q_dtype, k_dtype):
    """Rotate (forward) or rotate back (backward) [B, heads, T, D] q and k (any strides) and [B or 1, T, D]
    cos/sin; the outputs get the layout of q and k."""
    q, k = _last_dim_dense(q), _last_dim_dense(k)
    batch, n_q, n_tokens, dim = q.shape
    n_k = k.shape[1]
    q_out = torch.empty_like(q, dtype=q_dtype)  # keeps q's strides: [B, T, heads, D] in memory
    k_out = torch.empty_like(k, dtype=k_dtype)
    cos = cos.expand(batch, n_tokens, dim)  # batch stride 0 when position_ids had batch 1
    sin = sin.expand(batch, n_tokens, dim)
    if cos.stride() != sin.stride() or cos.stride(-1) != 1:
        cos, sin = cos.contiguous(), sin.contiguous()
    half = dim // 2
    sin_a_off, sin_b_off, sign = (half, 0, -1.0) if backward else (0, half, 1.0)
    rope_kernel[(batch * n_tokens,)](
        q, k, cos, sin, q_out, k_out, n_tokens, sin_a_off, sin_b_off, sign,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), cos.stride(0), cos.stride(1),
        N_Q=n_q, N_K=n_k, BLOCK_Q=triton.next_power_of_2(n_q),
        BLOCK_K=triton.next_power_of_2(n_k), HALF=half, num_warps=4)
    return q_out, k_out


class RoPEFunction(torch.autograd.Function):
    """(q * cos + rotate_half(q) * sin, same for k), one launch each way; saves only cos, sin."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q_out, k_out = _rope_launch(False, q, k, cos, sin,
                                    torch.result_type(q, cos), torch.result_type(k, cos))
        ctx.save_for_backward(cos, sin)
        ctx.dtypes = (q.dtype, k.dtype)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        dq_in, dk_in = _rope_launch(True, dq, dk, cos, sin, *ctx.dtypes)
        return (dq_in if ctx.needs_input_grad[0] else None,
                dk_in if ctx.needs_input_grad[1] else None, None, None)


def apply_rope(q, k, cos, sin, unsqueeze_dim=1):
    """Same contract as transformers' apply_rotary_pos_emb for [B, heads, T, D] q/k."""
    return RoPEFunction.apply(q, k, cos, sin)


_SUPPORTED = (torch.float16, torch.float32)


def _kernel_can_take(q, k, cos, sin, unsqueeze_dim):
    tensors = (q, k, cos, sin)
    if not all(isinstance(x, torch.Tensor) and x.is_cuda and x.dtype in _SUPPORTED
               for x in tensors):
        return False
    if unsqueeze_dim != 1 or q.dim() != 4 or k.dim() != 4 or cos.dim() != 3:
        return False
    batch, _, n_tokens, dim = q.shape
    return (dim == 128 and k.shape[0] == batch and k.shape[2:] == q.shape[2:]
            and cos.shape == sin.shape and cos.shape[0] in (1, batch)
            and tuple(cos.shape[1:]) == (n_tokens, dim) and cos.dtype == sin.dtype
            and not cos.requires_grad and not sin.requires_grad and q.numel() > 0)


def patch_rope():
    """Send modeling_qwen2.apply_rotary_pos_emb (called by name from Qwen2Attention.forward)
    through the kernel; anything unusual (CPU, other dtypes or shapes, trainable cos/sin) takes
    the original path."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.apply_rotary_pos_emb
    if getattr(original, "_mindx_patched", False):
        return

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if not _kernel_can_take(q, k, cos, sin, unsqueeze_dim):
            return original(q, k, cos, sin, unsqueeze_dim)
        return apply_rope(q, k, cos, sin)

    apply_rotary_pos_emb._mindx_patched = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
