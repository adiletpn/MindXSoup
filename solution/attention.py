"""Qwen2 attention with fp16 q/k/v: one input cast, one kernel for LoRA sum + RoPE + cast.

What the base run does per layer: each 4-bit projection (bitsandbytes Linear4bit) casts the fp32
input to fp16, multiplies in fp16 and casts the result back to fp32; peft clones that result and
adds the LoRA output (fp16 under autocast) in fp32; RoPE rotates fp32 q and k; then SDPA, under
autocast, casts q, k and v to fp16 again. Every one of those fp32 tensors holds an fp16 matmul
result, and backward walks the same chain of casts in reverse.

Here the input is cast to fp16 once for the three base projections, which then return fp16. The
LoRA branches run exactly as peft runs them (dropout on the same fp32 input, in the same order,
so the masks do not change). One Triton launch then takes, for each position, the fp16 base and
LoRA outputs, adds them in fp32, rotates q and k in fp32 and writes fp16 q, k, v: the same
arithmetic as the base, which also added in fp32, rotated in fp32 and rounded to fp16 before SDPA.
Backward is the same kernel: it rotates dq and dk back in fp32 and writes fp16 grads in the
projections' [B, T, heads * D] layout, which the base also rounded to fp16 before the matmuls.
"""

import math

import torch
import triton
import triton.language as tl

from . import direct
from .mlp import _is_fp16_4bit


@triton.jit
def _qkv_heads(a_ptr, b_ptr, y_ptr, bi, t, sab, sah, sat, sbb, sbh, sbt, syb, syh, syt,
               c1, c2, p, r, b_scale, N_HEADS: tl.constexpr, BLOCK_H: tl.constexpr, HALF: tl.constexpr,
               ADD: tl.constexpr, ROTATE: tl.constexpr):
    # x = a (+ b * b_scale), then y1 = x1 * c1 + x2 * p,  y2 = x2 * c2 + x1 * r  (or y = x), in fp32
    heads = tl.arange(0, BLOCK_H)[:, None]
    cols = tl.arange(0, HALF)[None, :]
    mask = heads < N_HEADS
    a = a_ptr + bi * sab + t * sat + heads * sah + cols
    x1 = tl.load(a, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(a + HALF, mask=mask, other=0.0).to(tl.float32)
    if ADD:
        b = b_ptr + bi * sbb + t * sbt + heads * sbh + cols
        x1 += tl.load(b, mask=mask, other=0.0).to(tl.float32) * b_scale
        x2 += tl.load(b + HALF, mask=mask, other=0.0).to(tl.float32) * b_scale
    if ROTATE:
        y1 = x1 * c1 + x2 * p
        y2 = x2 * c2 + x1 * r
    else:
        y1 = x1
        y2 = x2
    y = y_ptr + bi * syb + t * syt + heads * syh + cols
    tl.store(y, y1.to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(y + HALF, y2.to(y_ptr.dtype.element_ty), mask=mask)


# One program per position (b, t): q and v heads as base (+ LoRA), q and k rotated, v copied.
# The rotation is the one of solution/rope.py: forward sign = +1, sin halves (1, 2); backward
# sign = -1, sin halves (2, 1). n_tokens and the scalars change between calls and are not
# specialized; every stride is a multiple of 16 for any T, so new lengths never recompile.
@triton.jit(do_not_specialize=["n_tokens", "sin_a_off", "sin_b_off", "sign", "q_scale",
                               "v_scale"])
def qkv_rope_kernel(qa_ptr, qb_ptr, k_ptr, va_ptr, vb_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr,
                    vo_ptr, n_tokens, sin_a_off, sin_b_off, sign, q_scale, v_scale,
                    sqab, sqah, sqat, sqbb, sqbh, sqbt, skb, skh, skt,
                    svab, svah, svat, svbb, svbh, svbt,
                    sqob, sqoh, sqot, skob, skoh, skot, svob, svoh, svot, scb, sct,
                    N_Q: tl.constexpr, N_KV: tl.constexpr, BLOCK_Q: tl.constexpr,
                    BLOCK_KV: tl.constexpr, HALF: tl.constexpr, ADD_Q: tl.constexpr,
                    ADD_V: tl.constexpr):
    pid = tl.program_id(0)
    bi = pid // n_tokens
    t = pid % n_tokens
    cols = tl.arange(0, HALF)[None, :]
    row = bi * scb + t * sct + cols
    c1 = tl.load(cos_ptr + row).to(tl.float32)
    c2 = tl.load(cos_ptr + row + HALF).to(tl.float32)
    p = -sign * tl.load(sin_ptr + row + sin_a_off).to(tl.float32)
    r = sign * tl.load(sin_ptr + row + sin_b_off).to(tl.float32)
    _qkv_heads(qa_ptr, qb_ptr, qo_ptr, bi, t, sqab, sqah, sqat, sqbb, sqbh, sqbt,
               sqob, sqoh, sqot, c1, c2, p, r, q_scale, N_Q, BLOCK_Q, HALF, ADD_Q, True)
    _qkv_heads(k_ptr, k_ptr, ko_ptr, bi, t, skb, skh, skt, skb, skh, skt,
               skob, skoh, skot, c1, c2, p, r, 1.0, N_KV, BLOCK_KV, HALF, False, True)
    _qkv_heads(va_ptr, vb_ptr, vo_ptr, bi, t, svab, svah, svat, svbb, svbh, svbt,
               svob, svoh, svot, c1, c2, p, r, v_scale, N_KV, BLOCK_KV, HALF, ADD_V, False)


def _heads(x, n_heads):
    """[B, T, heads * D] or [B, heads, T, D] -> [B, heads, T, D] view (no copy)."""
    if x.dim() == 3:
        batch, tokens, _ = x.shape
        x = x.view(batch, tokens, n_heads, -1).transpose(1, 2)
    return x if x.stride(-1) == 1 else x.contiguous()


def _bhtd_strides(x):
    return x.stride(0), x.stride(1), x.stride(2)


def _launch(backward, q_a, q_b, k, v_a, v_b, cos, sin, outs, q_scale=1.0, v_scale=1.0):
    """q_a/k/v_a (+ q_b, v_b): [B, heads, T, D] views; outs: q, k, v destinations, same shapes."""
    batch, n_q, n_tokens, dim = q_a.shape
    n_kv = k.shape[1]
    half = dim // 2
    cos = cos.expand(batch, n_tokens, dim)
    sin = sin.expand(batch, n_tokens, dim)
    if cos.stride() != sin.stride() or cos.stride(-1) != 1:
        cos, sin = cos.contiguous(), sin.contiguous()
    sin_a_off, sin_b_off, sign = (half, 0, -1.0) if backward else (0, half, 1.0)
    q_o, k_o, v_o = outs
    qkv_rope_kernel[(batch * n_tokens,)](
        q_a, q_a if q_b is None else q_b, k, v_a, v_a if v_b is None else v_b, cos, sin,
        q_o, k_o, v_o, n_tokens, sin_a_off, sin_b_off, sign, q_scale, v_scale,
        *_bhtd_strides(q_a), *_bhtd_strides(q_a if q_b is None else q_b), *_bhtd_strides(k),
        *_bhtd_strides(v_a), *_bhtd_strides(v_a if v_b is None else v_b),
        *_bhtd_strides(q_o), *_bhtd_strides(k_o), *_bhtd_strides(v_o),
        cos.stride(0), cos.stride(1),
        N_Q=n_q, N_KV=n_kv, BLOCK_Q=triton.next_power_of_2(n_q),
        BLOCK_KV=triton.next_power_of_2(n_kv), HALF=half, ADD_Q=q_b is not None,
        ADD_V=v_b is not None, num_warps=4)


def _empty_bhtd(batch, n_heads, n_tokens, dim, dtype, device):
    """[B, heads, T, D] tensor stored as [B, T, heads, D], like a projection viewed per head."""
    return torch.empty_strided((batch, n_heads, n_tokens, dim),
                               (n_tokens * n_heads * dim, dim, n_heads * dim, 1),
                               dtype=dtype, device=device)


class QKVRoPEFunction(torch.autograd.Function):
    """fp16 (q_base + q_lora) rotated, k rotated, (v_base + v_lora); inputs are the projections'
    [B, T, heads * D] outputs (LoRA parts may be None, and come times q_scale / v_scale: a power
    of two, so the product is exact either way). Saves only cos and sin."""

    @staticmethod
    def forward(ctx, q_base, q_lora, k, v_base, v_lora, cos, sin, n_q, n_kv, q_scale=1.0,
                v_scale=1.0):
        q_a, k_h, v_a = _heads(q_base, n_q), _heads(k, n_kv), _heads(v_base, n_kv)
        q_b = None if q_lora is None else _heads(q_lora, n_q)
        v_b = None if v_lora is None else _heads(v_lora, n_kv)
        batch, _, n_tokens, dim = q_a.shape
        outs = tuple(_empty_bhtd(batch, n, n_tokens, dim, torch.float16, q_a.device)
                     for n in (n_q, n_kv, n_kv))
        _launch(False, q_a, q_b, k_h, v_a, v_b, cos, sin, outs, q_scale, v_scale)
        ctx.save_for_backward(cos, sin)
        ctx.scales = (1.0, q_scale, 1.0, 1.0, v_scale)
        ctx.meta = (n_q, n_kv, [None if x is None else x.dtype
                                for x in (q_base, q_lora, k, v_base, v_lora)])
        return outs

    @staticmethod
    def backward(ctx, dq, dk, dv):
        cos, sin = ctx.saved_tensors
        n_q, n_kv, dtypes = ctx.meta
        dq, dk, dv = _heads(dq, n_q), _heads(dk, n_kv), _heads(dv, n_kv)
        batch, _, n_tokens, dim = dq.shape
        grads = tuple(_empty_bhtd(batch, n, n_tokens, dim, torch.float16, dq.device)
                      for n in (n_q, n_kv, n_kv))
        _launch(True, dq, None, dk, dv, None, cos, sin, grads)
        # back to the projections' [B, T, heads * D] shape; the storage already is that layout
        g_q, g_k, g_v = (g.transpose(1, 2).reshape(batch, n_tokens, -1) for g in grads)
        out = []
        for i, (grad, dtype) in enumerate(zip((g_q, g_q, g_k, g_v, g_v), dtypes)):
            if dtype is None or not ctx.needs_input_grad[i]:
                out.append(None)
                continue
            grad = grad.to(dtype)
            out.append(grad * ctx.scales[i] if ctx.scales[i] != 1.0 else grad)
        return (*out, None, None, None, None, None, None)


def qkv_rope(q_base, q_lora, k, v_base, v_lora, cos, sin, n_q, n_kv, q_scale=1.0, v_scale=1.0):
    return direct.run(QKVRoPEFunction, q_base, q_lora, k, v_base, v_lora, cos, sin, n_q, n_kv,
                      q_scale, v_scale)


def _split_projection(proj):
    """(fp16 4-bit base layer, LoRA function or None, its scale) for a bitsandbytes Linear4bit
    with fp16 compute, bare or wrapped by peft's LoRA Linear4bit with one plain adapter; else
    None. A power-of-two LoRA scaling is left to qkv_rope_kernel (x * 2^k is exact in fp16 and
    fp32 alike, so the sum is the same bits) and saves one launch each way."""
    lora_a = getattr(proj, "lora_A", None)
    if lora_a is None:  # checked first: peft's wrapper also exposes the base layer's weight
        return (proj, None, 1.0) if _is_fp16_4bit(proj) else None
    base = getattr(proj, "base_layer", None)
    if not _is_fp16_4bit(base):
        return None
    if getattr(proj, "disable_adapters", True) or getattr(proj, "merged", True):
        return None
    active = list(getattr(proj, "active_adapters", []))
    if len(active) != 1 or active[0] not in lora_a.keys() \
            or active[0] in getattr(proj, "lora_variant", {}):
        return None
    name = active[0]

    scaling = proj.scaling[name]
    exact = isinstance(scaling, (int, float)) and scaling > 0 and \
        math.frexp(scaling)[0] == 0.5 and 2.0 ** -8 <= scaling <= 2.0 ** 8

    def lora(x):  # peft's vanilla LoRA branch under autocast (no dtype conversion there)
        out = proj.lora_B[name](proj.lora_A[name](proj.lora_dropout[name](x)))
        return out if exact else out * scaling

    return base, lora, (float(scaling) if exact else 1.0)


def _fast_path_ok(module, hidden_states, position_embeddings, past_key_values, kwargs):
    if (not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda
            or hidden_states.dtype not in (torch.float32, torch.float16) or hidden_states.dim() != 3
            or past_key_values is not None or kwargs.get("output_attentions", False)):
        return False
    if not (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16):
        return False  # without fp16 autocast the base runs SDPA in fp32: keep it
    config = getattr(module, "config", None)
    if getattr(config, "_attn_implementation", None) != "sdpa" or \
            getattr(module, "head_dim", None) != 128:
        return False
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        return False
    cos, sin = position_embeddings
    batch, n_tokens, _ = hidden_states.shape
    return all(isinstance(x, torch.Tensor) and x.is_cuda and x.dim() == 3
               and x.dtype in (torch.float32, torch.float16) and not x.requires_grad
               and x.shape[0] in (1, batch) and tuple(x.shape[1:]) == (n_tokens, 128)
               for x in (cos, sin))


def patch_qwen2_attention_fp16():
    """Send Qwen2Attention.forward through the fp16 path; anything else (CPU, fake modules in
    Soup's tests, a KV cache, other attention backends, projections that are not fp16 4-bit,
    no fp16 autocast) takes the original forward."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2Attention
    if getattr(cls, "_mindx_fp16", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        parts = None
        if _fast_path_ok(self, hidden_states, position_embeddings, past_key_values, kwargs):
            parts = [_split_projection(getattr(self, name, None))
                     for name in ("q_proj", "k_proj", "v_proj")]
        if parts is None or any(p is None for p in parts) or parts[1][1] is not None:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kwargs)
        (q_base_layer, q_lora, q_scale), (k_layer, _, _), (v_base_layer, v_lora, v_scale) = parts
        input_shape = hidden_states.shape[:-1]
        x16 = hidden_states.to(torch.float16)
        # same order as the base: q (base, LoRA dropout), k, v (base, LoRA dropout)
        q_base = q_base_layer(x16)
        q_add = None if q_lora is None else q_lora(hidden_states)
        k = k_layer(x16)
        v_base = v_base_layer(x16)
        v_add = None if v_lora is None else v_lora(hidden_states)
        n_q, n_kv = q_base.shape[-1] // self.head_dim, k.shape[-1] // self.head_dim
        cos, sin = position_embeddings
        query, key, value = qkv_rope(q_base, q_add, k, v_base, v_add, cos, sin, n_q, n_kv,
                                     q_scale, v_scale)

        attention_interface = modeling_qwen2.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen2.eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query, key, value, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights

    forward._mindx_original = original
    cls.forward = forward
    cls._mindx_fp16 = True


def compile_qkv_rope():
    """Warm-up job (solution/warmup.py): compile the forward and backward variants on tiny
    tensors with the training call's dtypes and layouts. Zeros only, no random draws."""
    tokens, n_q, n_kv, dim = 16, 12, 2, 128
    zeros = lambda n: torch.zeros(1, tokens, n * dim, dtype=torch.float16, device="cuda",  # noqa: E731
                                  requires_grad=True)
    cos = torch.ones(1, tokens, dim, device="cuda")
    sin = torch.zeros(1, tokens, dim, device="cuda")
    outs = qkv_rope(zeros(n_q), zeros(n_q), zeros(n_kv), zeros(n_kv), zeros(n_kv), cos, sin,
                    n_q, n_kv)
    torch.autograd.backward(outs, [torch.zeros_like(o) for o in outs])
