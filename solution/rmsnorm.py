"""Fused RMSNorm for Qwen2, forward and backward in Triton.

Qwen2RMSNorm runs five PyTorch ops per call (cast, pow, mean, rsqrt, two muls) and autograd keeps
several full fp32 copies of the row for backward. In this task the residual stream is fp32
(prepare_model_for_kbit_training upcasts norms and embeddings), so each copy is 2048 x 1536 x 4 B.
Here one program normalizes one row in registers; backward keeps only the input and one fp32
scalar per row (1/rms) and recomputes the normalized row.

Numerics follow transformers' Qwen2RMSNorm exactly: the normalized row is rounded to the input
dtype before it is multiplied by the weight, and the output dtype is the promotion of the two.
Idea: Triton's layer-norm tutorial (https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html);
the code is ours.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_fwd_kernel(x_ptr, w_ptr, y_ptr, rstd_ptr, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x_raw = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    x = x_raw.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    # HF casts the normalized row back to the input dtype before the weight multiply.
    x_hat = (x * rstd).to(x_raw.dtype).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * n_cols + cols, (w * x_hat).to(y_ptr.dtype.element_ty), mask=mask)
    tl.store(rstd_ptr + row, rstd)


@triton.jit
def rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    dy = tl.load(dy_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row)
    # d/dx of w * x * rstd(x):  rstd * (g - x_hat * mean(g * x_hat)),  g = w * dy
    g = w * dy
    x_hat = x * rstd
    proj = tl.sum(g * x_hat, axis=0) / n_cols
    dx = rstd * (g - x_hat * proj)
    tl.store(dx_ptr + row * n_cols + cols, dx.to(dx_ptr.dtype.element_ty), mask=mask)


def _launch_config(n_cols):
    block = triton.next_power_of_2(n_cols)
    return block, (8 if block >= 2048 else 4)


class RMSNormFunction(torch.autograd.Function):
    """y = weight * (x / rms(x)).to(x.dtype), one launch each way."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        rows, cols = x2.shape
        out_dtype = torch.result_type(weight, x)
        y = torch.empty((rows, cols), dtype=out_dtype, device=x.device)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block, warps = _launch_config(cols)
        rmsnorm_fwd_kernel[(rows,)](x2, weight, y, rstd, cols, eps, BLOCK=block, num_warps=warps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = shape
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        rows, cols = x2.shape
        dy2 = dy.reshape(rows, cols).contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty_like(x2)
            block, warps = _launch_config(cols)
            rmsnorm_bwd_kernel[(rows,)](dy2, x2, weight, rstd, dx, cols, BLOCK=block,
                                        num_warps=warps)
            dx = dx.view(ctx.shape)
        dw = None
        if ctx.needs_input_grad[1]:  # norms are frozen in this task; kept for completeness
            x_hat = (x2.float() * rstd[:, None]).to(x2.dtype).float()
            dw = (dy2.float() * x_hat).sum(0).to(weight.dtype)
        return dx, dw, None


def rmsnorm(x, weight, eps):
    return RMSNormFunction.apply(x, weight, eps)


_SUPPORTED = (torch.float16, torch.float32)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel; anything unusual (CPU, other dtypes,
    a weight that does not match the last dim) takes the original path."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls, "_mindx_patched", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if (not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda
                or not isinstance(weight, torch.Tensor) or not weight.is_cuda
                or hidden_states.dtype not in _SUPPORTED or weight.dtype not in _SUPPORTED
                or weight.dim() != 1 or weight.numel() != hidden_states.shape[-1]
                or hidden_states.numel() == 0):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, float(self.variance_epsilon))

    cls.forward = forward
    cls._mindx_patched = True
