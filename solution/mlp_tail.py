"""The MLP in a rerun layer computes only what its backward reads.

When recompute.py reruns a layer inside backward, nothing reads the values of the MLP's output: a
linear layer's backward needs its weight and the incoming gradient, not its output, and the
SwiGLU backward needs gate and up, not silu(gate) * up. (The real output came from the first,
no-grad pass.) So the rerun computes gate and up and returns a placeholder of the output's shape:
no down_proj matmul, no unpack of its weight, no SwiGLU forward. Backward does what autograd did:
grad @ W_down, the SwiGLU adjoint over gate and up, and dx = d(up) @ W_up + d(gate) @ W_gate.
The gradients are the same bits.

Only for weights streamed by stream.py (their guards already hold), inside a rerun.
"""

import torch

from . import linear4bit, stream
from .mlp_recompute import _in_backward
from .silu_mul import COLS_PER_PROGRAM, WARPS, _grid, silu_mul_kernel


def _adjoint(layer, grad):
    return torch.matmul(grad, linear4bit.streamed_weight(layer, layer.weight.quant_state)
                        .to(grad.dtype))


class MlpTailRerun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mlp):
        x16 = x.to(torch.float16)
        ctx.mlp, ctx.x_dtype = mlp, x.dtype
        ctx.gate = mlp.gate_proj(x16).contiguous()
        ctx.up = mlp.up_proj(x16).contiguous()
        return torch.zeros((), dtype=x.dtype, device=x.device).expand(x.shape)

    @staticmethod
    def backward(ctx, grad):
        gate, up, mlp = ctx.gate, ctx.up, ctx.mlp
        ctx.gate = ctx.up = None
        d = _adjoint(mlp.down_proj, grad.to(torch.float16)).contiguous()
        grid, n_cols = _grid(gate)
        silu_mul_kernel[grid](d, gate, up, gate, n_cols, COLS=COLS_PER_PROGRAM, BACKWARD=True,
                              num_warps=WARPS)  # writes d(gate) over gate and d(up) over up
        del d
        dx = _adjoint(mlp.up_proj, up)
        del up
        dx += _adjoint(mlp.gate_proj, gate)
        return dx.to(ctx.x_dtype), None


def _tail_ok(mlp, x):
    if not (isinstance(x, torch.Tensor) and x.is_cuda and x.dtype == torch.float32 and x.numel() > 0
            and getattr(mlp, "training", False) and torch.is_grad_enabled() and _in_backward()
            and stream.active()):
        return False
    if getattr(getattr(mlp, "config", None), "hidden_act", "silu") != "silu":
        return False
    return all(getattr(layer, "bias", 0) is None and stream.offloaded(layer)
               and stream.streamer().owns(layer)
               for layer in (getattr(mlp, name, None)
                             for name in ("gate_proj", "up_proj", "down_proj")))


def patch_mlp_tail():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_mindx_tail", False):
        return
    previous = cls.forward

    def forward(self, x):
        return MlpTailRerun.apply(x, self) if _tail_ok(self, x) else previous(self, x)

    cls.forward = forward
    cls._mindx_tail = True
