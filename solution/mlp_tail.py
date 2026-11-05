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
