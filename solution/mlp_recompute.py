"""Keep the MLP's fp16 input instead of gate and up for backward, and redo the two matmuls there.

Measured at the solution's peak (tools/mem_peak.py, main 1d3b11e): the largest single owner of
GPU memory is gate and up of the decoder layers that recompute.py does not rerun, kept by the
SwiGLU kernel for backward: 2 x [N, 8960] fp16 per layer, 48.8 MiB at N = 1 430 tokens, 246 MiB
over the five such layers of the longest batch. Their input is [N, 1536] fp16, 4.2 MiB.

Here, in a layer that is not rerun, gate_proj, up_proj and SwiGLU run inside a non-reentrant
torch.utils.checkpoint: the forward keeps only the fp16 input; backward redoes the two 4-bit
matmuls (the same fp16 values: same bytes, same kernels) before SwiGLU's backward. Nothing in
there draws random numbers. Layers that recompute.py reruns are left alone: their forward runs
without grad and their rerun happens inside backward, where keeping gate/up costs nothing extra.
Recomputation in our own code (organizers: allowed, its time counts).
"""

import torch
from torch.utils.checkpoint import checkpoint

from .mlp import _is_fp16_4bit
from .swiglu import swiglu

patched_forward = previous_forward = None  # set by patch_qwen2_mlp_gateup_recompute()


def _in_backward():
    graph_task_id = getattr(torch._C, "_current_graph_task_id", None)
    return graph_task_id is not None and graph_task_id() != -1


def patch_qwen2_mlp_gateup_recompute():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_mindx_gateup_recompute", False):
        return
    previous = cls.forward  # mlp.py's fp16 forward when it is enabled

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (not self.training or not torch.is_grad_enabled() or _in_backward()
                or act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float32
                or not all(_is_fp16_4bit(getattr(self, name, None))
                           for name in ("gate_proj", "up_proj", "down_proj"))):
            return previous(self, x)
        x16 = x.to(torch.float16)
        hidden = checkpoint(lambda t: swiglu(self.gate_proj(t), self.up_proj(t)), x16,
                            use_reentrant=False, preserve_rng_state=False)
        return self.down_proj(hidden).to(x.dtype)

    global patched_forward, previous_forward
    patched_forward, previous_forward = forward, previous
    cls.forward = forward
    cls._mindx_gateup_recompute = True
