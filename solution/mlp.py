"""Qwen2 MLP on fp16 activations end to end.

In this task the residual stream is fp32, so every 4-bit projection (bitsandbytes Linear4bit)
casts its fp32 input to fp16, multiplies in fp16 and casts the result back to fp32. The MLP then
keeps gate and up as fp32 tensors for backward: 2 x 2048 x 8960 x 4 B = 147 MB per layer, 4.1 GB
over 28 layers, although every one of those values came out of an fp16 matmul.

Here the input is cast to fp16 once, the projections return fp16 (Linear4bit returns its input
dtype), the SwiGLU kernel keeps fp16 gate/up for backward, and only the MLP output goes back to
fp32 for the residual add. The values are the same as in the base run: the matmuls ran in fp16
there too, and the base rounds silu(gate) * up to fp16 before down_proj just as the kernel does.
Saved activations per layer drop from 147 MB to 73 MB, and two fp32 round trips disappear.
"""

import torch

from .swiglu import swiglu


def _is_fp16_4bit(linear):
    weight = getattr(linear, "weight", None)
    return type(weight).__name__ == "Params4bit" and \
        getattr(linear, "compute_dtype", None) == torch.float16


def patch_qwen2_mlp_fp16():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_mindx_fp16", False):
        return
    previous = cls.forward  # the example's SwiGLU forward when it is enabled

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float32
                or not all(_is_fp16_4bit(getattr(self, name, None))
                           for name in ("gate_proj", "up_proj", "down_proj"))):
            return previous(self, x)
        x16 = x.to(torch.float16)
        hidden = swiglu(self.gate_proj(x16), self.up_proj(x16))
        return self.down_proj(hidden).to(x.dtype)

    cls.forward = forward
    cls._mindx_fp16 = True
