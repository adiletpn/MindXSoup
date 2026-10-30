"""Keep only each decoder layer's input for backward and run the layer again there.

Without it every layer keeps its activations (attention inputs and outputs, the MLP's fp16
gate/up, norms) from the forward until its backward: that is most of the peak. With it a layer
keeps one [N, hidden] fp32 input; backward reruns that layer's forward first (one extra forward
per step) and frees its activations again before the next layer. torch.utils.checkpoint restores
the random generator state for the rerun, so LoRA dropout draws the same masks; a packed batch
(unpad.py) reruns with its row layout. This is recomputation in our own patch: Soup's
gradient_checkpointing switch stays off.
"""

import torch
from torch.utils.checkpoint import checkpoint

from . import unpad

enabled = False  # set by patch_qwen2_layer_recompute()


def patch_qwen2_layer_recompute():
    from transformers.models.qwen2 import modeling_qwen2

    global enabled
    enabled = True
    cls = modeling_qwen2.Qwen2DecoderLayer
    if getattr(cls, "_mindx_recompute", False):
        return
    original = cls.forward

    def forward(self, hidden_states, *args, **kwargs):
        if not (enabled and self.training and torch.is_grad_enabled() and isinstance(hidden_states, torch.Tensor)
                and hidden_states.is_cuda and kwargs.get("past_key_values") is None):
            return original(self, hidden_states, *args, **kwargs)
        rows = unpad._rows

        def run(hidden):
            if rows is None:
                return original(self, hidden, *args, **kwargs)
            with unpad.packed(*rows):
                return original(self, hidden, *args, **kwargs)

        return checkpoint(run, hidden_states, use_reentrant=False, preserve_rng_state=True)

    cls.forward = forward
    cls._mindx_recompute = True
