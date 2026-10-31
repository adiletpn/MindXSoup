"""Keep only a decoder layer's input for backward and run the layer again there, in as many
layers as a batch needs to stay within a fixed activation budget.

Without it every layer keeps its activations (attention inputs and outputs, the MLP's fp16
gate/up, norms) from the forward until its backward: that is most of the peak. With it a layer
keeps one [N, hidden] fp32 input; backward reruns that layer's forward first (one extra forward
per step) and frees its activations again before the next layer. torch.utils.checkpoint restores
the random generator state for the rerun, so LoRA dropout draws the same masks; a packed batch
(unpad.py) reruns with its row layout. This is recomputation in our own patch: Soup's
gradient_checkpointing switch stays off.

The peak is set by the batches with the most real tokens; most batches are far shorter. A batch
of N tokens reruns the first ceil(L * (1 - BUDGET_TOKENS / N)) of its L layers, so what the
others keep never exceeds what all L layers keep for BUDGET_TOKENS tokens; batches up to
BUDGET_TOKENS rerun nothing and cost no time.
"""

import math

import torch
from torch.utils.checkpoint import checkpoint

from . import stream, unpad

enabled = False  # set by patch_qwen2_layer_recompute()
BUDGET_TOKENS = 512  # layers kept for batches up to 512 tokens; longer ones rerun the rest (Dilya's map: 53 s at 2.07 GB vs 63 s at 1.21 GB)


def layers_to_rerun(n_tokens, n_layers, budget=None):
    budget = BUDGET_TOKENS if budget is None else budget
    if n_tokens <= budget:
        return 0
    return min(n_layers, math.ceil(n_layers * (1.0 - budget / n_tokens)))


def _backward_begins(layer):
    if stream.active():
        stream.streamer().layer_begins(layer, forward=False)


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
        attn = getattr(self, "self_attn", None)
        layer, n_layers = getattr(attn, "layer_idx", None), \
            getattr(getattr(attn, "config", None), "num_hidden_layers", None)
        if layer is None or n_layers is None or \
                layer >= layers_to_rerun(hidden_states.shape[0] * hidden_states.shape[1], n_layers):
            if layer is None or not stream.active():
                return original(self, hidden_states, *args, **kwargs)
            stream.streamer().layer_begins(layer, forward=True)
            out = original(self, hidden_states, *args, **kwargs)
            hidden = out[0] if isinstance(out, tuple) else out
            if isinstance(hidden, torch.Tensor) and hidden.requires_grad:
                # backward of this (not recomputed) layer starts when its output's gradient is
                # ready: bring its weights in and prefetch the layers below, as recompute does
                hidden.register_hook(lambda grad, layer=layer: _backward_begins(layer))
            return out
        rows = unpad._rows
        phase = {"forward": True}  # the first call of run is the forward, later ones recompute

        def run(hidden):
            if stream.active():
                stream.streamer().layer_begins(layer, forward=phase["forward"])
            if rows is None:
                return original(self, hidden, *args, **kwargs)
            with unpad.packed(*rows):
                return original(self, hidden, *args, **kwargs)

        out = checkpoint(run, hidden_states, use_reentrant=False, preserve_rng_state=True)
        phase["forward"] = False
        return out

    cls.forward = forward
    cls._mindx_recompute = True
