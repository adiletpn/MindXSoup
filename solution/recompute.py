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
fp16_inputs = False  # set by patch_qwen2_layer_recompute(fp16_inputs=True)
BUDGET_TOKENS = 256  # map 31.10 (one T4, empty Triton cache): 0 -> 63 s / 1 209 MiB, 256 -> 55 s / 1 603 MiB, 512 -> 54 s / 2 067 MiB


def layers_to_rerun(n_tokens, n_layers, budget=None):
    budget = BUDGET_TOKENS if budget is None else budget
    if n_tokens <= budget:
        return 0
    return min(n_layers, math.ceil(n_layers * (1.0 - budget / n_tokens)))


class Fp16InputCheckpoint(torch.autograd.Function):
    """Recompute a layer in backward from an fp16 copy of its fp32 input.

    The input is the fp32 residual stream; keeping it only for the rerun, it is stored as fp16
    rows scaled by one fp32 factor per row (so no value can overflow fp16), half the bytes of
    torch.utils.checkpoint's fp32 copy. Backward rebuilds the fp32 input, reruns the layer under
    the forward's autocast and random generator state (LoRA dropout draws the same masks), and
    backpropagates through it. Organizers (31.10 11:27): a tensor kept only for backward may be
    stored in fp16; the forward, the loss and the metrics are unchanged."""

    @staticmethod
    def forward(ctx, hidden, run):
        ctx.run = run
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.rng = torch.cuda.get_rng_state(hidden.device)
        ctx.device = hidden.device
        with torch.no_grad():
            out = run(hidden)
        scale = hidden.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(1.0) / 32768.0
        ctx.save_for_backward((hidden.detach() / scale).to(torch.float16), scale)
        ctx.dtype = hidden.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        half, scale = ctx.saved_tensors
        hidden = (half.float() * scale).to(ctx.dtype).requires_grad_(True)
        enabled, dtype = ctx.autocast
        with torch.random.fork_rng(devices=[ctx.device]):
            torch.cuda.set_rng_state(ctx.rng, ctx.device)
            with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
                out = ctx.run(hidden)
            torch.autograd.backward(out, grad)
        return hidden.grad, None


def _backward_begins(layer):
    if stream.active():
        stream.streamer().layer_begins(layer, forward=False)


def patch_qwen2_layer_recompute(fp16=False):
    from transformers.models.qwen2 import modeling_qwen2

    global enabled, fp16_inputs
    enabled = True
    fp16_inputs = fp16
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

        if fp16_inputs and hidden_states.dtype == torch.float32:
            form = {}

            def run_tensor(hidden):  # the layer returns a tensor (a 1-tuple in older versions)
                out = run(hidden)
                form["tuple"] = isinstance(out, tuple)
                return out[0] if form["tuple"] else out

            hidden = Fp16InputCheckpoint.apply(hidden_states, run_tensor)
            phase["forward"] = False
            return (hidden,) if form["tuple"] else hidden
        out = checkpoint(run, hidden_states, use_reentrant=False, preserve_rng_state=True)
        phase["forward"] = False
        return out

    cls.forward = forward
    cls._mindx_recompute = True
