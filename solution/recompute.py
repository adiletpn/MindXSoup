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
host_inputs = False  # set by patch_qwen2_layer_recompute(host=True): fp16 inputs wait in pinned host memory
BUDGET_TOKENS = 0  # every layer reruns; 31.10 evening, one T4, host inputs: 0 -> 59.2 s / 675 MiB, 256 -> 58.3 s / 957 MiB


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
    def forward(ctx, hidden, run, _anchor):
        # _anchor: an empty tensor that requires grad, so autograd calls backward even when the
        # layer input itself does not require grad (the first layer's input comes from the frozen
        # embedding); the LoRA weights inside the layer are not inputs of this function
        ctx.run = run
        ctx.autocast = (torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        ctx.rng = torch.cuda.get_rng_state(hidden.device)
        ctx.device = hidden.device
        with torch.no_grad():
            out = run(hidden)
        scale = hidden.detach().abs().amax(dim=-1, keepdim=True).float().clamp_min(1.0) / 32768.0
        half = (hidden.detach() / scale).to(torch.float16)
        if host_inputs:  # copied on the compute stream; the GPU block is reused only after it
            host = torch.empty(half.shape, dtype=half.dtype, pin_memory=True)
            host.copy_(half, non_blocking=True)
            half = host
        ctx.save_for_backward(half, scale)
        ctx.dtype = hidden.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        half, scale = ctx.saved_tensors
        half = half.to(ctx.device, non_blocking=True)
        hidden = (half.float() * scale).to(ctx.dtype).requires_grad_(True)
        enabled, dtype = ctx.autocast
        with torch.random.fork_rng(devices=[ctx.device]):
            torch.cuda.set_rng_state(ctx.rng, ctx.device)
            with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
                out = ctx.run(hidden)
            torch.autograd.backward(out, grad)
        return (hidden.grad if ctx.needs_input_grad[0] else None), None, None


offload = False  # set by patch_qwen2_layer_recompute(offload=True): no rerun, activations wait on the host
OFFLOAD_MIN_BYTES = 256 * 1024  # smaller saved tensors (norm weights, cos/sin, scales) stay on the GPU
KEEP_LAST = 2  # the last layers' activations stay on the GPU: their backward starts right away


def _empty_like(tensor, device):
    """Same shape, dtype and strides (when they are dense), pinned when on the host."""
    pin = str(device) == "cpu"
    try:
        from torch._prims_common import is_non_overlapping_and_dense
        dense = is_non_overlapping_and_dense(tensor)
    except Exception:  # noqa: BLE001 - private helper; contiguous is always valid
        dense = tensor.is_contiguous()
    if dense:
        return torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype,
                                   device=device, pin_memory=pin)
    return torch.empty(tensor.size(), dtype=tensor.dtype, device=device, pin_memory=pin)


class _Stashed:
    """One saved tensor that waits for backward in pinned host memory."""
    __slots__ = ("host", "copied", "gpu", "ready", "device")


class ActivationStash:
    """Offloads what a decoder layer saves for backward (saved_tensors_hooks) and brings it back.

    Forward: each large saved tensor is copied to pinned host memory on a side stream; the GPU
    block is released once that copy has run (record_stream). Backward: when layer i's output
    gradient is ready, layer i - 1's tensors start coming back on another side stream, so the
    copies overlap layer i's backward. Unpacking waits for the copy, so correctness never depends
    on the prefetch; the values are the same bytes."""

    def __init__(self):
        self.d2h = torch.cuda.Stream()
        self.h2d = torch.cuda.Stream()
        self.layers = {}  # layer -> list of _Stashed of the current step

    def pack(self, layer):
        def pack_hook(tensor):
            if not tensor.is_cuda or tensor.numel() * tensor.element_size() < OFFLOAD_MIN_BYTES \
                    or isinstance(tensor, torch.nn.Parameter):
                return tensor
            item = _Stashed()
            item.device, item.gpu, item.ready = tensor.device, None, None
            item.host = _empty_like(tensor, "cpu")
            produced = torch.cuda.Event()
            produced.record(torch.cuda.current_stream())
            with torch.cuda.stream(self.d2h):
                self.d2h.wait_event(produced)
                item.host.copy_(tensor, non_blocking=True)
                tensor.record_stream(self.d2h)
                item.copied = torch.cuda.Event()
                item.copied.record(self.d2h)
            self.layers.setdefault(layer, []).append(item)
            return item
        return pack_hook

    def unpack(self, packed):
        if not isinstance(packed, _Stashed):
            return packed
        compute = torch.cuda.current_stream()
        if packed.gpu is None:
            self._fetch(packed)
        compute.wait_event(packed.ready)
        tensor, packed.gpu = packed.gpu, None
        tensor.record_stream(compute)
        return tensor

    def _fetch(self, item):
        with torch.cuda.stream(self.h2d):
            self.h2d.wait_event(item.copied)
            item.gpu = _empty_like(item.host, item.device)
            item.gpu.copy_(item.host, non_blocking=True)
            item.ready = torch.cuda.Event()
            item.ready.record(self.h2d)

    def prefetch(self, layer):
        for item in self.layers.pop(layer, ()):
            if item.gpu is None:
                self._fetch(item)


_stash = None


def _backward_begins(layer):
    if stream.active():
        stream.streamer().layer_begins(layer, forward=False)
    if offload and _stash is not None:
        _stash.layers.pop(layer, None)  # its tensors are unpacked (or fetched) from here on
        _stash.prefetch(layer - 1)


def patch_qwen2_layer_recompute(fp16=False, host=False, offload_activations=False):
    from transformers.models.qwen2 import modeling_qwen2

    global enabled, fp16_inputs, host_inputs, offload, BUDGET_TOKENS
    enabled = True
    fp16_inputs = fp16
    host_inputs = fp16 and host
    offload = offload_activations
    if offload:
        BUDGET_TOKENS = 1 << 30  # nothing reruns
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
            if offload and layer < n_layers - KEEP_LAST:
                global _stash
                if _stash is None:
                    _stash = ActivationStash()
                with torch.autograd.graph.saved_tensors_hooks(_stash.pack(layer), _stash.unpack):
                    out = original(self, hidden_states, *args, **kwargs)
            else:
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

            anchor = torch.empty(0, device=hidden_states.device, requires_grad=True)
            hidden = Fp16InputCheckpoint.apply(hidden_states, run_tensor, anchor)
            phase["forward"] = False
            return (hidden,) if form["tuple"] else hidden
        out = checkpoint(run, hidden_states, use_reentrant=False, preserve_rng_state=True)
        phase["forward"] = False
        return out

    cls.forward = forward
    cls._mindx_recompute = True
