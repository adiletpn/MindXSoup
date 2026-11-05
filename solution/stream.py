"""Keep the decoder layers' 4-bit weights (packed bytes and block scales) in pinned host memory and bring each layer's weights to
the GPU only while it runs (offloading in our own code: allowed by the rules, "recomputation or
offloading in your own code (its time counts)").

The packed NF4 bytes of the 7 Linear4bit modules of a decoder layer (≈ 24 MiB) move once into a
pinned host buffer; on the GPU each module keeps a 1-byte placeholder of the same shape, so peft,
Soup and transformers still see a CUDA uint8 weight. A ring of a few GPU slots holds the layers in
use: while layer i runs, a side CUDA stream copies layer i ± PREFETCH (forward / backward order)
over PCIe (12.4 GB/s on Kaggle's T4, ≈ 2 ms per layer, hidden behind the layer's own matmuls).

Correctness never depends on the prefetching: every 4-bit matmul asks the streamer for its
module's bytes right before use (`packed_for`), which loads them synchronously if they are not
resident; a slot is overwritten only after all GPU work enqueued so far has finished. The values
are the same bytes, so results are bit-identical to keeping the weights on the GPU.
"""

import threading

import torch

SLOTS = 2
COALESCE_TRANSFERS = True
PREFETCH = 2

_streamer = None
enabled = False  # set by patch_stream_layers()


class LayerStreamer:
    def __init__(self, groups, slots=SLOTS):
        self.groups = groups  # per decoder layer, its Linear4bit modules in call order
        self.lock = threading.RLock()
        self.copy_stream = torch.cuda.Stream()
        self.entries = {}  # id(module) -> (layer, offset in the slot, numel, shape)
        sizes = []
        for layer, modules in enumerate(groups):
            offset = 0
            for module in modules:
                offload(module.weight)  # no-op when it already left the GPU at load time
                numel = module.weight._mindx_host.numel()
                self.entries[id(module)] = [layer, offset, numel, module.weight._mindx_shape, 0, 0]
                offset += numel
            for module in modules:  # each block's 8-bit scale follows all the layer's packed bytes
                numel = module.weight._mindx_absmax.numel()
                self.entries[id(module)][4:] = [offset, numel]
                offset += numel
            sizes.append(offset)
        self.slot_bytes = max(sizes)
        # One pinned allocation and H2D transfer per layer instead of one per projection.
        # Keep module views so fallback callers still see precisely the original bytes.
        self.host_layers = []
        for layer, modules in enumerate(groups if COALESCE_TRANSFERS else []):
            host = torch.empty(sizes[layer], dtype=torch.uint8, pin_memory=True)
            for module in modules:
                _, offset, numel, _, abs_offset, abs_numel = self.entries[id(module)]
                view = host[offset:offset + numel]
                view.copy_(module.weight._mindx_host)
                module.weight._mindx_host = view
                view = host[abs_offset:abs_offset + abs_numel]
                view.copy_(module.weight._mindx_absmax)
                module.weight._mindx_absmax = view
            self.host_layers.append(host)
        self.slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device="cuda")
                      for _ in range(min(slots, len(groups)))]
        self.slot_layer = [None] * len(self.slots)
        self.slot_ready = [None] * len(self.slots)
        self.slot_used = [0] * len(self.slots)
        self.layer_slot = {}
        self.clock = 0
        self.current = None

    def _victim(self, keep):
        free = [i for i, layer in enumerate(self.slot_layer) if layer is None]
        if free:
            return free[0]
        candidates = [i for i in range(len(self.slots)) if self.slot_layer[i] not in keep]
        if not candidates:
            return None
        return min(candidates, key=lambda i: self.slot_used[i])

    def _load(self, layer, keep):
        slot = self._victim(keep)
        if slot is None:  # every slot holds a layer we keep: take the least recently used one
            slot = min(range(len(self.slots)), key=lambda i: self.slot_used[i])
        old = self.slot_layer[slot]
        if old is not None:
            del self.layer_slot[old]
        # the slot may still be read by work already enqueued on the compute stream
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            self.copy_stream.wait_event(done)
            if self.host_layers:
                host = self.host_layers[layer]
                self.slots[slot][:host.numel()].copy_(host, non_blocking=True)
            else:
                for module in self.groups[layer]:
                    _, offset, numel, _, abs_offset, abs_numel = self.entries[id(module)]
                    self.slots[slot][offset:offset + numel].copy_(
                        module.weight._mindx_host, non_blocking=True)
                    self.slots[slot][abs_offset:abs_offset + abs_numel].copy_(
                        module.weight._mindx_absmax, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self.copy_stream)
        self.slot_layer[slot] = layer
        self.slot_ready[slot] = ready
        self.layer_slot[layer] = slot
        return slot

    def _acquire(self, layer):
        slot = self.layer_slot.get(layer)
        if slot is None:
            slot = self._load(layer, keep={layer, self.current})
        torch.cuda.current_stream().wait_event(self.slot_ready[slot])
        self.clock += 1
        self.slot_used[slot] = self.clock
        return slot

    def bytes_for(self, module):
        """The module's packed NF4 bytes and 8-bit block scales on the GPU, ready for the current
        stream."""
        with self.lock:
            layer, offset, numel, shape, abs_offset, abs_numel = self.entries[id(module)]
            slot = self._acquire(layer)
            data = self.slots[slot]
            return data[offset:offset + numel].view(shape), data[abs_offset:abs_offset + abs_numel]

    def layer_begins(self, layer, forward):
        """Hint from the decoder layer: it is about to run (forward, or its recomputation in
        backward); bring it in and prefetch the next ones in the order they will run."""
        with self.lock:
            if not 0 <= layer < len(self.groups):
                return
            self.current = layer
            self._acquire(layer)
            step = 1 if forward else -1
            keep = {layer}
            for ahead in range(1, min(PREFETCH, len(self.slots) - 1) + 1):
                nxt = layer + step * ahead
                if not 0 <= nxt < len(self.groups):
                    break
                if nxt not in self.layer_slot:
                    if self._victim(keep | {nxt}) is None:
                        break  # no slot to spare: the layer is loaded when it is used
                    self._load(nxt, keep=keep | {nxt})
                keep.add(nxt)

    def owns(self, module):
        return id(module) in self.entries


def offload(param, pin=True):
    """Move a quantized Params4bit's packed bytes and 8-bit block scales to pinned host memory,
    once; 1-byte placeholders of the same shapes stay on the GPU, so the module still looks like a
    CUDA weight."""
    if hasattr(param, "_mindx_host"):
        return
    data = param.data
    host = torch.empty(data.numel(), dtype=torch.uint8, pin_memory=pin)
    host.copy_(data.reshape(-1))
    param._mindx_host, param._mindx_shape = host, tuple(data.shape)
    param.data = torch.empty(1, dtype=torch.uint8, device=data.device).expand(data.shape)
    state = param.quant_state
    absmax = state.absmax
    param._mindx_absmax = torch.empty(absmax.numel(), dtype=torch.uint8, pin_memory=pin)
    param._mindx_absmax.copy_(absmax.reshape(-1))
    state.absmax = torch.empty(1, dtype=torch.uint8, device=absmax.device).expand(absmax.shape)


def offloaded(module):
    return hasattr(getattr(module, "weight", None), "_mindx_host")


def bytes_for(module):
    """The module's packed NF4 bytes and block scales on the GPU: from the streamer's slots, or
    (before the streamer exists, e.g. a forward before training) a one-off synchronous copy."""
    if _streamer is not None and _streamer.owns(module):
        return _streamer.bytes_for(module)
    weight = module.weight
    return (weight._mindx_host.to(weight.device).view(weight._mindx_shape),
            weight._mindx_absmax.to(weight.device))


def weight_for(module):
    """The module's weight already unpacked for the running layer, or None."""
    if _streamer is not None and _streamer.owns(module):
        return _streamer.weight_for(module)
    return None


def active():
    return _streamer is not None


def streamer():
    return _streamer


def _layer_modules(layer):
    import bitsandbytes as bnb

    return [m for m in layer.modules() if isinstance(m, bnb.nn.Linear4bit)]


def _streamable(module):
    weight = module.weight
    state = getattr(weight, "quant_state", None)
    return (weight.is_cuda and weight.dtype == torch.uint8
            and (hasattr(weight, "_mindx_host") or weight.data.is_contiguous())
            and state is not None and getattr(state, "nested", False)
            and state.quant_type == "nf4" and state.blocksize == 64
            and getattr(state, "state2", None) is not None and state.state2.blocksize == 256
            and state.dtype == torch.float16 and module.compute_dtype == torch.float16
            and not getattr(state, "packing_format_for_cpu", False))


def setup(causal_lm):
    """Move the decoder layers' 4-bit weights to the host, once; only when every layer is a plain
    fp16-compute NF4 model on one GPU, else nothing changes."""
    global _streamer
    if _streamer is not None or not enabled:
        return
    base = getattr(causal_lm, "model", None)
    layers = getattr(base, "layers", None)
    if layers is None or not torch.cuda.is_available():
        return
    groups = [_layer_modules(layer) for layer in layers]
    if not groups or any(len(g) == 0 for g in groups) or \
            not all(_streamable(m) for g in groups for m in g):
        return
    for g in groups:
        for m in g:
            m.compute_type_is_set = True  # fp16 inputs only reach them; nothing to switch
    _streamer = LayerStreamer(groups)


def _offload_at_load(name, value):
    state = getattr(value, "quant_state", None)
    return (".layers." in name and getattr(value, "bnb_quantized", False) and value.is_cuda
            and value.dtype == torch.uint8 and state is not None
            and getattr(state, "nested", False) and state.quant_type == "nf4"
            and state.blocksize == 64 and state.dtype == torch.float16)


def patch_offload_at_load():
    """While transformers loads the model, move each decoder weight to the host right after
    bitsandbytes quantizes it, so the 4-bit layers never sit on the GPU all at once (the memory
    peak covers loading too). transformers' caching_allocator_warmup, which reserves GPU memory for
    the whole model up front only to speed up loading, is skipped: that reservation would stay in
    PyTorch's pool and in nvidia-smi's peak."""
    from transformers import modeling_utils
    from transformers.integrations import bitsandbytes as hf_bnb

    cls = hf_bnb.Bnb4bitQuantize
    if not getattr(cls, "_mindx_offload", False):
        original = cls.convert

        def convert(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if enabled and isinstance(result, dict):
                for name, value in result.items():
                    if _offload_at_load(name, value):
                        # plain memory for now: LayerStreamer copies it once into pinned layer blocks
                        offload(value, pin=not COALESCE_TRANSFERS)
            return result

        cls.convert = convert
        cls._mindx_offload = True
    if not getattr(modeling_utils.caching_allocator_warmup, "_mindx_patched", False):
        original_warmup = modeling_utils.caching_allocator_warmup

        def caching_allocator_warmup(model, expanded_device_map, hf_quantizer):
            if enabled and hf_quantizer is not None and \
                    getattr(getattr(hf_quantizer, "quantization_config", None),
                            "load_in_4bit", False):
                return None
            return original_warmup(model, expanded_device_map, hf_quantizer)

        caching_allocator_warmup._mindx_patched = True
        modeling_utils.caching_allocator_warmup = caching_allocator_warmup


def patch_stream_layers():
    global enabled
    enabled = True
    patch_offload_at_load()
