"""Keep the decoder layers' 4-bit weights in pinned host memory and bring each layer's weights to
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

SLOTS = 4
PREFETCH = 2

_streamer = None
enabled = False  # set by patch_stream_layers()


class LayerStreamer:
    def __init__(self, groups, slots=SLOTS):
        self.groups = groups  # per decoder layer, its Linear4bit modules in call order
        self.lock = threading.RLock()
        self.copy_stream = torch.cuda.Stream()
        self.entries = {}  # id(module) -> (layer, offset, numel, shape)
        self.host = []
        sizes = []
        for layer, modules in enumerate(groups):
            total = sum(m.weight.data.numel() for m in modules)
            sizes.append(total)
            host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
            offset = 0
            for module in modules:
                data = module.weight.data
                host[offset:offset + data.numel()].copy_(data.reshape(-1))
                self.entries[id(module)] = (layer, offset, data.numel(), tuple(data.shape))
                offset += data.numel()
            self.host.append(host)
        self.slot_bytes = max(sizes)
        self.slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device="cuda")
                      for _ in range(min(slots, len(groups)))]
        self.slot_layer = [None] * len(self.slots)
        self.slot_ready = [None] * len(self.slots)
        self.slot_used = [0] * len(self.slots)
        self.layer_slot = {}
        self.clock = 0
        self.current = None
        # all bytes are on the host now: replace the GPU copies by 1-byte placeholders
        for modules in groups:
            for module in modules:
                shape = module.weight.data.shape
                module.weight.data = torch.empty(1, dtype=torch.uint8, device="cuda").expand(shape)

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
            size = self.host[layer].numel()
            self.slots[slot][:size].copy_(self.host[layer], non_blocking=True)
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

    def packed_for(self, module):
        """The module's packed NF4 bytes on the GPU, ready for the current stream."""
        with self.lock:
            layer, offset, numel, shape = self.entries[id(module)]
            slot = self._acquire(layer)
            return self.slots[slot][offset:offset + numel].view(shape)

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
    return (weight.is_cuda and weight.dtype == torch.uint8 and weight.data.is_contiguous()
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


def patch_stream_layers():
    global enabled
    enabled = True
