"""The tied embedding / lm_head (151936 x 1536 fp16, 445 MiB) in pinned host memory.

Our own offloading (organizers: allowed, the copies count in the time). The table is the largest
tensor left on the GPU, and the training step reads it twice only: the embedding lookup of the
batch's tokens, and the head over the vocabulary.

* Lookup: the batch's token rows are gathered on the host and copied to the GPU (a few MB).
* Head: the vocabulary is walked in slices of `SLICE` rows. Each slice of the weight is copied
  over PCIe into one of two GPU buffers while the previous slice is used; cuBLAS computes the fp16
  logits of all the step's tokens for that slice, and `head_slice_kernel` carries each row's
  online-softmax state (running max, sum of exp, sum of exp * x, first argmax, label logit) from
  slice to slice in fp32. A second walk turns each slice's logits into the scaled gradient
  softmax - onehot in place and accumulates the hidden-state gradient. The slices start at
  multiples of the kernel's block, so every row sees the same blocks in the same order as
  fused_ce.py's whole-row kernel: same loss, entropy and argmax.

Anything else that reads the table (a forward outside training, a lookup before the first step)
gets a temporary GPU copy of it, so results never depend on where the table lives.
"""

import torch
import triton
import triton.language as tl

SLICE = 4096   # vocabulary rows per slice: 4096 x 1536 fp16 = 12 MiB per buffer
BLOCK = 4096   # = fused_ce.VOCAB_BLOCK; SLICE is a multiple, so the block boundaries agree
ONE_WALK = True  # gradient accumulated in the same walk (FlashAttention-style rescaling)

enabled = False  # set by patch_head_stream()
_hosts_at_load = {}  # GPU placeholder storage pointer -> the table's pinned host copy


@triton.jit
def head_slice_kernel(logits_ptr, target_ptr, m_ptr, s_ptr, t_ptr, best_ptr, best_idx_ptr,
                      x_target_ptr, lse_ptr, alpha_ptr, n_cols, col0, grad_scale,
                      GRAD: tl.constexpr, ACC: tl.constexpr, BLOCK: tl.constexpr):
    # One program per token row; logits: [rows, n_cols] fp16, the vocabulary columns
    # col0 .. col0 + n_cols. Without GRAD the row's online-softmax state is advanced over the
    # slice; with GRAD the slice is overwritten with (softmax - onehot) * grad_scale. With ACC
    # (one walk), after the state update the slice is overwritten with exp(x - m), m the row's
    # running max after this slice, and alpha = exp(m_before - m) rescales what was accumulated.
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * n_cols
    target = tl.load(target_ptr + row)
    labelled = target != -100
    cols = tl.arange(0, BLOCK)
    if GRAD:
        lse = tl.load(lse_ptr + row)
        for start in range(0, n_cols, BLOCK):
            idx = start + cols
            valid = idx < n_cols
            x = tl.load(row_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(col0 + idx == target, 1.0, 0.0)
            g = tl.where(labelled, g * grad_scale, 0.0)
            tl.store(row_ptr + idx, g.to(logits_ptr.dtype.element_ty), mask=valid)
    else:
        m = tl.load(m_ptr + row)
        m_before = m
        s = tl.load(s_ptr + row)
        t = tl.load(t_ptr + row)
        best = tl.load(best_ptr + row)
        best_idx = tl.load(best_idx_ptr + row)
        for start in range(0, n_cols, BLOCK):
            idx = start + cols
            valid = idx < n_cols
            x = tl.load(row_ptr + idx, mask=valid, other=-float("inf")).to(tl.float32)
            block_max = tl.max(x, axis=0)
            block_idx = col0 + tl.min(tl.where(x == block_max, idx, n_cols), axis=0)
            best_idx = tl.where(block_max > best, block_idx, best_idx)  # first maximum wins
            best = tl.maximum(best, block_max)
            new_m = tl.maximum(m, block_max)
            e = tl.where(valid, tl.exp(x - new_m), 0.0)
            alpha = tl.exp(m - new_m)
            s = s * alpha + tl.sum(e, axis=0)
            t = t * alpha + tl.sum(tl.where(valid, e * x, 0.0), axis=0)
            m = new_m
        tl.store(m_ptr + row, m)
        tl.store(s_ptr + row, s)
        tl.store(t_ptr + row, t)
        tl.store(best_ptr + row, best)
        tl.store(best_idx_ptr + row, best_idx)
        here = labelled & (target >= col0) & (target < col0 + n_cols)
        safe = tl.where(here, target - col0, 0)
        x_target = tl.load(row_ptr + safe).to(tl.float32)
        tl.store(x_target_ptr + row, x_target, mask=here)
        if ACC:
            tl.store(alpha_ptr + row, tl.exp(m_before - m))
            for start in range(0, n_cols, BLOCK):
                idx = start + cols
                valid = idx < n_cols
                x = tl.load(row_ptr + idx, mask=valid, other=-float("inf")).to(tl.float32)
                p = tl.where(valid & labelled, tl.exp(x - m), 0.0)
                tl.store(row_ptr + idx, p.to(logits_ptr.dtype.element_ty), mask=valid)


def launch_slice(logits, target, state, col0, grad_scale, grad, acc=False):
    m, s, t, best, best_idx, x_target, lse, alpha = state
    head_slice_kernel[(logits.shape[0],)](
        logits, target, m, s, t, best, best_idx, x_target, lse, alpha, logits.shape[1], col0,
        grad_scale, GRAD=grad, ACC=acc, BLOCK=BLOCK, num_warps=8)


class HostTable:
    """The table's pinned host copy and two GPU slice buffers fed by a side stream."""

    def __init__(self, host, device):
        self.host = host  # [vocab, width] fp16, pinned
        self.vocab, self.width = host.shape
        self.device = device
        self.copy_stream = torch.cuda.Stream(device)
        self.buffers = None

    def _buffers(self):
        if self.buffers is None:
            self.buffers = [torch.empty((SLICE, self.width), dtype=self.host.dtype,
                                        device=self.device) for _ in range(2)]
        return self.buffers

    def slices(self):
        """Yield (col0, weight slice on the GPU) over the vocabulary, the next slice copying
        while the current one is in use. Each buffer is refilled only after the compute stream
        is done with it."""
        buffers = self._buffers()
        compute = torch.cuda.current_stream(self.device)
        starts = list(range(0, self.vocab, SLICE))
        ready = [None, None]
        free = [None, None]
        # the buffers may still be read by work already enqueued (the previous walk): the first
        # copies wait for everything the compute stream has so far
        start = torch.cuda.Event()
        start.record(compute)
        self.copy_stream.wait_event(start)

        def issue(i):
            col0 = starts[i]
            rows = min(SLICE, self.vocab - col0)
            k = i % 2
            with torch.cuda.stream(self.copy_stream):
                if free[k] is not None:
                    self.copy_stream.wait_event(free[k])
                buffers[k][:rows].copy_(self.host[col0:col0 + rows], non_blocking=True)
                ready[k] = torch.cuda.Event()
                ready[k].record(self.copy_stream)

        issue(0)
        for i, col0 in enumerate(starts):
            if i + 1 < len(starts):
                issue(i + 1)
            k = i % 2
            compute.wait_event(ready[k])
            yield col0, buffers[k][:min(SLICE, self.vocab - col0)]
            free[k] = torch.cuda.Event()
            free[k].record(compute)
        self.buffers = None  # back to the allocator for the decoder's backward (the training peak)

    def gather(self, ids):
        """table[ids] on the GPU: the rows are gathered on the host, then copied."""
        flat = ids.reshape(-1).to("cpu", torch.int64)
        rows = torch.index_select(self.host, 0, flat)
        return rows.to(self.device).view(*ids.shape, self.width)

    def full(self):
        return self.host.to(self.device)


def table_at_load(source, device, dtype):
    """While transformers loads the checkpoint: put the table straight into pinned host memory and
    give the model a 1-element GPU placeholder of its shape, so the 445 MiB never sit on the GPU
    (the nvidia-smi peak covers loading). offload_table() picks the host copy up later. The values
    are the ones `source.to(device, dtype)` would give: the same conversion, on the CPU."""
    host = torch.empty(source.shape, dtype=dtype, pin_memory=True)
    host.copy_(source.to(dtype))
    placeholder = torch.empty((1, 1), dtype=dtype, device=device).expand(source.shape)
    _hosts_at_load[placeholder.untyped_storage().data_ptr()] = host
    return placeholder


def pending_at_load(weight):
    return weight.untyped_storage().data_ptr() in _hosts_at_load


def ensure_table(causal_lm):
    """Before any forward reads a table that was loaded straight to the host: hand it to
    offload_table(), or, where that does not apply, put the real weight back on the GPU."""
    from .embedding import _tied_parts

    parts = _tied_parts(causal_lm)
    if parts is None or not pending_at_load(parts[0].weight):
        return
    if enabled and offload_table(causal_lm):
        return
    weight = parts[0].weight
    host = _hosts_at_load.pop(weight.untyped_storage().data_ptr())
    weight.data = host.to(weight.device)


def table_of(weight):
    return getattr(weight, "_mindx_table", None)


def head_ce(flat, table, target, grad_scale, with_grad):
    """Per-row loss / entropy / argmax and the (grad_scale-scaled) hidden-state gradient of
    CE(flat @ table.T, target), streaming the table slice by slice. flat: [rows, width] fp16."""
    rows = flat.shape[0]
    dev = flat.device
    m = torch.full((rows,), -float("inf"), dtype=torch.float32, device=dev)
    best = torch.full((rows,), -float("inf"), dtype=torch.float32, device=dev)
    s = torch.zeros(rows, dtype=torch.float32, device=dev)
    t = torch.zeros(rows, dtype=torch.float32, device=dev)
    x_target = torch.zeros(rows, dtype=torch.float32, device=dev)
    best_idx = torch.zeros(rows, dtype=torch.int32, device=dev)
    lse = torch.empty(rows, dtype=torch.float32, device=dev)
    alpha = torch.empty(rows, dtype=torch.float32, device=dev)
    state = (m, s, t, best, best_idx, x_target, lse, alpha)
    logits = torch.empty(rows * SLICE, dtype=torch.float16, device=dev)

    def contiguous_buffer(width):  # the kernel walks rows of exactly `width` columns
        return logits[:rows * width].view(rows, width)

    one_walk = with_grad and ONE_WALK
    acc = torch.zeros((rows, flat.shape[1]), dtype=torch.float32, device=dev) if one_walk \
        else None
    for col0, w in table.slices():
        buf = contiguous_buffer(w.shape[0])
        torch.mm(flat, w.t(), out=buf)
        launch_slice(buf, target, state, col0, grad_scale, grad=False, acc=one_walk)
        if one_walk:  # buf now holds exp(x - m): accumulate sum_j exp(x_j - m) * W_j
            acc.mul_(alpha.unsqueeze(1))
            acc += torch.mm(buf, w).float()
    torch.add(m, torch.log(s), out=lse)
    entropy = lse - t / s
    labelled = target != -100
    token_loss = torch.where(labelled, lse - x_target, torch.zeros_like(lse))
    grad = None
    if one_walk:
        # softmax @ W - W[target] for labelled rows, scaled like the two-walk gradient
        target_rows = table.gather(torch.where(labelled, target, 0)).float()
        grad = (acc / s.unsqueeze(1) - target_rows) * grad_scale
        grad.mul_(labelled.unsqueeze(1))
    elif with_grad:
        grad = torch.zeros((rows, flat.shape[1]), dtype=torch.float32, device=dev)
        for col0, w in table.slices():
            buf = contiguous_buffer(w.shape[0])
            torch.mm(flat, w.t(), out=buf)
            launch_slice(buf, target, state, col0, grad_scale, grad=True)
            grad += torch.mm(buf, w).float()
    return token_loss, entropy, best_idx, grad


def offload_table(causal_lm):
    """Move the tied input embedding / lm_head weight to pinned host memory, once; the GPU keeps
    a 1-element placeholder of the same shape, and the two modules read the host copy."""
    from .embedding import _tied_parts

    parts = _tied_parts(causal_lm)
    if parts is None or parts[1] is None:
        return False
    embedding, head = parts
    weight = embedding.weight
    if table_of(weight) is not None:
        return True
    if not (weight.is_cuda and weight.dtype == torch.float16 and not weight.requires_grad
            and weight.dim() == 2 and embedding.max_norm is None and head.bias is None):
        return False
    host = _hosts_at_load.pop(weight.untyped_storage().data_ptr(), None)
    if host is None:
        host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
        host.copy_(weight.data)
        weight.data = torch.empty((1, 1), dtype=weight.dtype, device=weight.device).expand(
            weight.shape)
    table = HostTable(host, weight.device)
    weight._mindx_table = table

    def embed_forward(input_ids, _module=embedding):
        return table.gather(input_ids)

    def head_forward(hidden, _module=head):
        # fp16, as the fp16 weight gives; embedding.py's hooks cast it outside autocast
        return torch.nn.functional.linear(hidden.to(table.host.dtype), table.full())

    embedding.forward = embed_forward
    head.forward = head_forward
    torch.cuda.empty_cache()  # give the 445 MiB back to the driver (nvidia-smi counts the pool)
    return True


def patch_head_stream():
    global enabled
    enabled = True
