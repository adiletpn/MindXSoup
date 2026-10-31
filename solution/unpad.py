"""Train on the real tokens only: the pad tokens of a batch never enter the decoder.

A batch of 4 rows is right-padded to its longest row; on this data about 43% of the [4, T]
positions are padding, and every matmul, norm and MLP of the base runs on them too. With right
padding and a causal mask a real token never attends to a pad (pads come after it), so dropping
the pads changes nothing for the real tokens:

* the real tokens of all rows are packed into one [1, N] sequence; position ids restart per row;
* attention runs per row, causal, on that row's slice (`_rows` holds the row lengths while the
  packed forward runs); mask-free causal SDPA is what the base computes for the real tokens;
* LoRA dropout draws its mask for the full [B, T, hidden] input, exactly as the base does (same
  shape, same generator call, so the same random numbers), and keeps the rows of the real tokens;
* the head and loss run on the packed hidden states with the targets shifted inside each row
  first; the per-token values are scattered back to [B, T] for TRL's metrics and Soup's rewind.

Any batch that is not right-padded, or any step that is not a plain training forward, keeps the
padded path.
"""

import contextlib

import torch

from . import dropout

enabled = False  # set by patch_unpad()
_rows = None  # (row lengths, batch, padded length, real-token index) during a packed forward


def layout(attention_mask):
    """(lengths, index of real tokens in the flattened [B, T]) for a right-padded mask, else None."""
    if attention_mask is None or attention_mask.dim() != 2 or not attention_mask.is_cuda:
        return None
    batch, length = attention_mask.shape
    keep = attention_mask.bool()
    lengths = keep.sum(1)
    right_padded = torch.arange(length, device=keep.device)[None, :] < lengths[:, None]
    if not torch.equal(keep, right_padded) or int(lengths.min()) == 0:
        return None
    return lengths.tolist(), keep.reshape(-1).nonzero().squeeze(1)


@contextlib.contextmanager
def packed(lengths, batch, length, index):
    global _rows
    outer, _rows = _rows, (lengths, batch, length, index)
    try:
        yield
    finally:
        _rows = outer


def pack(x, index):
    """[B, T, ...] -> [1, N, ...] rows of the real tokens."""
    return x.reshape(-1, *x.shape[2:])[index].unsqueeze(0)


def scatter(values, index, batch, length):
    """[1, N] or [N] -> [B, T], zeros on the pads."""
    full = values.new_zeros(batch * length)
    full[index] = values.reshape(-1)
    return full.view(batch, length)


def patch_sdpa_per_row():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(original, "_mindx_patched", False):
        return

    def sdpa_per_row(module, query, key, value, attention_mask, **kwargs):
        rows = _rows
        if rows is None or attention_mask is not None or query.shape[0] != 1 \
                or query.shape[2] != sum(rows[0]):
            return original(module, query, key, value, attention_mask, **kwargs)
        direct = (kwargs.get("sliding_window") is None and kwargs.get("position_bias") is None
                  and not kwargs.get("output_attentions", False)
                  and query.shape[1] % key.shape[1] == 0 and key.shape == value.shape)
        if direct and query.shape[1] != key.shape[1]:
            # Without a mask transformers asks SDPA for enable_gqa, which has no fused kernel on
            # a T4: SDPA falls back to the math path in fp32 (sgemm + fp32 softmax), about 2.5x
            # slower than the fp16 memory-efficient kernel the padded base runs on repeated k/v.
            # Repeat k/v to the query heads once, as transformers' repeat_kv does with a mask.
            groups = query.shape[1] // key.shape[1]
            key, value = (x[:, :, None].expand(-1, -1, groups, -1, -1).reshape(
                x.shape[0], -1, x.shape[2], x.shape[3]) for x in (key, value))
        # split, not slicing: backward of a split is one cat; each slice's backward would be a
        # zero-filled full-size gradient plus an add, 3 tensors x rows x layers launches per step
        outputs = []
        parts = [x.split(list(rows[0]), dim=2) for x in (query, key, value)]
        for n, q, k, v in zip(rows[0], *parts):
            if direct:  # what sdpa_attention_forward runs for a mask-free causal row
                outputs.append(torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=kwargs.get("dropout", 0.0), scale=kwargs.get("scaling"),
                    is_causal=n > 1))  # [1, heads, n, head_dim]
            else:
                out, _ = original(module, q, k, v, None, **kwargs)
                outputs.append(out.transpose(1, 2))  # [1, n, heads, head_dim] -> heads first
        # one cat along the tokens and one transpose copy for all rows, not one copy per row
        return torch.cat(outputs, dim=2).transpose(1, 2).contiguous(), None

    sdpa_per_row._mindx_patched = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_per_row


def tag_lora_dropouts(model):
    """Mark peft's LoRA dropout modules once: their output feeds lora_A, which under fp16 autocast
    casts it to fp16 first, so our kernel can write fp16 directly."""
    if getattr(model, "_mindx_lora_tagged", False):
        return
    for module in model.modules():
        dropouts = getattr(module, "lora_dropout", None)
        if isinstance(dropouts, torch.nn.ModuleDict):
            for drop in dropouts.values():
                if isinstance(drop, torch.nn.Dropout):
                    drop._mindx_lora = True
    model._mindx_lora_tagged = True


def patch_dropout_full_mask():
    cls = torch.nn.Dropout
    if getattr(cls, "_mindx_patched", False):
        return
    original = cls.forward

    def forward(self, x):
        rows = _rows
        if rows is None or not self.training or self.p <= 0 or self.inplace \
                or not isinstance(x, torch.Tensor) or x.dim() != 3 or x.shape[0] != 1 \
                or x.shape[1] != rows[3].numel():
            return original(self, x)
        lengths, batch, length, index = rows
        # the base drops on the padded [B, T, hidden] input: draw the same numbers, keep our rows
        out_dtype = x.dtype
        if getattr(self, "_mindx_lora", False) and x.dtype == torch.float32 \
                and torch.is_autocast_enabled("cuda") \
                and torch.get_autocast_dtype("cuda") == torch.float16:
            out_dtype = torch.float16  # the cast lora_A's autocast would do next
        return dropout.packed_dropout(x, self.p, batch, length, index, out_dtype)

    cls.forward = forward
    cls._mindx_patched = True


def patch_unpad():
    global enabled
    patch_sdpa_per_row()
    patch_dropout_full_mask()
    enabled = True
