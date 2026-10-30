"""Fused output head + cross-entropy + Soup/TRL metrics, without the full logits tensor.

The base step builds logits [batch, seq, 151936] in fp16, upcasts them to fp32 for the loss, keeps
more full-size buffers for backward, and then TRL reads them again for `entropy` and
`mean_token_accuracy` (with one more `.contiguous()` copy) and Soup reads them for the per-row
losses in rewind.jsonl. That is 3-4 GB at the peak.

Here the head runs on `CHUNK` rows at a time. For each chunk cuBLAS computes the fp16 logits (the
same matmul autocast does in the base run), then one Triton program per row makes two passes over
the vocabulary: an online softmax that yields log-sum-exp, the label's loss, the entropy and the
argmax, and a second pass that overwrites the logits in place with their gradient. The gradient on
the hidden states is taken in the forward pass and scaled by the incoming gradient in backward, so
no logits outlive their chunk.

Soup and TRL still get every output they log: the trainer patch below logs `entropy`,
`num_tokens` and `mean_token_accuracy` by TRL's formulas, and hands the per-token losses to Soup's
rewind recorder through the same `ChunkedStep` record Soup's own chunked step uses.

Ideas: gradient in the forward pass and chunking from Liger Kernel's fused linear cross-entropy
(arXiv 2410.10989); never materializing full logits from Apple's Cut Cross-Entropy
(arXiv 2411.09009). The code is ours.
"""

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
CHUNK = 512          # rows per head matmul: 512 x 151936 fp16 logits = 155 MB
VOCAB_BLOCK = 4096
# The gradient is written into the fp16 logits buffer scaled by GRAD_SCALE so that small
# probabilities stay representable; it is divided out in fp32 after the matmul.
GRAD_SCALE = 2048.0


@triton.jit
def head_ce_rows_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, argmax_ptr, n_cols,
                        grad_scale, WRITE_GRAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * n_cols
    target = tl.load(target_ptr + row)
    cols = tl.arange(0, BLOCK)

    # Pass 1: running max m, sum of exp(x - m), sum of exp(x - m) * x, and the first argmax.
    m = -float("inf")
    s = 0.0
    t = 0.0
    best = -float("inf")
    best_idx = 0
    for start in range(0, n_cols, BLOCK):
        idx = start + cols
        valid = idx < n_cols
        x = tl.load(row_ptr + idx, mask=valid, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        block_idx = tl.min(tl.where(x == block_max, idx, n_cols), axis=0)
        # strictly greater keeps the first maximum, as torch.argmax does
        best_idx = tl.where(block_max > best, block_idx, best_idx)
        best = tl.maximum(best, block_max)
        new_m = tl.maximum(m, block_max)
        e = tl.where(valid, tl.exp(x - new_m), 0.0)
        alpha = tl.exp(m - new_m)
        s = s * alpha + tl.sum(e, axis=0)
        t = t * alpha + tl.sum(tl.where(valid, e * x, 0.0), axis=0)
        m = new_m
    lse = m + tl.log(s)
    tl.store(entropy_ptr + row, lse - t / s)
    tl.store(argmax_ptr + row, best_idx)

    labelled = target != -100
    safe_target = tl.where(labelled, target, 0)
    x_target = tl.load(row_ptr + safe_target).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - x_target, 0.0))

    # Pass 2: d loss / d logits = softmax - onehot, written over the logits it was computed from.
    if WRITE_GRAD:
        for start in range(0, n_cols, BLOCK):
            idx = start + cols
            valid = idx < n_cols
            x = tl.load(row_ptr + idx, mask=valid, other=0.0).to(tl.float32)
            g = tl.exp(x - lse) - tl.where(idx == target, 1.0, 0.0)
            g = tl.where(labelled, g * grad_scale, 0.0)
            tl.store(row_ptr + idx, g.to(logits_ptr.dtype.element_ty), mask=valid)


class FusedHeadCE(torch.autograd.Function):
    """sum over labelled positions of CE(hidden @ weight.T, target) / denominator, plus the
    per-position entropy, loss and argmax that TRL and Soup log."""

    @staticmethod
    def forward(ctx, hidden, weight, target, denominator):
        rows, width = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
        flat = hidden.reshape(rows, width)
        flat_target = target.reshape(rows).to(torch.int32).contiguous()
        weight16 = weight.detach().to(torch.float16)  # what autocast feeds the base matmul
        vocab = weight16.shape[0]
        with_grad = ctx.needs_input_grad[0]

        token_loss = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        token_entropy = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        argmax = torch.empty(rows, dtype=torch.int32, device=hidden.device)
        grad = torch.empty((rows, width), dtype=torch.float32, device=hidden.device) \
            if with_grad else None
        logits = torch.empty((min(CHUNK, rows), vocab), dtype=torch.float16, device=hidden.device)
        with torch.autocast("cuda", enabled=False):  # dtypes are explicit here
            for start in range(0, rows, CHUNK):
                stop = min(start + CHUNK, rows)
                buf = logits[: stop - start]
                torch.mm(flat[start:stop].to(torch.float16), weight16.t(), out=buf)
                head_ce_rows_kernel[(stop - start,)](
                    buf, flat_target[start:stop], token_loss[start:stop],
                    token_entropy[start:stop], argmax[start:stop], vocab, GRAD_SCALE,
                    WRITE_GRAD=with_grad, BLOCK=VOCAB_BLOCK, num_warps=8)
                if with_grad:
                    grad[start:stop] = torch.mm(buf, weight16).float()
        del logits, weight16

        denominator = denominator.to(torch.float32)
        loss = token_loss.sum() / denominator
        if with_grad:
            grad.mul_(1.0 / GRAD_SCALE).div_(denominator)
            ctx.save_for_backward(grad)
        ctx.hidden_shape, ctx.hidden_dtype = hidden.shape, hidden.dtype
        shape = hidden.shape[:-1]
        token_entropy, token_loss, argmax = (token_entropy.view(shape), token_loss.view(shape),
                                             argmax.view(shape))
        ctx.mark_non_differentiable(token_entropy, token_loss, argmax)
        return loss, token_entropy, token_loss, argmax

    @staticmethod
    def backward(ctx, grad_loss, _grad_entropy, _grad_token_loss, _grad_argmax):
        (grad,) = ctx.saved_tensors
        dh = (grad * grad_loss).to(ctx.hidden_dtype).view(ctx.hidden_shape)
        return dh, None, None, None


def shift_targets(labels):
    """labels moved one position left, as ForCausalLMLoss does: position t predicts t + 1."""
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def fused_head_ce(hidden, weight, labels, num_items_in_batch=None):
    target = shift_targets(labels).to(hidden.device)
    labelled = target != IGNORE_INDEX
    if num_items_in_batch is None:
        denominator = labelled.sum()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = torch.tensor(float(num_items_in_batch), device=hidden.device)
    loss, entropy, token_loss, argmax = FusedHeadCE.apply(hidden, weight, target, denominator)
    correct = ((argmax == target) & labelled).sum()
    return loss, entropy, token_loss, correct, labelled.sum()


# ----------------------------------------------------------------------------------------------
# Patches: the causal-LM forward (no logits during training) and TRL's metric logging.

STEP_ATTR = "_mindx_last_step"


def _head_is_fusable(causal_lm):
    head = causal_lm.get_output_embeddings() if hasattr(causal_lm, "get_output_embeddings") \
        else None
    return (isinstance(head, torch.nn.Linear) and head.bias is None
            and not head.weight.requires_grad and head.weight.is_cuda
            and getattr(causal_lm.config, "final_logit_softcapping", None) is None)


def patch_qwen2_causal_lm():
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls, "_mindx_patched", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        keep_all = logits_to_keep is None or (isinstance(logits_to_keep, int)
                                              and logits_to_keep == 0)
        if labels is None or not self.training or not keep_all or "shift_labels" in kwargs \
                or not _head_is_fusable(self):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        loss, entropy, token_loss, correct, labelled = fused_head_ce(
            hidden, self.lm_head.weight, labels, kwargs.get("num_items_in_batch"))
        setattr(self, STEP_ATTR, (entropy, token_loss, correct, labelled))
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    cls.forward = forward
    cls._mindx_patched = True


def _causal_lm_of(model):
    model = getattr(model, "module", model)  # accelerate / DDP wrappers
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def patch_trl_metrics():
    """When our head produced the step, log TRL's entropy / num_tokens / mean_token_accuracy
    from what the kernel collected and give Soup's rewind recorder its per-token losses. Any
    other step (no labels, eval, a model we did not patch) takes TRL's own path."""
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    if getattr(cls, "_mindx_patched", False):
        return
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        causal_lm = _causal_lm_of(model)
        if (not isinstance(causal_lm, torch.nn.Module) or not getattr(model, "training", False)
                or "labels" not in inputs or "shift_labels" in inputs
                or getattr(self.args, "use_liger_kernel", False)
                or not getattr(type(causal_lm), "_mindx_patched", False)
                or not _head_is_fusable(causal_lm)):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        setattr(causal_lm, STEP_ATTR, None)
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        step = getattr(causal_lm, STEP_ATTR, None)
        setattr(causal_lm, STEP_ATTR, None)
        if step is None:  # pragma: no cover - the patched forward always leaves one
            raise RuntimeError("fused head did not run")
        _log_trl_metrics(self, mode, inputs, *step)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss
    cls._mindx_patched = True


def _log_trl_metrics(trainer, mode, inputs, entropy, token_loss, correct, labelled):
    """Same values and order as trl 0.29.1 SFTTrainer.compute_loss (non-Liger path)."""
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR as SOUP_STEP_ATTR, ChunkedStep

    gather = trainer.accelerator.gather_for_metrics
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        mean_entropy = torch.sum(entropy * attention_mask) / attention_mask.sum()
        trainer._metrics[mode]["entropy"].append(gather(mean_entropy).mean().item())
        trainer._total_train_tokens += gather(attention_mask.sum()).sum().item()
        trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
        total = gather(labelled).sum()
        accuracy = (gather(correct).sum() / total).item() if total > 0 else 0.0
        trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)
    # Soup's rewind recorder reads per-token losses from this record when a step has no logits.
    setattr(trainer, SOUP_STEP_ATTR, ChunkedStep(token_entropy=entropy, token_loss=token_loss,
                                                 correct=correct, labelled=labelled))
