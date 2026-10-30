"""Fused head + cross-entropy against the base path (fp16 logits, fp32 loss), forward and backward,
plus the metrics TRL and Soup take from the logits.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(hidden, weight, labels, num_items=None):
    """What the base step computes: autocast head in fp16, ForCausalLMLoss in fp32, TRL's
    entropy and accuracy, Soup's per-token loss."""
    from solution.fused_ce import shift_targets

    logits = hidden.to(torch.float16) @ weight.to(torch.float16).t()
    target = shift_targets(labels)
    flat = logits.float().view(-1, logits.shape[-1])
    reduction = "sum" if num_items is not None else "mean"
    loss = torch.nn.functional.cross_entropy(flat, target.reshape(-1), ignore_index=-100,
                                             reduction=reduction)
    if num_items is not None:
        loss = loss / num_items
    with torch.no_grad():
        logp = torch.log_softmax(logits.float(), -1)
        entropy = -(logp.exp() * logp).sum(-1)
        labelled = target != -100
        token_loss = -logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1) * labelled
        correct = ((logits.argmax(-1) == target) & labelled).sum()
    return loss, entropy, token_loss, correct, labelled.sum()


@pytest.mark.parametrize("shape", [(2, 37, 64, 1000), (4, 300, 1536, 151936), (1, 5, 32, 4097)])
@pytest.mark.parametrize("num_items", [None, 123])
def test_fused_head_ce_matches_reference(shape, num_items):
    from solution.fused_ce import fused_head_ce

    batch, seq, width, vocab = shape
    torch.manual_seed(0)
    hidden = (torch.randn(batch, seq, width, device="cuda") * 0.5).requires_grad_(True)
    weight = torch.randn(vocab, width, device="cuda") * width ** -0.5
    labels = torch.randint(0, vocab, (batch, seq), device="cuda")
    labels[:, : seq // 3] = -100  # a prompt that is not trained on
    labels[-1, -2:] = -100        # padding at the end of the last row
    num = None if num_items is None else torch.tensor(num_items, device="cuda")

    # Training runs under the fp16 GradScaler (initial scale 2**16); without it the reference's fp16
    # logit gradients underflow, so both sides backpropagate the scaled loss, as the trainer does.
    grad_scale = 2.0 ** 16
    ref_loss, ref_entropy, ref_token_loss, ref_correct, ref_labelled = reference(
        hidden, weight, labels, num)
    (ref_loss * grad_scale).backward()
    ref_grad = hidden.grad.clone()
    hidden.grad = None

    loss, entropy, token_loss, correct, labelled = fused_head_ce(hidden, weight, labels, num)
    (loss * grad_scale).backward()

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(token_loss, ref_token_loss, rtol=1e-3, atol=1e-3)
    assert int(labelled) == int(ref_labelled)
    assert abs(int(correct) - int(ref_correct)) <= max(1, int(ref_labelled) // 500)  # fp16 ties
    scale = ref_grad.abs().max()
    torch.testing.assert_close(hidden.grad / scale, ref_grad / scale, rtol=2e-2, atol=2e-3)


def test_no_grad_path_matches():
    from solution.fused_ce import fused_head_ce

    torch.manual_seed(1)
    hidden = torch.randn(2, 9, 64, device="cuda")
    weight = torch.randn(500, 64, device="cuda") * 0.125
    labels = torch.randint(0, 500, (2, 9), device="cuda")
    with torch.no_grad():
        loss, *_ = fused_head_ce(hidden, weight, labels)
    torch.testing.assert_close(loss, reference(hidden, weight, labels)[0], rtol=1e-4, atol=1e-4)
