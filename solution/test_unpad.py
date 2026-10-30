"""Training on the real tokens only gives the step of the padded batch: same loss, LoRA gradients,
per-token values and dropout masks.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENGTHS = [37, 20, 5, 30]


def tiny_lora_qwen2(dropout):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.fused_ce import patch_qwen2_causal_lm
    from solution.unpad import patch_unpad

    patch_qwen2_causal_lm()
    patch_unpad()
    config = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         attn_implementation="sdpa")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=dropout, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora)
    for name, param in model.named_parameters():
        if "lora_B" in name:  # non-zero B, so the LoRA path matters
            torch.nn.init.normal_(param, std=0.02)
    return model.train()


def batch(left_padded=False):
    torch.manual_seed(1)
    length = max(LENGTHS)
    ids = torch.randint(0, 1000, (len(LENGTHS), length), device="cuda")
    mask = torch.zeros_like(ids)
    for row, n in enumerate(LENGTHS):
        if left_padded:
            mask[row, length - n:] = 1
        else:
            mask[row, :n] = 1
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :3] = -100  # a prompt part, as in SFT
    return ids, mask, labels


def step(model, unpadded, ids, mask, labels):
    from solution import fused_ce, unpad

    unpad.enabled = unpadded
    torch.manual_seed(123)  # the same dropout draws in both runs
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    entropy, token_loss, correct, labelled = getattr(model.get_base_model(), fused_ce.STEP_ATTR)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    unpad.enabled = True
    return out.loss.detach(), entropy, token_loss, correct, labelled, grads


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_unpadded_step_matches_padded(dropout):
    model = tiny_lora_qwen2(dropout)
    ids, mask, labels = batch()
    ref = step(model, False, ids, mask, labels)
    got = step(model, True, ids, mask, labels)
    real = mask.bool()
    # the masked (padded) and per-row causal SDPA kernels round differently in the last fp32 bits,
    # which the fp16 head matmul can carry one fp16 step further
    torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[1][real], ref[1][real], rtol=1e-3, atol=1e-3)  # entropy
    torch.testing.assert_close(got[2], ref[2], rtol=1e-3, atol=1e-3)  # loss per token
    assert torch.equal(got[4], ref[4])  # labelled tokens
    assert abs(int(got[3]) - int(ref[3])) <= 1  # correct argmax: a near-tie may flip
    for name, grad in ref[5].items():
        torch.testing.assert_close(got[5][name], grad, rtol=2e-3, atol=1e-5, msg=name)


def test_dropout_keeps_the_base_mask():
    from solution import unpad

    unpad.patch_dropout_full_mask()
    drop = torch.nn.Dropout(0.05).train()
    x = torch.randn(4, 37, 128, device="cuda")
    _, mask, _ = batch()
    lengths, index = unpad.layout(mask)
    torch.manual_seed(5)
    ref = drop(x)
    torch.manual_seed(5)
    with unpad.packed(lengths, 4, 37, index):
        got = drop(unpad.pack(x, index))
    assert torch.equal(got, unpad.pack(ref, index))


def test_left_padding_and_full_rows_keep_the_padded_path():
    from solution import unpad

    _, mask, _ = batch(left_padded=True)
    assert unpad.layout(mask) is None
    lengths, index = unpad.layout(torch.ones(2, 9, dtype=torch.long, device="cuda"))
    assert lengths == [9, 9] and index.numel() == 18
    assert unpad.layout(mask.cpu()) is None
