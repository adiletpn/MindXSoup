"""fp16 storage of the tied embedding gives the same training forward and backward, bit for bit.

Run on a GPU: `python -m pytest solution -q`.
"""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def tiny_qwen2(tied=True):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=tied)
    torch.manual_seed(0)
    # an fp16 checkpoint upcast to fp32, as after prepare_model_for_kbit_training
    model = Qwen2ForCausalLM(config).half().float().cuda()
    for name, param in model.named_parameters():
        param.requires_grad_("q_proj" in name)  # stands in for the LoRA weights
    return model.train()


def step(model, ids, autocast):
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad()
    return out.loss.detach(), out.logits.detach(), grads


@pytest.mark.parametrize("tied", [True, False])
def test_training_step_is_bit_identical_under_autocast(tied):
    from solution.embedding import shrink_embedding

    base = tiny_qwen2(tied)
    ours = copy.deepcopy(base)
    shrink_embedding(ours)
    assert ours.get_input_embeddings().weight.dtype == torch.float16
    assert (ours.get_output_embeddings().weight.dtype == torch.float16) == tied
    ids = torch.randint(0, 1000, (2, 37), device="cuda")
    ref, got = step(base, ids, True), step(ours, ids, True)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])
    assert all(torch.equal(a, b) for a, b in zip(ref[2], got[2]))
    assert ours.model.embed_tokens(ids).dtype == torch.float32


def test_without_autocast_still_runs_close():
    from solution.embedding import shrink_embedding

    base = tiny_qwen2()
    ours = copy.deepcopy(base)
    shrink_embedding(ours)
    ids = torch.randint(0, 1000, (2, 37), device="cuda")
    ref, got = step(base, ids, False), step(ours, ids, False)
    assert got[1].dtype == torch.float32
    torch.testing.assert_close(got[0], ref[0], rtol=2e-3, atol=2e-3)


def test_inexact_weights_stay_fp32():
    from solution.embedding import shrink_embedding

    model = tiny_qwen2()
    with torch.no_grad():
        model.model.embed_tokens.weight.add_(1e-7)  # no longer fp16 numbers
    shrink_embedding(model)
    assert model.get_input_embeddings().weight.dtype == torch.float32
