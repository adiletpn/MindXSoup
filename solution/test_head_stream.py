"""The head over vocabulary slices of a host table against fused_ce.py's whole-row head, and the
offloaded table against the GPU one in a tiny tied Qwen2.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _host_weight(weight):
    from solution.head_stream import HostTable

    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
    host.copy_(weight)
    placeholder = torch.nn.Parameter(
        torch.empty((1, 1), dtype=weight.dtype, device="cuda").expand(weight.shape),
        requires_grad=False)
    placeholder._mindx_table = HostTable(host, weight.device)
    return placeholder


@pytest.mark.parametrize("vocab", [3 * 8192 + 384, 151936])
@pytest.mark.parametrize("rows", [5, 300, 1203])
def test_slices_match_the_whole_row_head(vocab, rows):
    from solution.fused_ce import fused_head_ce_shifted

    torch.manual_seed(0)
    weight = (torch.randn(vocab, 256, device="cuda") * 0.05).half()
    hidden = torch.randn(1, rows, 256, device="cuda", requires_grad=True)
    target = torch.randint(0, vocab, (1, rows), device="cuda")
    target[0, ::7] = -100
    target[0, 1] = vocab - 1  # the last slice's last column
    got = fused_head_ce_shifted(hidden, _host_weight(weight), target)
    got[0].backward()
    grad_got = hidden.grad.clone()
    hidden.grad = None
    want = fused_head_ce_shifted(hidden, weight, target)
    want[0].backward()
    assert torch.allclose(got[0], want[0], rtol=1e-4, atol=0)
    for a, b in zip(got[1:3], want[1:3]):  # entropy, per-token loss
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)
    assert torch.equal(got[3], want[3]) and torch.equal(got[4], want[4])  # correct, labelled
    scale = hidden.grad.abs().max()
    assert (grad_got - hidden.grad).abs().max() <= 2e-3 * scale


def test_offloaded_table_gives_the_same_model_outputs():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.embedding import shrink_embedding
    from solution.head_stream import offload_table, table_of

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1000, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).cuda()
    for param in model.parameters():
        param.requires_grad_(False)
    table = model.get_input_embeddings().weight
    table.data = table.data.half().float()  # fp16 numbers, as the checkpoint's
    shrink_embedding(model)
    ids = torch.randint(0, 1000, (2, 17), device="cuda")
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids)
        logits = model(input_ids=ids).logits
    assert offload_table(model)
    assert table_of(model.get_input_embeddings().weight) is not None
    assert model.get_input_embeddings().weight.untyped_storage().nbytes() <= 2
    with torch.no_grad():
        assert torch.equal(model.get_input_embeddings()(ids), embeds)
        assert torch.equal(model(input_ids=ids).logits, logits)
