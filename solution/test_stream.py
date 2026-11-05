"""Streaming the 4-bit weights from the host gives bit-identical outputs and gradients to keeping
them on the GPU, with recomputation, with fewer GPU slots than layers (constant eviction) and with
or without the layer-order hints.

Run on a GPU: `python -m pytest solution -q`.
"""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOKENS = 1600  # past bitsandbytes' fused-kernel range, so both sides unpack + F.linear


def nf4_linear(n_in, n_out, bias):
    import bitsandbytes as bnb

    layer = bnb.nn.Linear4bit(n_in, n_out, bias=bias, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight.data = layer.weight.data.half()
    if bias:
        layer.bias.data = layer.bias.data.float()
    layer = layer.cuda()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nf4_linear(512, 512, True)
        self.up = nf4_linear(512, 1024, False)
        self.down = nf4_linear(1024, 512, False)
        self.scale = torch.nn.Parameter(torch.tensor(0.5, device="cuda"))

    def forward(self, h):
        h16 = h.half()
        mlp = self.down(torch.nn.functional.silu(self.up(h16)))
        return h + mlp.float() * self.scale + self.q(h16).float() * 0.1


def blocks(n=5):
    torch.manual_seed(0)
    return torch.nn.ModuleList(Block() for _ in range(n))


def run(model, hints):
    from solution import stream

    for block in model:
        block.scale.grad = None
    torch.manual_seed(1)
    h = torch.randn(1, TOKENS, 512, device="cuda", requires_grad=True)
    out = h
    for i, block in enumerate(model):
        phase = {"forward": True}

        def body(x, block=block, i=i, phase=phase):
            if hints and stream.active():
                stream.streamer().layer_begins(i, forward=phase["forward"])
            return block(x)

        out = checkpoint(body, out, use_reentrant=False)
        phase["forward"] = False
    out.square().mean().backward()
    return out.detach(), h.grad, [b.scale.grad.clone() for b in model]


@pytest.mark.parametrize("hints", [True, False])
@pytest.mark.parametrize("slots", [1, 2, 3, 4])
@pytest.mark.parametrize("coalesce", [True, False])
def test_streamed_is_bit_identical(hints, slots, coalesce, monkeypatch):
    from solution import linear4bit, stream
    from solution.nf4 import patch_bnb_nf4

    monkeypatch.setattr(stream, "COALESCE_TRANSFERS", coalesce)
    patch_bnb_nf4()
    linear4bit.patch_bnb_linear4bit()
    model = blocks()
    reference = copy.deepcopy(model)
    for layer in list(model.modules()) + list(reference.modules()):
        if hasattr(layer, "compute_type_is_set"):
            layer.compute_type_is_set = True
    ref = run(reference, hints=False)
    groups = [[b.q, b.up, b.down] for b in model]
    try:
        stream._streamer = stream.LayerStreamer(groups, slots=slots)
        assert all(m.weight.data.untyped_storage().nbytes() == 1 for g in groups for m in g)
        assert len(stream._streamer.host_layers) == (len(groups) if coalesce else 0)
        for group, host in zip(groups, stream._streamer.host_layers):
            assert host.is_pinned()
            assert host.numel() == sum(m.weight._mindx_host.numel()
                                       + m.weight._mindx_absmax.numel() for m in group)
            assert all(m.weight._mindx_host.untyped_storage().data_ptr()
                       == host.untyped_storage().data_ptr() for m in group)
            assert all(m.weight.quant_state.absmax.untyped_storage().nbytes() == 1 for m in group)
        got = run(model, hints)
        got2 = run(model, hints)  # a second step reuses the host copies
    finally:
        stream._streamer = None
    for a, b in ((got, ref), (got2, ref)):
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert all(torch.equal(x, y) for x, y in zip(a[2], b[2]))


def test_setup_skips_models_it_cannot_stream():
    from solution import stream

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

    stream.enabled = True
    try:
        stream.setup(Fake())
        assert not stream.active()
    finally:
        stream.enabled = False


def test_offload_at_load_gives_the_same_model(tmp_path):
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig, Qwen2Config, Qwen2ForCausalLM

    from solution import linear4bit, stream
    from solution.nf4 import patch_bnb_nf4

    patch_bnb_nf4()
    linear4bit.patch_bnb_linear4bit()
    config = Qwen2Config(vocab_size=512, hidden_size=256, intermediate_size=512,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).half().save_pretrained(tmp_path)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    load = lambda: Qwen2ForCausalLM.from_pretrained(  # noqa: E731
        tmp_path, quantization_config=quant, device_map={"": 0}, dtype=torch.float16)
    reference = load()
    stream.enabled = True
    stream.patch_offload_at_load()
    try:
        got = load()
    finally:
        stream.enabled = False
    layers = [m for m in got.model.layers.modules() if isinstance(m, bnb.nn.Linear4bit)]
    assert len(layers) == 14 and all(stream.offloaded(m) for m in layers)
    assert all(m.weight.data.untyped_storage().nbytes() == 1 for m in layers)
    ids = torch.randint(0, 512, (1, TOKENS), device="cuda")
    with torch.no_grad():
        assert torch.equal(reference(input_ids=ids).logits, got(input_ids=ids).logits)
