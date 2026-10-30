"""The direct NF4 linear path gives bit-identical outputs and input gradients to bitsandbytes'
own Linear4bit path (which, with our NF4 patch, runs the same unpack kernel).

Run on a GPU: `python -m pytest solution -q`.
"""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def fp16_layer(out_features, bias):
    import bitsandbytes as bnb

    torch.manual_seed(0)
    layer = bnb.nn.Linear4bit(1536, out_features, bias=bias, compute_dtype=torch.float16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight.data = layer.weight.data.half()  # Soup loads the checkpoint in fp16 on a T4
    if bias:
        layer.bias.data = layer.bias.data.float()  # prepare_model_for_kbit_training upcasts it
    layer = layer.cuda()
    for param in layer.parameters():
        param.requires_grad_(False)
    return layer


def run(layer, x):
    x = x.detach().requires_grad_(True)
    out = layer(x)
    out.backward(torch.randn_like(out))
    return out.detach(), x.grad


@pytest.mark.parametrize("out_features,bias", [(1536, True), (256, True), (8960, False),
                                               (1536, False)])
@pytest.mark.parametrize("tokens", [37, 1203])
def test_direct_path_is_bit_identical(out_features, bias, tokens):
    from solution.linear4bit import patch_bnb_linear4bit
    from solution.nf4 import patch_bnb_nf4

    patch_bnb_nf4()
    patch_bnb_linear4bit()
    layer = fp16_layer(out_features, bias)
    reference = copy.deepcopy(layer)
    x = torch.randn(1, tokens, 1536, device="cuda", dtype=torch.float16)
    layer(x)  # first call sets the compute dtype in both
    reference(x)
    torch.manual_seed(1)
    got = run(layer, x)
    # the reference goes through bitsandbytes' own forward
    from solution import linear4bit

    torch.manual_seed(1)
    with linear4bit.disabled():
        ref = run(reference, x)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_fp32_input_takes_the_original_path():
    from solution import linear4bit

    linear4bit.patch_bnb_linear4bit()
    layer = fp16_layer(256, True)
    x = torch.randn(1, 9, 1536, device="cuda")
    layer(x)
    assert not linear4bit._direct_ok(layer, x)
