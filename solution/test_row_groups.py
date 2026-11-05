"""A rerun layer by groups of rows gives the step of the whole batch: same loss, per-token values,
LoRA gradients and random generator state afterwards.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

from solution.recompute import row_groups


def test_row_groups_cover_rows_in_order():
    assert row_groups([512, 512, 300, 106], 512) == [(0, 1, 0, 512), (1, 2, 512, 1024),
                                                     (2, 4, 1024, 1430)]
    assert row_groups([700, 10], 512) == [(0, 1, 0, 700), (1, 2, 700, 710)]
    assert row_groups([5, 5, 5], 100) == [(0, 3, 0, 15)]
    assert row_groups([3], 1) == [(0, 1, 0, 3)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("cap", [1, 40, 60])
def test_grouped_rerun_matches_whole_batch(cap):
    from solution import recompute
    from solution.test_unpad import batch, step, tiny_lora_qwen2

    model = tiny_lora_qwen2(0.05)
    ids, mask, labels = batch()

    def run(group_tokens):
        original = recompute.patch_qwen2_layer_recompute

        def patched(fp16=False, host=False, offload_activations=False, group_tokens_=0):
            original(fp16=fp16, host=host, offload_activations=offload_activations,
                     group_tokens=group_tokens)

        recompute.patch_qwen2_layer_recompute = patched
        try:
            got = step(model, True, ids, mask, labels, recompute=True, fp16_inputs=True, host=True)
        finally:
            recompute.patch_qwen2_layer_recompute = original
            recompute.GROUP_TOKENS = 0
        return got, torch.cuda.get_rng_state()

    ref, ref_rng = run(0)
    got, got_rng = run(cap)
    assert torch.equal(ref_rng, got_rng)  # the masks drew the same numbers, the rerun restored them
    assert got[0].item() == pytest.approx(ref[0].item(), rel=1e-5)
    torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-4)  # entropy
    torch.testing.assert_close(got[2], ref[2], rtol=1e-4, atol=1e-4)  # loss per token
    assert torch.equal(got[4], ref[4])
    for name, grad in ref[5].items():
        torch.testing.assert_close(got[5][name], grad, rtol=1e-4, atol=1e-5, msg=name)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_direct_nograd_gives_the_same_output():
    from solution import direct
    from solution.rmsnorm import rmsnorm

    x = torch.randn(4, 64, device="cuda")
    w = torch.ones(64, device="cuda")
    with torch.no_grad():
        ref = rmsnorm(x, w, 1e-6)
        direct.enabled = True
        try:
            got = rmsnorm(x, w, 1e-6)
        finally:
            direct.enabled = False
    assert torch.equal(ref, got)
    assert not vars(direct._context)  # nothing kept alive between calls
