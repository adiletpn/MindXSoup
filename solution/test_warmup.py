"""The warm-up dummies compile the same variants as the real training calls: after a warm-up job,
a call with training shapes must not compile anything new.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

BATCH, TOKENS = 4, 301  # an odd length, as real batches have


def variants(kernel):
    return sum(len(cache[0]) for cache in kernel.device_caches.values())


def check_no_recompile(kernels, warm, real):
    for kernel in kernels:
        if not hasattr(kernel, "device_caches"):
            pytest.skip("Triton's cache API differs")
        kernel.device_caches.clear()
    warm()
    torch.cuda.synchronize()
    after_warm = [variants(k) for k in kernels]
    assert all(after_warm), "warm-up compiled nothing"
    real()
    torch.cuda.synchronize()
    assert [variants(k) for k in kernels] == after_warm


def test_rmsnorm():
    from solution import warmup
    from solution.rmsnorm import rmsnorm, rmsnorm_bwd_kernel, rmsnorm_fwd_kernel

    def real():
        x = torch.randn(BATCH, TOKENS, 1536, device="cuda", requires_grad=True)
        weight = torch.ones(1536, device="cuda")
        rmsnorm(x, weight, 1e-6).sum().backward()

    check_no_recompile([rmsnorm_fwd_kernel, rmsnorm_bwd_kernel], warmup.compile_rmsnorm, real)


def test_rope():
    from solution import warmup
    from solution.rope import apply_rope, rope_kernel

    def real():
        q_in = torch.randn(BATCH, TOKENS, 1536, device="cuda", requires_grad=True)
        k_in = torch.randn(BATCH, TOKENS, 256, device="cuda", requires_grad=True)
        q = q_in.view(BATCH, TOKENS, 12, 128).transpose(1, 2)
        k = k_in.view(BATCH, TOKENS, 2, 128).transpose(1, 2)
        for cos_batch in (1, BATCH):  # position_ids given per row or broadcast
            cos = torch.randn(cos_batch, TOKENS, 128, device="cuda")
            sin = torch.randn(cos_batch, TOKENS, 128, device="cuda")
            q_out, k_out = apply_rope(q, k, cos, sin)
            # SDPA under autocast takes fp16 copies; their backward hands back fp32 grads
            (q_out.half().float().sum() + k_out.half().float().sum()).backward()

    check_no_recompile([rope_kernel], warmup.compile_rope, real)


def test_head_ce():
    from solution import warmup
    from solution.fused_ce import fused_head_ce, head_ce_rows_kernel

    def real():
        hidden = torch.randn(BATCH, TOKENS, 1536, device="cuda", requires_grad=True)
        weight = torch.randn(151936, 1536, device="cuda", dtype=torch.float16) * 0.02
        labels = torch.randint(0, 151936, (BATCH, TOKENS), device="cuda")
        labels[:, :50] = -100
        loss = fused_head_ce(hidden, weight, labels)[0]
        loss.backward()

    check_no_recompile([head_ce_rows_kernel], warmup.compile_head_ce, real)


def test_nf4():
    import bitsandbytes as bnb

    from solution import warmup
    from solution.nf4 import nf4_unpack_kernel, patch_bnb_nf4

    patch_bnb_nf4()

    def real():
        # M > 1536 rows takes bitsandbytes' dequantize + F.linear path, backward always does
        for out_features in (1536, 256, 8960):
            layer = bnb.nn.Linear4bit(1536, out_features, bias=False,
                                      compute_dtype=torch.float16, compress_statistics=True,
                                      quant_type="nf4").cuda()
            x = torch.randn(BATCH, TOKENS, 1536, device="cuda", dtype=torch.float16,
                            requires_grad=True)
            layer(x).sum().backward()

    check_no_recompile([nf4_unpack_kernel], warmup.compile_nf4, real)


def test_example_swiglu():
    from solution import warmup
    from solution.swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel, swiglu

    def real():
        gate = torch.randn(BATCH, TOKENS, 8960, device="cuda", dtype=torch.float16,
                           requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        swiglu(gate, up).float().sum().backward()

    check_no_recompile([example_swiglu_fwd_kernel, example_swiglu_bwd_kernel],
                       warmup.compile_example_swiglu, real)


def test_start_and_wait_in_a_thread():
    from solution import warmup

    warmup._thread = None
    warmup.start([warmup.compile_rmsnorm, warmup.compile_rope])
    assert warmup._thread is not None
    warmup.wait()
    assert not warmup._thread.is_alive()
