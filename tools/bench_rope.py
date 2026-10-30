"""Benchmark the RoPE kernel against transformers' apply_rotary_pos_emb on the task's shapes.

    python tools/bench_rope.py

Prints: first-call (compile) time of the forward and backward kernels with an empty Triton cache,
whether new sequence lengths compile again, and forward / forward+backward time of both versions
(10 warm-up iterations, 100 timed, CUDA events).
"""

import os
import sys
import tempfile
import time
from pathlib import Path

# An empty Triton cache, as in the jury's run, so the first call shows the real compile time.
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_bench_rope_")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from solution.rope import apply_rope, rope_bwd_kernel, rope_fwd_kernel  # noqa: E402
from solution.test_rope import model_like_cos_sin, model_like_qk, reference  # noqa: E402

BATCH, N_Q, N_K, N_TOKENS, LAYERS = 4, 12, 2, 512, 28


def compiled_variants(kernel):
    """Number of compiled variants Triton holds for a kernel (None if the API differs)."""
    try:
        return sum(len(cache[0]) for cache in kernel.device_caches.values())
    except Exception:  # noqa: BLE001 - internal Triton API, only for the report
        return None


def fwd_bwd(fn, q, k, cos, sin, grads):
    out = fn(q, k, cos, sin)
    torch.autograd.grad(out, (q, k), grads)


def timed_first_call(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def cuda_ms(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def inputs(n_tokens, dtype):
    q, k = model_like_qk(BATCH, N_Q, N_K, n_tokens, dtype)
    cos, sin = model_like_cos_sin(BATCH, n_tokens, dtype)
    grads = (torch.randn_like(q), torch.randn_like(k))
    return q, k, cos, sin, grads


def main():
    torch.manual_seed(0)
    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    import triton

    print("triton", triton.__version__)

    print("\n== First call with an empty Triton cache (fp32, T=512) ==")
    q, k, cos, sin, grads = inputs(N_TOKENS, torch.float32)
    holder = {}
    fwd_s = timed_first_call(lambda: holder.setdefault("out", apply_rope(q, k, cos, sin)))
    bwd_s = timed_first_call(lambda: torch.autograd.grad(holder["out"], (q, k), grads))
    print(f"forward kernel  first call: {fwd_s:6.3f} s")
    print(f"backward kernel first call: {bwd_s:6.3f} s")
    print(f"total compile + first run : {fwd_s + bwd_s:6.3f} s")

    print("\n== New sequence lengths must not compile again (fp32) ==")
    before = compiled_variants(rope_fwd_kernel), compiled_variants(rope_bwd_kernel)
    for n_tokens in (100, 200, 300, 77, 511):
        args = inputs(n_tokens, torch.float32)
        seconds = timed_first_call(lambda: fwd_bwd(apply_rope, *args))
        print(f"T={n_tokens:4d}: first fwd+bwd call {1000 * seconds:7.2f} ms")
    after = compiled_variants(rope_fwd_kernel), compiled_variants(rope_bwd_kernel)
    print(f"compiled variants fwd/bwd: before {before}, after {after}"
          + ("  -> no recompilation" if before == after and None not in before else ""))

    print("\n== Speed, B=4 T=512, 12 q heads + 2 k heads, head_dim 128 ==")
    print(f"{'dtype':6} {'pass':8} {'PyTorch, ms':>12} {'ours, ms':>9} {'speedup':>8} "
          f"{'saved per step (28 layers), ms':>31}")
    for dtype in (torch.float32, torch.float16):
        q, k, cos, sin, grads = inputs(N_TOKENS, dtype)
        with torch.no_grad():
            ref_f = cuda_ms(lambda: reference(q, k, cos, sin))
            our_f = cuda_ms(lambda: apply_rope(q, k, cos, sin))
        ref_fb = cuda_ms(lambda: fwd_bwd(reference, q, k, cos, sin, grads))
        our_fb = cuda_ms(lambda: fwd_bwd(apply_rope, q, k, cos, sin, grads))
        name = str(dtype).replace("torch.", "")
        for label, ref_ms, our_ms in (("fwd", ref_f, our_f), ("fwd+bwd", ref_fb, our_fb)):
            print(f"{name:6} {label:8} {ref_ms:12.3f} {our_ms:9.3f} {ref_ms / our_ms:7.2f}x "
                  f"{LAYERS * (ref_ms - our_ms):31.2f}")


if __name__ == "__main__":
    main()
