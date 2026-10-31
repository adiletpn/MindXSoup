"""Backward input-gradient GEMMs of the frozen 4-bit layers on the T4: fp16 matmul vs int8
torch._int_mm (per-token gradient scale, weight column-major) plus the quantize / rescale passes
(here as plain torch ops, an upper bound on what kernels of our own would cost)."""

import torch

H, I, KV = 1536, 8960, 256
# (name, N = output dim of the layer = reduction dim of grad @ W, K = input dim)
SHAPES = [("q/o", H, H), ("k/v", KV, H), ("gate/up", I, H), ("down", H, I)]


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


for m in (1168, 2048):
    for name, n, k in SHAPES:
        grad = torch.randn(m, n, device="cuda", dtype=torch.float16)
        w = torch.randn(n, k, device="cuda", dtype=torch.float16)
        qg = torch.randint(-127, 127, (m, n), device="cuda", dtype=torch.int8)
        qwt = torch.randint(-127, 127, (k, n), device="cuda", dtype=torch.int8)  # [K, N] -> .t()
        sg = torch.rand(m, 1, device="cuda")
        fp16 = timed(lambda: grad @ w)
        gemm = timed(lambda: torch._int_mm(qg, qwt.t()))
        c = torch._int_mm(qg, qwt.t())
        quant = timed(lambda: (grad.float() / sg).round().to(torch.int8))
        rescale = timed(lambda: (c.float() * sg).half())
        # memory-bound lower bound for the two passes (fused kernels): bytes / 300 GB/s
        bound = (m * n * 3 + m * k * 6) / 300e9 * 1e3
        print(f"M={m:5d} {name:8s} fp16 {fp16:6.3f} ms | int8 gemm {gemm:6.3f} ms "
              f"(x{fp16 / gemm:4.2f}) | torch quant {quant:6.3f} + rescale {rescale:6.3f} | "
              f"fused-pass bound {bound:6.3f} | net x{fp16 / (gemm + bound):4.2f}", flush=True)
