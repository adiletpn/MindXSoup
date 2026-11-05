"""T4 diagnostics for the floor work: GEMM layout/alignment and what fills the CUDA context.

    python tools/floor_bench.py gemm
    python tools/floor_bench.py context

Not part of the solution; numbers go to docs/results.md.
"""

import subprocess
import sys
import time


def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                          "-i", "0"], capture_output=True, text=True).stdout
    return int(out.split()[0])


def gemm():
    import torch

    def timed(fn, reps=30):
        fn()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t) / reps)
        return best * 1e3

    dev = "cuda"
    print("backward dX = dY @ W, W row-major [out, in] vs the same W stored column-major", flush=True)
    print("shape | row-major ms | col-major ms | dY.t view (W @ dY^T) ms")
    for out_dim, in_dim in ((8960, 1536), (1536, 8960), (1536, 1536), (256, 1536)):
        w = torch.randn(out_dim, in_dim, dtype=torch.float16, device=dev)
        w_col = w.t().contiguous().t()  # same values, column-major strides
        for n in (640, 1024, 1430, 2048):
            dy = torch.randn(n, out_dim, dtype=torch.float16, device=dev)
            a = timed(lambda: torch.matmul(dy, w))
            b = timed(lambda: torch.matmul(dy, w_col))
            c = timed(lambda: torch.matmul(w.t(), dy.t()))
            print(f"N={n} W=[{out_dim},{in_dim}] | {a:.3f} | {b:.3f} | {c:.3f}", flush=True)

    print("\nforward F.linear: alignment of N (padding to a multiple helps cuBLAS?)", flush=True)
    for out_dim, in_dim in ((8960, 1536), (1536, 8960), (1536, 1536)):
        w = torch.randn(out_dim, in_dim, dtype=torch.float16, device=dev)
        for n in (1001, 1008, 1016, 1024, 1025, 1040, 1417, 1424, 1432, 1440):
            x = torch.randn(n, in_dim, dtype=torch.float16, device=dev)
            ms = timed(lambda: torch.nn.functional.linear(x, w))
            print(f"N={n} W=[{out_dim},{in_dim}] {ms:.3f} ms  {ms / n * 1000:.3f} us/token", flush=True)

    print("\nhead: fp16 256 rows x 151936", flush=True)
    w = torch.randn(151936, 1536, dtype=torch.float16, device=dev)
    w_col = w.t().contiguous().t()
    x = torch.randn(256, 1536, dtype=torch.float16, device=dev)
    g = torch.randn(256, 151936, dtype=torch.float16, device=dev)
    print(f"logits {timed(lambda: torch.mm(x, w.t()), 5):.2f} ms, "
          f"grad row {timed(lambda: torch.mm(g, w), 5):.2f} ms, "
          f"grad col {timed(lambda: torch.mm(g, w_col), 5):.2f} ms", flush=True)


def context():
    print("start", smi(), flush=True)
    import torch
    print("import torch", smi(), flush=True)
    torch.empty(1, device="cuda")
    time.sleep(0.5)
    print("first allocation (context)", smi(), torch.cuda.memory_reserved() >> 20, flush=True)
    a = torch.randn(16, 16, dtype=torch.float16, device="cuda")
    time.sleep(0.5)
    print("randn", smi(), torch.cuda.memory_reserved() >> 20, flush=True)
    a @ a
    torch.cuda.synchronize()
    time.sleep(0.5)
    print("first GEMM (cuBLAS)", smi(), torch.cuda.memory_reserved() >> 20, flush=True)
    torch.nn.functional.scaled_dot_product_attention(a[None, None], a[None, None], a[None, None])
    torch.cuda.synchronize()
    time.sleep(0.5)
    print("first SDPA", smi(), torch.cuda.memory_reserved() >> 20, flush=True)
    import triton
    import triton.language as tl

    @triton.jit
    def probe(p, n, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        tl.store(p + i, tl.exp(tl.load(p + i, mask=i < n, other=0.0)), mask=i < n)

    x = torch.zeros(1024, device="cuda")
    probe[(1,)](x, 1024, B=1024)
    torch.cuda.synchronize()
    time.sleep(0.5)
    print("one Triton kernel", smi(), torch.cuda.memory_reserved() >> 20, flush=True)
    torch.cuda.empty_cache()
    time.sleep(0.5)
    print("empty_cache", smi(), torch.cuda.memory_reserved() >> 20, flush=True)


def sweep():
    """GEMM cost per token against N; is a split into pieces faster than one call?"""
    import torch

    def timed(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(4):
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t) / reps)
        return best * 1e3

    shapes = ((8960, 1536), (1536, 8960), (1536, 1536), (256, 1536))
    for out_dim, in_dim in shapes:
        w = torch.randn(out_dim, in_dim, dtype=torch.float16, device="cuda")
        print(f"W=[{out_dim},{in_dim}] N ms us/token", flush=True)
        for n in list(range(32, 256, 32)) + list(range(256, 2049, 64)):
            x = torch.randn(n, in_dim, dtype=torch.float16, device="cuda")
            out = torch.empty(n, out_dim, dtype=torch.float16, device="cuda")
            ms = timed(lambda: torch.mm(x, w.t(), out=out))
            print(f"{n} {ms:.3f} {ms / n * 1000:.3f}", flush=True)
        for n, pieces in ((1430, 1), (1430, 2), (1430, 3), (1430, 4), (1000, 2), (1000, 4),
                          (830, 2), (2048, 2), (2048, 4)):
            x = torch.randn(n, in_dim, dtype=torch.float16, device="cuda")
            out = torch.empty(n, out_dim, dtype=torch.float16, device="cuda")
            cuts = [n * i // pieces for i in range(pieces + 1)]
            def run():
                for a, b in zip(cuts, cuts[1:]):
                    torch.mm(x[a:b], w.t(), out=out[a:b])
            print(f"split N={n} pieces={pieces} {timed(run):.3f} ms", flush=True)
