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
