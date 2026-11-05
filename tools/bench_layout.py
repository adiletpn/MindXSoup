"""Do gate/up matmuls get faster with W^T stored row-major? (PERF_IDEAS idea 1.) T4, same session."""
import torch


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


w = torch.randn(8960, 1536, dtype=torch.float16, device="cuda")
wt = w.t().contiguous()  # [1536, 8960], row-major
print(f"transpose copy {timed(lambda: w.t().contiguous(), 5):.3f} ms")
for rounds in range(2):
    for n in (354, 512, 640, 775, 832, 1024, 1296, 1430):
        x = torch.randn(n, 1536, dtype=torch.float16, device="cuda")
        g = torch.randn(n, 8960, dtype=torch.float16, device="cuda")
        print(n, f"fwd {timed(lambda: torch.mm(x, w.t())):.3f} -> {timed(lambda: torch.mm(x, wt)):.3f}",
              f"bwd {timed(lambda: torch.mm(g, w)):.3f} -> {timed(lambda: torch.mm(g, wt.t())):.3f}",
              flush=True)
