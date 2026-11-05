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


def launchcost():
    """CPU cost of each kind of call on this machine, and which calls wait for the GPU."""
    import torch
    import torch.nn.functional as F
    import triton
    import triton.language as tl

    @triton.jit
    def probe(p, n, B: tl.constexpr):
        i = tl.program_id(0) * n + tl.arange(0, B)
        tl.store(p + i, tl.load(p + i) + 1.0)

    d = "cuda"
    n = 1000
    x16 = torch.randn(1, n, 1536, dtype=torch.float16, device=d)
    x32 = x16.float()
    w = torch.randn(8640, 1536, dtype=torch.float16, device=d)
    w_small = torch.randn(1536, 1536, dtype=torch.float16, device=d)
    big = torch.randn(4096, 4096, dtype=torch.float16, device=d)
    small = torch.zeros(1024, device=d)
    q = torch.randn(1, 12, 300, 128, dtype=torch.float16, device=d)
    pinned = torch.empty(4 << 20, dtype=torch.uint8, pin_memory=True)
    slot = torch.empty(4 << 20, dtype=torch.uint8, device=d)
    rng = torch.cuda.get_rng_state(0)
    event = torch.cuda.Event()
    event.record()
    other = torch.cuda.Stream()
    shapes = iter(range(1100, 100000))

    def fresh_shape():
        F.linear(torch.empty(1, next(shapes) % 1400 + 1, 1536, dtype=torch.float16, device=d), w)

    def grad_node():
        a = x32.detach().requires_grad_(True)
        (a * 2.0).sum().backward()

    def autocast_linear():
        with torch.autocast("cuda", dtype=torch.float16):
            F.linear(x16, w)

    def fork():
        with torch.random.fork_rng(devices=[0]):
            torch.cuda.set_rng_state(rng, 0)

    calls = {
        "python no-op": lambda: None,
        "empty fp16 [1,n,1536]": lambda: torch.empty_like(x16),
        "view": lambda: x16.view(n, 1536),
        "fill_ small": lambda: small.fill_(1.0),
        "add_ small": lambda: small.add_(1.0),
        "cast fp32->fp16": lambda: x32.to(torch.float16),
        "triton launch small": lambda: probe[(1,)](small, 16, B=16),
        "F.linear small weight, same shape": lambda: F.linear(x16, w_small),
        "F.linear gate weight, same shape": lambda: F.linear(x16, w),
        "F.linear, rows never seen": fresh_shape,
        "F.linear under autocast": autocast_linear,
        "matmul grad @ W": lambda: torch.matmul(x16[:, :, :8640 // 8640 * 1536], w_small),
        "sdpa causal": lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True),
        "dropout draw": lambda: F.dropout(torch.ones(4, 512, 1536, device=d), 0.05, True),
        "autograd mul+sum backward": grad_node,
        "get_rng_state": lambda: torch.cuda.get_rng_state(0),
        "set_rng_state": lambda: torch.cuda.set_rng_state(rng, 0),
        "fork_rng + set": fork,
        "Event.record": lambda: torch.cuda.Event().record(),
        "wait_event": lambda: torch.cuda.current_stream().wait_event(event),
        "pinned empty 4 MiB": lambda: torch.empty(4 << 20, dtype=torch.uint8, pin_memory=True),
        "HtoD non_blocking 4 MiB": lambda: slot.copy_(pinned, non_blocking=True),
        "pinned 4 MiB + DtoH": lambda: torch.empty(4 << 20, dtype=torch.uint8, pin_memory=True)
        .copy_(slot, non_blocking=True),
    }
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    print(f"{'call':38s} {'cpu us':>9s} {'median':>9s} {'after long gemm us':>19s}")
    for name, fn in calls.items():
        times = []
        for _ in range(200):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        times.sort()
        # a call issued right after a long GPU job: does it return at once, or wait for the job?
        waits = []
        for _ in range(5):
            torch.cuda.synchronize()
            torch.matmul(big, big)
            torch.matmul(big, big)
            t = time.perf_counter()
            fn()
            waits.append(time.perf_counter() - t)
        waits.sort()
        print(f"{name:38s} {sum(times) / len(times) * 1e6:9.1f} {times[100] * 1e6:9.1f} {waits[2] * 1e6:19.1f}")
    torch.cuda.synchronize()
    t = time.perf_counter()
    torch.matmul(big, big)
    torch.matmul(big, big)
    torch.cuda.synchronize()
    print(f"long job itself {(time.perf_counter() - t) * 1e6:.0f} us")


def masks():
    """Same random numbers from a cheaper dropout input? Compared bit for bit with fp32 ones."""
    import torch

    shape = (4, 512, 1536)
    def draw(x):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)
        return torch.nn.functional.dropout(x, 0.05, True) != 0
    ref = draw(torch.ones(shape, device="cuda"))
    print("kept share", ref.float().mean().item())
    for name, x in (("fp16 ones", torch.ones(shape, dtype=torch.float16, device="cuda")),
                    ("fp32 expanded", torch.ones(1, device="cuda").expand(shape)),
                    ("fp16 expanded", torch.ones(1, dtype=torch.float16, device="cuda").expand(shape))):
        try:
            print(name, "identical:", torch.equal(draw(x), ref), flush=True)
        except Exception as exc:
            print(name, "error", repr(exc), flush=True)
    state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    a = torch.nn.functional.dropout(torch.ones(shape, device="cuda"), 0.05, True)
    after_a = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    b = torch.nn.functional.dropout(torch.ones(1, dtype=torch.float16, device="cuda").expand(shape), 0.05, True)
    print("rng state after equal:", torch.equal(after_a, torch.cuda.get_rng_state()), flush=True)


if __name__ == "__main__":
    {"gemm": gemm, "context": context, "sweep": sweep, "masks": masks, "launchcost": launchcost}[sys.argv[1]]()
