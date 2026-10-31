"""Benchmark NF4 unpacking on the task's weight shapes: bitsandbytes, our 1-byte kernel, our wide kernel.

    python tools/bench_nf4.py

For each shape: time per unpack (CUDA events, 50 iterations after 10 warm-up), effective bandwidth
(packed bytes read + fp16 bytes written + 8-bit absmax read), and whether the result is
bit-identical to our current kernel and to bitsandbytes. Then the configurations of the wide kernel
(blocks per program x warps) and an estimate per training step: every Linear4bit of the 28 layers
is unpacked three times per step with full recomputation (forward, rerun, backward).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import triton  # noqa: E402

from solution import nf4  # noqa: E402

# (out, in) of q, k, v, o, gate, up, down in Qwen2.5-1.5B; one layer has each once
LAYER = {"q/o 1536x1536": ((1536, 1536), 2), "k/v 256x1536": ((256, 1536), 2),
         "gate/up 8960x1536": ((8960, 1536), 2), "down 1536x8960": ((1536, 8960), 1)}
LAYERS, UNPACKS_PER_STEP = 28, 3
CONFIGS = [(32, 4), (64, 8), (128, 8)]  # (blocks per program, warps); 64 x 8 is the default


def quantized(shape):
    import bitsandbytes.functional as F

    torch.manual_seed(0)
    weight = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02
    return F.quantize_4bit(weight, blocksize=64, quant_type="nf4", compress_statistics=True)


def bnb_unpack(packed, state):
    import bitsandbytes.functional as F

    absmax = F.dequantize_blockwise(state.absmax, state.state2) + state.offset
    return torch.ops.bitsandbytes.dequantize_4bit.default(
        packed, absmax.float(), state.blocksize, state.quant_type, state.shape, state.dtype)


def old_unpack(packed, state):
    n_bytes = packed.numel()
    out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    nf4.nf4_unpack_kernel[(triton.cdiv(n_bytes, nf4.BYTES_PER_PROGRAM),)](
        packed, out, nf4._nf4_table(packed.device), state.absmax, state.state2.code,
        state.state2.absmax, state.offset, n_bytes, NESTED=True, BYTES=nf4.BYTES_PER_PROGRAM,
        num_warps=4)
    return out


def wide_unpack(packed, state, blocks, warps):
    out = torch.empty(state.shape, dtype=state.dtype, device=packed.device)
    nf4._launch_wide(packed, out, state.state2.absmax, state.absmax, state.state2.code,
                     state.offset, packed.numel() // 32, blocks, warps)
    return out


def ms(fn, warmup=10, iters=50):
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


def first_call_s(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def main():
    print(torch.cuda.get_device_name(), "| torch", torch.__version__, "| triton", triton.__version__)
    weights = {name: quantized(shape) for name, (shape, _) in LAYER.items()}
    # compile both kernels outside the timings (the first call of each config compiles)
    packed, state = weights["k/v 256x1536"]
    print(f"compile old kernel: {first_call_s(lambda: old_unpack(packed, state)):.2f} s")
    print(f"compile wide kernel (default {nf4.WIDE_BLOCKS}x{nf4.WIDE_WARPS}): "
          f"{first_call_s(lambda: wide_unpack(packed, state, nf4.WIDE_BLOCKS, nf4.WIDE_WARPS)):.2f} s")

    rows, per_step = [], {"bitsandbytes": 0.0, "old": 0.0}
    per_step.update({f"wide {b}x{w}": 0.0 for b, w in CONFIGS})
    for name, (shape, count) in LAYER.items():
        packed, state = weights[name]
        moved = packed.numel() + 2 * state.shape[0] * state.shape[1] + state.absmax.numel()
        ref_bnb = bnb_unpack(packed, state)
        ref_old = old_unpack(packed, state)
        variants = {"bitsandbytes": lambda: bnb_unpack(packed, state),
                    "old": lambda: old_unpack(packed, state)}
        for b, w in CONFIGS:
            variants[f"wide {b}x{w}"] = (lambda b=b, w=w: wide_unpack(packed, state, b, w))
        for label, fn in variants.items():
            t = ms(fn)
            out = fn()
            same_old = torch.equal(out, ref_old)
            ulp_bnb = int((out.view(torch.int16).int() - ref_bnb.view(torch.int16).int())
                          .abs().max())
            rows.append((name, label, t, moved / t / 1e6, same_old, ulp_bnb))
            per_step[label] += count * t
    print(f"\n{'shape':20} {'variant':18} {'ms':>8} {'GB/s':>7} {'= old':>6} {'max ulp vs bnb':>15}")
    for name, label, t, gbs, same_old, ulp in rows:
        print(f"{name:20} {label:18} {t:8.4f} {gbs:7.1f} {str(same_old):>6} {ulp:15d}")
    print(f"\n== per training step: {LAYERS} layers x {UNPACKS_PER_STEP} unpacks of q,k,v,o,gate,up,down ==")
    old_step = per_step["old"] * LAYERS * UNPACKS_PER_STEP
    for label, t in per_step.items():
        step = t * LAYERS * UNPACKS_PER_STEP
        print(f"{label:18} {step:7.1f} ms per step   ({old_step - step:+6.1f} ms vs old)")


if __name__ == "__main__":
    main()
