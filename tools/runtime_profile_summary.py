"""Summarize an unchanged harness --only solution --profile run.

GPU categories are name-based and include overlapping transfers. Their sum is
not wall time, and it cannot establish GPU idle time. Timings under the profiler
are never substituted for the unprofiled benchmark.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def category(name):
    name = name.lower().replace("nocast", "")
    groups = (
        ("host_to_device", ("memcpy htod", "memcpy hto d", "host to device")),
        ("device_to_host", ("memcpy dtoh", "device to host")),
        ("nf4_unpack", ("nf4_unpack", "dequantize")),
        ("head_loss_reduction", ("head_ce", "head_online", "cross_entropy", "softmax", "nll")),
        ("attention", ("fmha", "attention", "efficient")),
        ("matmul", ("gemm", "gemv", "cutlass", "sm75_xmma")),
        ("rmsnorm", ("rmsnorm",)),
        ("rope_qkv", ("rope", "qkv")),
        ("dropout", ("dropout",)),
        ("swiglu", ("swiglu",)),
        ("cast_copy", ("copy", "cast")),
        ("reduction", ("reduce",)),
        ("elementwise", ("elementwise", "vectorized",)),
    )
    return next((group for group, words in groups if any(w in name for w in words)), "other")


def summarize(root):
    root = Path(root)
    result = json.loads((root / "result.json").read_text())
    run = result["runs"]["solution"][0]
    metrics = json.loads((root / run["run"] / "metrics.json").read_text())
    profile_run = result["profile_run"]
    if run["exit_code"] != 0 or not profile_run or profile_run["exit_code"] != 0:
        raise ValueError("Training or profile failed; inspect logs")
    profile = json.loads((root / profile_run["run"] / "profile.json").read_text())
    total = profile["total_gpu_us"]
    if total <= 0:
        raise ValueError("Empty GPU profile")
    groups = defaultdict(lambda: {"us": 0, "calls": 0})
    for row in profile["kernels"]:
        group = groups[category(row["name"])]
        group["us"] += row["time_us"]
        group["calls"] += row["count"]
    steps = profile["active_steps"]
    return {
        "unprofiled": {
            "wall_s": run["wall_s"],
            "train_loop_s": metrics["train_loop_s"],
            "outside_loop_s": round(run["wall_s"] - metrics["train_loop_s"], 3),
            "apply_s_included_in_wall": metrics["apply_s"],
            "steps": metrics["steps"],
            "median_step_s": metrics["median_step_s"],
            "first_step_s": metrics["step_s"][0],
            "gpu_peak_mib": run["gpu_peak_mib"],
            "eval_loss": result["evals"]["solution"].get("mean_loss"),
        },
        "profile_active_steps": steps,
        "summed_gpu_ms_per_step": total / steps / 1000,
        "groups": [{"name": name, "ms_per_step": data["us"] / steps / 1000,
                    "percent_gpu_sum": 100 * data["us"] / total,
                    "calls_per_step": data["calls"] / steps}
                   for name, data in sorted(groups.items(), key=lambda item: -item[1]["us"])],
        "top_kernels": sorted(profile["kernels"], key=lambda row: -row["time_us"])[:20],
        "limitations": ["One unprofiled run, not a paired speedup comparison",
                        "GPU event durations can overlap across streams",
                        "Kernel-name categories do not distinguish head GEMMs from decoder GEMMs",
                        "Outside-loop time includes imports, loading, preparation and saving",
                        "No base run; no claim of all ten rule checks"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.runs), indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
