"""Summarize a harness CUDA profile; optionally collect an unpatched baseline profile."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def category(name):
    name = name.lower().replace("nocast", "")
    # Specific operations precede generic elementwise names.
    groups = (
        ("nf4 dequant", ("dequantize", "kdequantize")),
        ("attention", ("fmha", "attention", "efficient")),
        ("matmul", ("gemm", "gemv", "cutlass", "sm75_xmma")),
        ("softmax/loss", ("softmax", "log_softmax", "nll")),
        ("cast/copy", ("copy", "cast", "to_copy")),
        ("reduce", ("reduce",)),
        ("elementwise", ("elementwise", "vectorized", "mul", "add")),
    )
    return next((group for group, words in groups if any(w in name for w in words)), "other")


def summarize(profile, top=30):
    kernels = sorted(profile["kernels"], key=lambda row: -row["time_us"])
    total = float(profile["total_gpu_us"])
    if total <= 0 or not kernels:
        raise ValueError("Profile contains no positive GPU time")
    groups = defaultdict(float)
    for row in kernels:
        if row["time_us"] < 0:
            raise ValueError("Negative kernel duration")
        groups[category(row["name"])] += row["time_us"]
    lines = ["# Baseline GPU profile", "",
             f"Active profiled steps: {profile.get('active_steps', 'not recorded')}",
             f"Total GPU time: {total / 1000:.3f} ms", "",
             "Groups use kernel-name heuristics; this is GPU time, not wall time or own-kernel share.",
             "", "| Group | ms | % of GPU time |", "|---|---:|---:|"]
    for group, duration in sorted(groups.items(), key=lambda item: -item[1]):
        lines.append(f"| {group} | {duration / 1000:.3f} | {100 * duration / total:.2f} |")
    lines += ["", f"Top {min(top, len(kernels))} kernels", "",
              "| Kernel | ms | % | Calls |", "|---|---:|---:|---:|"]
    for row in kernels[:top]:
        name = row['name'].replace('|', '\\|').replace('\n', ' ').replace('`', "'")
        lines.append(f"| `{name}` | {row['time_us'] / 1000:.3f} | "
                     f"{100 * row['time_us'] / total:.2f} | {row['count']} |")
    return '\n'.join(lines) + '\n'


def collect_baseline(out):
    """The no-op solution exists only in a disposable copy, never in the team checkout."""
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="mindx-base-profile-") as scratch:
        copy = Path(scratch) / "repo"
        shutil.copytree(repo, copy, ignore=shutil.ignore_patterns(
            ".git", "runs*", "kaggle_out", "__pycache__", ".pytest_cache"))
        (copy / "solution" / "__init__.py").write_text("def apply():\n    pass\n", encoding="utf-8")
        subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution",
                        "--out", str(out / "baseline_profile"), "--profile"], cwd=copy, check=True)
    path = out / "baseline_profile/solution_profile/profile.json"
    if not path.is_file():
        raise RuntimeError("Baseline profile failed; inspect baseline_profile logs")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?", type=Path)
    parser.add_argument("--run-base-profile", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/out"))
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    if not args.profile and not args.run_base_profile:
        parser.error("provide a profile.json or --run-base-profile")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    path = collect_baseline(args.out) if args.run_base_profile else args.profile
    report = summarize(json.loads(path.read_text(encoding="utf-8")), args.top)
    (args.out / "profile_base.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
