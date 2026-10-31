"""Compare budget 512 with unchanged budget 256 on one T4, in disposable copies.

The initial A measurement is an existing --only solution run. This adds B, B,
then A with the official harness and separate cold compiler caches. It does
not edit the source checkout, task, tests or harness, or claim official checks.
Run after GPU diagnostics have finished; no concurrent GPU work.
"""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--initial-runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier results")
    initial = json.loads((args.initial_runs / "result.json").read_text())
    if initial["runs"]["solution"][0]["exit_code"] != 0:
        raise ValueError("Initial A run failed")
    original = (source / "solution/recompute.py").read_text()
    marker = "BUDGET_TOKENS = 256"
    if original.count(marker) != 1:
        raise ValueError("Source is not the expected 256-token baseline")
    out.mkdir(parents=True)
    reports = {"A_initial": initial}
    for label, budget, repeats in (("B", 512, 2), ("A_final", 256, 1)):
        checkout = out / (label + "_source")
        shutil.copytree(source, checkout, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "runs*", "kaggle_out"))
        if budget != 256:
            (checkout / "solution/recompute.py").write_text(
                original.replace(marker, "BUDGET_TOKENS = 512"))
        run_out = out / (label + "_runs")
        command = [sys.executable, "harness/run_task.py", "--only", "solution",
                   "--repeats", str(repeats), "--out", str(run_out)]
        print("BUDGET_COMPARISON_START", label, budget, flush=True)
        with (out / (label + ".log")).open("w") as log:
            proc = subprocess.run(command, cwd=checkout, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"{label} failed; see {out / (label + '.log')}")
        report = json.loads((run_out / "result.json").read_text())
        if any(r["exit_code"] for r in report["runs"]["solution"]) or report["evals"]["solution"]["exit_code"]:
            raise RuntimeError(f"{label} training/evaluation failed")
        reports[label] = report
        print(label, json.dumps(report["summary"]), flush=True)
    a_runs = reports["A_initial"]["runs"]["solution"][:1] + reports["A_final"]["runs"]["solution"]
    b_runs = reports["B"]["runs"]["solution"]
    summary = {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "order": "A, diagnostics, B, B, A; evaluations between groups",
        "variants": {}, "limitations": ["No base run or ten-rule check",
        "One machine and seed; two training repeats per variant",
        "GPU/CPU diagnostics occurred between initial A and B",
        "Higher budget trades memory for speed; not a submission recommendation by itself"]}
    for label, budget, runs in (("A", 256, a_runs), ("B", 512, b_runs)):
        summary["variants"][label] = dict(budget=budget,
            wall_samples_s=[r["wall_s"] for r in runs],
            median_wall_s=statistics.median(r["wall_s"] for r in runs),
            peak_mib=max(r["gpu_peak_mib"] for r in runs),
            loop_samples_s=[r["metrics"]["train_loop_s"] for r in runs],
            step_medians_s=[r["metrics"]["median_step_s"] for r in runs])
    summary["eval_losses"] = {k: r["evals"]["solution"]["mean_loss"] for k, r in reports.items()}
    a, b = summary["variants"]["A"], summary["variants"]["B"]
    summary["speedup_A_over_B"] = a["median_wall_s"] / b["median_wall_s"]
    summary["wall_reduction_percent"] = 100 * (1 - b["median_wall_s"] / a["median_wall_s"])
    (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert (source / "solution/recompute.py").read_text() == original
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
