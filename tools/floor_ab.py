"""A/B of solution profiles on one machine: the given order of harness runs in one session.

    python tools/floor_ab.py rules_balanced rules_x rules_x rules_balanced

Each run is `harness/run_task.py --only solution` with PROFILE set in this checkout's copy (Kaggle:
a disposable one) and restored afterwards. Prints wall, peak and the median per-step ratio against
the first profile: batches are the same, so paired steps cancel most of the machine noise.
"""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INIT = REPO / "solution" / "__init__.py"


def main():
    original = INIT.read_text()
    rows = []
    try:
        for i, name in enumerate(sys.argv[1:]):
            INIT.write_text(re.sub(r'^PROFILE = .*$', f'PROFILE = "{name}"', original, flags=re.M))
            out = REPO / f"runs_ab_{i}"
            subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution", "--out", str(out)],
                           cwd=REPO, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result = json.loads((out / "result.json").read_text())["summary"]["solution"]
            if "wall_s" not in result:  # the run failed: show why, then stop
                print(name, "FAILED", result, flush=True)
                for log in sorted(out.rglob("*.log")) + sorted(out.rglob("*.txt")):
                    print(f"--- {log}", "\n".join(log.read_text(errors="replace").splitlines()[-40:]), flush=True)
                return
            metrics = json.loads((out / "solution_1" / "metrics.json").read_text())
            rows.append((name, result, metrics))
            print(name, result["wall_s"], "s", result["gpu_peak_mib"], "MiB eval", result["eval_loss"],
                  "loop", metrics["train_loop_s"], "median", metrics["median_step_s"],
                  "max_alloc", metrics["torch_max_allocated_mib"], "reserved",
                  metrics["torch_max_reserved_mib"], flush=True)
    finally:
        INIT.write_text(original)
    first = rows[0][2]["step_s"][2:]
    groups = {}
    for name, result, metrics in rows:
        ratio = statistics.median(b / a for a, b in zip(first, metrics["step_s"][2:]))
        groups.setdefault(name, []).append((ratio, result["wall_s"], result["gpu_peak_mib"]))
    for name, values in groups.items():
        print(f"{name}: step ratio {[round(v[0], 3) for v in values]} wall {[v[1] for v in values]} "
              f"peak {[v[2] for v in values]}", flush=True)


if __name__ == "__main__":
    main()
