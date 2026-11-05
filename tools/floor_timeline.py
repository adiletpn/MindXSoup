"""GPU timeline of the solution's training step: busy, overlap, idle gaps, with low overhead.

    python tools/floor_timeline.py run       # harness run with --profile, CUDA activity only
    python tools/floor_timeline.py analyse <trace.json>

`run` puts a sitecustomize hook on PYTHONPATH for harness/_train.py: torch.profiler records CUDA
activity only (the CPU activity of the harness's profiler makes a run over twice slower, which
would show up as idle GPU) and the trace is exported when the profiler stops. Nothing in harness/
changes.
"""

import collections
import json
import os
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent / "_timeline_hook"
TRACE = os.environ.get("MINDX_TRACE", "/kaggle/working/out/trace.json")


def run():
    HOOK.mkdir(exist_ok=True)
    (HOOK / "sitecustomize.py").write_text(
        "import os, sys\n"
        "if os.environ.get('MINDX_TRACE') and '--profile' in sys.argv and '_train.py' in ' '.join(sys.argv):\n"
        "    import torch.profiler as p\n"
        "    init = p.profile.__init__\n"
        "    def save(prof):\n"
        "        prof.export_chrome_trace(os.environ['MINDX_TRACE'])\n"
        "    def __init__(self, *a, **k):\n"
        "        k['activities'] = [p.ProfilerActivity.CUDA]\n"
        "        k['on_trace_ready'] = save\n"
        "        init(self, *a, **k)\n"
        "    p.profile.__init__ = __init__\n")
    env = dict(os.environ, MINDX_TRACE=TRACE, PYTHONPATH=f"{HOOK}:{os.environ.get('PYTHONPATH', '')}")
    subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution", "--out", "runs_tl",
                    "--profile"], env=env, check=False)
    analyse(TRACE)


def launches():
    """Which Python code issues each CUDA launch: 4 steps with stacks (slow, so counts only)."""
    HOOK.mkdir(exist_ok=True)
    (HOOK / "sitecustomize.py").write_text(
        "import os, sys\n"
        "if os.environ.get('MINDX_TRACE') and '--profile' in sys.argv and '_train.py' in ' '.join(sys.argv):\n"
        "    import torch.profiler as p\n"
        "    init = p.profile.__init__\n"
        "    def save(prof):\n"
        "        prof.export_chrome_trace(os.environ['MINDX_TRACE'])\n"
        "    def __init__(self, *a, **k):\n"
        "        k['activities'] = [p.ProfilerActivity.CPU, p.ProfilerActivity.CUDA]\n"
        "        k['with_stack'] = True\n"
        "        k['schedule'] = p.schedule(wait=2, warmup=1, active=4, repeat=1)\n"
        "        k['on_trace_ready'] = save\n"
        "        init(self, *a, **k)\n"
        "    p.profile.__init__ = __init__\n")
    env = dict(os.environ, MINDX_TRACE=TRACE, PYTHONPATH=f"{HOOK}:{os.environ.get('PYTHONPATH', '')}")
    subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution", "--out", "runs_tl",
                    "--profile"], env=env, check=False)
    analyse_launches(TRACE)
    os.remove(TRACE)


def _frame(name):
    """'/x/y/solution/stream.py(12): f' -> 'solution/stream.py:12 f', or None for library plumbing."""
    if "(" not in name or ".py" not in name:
        return None
    path, _, rest = name.partition("(")
    line, _, func = rest.partition("): ")
    for root in ("solution/", "transformers/", "peft/", "bitsandbytes/", "trl/", "soup_cli/", "accelerate/"):
        if root in path:
            short = root + path.split(root, 1)[1]
            if short.endswith("torch/nn/modules/module.py"):
                return None
            return f"{short}:{line} {func}"
    return None


def analyse_launches(path, steps=4):
    events = json.load(open(path))["traceEvents"]
    kernel_of = {}
    for e in events:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "correlation" in e.get("args", {}):
            kernel_of[e["args"]["correlation"]] = e["name"]
    per_tid = collections.defaultdict(list)
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        if cat in ("python_function", "cpu_op"):
            per_tid[e.get("tid")].append((e["ts"], e["ts"] + e["dur"], cat, e["name"], None))
        elif cat == "cuda_runtime" and "correlation" in e.get("args", {}):
            corr = e["args"]["correlation"]
            if corr in kernel_of:
                per_tid[e.get("tid")].append((e["ts"], e["ts"], "launch", kernel_of[corr], None))
    by_kernel = collections.Counter()
    by_site = collections.Counter()
    by_pair = collections.Counter()
    by_op = collections.Counter()
    tids = collections.Counter()
    for tid, items in per_tid.items():
        items.sort(key=lambda i: (i[0], -i[1]))
        stack = []
        for start, end, cat, name, _ in items:
            while stack and stack[-1][0] <= start:
                stack.pop()
            if cat != "launch":
                stack.append((end, cat, name))
                continue
            tids[tid] += 1
            kernel = name.replace("at::native::", "").replace("(anonymous namespace)::", "")[:90]
            frames = [f for f in (_frame(s[2]) for s in stack if s[1] == "python_function") if f]
            ops = [s[2] for s in stack if s[1] == "cpu_op"]
            site = frames[-1] if frames else "?"
            by_kernel[kernel] += 1
            by_site[site] += 1
            by_pair[f"{kernel} <- {ops[0] if ops else '?'} <- {site}"] += 1
            by_op[ops[0] if ops else "?"] += 1
    total = sum(by_kernel.values())
    print(f"launches/step {total / steps:.0f}; by thread {[round(v / steps) for v in tids.values()]}")
    for title, counter, n in (("top ops", by_op, 30), ("by python site", by_site, 60),
                              ("kernel <- op <- site", by_pair, 120)):
        print(f"\n{title} (launches/step):")
        for key, value in counter.most_common(n):
            print(f"{value / steps:8.1f}  {key}")
