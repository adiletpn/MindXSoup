"""What is alive on the GPU at the solution's memory peak: a CUDA allocator history, analysed.

    python tools/peak_memory.py run       # solution run with the allocator history recorded
    python tools/peak_memory.py analyse   # replay the history, print what is alive at the peak

`run` starts `harness/run_task.py --only solution` with this directory's `_peak_hook/sitecustomize.py`
on PYTHONPATH: in the training subprocess it turns on `torch.cuda.memory._record_memory_history`
before the first CUDA allocation and dumps the snapshot at exit to /kaggle/working/peak.pickle.
Nothing in `harness/` changes. Recording python stacks slows the run; its time means nothing.

`analyse` replays every alloc/free of the history, finds the moment the allocated total is highest
and groups the blocks alive at that moment by the innermost frame of our code (`solution/`), or of
the library that allocated them, and by size. Prints also the stack of the allocation that made
the peak, i.e. what the step was doing then.
"""

import collections
import os
import pickle
import subprocess
import sys
from pathlib import Path

SNAP = os.environ.get("MINDX_PEAK", "/kaggle/working/peak.pickle")
HOOK = Path(__file__).resolve().parent / "_peak_hook"
MIB = 2 ** 20


def run():
    HOOK.mkdir(exist_ok=True)
    (HOOK / "sitecustomize.py").write_text(
        "import os, sys\n"
        "if os.environ.get('MINDX_PEAK') and '_train.py' in ' '.join(sys.argv):\n"
        "    import atexit, torch\n"
        "    torch.cuda.memory._record_memory_history(context='alloc', stacks='python',\n"
        "                                             max_entries=4_000_000)\n"
        "    def _dump():\n"
        "        try:\n"
        "            torch.cuda.memory._dump_snapshot(os.environ['MINDX_PEAK'])\n"
        "            print('[peak] wrote', os.environ['MINDX_PEAK'], flush=True)\n"
        "        except Exception as exc:\n"
        "            print('[peak] dump failed', exc, flush=True)\n"
        "    atexit.register(_dump)\n")
    env = dict(os.environ, MINDX_PEAK=SNAP,
               PYTHONPATH=f"{HOOK}:{os.environ.get('PYTHONPATH', '')}")
    subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution",
                    "--out", "runs_peak"], env=env, check=False)


def owner(frames):
    """Innermost frame in solution/, else the innermost frame outside torch internals."""
    for f in frames:
        if "/solution/" in f["filename"]:
            return f"{Path(f['filename']).name}:{f['line']} {f['name']}"
    for f in frames:
        name = f["filename"]
        if "/torch/" not in name or "/torch/optim/" in name or "/torch/utils/checkpoint" in name:
            return f"{'/'.join(Path(name).parts[-2:])}:{f['line']} {f['name']}"
    return frames[0]["name"] if frames else "(no stack)"


def _peak(trace, start):
    """(peak bytes, its event index, blocks alive there, free index of each) from event start on;
    blocks allocated before start still count while alive."""
    alive, total = {}, 0
    for e in trace[:start]:
        if e["action"] == "alloc":
            alive[e["addr"]] = e
            total += e["size"]
        elif e["action"] == "free_completed" and e["addr"] in alive:
            total -= alive.pop(e["addr"])["size"]
    peak, peak_i = total, start
    for i in range(start, len(trace)):
        e = trace[i]
        if e["action"] == "alloc":
            alive[e["addr"]] = e
            total += e["size"]
            if total > peak:
                peak, peak_i = total, i
        elif e["action"] == "free_completed" and e["addr"] in alive:
            total -= alive.pop(e["addr"])["size"]
    alive = {}
    for e in trace[:peak_i + 1]:
        if e["action"] == "alloc":
            alive[e["addr"]] = e
        elif e["action"] == "free_completed":
            alive.pop(e["addr"], None)
    return peak, peak_i, alive


def analyse(top=25):
    snap = pickle.load(open(SNAP, "rb"))
    trace = max(snap["device_traces"], key=len)
    print(f"{len(trace)} allocator events")
    start = 0
    for window in range(int(os.environ.get("MINDX_PEAK_WINDOWS", "2"))):
        _report(trace, start, top)
        # next window: from the moment the largest block alive at this peak is freed
        peak, peak_i, alive = _peak(trace, start)
        big = max(alive.values(), key=lambda e: e["size"])
        start = next((i + 1 for i in range(peak_i + 1, len(trace))
                      if trace[i]["action"] == "free_completed" and trace[i]["addr"] == big["addr"]),
                     None)
        if start is None:
            break
        print(f"\n\n### after the {big['size'] / MIB:.1f} MiB block of {owner(big.get('frames', []))} "
              f"is freed (event {start})")


def _report(trace, start, top):
    peak, peak_i, alive = _peak(trace, start)
    print(f"peak allocated (blocks seen in the history) {peak / MIB:.1f} MiB, "
          f"{len(alive)} blocks alive; event {peak_i} of {len(trace)}")
    by_owner = collections.Counter()
    count = collections.Counter()
    for e in alive.values():
        key = owner(e.get("frames", []))
        by_owner[key] += e["size"]
        count[key] += 1
    print(f"\n{'MiB':>8} {'blocks':>6}  allocated at")
    for key, size in by_owner.most_common(top):
        print(f"{size / MIB:8.1f} {count[key]:6d}  {key}")
    by_size = collections.Counter(e["size"] for e in alive.values())
    print(f"\n{'MiB':>8} {'blocks':>6}  block size")
    for size, n in sorted(by_size.items(), key=lambda kv: -kv[0] * kv[1])[:15]:
        print(f"{size * n / MIB:8.1f} {n:6d}  {size / MIB:.3f} MiB")
    print("\nallocation that made the peak:")
    for f in trace[peak_i].get("frames", [])[:30]:
        if "/torch/" in f["filename"] and "/solution/" not in f["filename"]:
            continue
        print(f"  {'/'.join(Path(f['filename']).parts[-2:])}:{f['line']} {f['name']}")


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()
