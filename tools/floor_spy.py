"""Wall-clock profile of one harness-style solution run: where the main thread is, by phase.

    pip install py-spy; python tools/floor_spy.py [--samples out.txt]

Runs harness/_train.py as run_task.py does (same env, empty Triton cache), under py-spy with idle
samples, so waits count too. The run under py-spy is a little slower; shares matter, not seconds.
"""

import collections
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
from common import OFFLINE_ENV  # noqa: E402
import run_task  # noqa: E402

RATE = 100


def main():
    out = Path(os.environ.get("OUT", "/kaggle/working/out"))
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp())
    config = run_task.make_config(run_dir, REPO / "data" / "train.jsonl", 1)
    raw = out / "spy_raw.txt"
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "e.db"),
               TRITON_CACHE_DIR=str(run_dir / "tc"), TORCHINDUCTOR_CACHE_DIR=str(run_dir / "ic"))
    cmd = ["py-spy", "record", "--idle", "--threads", "--nonblocking", "-r", str(RATE), "-f", "raw",
           "-o", str(raw), "--", sys.executable, str(REPO / "harness" / "_train.py"),
           "--config", str(config), "--metrics", str(out / "spy_metrics.json"), "--solution"]
    subprocess.run(cmd, env=env, check=True, stdout=open(out / "spy_train.log", "w"),
                   stderr=subprocess.STDOUT)
    report(raw)


def report(raw):
    threads = collections.Counter()
    phases = collections.Counter()
    leaves = collections.Counter()
    loop = collections.Counter()
    for line in open(raw, errors="ignore"):
        stack, _, count = line.rpartition(" ")
        frames = stack.split(";")
        if not count.strip().isdigit():
            continue
        count = int(count)
        threads[frames[0]] += count
        if "MainThread" not in frames[0]:
            continue
        joined = stack
        if "training_step" in joined or "_inner_training_loop" in joined:
            phase = "train loop"
            for name, label in (("training_step", "loop:training_step"),
                                ("_maybe_log_save_evaluate", "loop:log/save"),
                                ("optimizer_step", "loop:opt"), ("step (", "loop:opt"),
                                ("clip_grad", "loop:clip"), ("__next__", "loop:dataloader"),
                                ("on_step", "loop:callbacks")):
                if name in joined:
                    phase = label
                    break
            if "_engine_run_backward" in joined:
                phase = "loop:backward"
            elif "compute_loss" in joined:
                phase = "loop:forward"
        else:
            phase = "other"
            for name, label in (("solution/__init__.py", "apply"), ("apply (", "apply"),
                                ("from_pretrained", "load model"),
                                ("save_pretrained", "save"), ("save_model", "save"),
                                ("_map_text_sft", "tokenize"), ("map (", "tokenize"),
                                ("_find_and_load", "imports"), ("SFTTrainer", "trainer init")):
                if name in joined:
                    phase = label
                    break
        phases[phase] += count
        leaves[frames[-1]] += count
        if phase.startswith("loop"):
            for frame in frames[-1:]:
                loop[frame] += count
    def show(title, counter, n=45):
        print(f"\n== {title} (s of main-thread wall clock)")
        for key, c in counter.most_common(n):
            print(f"{c / RATE:8.2f}  {key}")
    show("threads", threads, 12)
    show("phases", phases)
    show("leaves, train loop", loop, 60)
    show("leaves, all", leaves, 30)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        main()
