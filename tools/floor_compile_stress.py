"""How often the background compile fails, and where a process hangs when it does.

    python tools/floor_compile_stress.py loop [seconds]
    python tools/floor_compile_stress.py real <seconds>

Each round is a fresh process with an empty Triton cache doing what a measured run does up to the
first step: solution.apply() (starts the compile thread), load the 4-bit model on the main thread,
join the thread, then compile on the main thread what the thread failed to compile. A round that
does not finish dumps all threads' stacks and is killed. Not part of the solution.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "harness"))
LIMIT = 150


def real_runs(seconds):
    """Full solution runs through the harness, with all threads' stacks dumped if one takes long."""
    hook = REPO / "tools" / "_hang_hook"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(
        "import sys\n"
        "if '_train.py' in ' '.join(sys.argv):\n"
        "    import faulthandler\n"
        "    faulthandler.dump_traceback_later(240, repeat=True)\n")
    env = dict(os.environ, PYTHONPATH=f"{hook}:{os.environ.get('PYTHONPATH', '')}")
    end = time.time() + seconds
    rounds = bad = 0
    while time.time() < end:
        rounds += 1
        out = REPO / f"runs_hang_{rounds}"
        subprocess.run([sys.executable, "harness/run_task.py", "--only", "solution", "--out", str(out)],
                       cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        text = "".join(log.read_text(errors="replace") for log in out.rglob("train.log"))
        result = (out / "result.json").read_text() if (out / "result.json").exists() else ""
        ok = result and json.loads(result)["summary"]["solution"].get("ok")
        if "warm-up of" in text or not ok:
            bad += 1
            print(f"=== run {rounds}", result[:300])
            lines = [l for l in text.splitlines() if not l.lstrip().startswith("%")]
            print("\n".join(l for l in lines if "warm-up of" in l))
            print("\n".join(lines[-150:]), flush=True)
        shutil.rmtree(out, ignore_errors=True)
        if rounds == 1 or rounds % 10 == 0:
            print(f"{rounds} runs, {bad} bad", flush=True)
    print(f"total {rounds} runs, {bad} bad", flush=True)


def child():
    import faulthandler

    faulthandler.enable()
    faulthandler.dump_traceback_later(LIMIT, exit=True)
    sys.path.insert(0, str(REPO))
    import solution

    solution.apply()
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from solution import warmup

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.float16))
    warmup.wait()
    print("thread joined", flush=True)
    if model is not None:
        torch.cuda.synchronize()
    print("again on the main thread", flush=True)
    for job in solution._warmup_jobs():
        job()
    torch.cuda.synchronize()
    print("round done", flush=True)


def loop(seconds):
    sys.path.insert(0, str(REPO / "harness"))
    from common import MODEL, OFFLINE_ENV
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    end = time.time() + seconds
    rounds = bad = 0
    while time.time() < end:
        rounds += 1
        with tempfile.TemporaryDirectory() as cache:
            env = {**os.environ, **OFFLINE_ENV, "TRITON_CACHE_DIR": cache}
            start = time.time()
            done = subprocess.run([sys.executable, __file__, "child"], env=env, cwd=REPO,
                                  capture_output=True, text=True, errors="replace")
        text = done.stdout + done.stderr
        failed = "warm-up of" in text or done.returncode != 0 or "round done" not in text
        if failed:
            bad += 1
            print(f"=== round {rounds}: exit {done.returncode}, {time.time() - start:.0f} s")
            lines = [l for l in text.splitlines() if not l.startswith(("      %", "    %", "  %"))]
            print("\n".join(lines[-90:]), flush=True)
        if rounds == 1 or rounds % 20 == 0:
            print(f"{rounds} rounds, {bad} bad, last {time.time() - start:.0f} s", flush=True)
    print(f"total {rounds} rounds, {bad} bad", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "child":
        child()
    elif sys.argv[1] == "real":
        real_runs(float(sys.argv[2]))
    else:
        loop(float(sys.argv[2]) if len(sys.argv) > 2 else 3000)
