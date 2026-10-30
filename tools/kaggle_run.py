"""Run this repository on a Kaggle T4 from the command line and fetch the results.

    python tools/kaggle_run.py tests            # pytest solution (kernels vs PyTorch)
    python tools/kaggle_run.py quick            # solution only, with the profiler table
    python tools/kaggle_run.py full             # base + solution, check_rules, score
    python tools/kaggle_run.py full --repeats 2 --seed 7
    python tools/kaggle_run.py cmd --cmd "python tools/memory_audit.py"   # any command

Files a command writes to /kaggle/working/out are downloaded too (kaggle_out/<mode>/out/).

The working tree (tracked files, as committed in HEAD plus uncommitted edits) is packed into the
kernel script itself, so the kernel needs no GitHub access. Needs the Kaggle CLI and
~/.kaggle/kaggle.json. The final, jury-style check is still the template notebook
(templates/kaggle_notebook.ipynb) with Run All; this is only the fast development loop.
"""

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = next((str(path) for path in (Path.home() / ".venvs" / "kaggle" / "bin" / "kaggle",
                                       Path.home() / ".venvs" / "kaggle" / "Scripts" / "kaggle.exe")
               if path.exists()), shutil.which("kaggle") or "kaggle")
PINS = ('"soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" '
        '"bitsandbytes==0.50.2" "accelerate==1.14.0" pytest')

SCRIPT = r'''
import base64, io, json, os, shutil, subprocess, sys, tarfile, time
PAYLOAD = "{payload}"
MODE, EXTRA = "{mode}", {extra!r}
REPO, OUT = "/tmp/repo", "/kaggle/working/out"
shutil.rmtree(REPO, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD)), mode="r:gz").extractall(REPO)

FAILED = []

def sh(cmd, log):
    started = time.time()
    with open(os.path.join(OUT, log), "w") as handle:
        proc = subprocess.run(cmd, shell=True, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[{{proc.returncode}}] {{time.time() - started:7.1f}} s  {{cmd}}", flush=True)
    if proc.returncode:
        FAILED.append(f"{{cmd}} (exit {{proc.returncode}}, see {{log}})")
    return proc.returncode

print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used", "--format=csv"],
                     capture_output=True, text=True).stdout)
sh('python -m pip install -q {pins}', "pip.log")
if MODE == "tests":
    sh("python -m pytest solution -q -p no:cacheprovider", "pytest_solution.log")
elif MODE == "cmd":
    sh("python task/prepare_data.py", "prepare_data.log")
    sh(EXTRA, "cmd.log")
else:
    sh("python task/prepare_data.py", "prepare_data.log")
    if MODE == "quick":
        sh("python harness/run_task.py --only solution --out runs_try --profile " + EXTRA,
           "run_task.log")
    else:
        sh("python -m pytest solution -q -p no:cacheprovider", "pytest_solution.log")
        sh("python harness/run_task.py " + EXTRA, "run_task.log")
        sh("python harness/check_rules.py", "check_rules.log")
        sh("python harness/score.py runs/result.json", "score.log")
for name in ("runs", "runs_try"):
    src = os.path.join(REPO, name)
    if not os.path.isdir(src):
        continue
    for root, _dirs, files in os.walk(src):
        if "cache" in root:
            continue
        for file in files:
            if file.endswith((".json", ".jsonl", ".log", ".txt")):
                dst = os.path.join(OUT, os.path.relpath(os.path.join(root, file), REPO))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(os.path.join(root, file), dst)
for log in sorted(os.listdir(OUT)):
    if log.endswith(".log") and log != "pip.log":
        print(f"\n===== {{log}} (tail) =====")
        with open(os.path.join(OUT, log)) as handle:
            print("".join(handle.readlines()[-40:]))
with open(os.path.join(OUT, "status.json"), "w") as handle:
    json.dump({{"failed": FAILED}}, handle, indent=2)
print("\nFAILED STEPS:\n  " + "\n  ".join(FAILED) if FAILED else "\nALL STEPS EXITED 0")
'''


def pack_tree():
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=REPO,
                           capture_output=True, text=True, check=True).stdout.split()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in files:
            path = REPO / name
            if path.is_file() and not name.startswith(("runs", "data/", "kaggle_out")) \
                    and name != "kaggle.json":
                tar.add(path, arcname=name)
    return base64.b64encode(buf.getvalue()).decode()


def kaggle(*args, check=True):
    proc = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and proc.returncode:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("tests", "quick", "full", "cmd"))
    parser.add_argument("--cmd", help="command to run in the repo for mode cmd")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--slug", default="mindx-dev")
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--fetch", action="store_true",
                        help="only wait for and download an already pushed run of --slug")
    args = parser.parse_args()

    user = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())["username"]
    kernel = f"{user}/{args.slug}"
    if args.fetch:
        return fetch(kernel, args.mode)
    extra = f"--repeats {args.repeats}" if args.mode == "full" else ""
    if args.mode == "cmd":
        if not args.cmd:
            sys.exit("mode cmd needs --cmd")
        extra = args.cmd
    if args.seed is not None and args.mode != "cmd":
        extra += f" --seed {args.seed}"
    work = REPO / "kaggle_out" / "push"
    work.mkdir(parents=True, exist_ok=True)
    (work / "run.py").write_text(SCRIPT.format(payload=pack_tree(), mode=args.mode,
                                               extra=extra.strip(), pins=PINS))
    (work / "kernel-metadata.json").write_text(json.dumps({
        "id": kernel, "title": args.slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": True, "dataset_sources": [], "competition_sources": [],
        "kernel_sources": []}, indent=2))
    print(kaggle("kernels", "push", "-p", str(work), "--accelerator", "NvidiaTeslaT4"))
    if not args.no_wait:
        fetch(kernel, args.mode)


def fetch(kernel, mode):
    started = time.time()
    while True:
        time.sleep(30)
        status = kaggle("kernels", "status", kernel, check=False)
        print(f"{time.time() - started:6.0f} s  {status.strip()}", flush=True)
        if any(word in status.lower() for word in ("complete", "error", "cancel")):
            break
    out = REPO / "kaggle_out" / mode
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", kernel, "-p", str(out), "--force", check=False))
    print(f"results in {out}")
    status = out / "out" / "status.json"
    failed = json.loads(status.read_text())["failed"] if status.exists() else ["no status.json"]
    if failed:
        sys.exit("FAILED:\n  " + "\n  ".join(failed))
    print("all steps exited 0")


if __name__ == "__main__":
    main()
