"""Time startup boundaries without changing the harness or training task.

Nested spans overlap and must not be summed. Replaces a cProfile experiment
whose self/cumulative times were inconsistent in this environment.
"""
import argparse
from collections import defaultdict
import contextlib
import functools
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    spans = []
    first_step = None

    @contextlib.contextmanager
    def timed(name):
        before = time.perf_counter()
        try:
            yield
        finally:
            spans.append(dict(name=name, seconds=time.perf_counter() - before,
                              before_first_step=first_step is None))

    def wrap(owner, name, label):
        original = getattr(owner, name, None)
        if original is None:
            return

        @functools.wraps(original)
        def measured(*positional, **keywords):
            with timed(label):
                return original(*positional, **keywords)
        setattr(owner, name, measured)

    # Same import order as the harness. No extra CUDA synchronization.
    wrap(importlib.metadata, "packages_distributions", "package_metadata_scan")
    with timed("import_torch"):
        import torch
    with timed("cuda_initialization"):
        torch.empty(1, device="cuda")
    with timed("import_transformers"):
        import transformers
    with timed("import_triton_runtime"):
        import triton.runtime.jit  # noqa: F401
    sys.path.insert(0, str(ROOT))
    with timed("import_solution"):
        import solution
    original_apply = solution.apply

    def apply():
        with timed("solution_apply"):
            original_apply()
        # Instrument after normal apply has imported and patched these modules.
        import datasets
        from trl import SFTTrainer
        wrap(datasets.Dataset, "map", "dataset_map")
        wrap(SFTTrainer, "_prepare_dataset", "sft_prepare_dataset")
        wrap(transformers.Trainer, "create_optimizer", "create_optimizer")
        wrap(transformers.Trainer, "save_model", "save_model")
        for owner, label in ((transformers.AutoModelForCausalLM, "load_model"),
                             (transformers.AutoTokenizer, "load_tokenizer")):
            original = owner.from_pretrained

            def factory(*positional, _original=original, _label=label, **keywords):
                with timed(_label):
                    return _original(*positional, **keywords)
            # Preserve arguments: original is already a bound class method.
            owner.from_pretrained = staticmethod(factory)
    solution.apply = apply

    def save_report():
        groups = defaultdict(lambda: {"seconds": 0.0, "calls": 0})
        for row in spans:
            key = row["name"] + ("_startup" if row["before_first_step"] else "_later")
            groups[key]["seconds"] += row["seconds"]
            groups[key]["calls"] += 1
        report = dict(startup_to_first_step_s=first_step, spans=spans, grouped=dict(groups),
                      limitations=["Nested spans overlap; do not sum",
                                   "Boundary instrumentation is diagnostic",
                                   "Imports before main and process teardown are not measured",
                                   "No attribution of asynchronous GPU work to CPU spans"])
        (args.out / "startup_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    class StartupBoundary(transformers.TrainerCallback):
        def on_step_begin(self, train_args, state, control, **kwargs):
            nonlocal first_step
            if first_step is None:
                first_step = time.perf_counter() - started
                save_report()
                print("STARTUP_BOUNDARY_SAVED", args.out, flush=True)

    original_init = transformers.Trainer.__init__

    def init(self, *positional, **keywords):
        with timed("trainer_init"):
            original_init(self, *positional, **keywords)
        self.add_callback(StartupBoundary())
    transformers.Trainer.__init__ = init
    sys.path.insert(0, str(ROOT / "harness"))
    sys.argv = [str(ROOT / "harness/_train.py"), "--config", str(args.config.resolve()),
                "--metrics", str(args.out / "metrics.json"), "--solution"]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        save_report()


if __name__ == "__main__":
    main()
