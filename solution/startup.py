"""Startup overhead of `soup train` itself (not the training step), allowed by the organizers in
the participants' chat (30.10, 18:31) and documented in the README.

importlib.metadata.packages_distributions() walks every installed package (~2.3 s on the Kaggle
image). transformers computes it when imported, before apply(); trl computes it again when Soup
(or our patch) imports it. Nothing is installed in between (the harness checks that packages do
not change during a run), so later callers get a copy of the mapping transformers already built:
a fresh dict of fresh lists per call, so no caller can change what another one sees.
"""

import importlib.metadata


def cache_package_distributions():
    if getattr(importlib.metadata.packages_distributions, "_mindx_patched", False):
        return
    try:
        from transformers.utils import import_utils
    except Exception:  # noqa: BLE001 - nothing to reuse
        return
    mapping = getattr(import_utils, "PACKAGE_DISTRIBUTION_MAPPING", None)
    if not isinstance(mapping, dict) or not mapping:
        return
    snapshot = {name: list(dists) for name, dists in mapping.items()}

    def packages_distributions():
        return {name: list(dists) for name, dists in snapshot.items()}

    packages_distributions._mindx_patched = True
    importlib.metadata.packages_distributions = packages_distributions


def skip_in_memory_map_hashing():
    """datasets hashes every map/filter transform (with dill, through the tokenizer it closes over)
    only to name a cache file. A dataset built in memory (Soup's Dataset.from_list) has no cache
    files and writes none, so the name is never used; datasets itself falls back to a random
    fingerprint whenever hashing fails. Give such calls a random fingerprint up front, from
    os.urandom so no random generator of the run moves. Rows, order and processing are unchanged."""
    import os

    from datasets import Dataset

    for name in ("map", "filter"):
        method = getattr(Dataset, name)
        if getattr(method, "_mindx_patched", False):
            continue

        def wrapped(self, *args, _method=method, **kwargs):
            if ("new_fingerprint" not in kwargs and not self.cache_files
                    and kwargs.get("cache_file_name") is None
                    and kwargs.get("num_proc") in (None, 0, 1)):
                kwargs["new_fingerprint"] = os.urandom(8).hex()
            return _method(self, *args, **kwargs)

        wrapped._mindx_patched = True
        wrapped.__name__, wrapped.__doc__ = method.__name__, method.__doc__
        setattr(Dataset, name, wrapped)


def skip_unused_optional_imports():
    """Two optional packages the run imports but never uses (organizers: allowed, 30.10 21:09):

    * sklearn (~1.7 s): transformers.generation.candidate_generator imports sklearn.metrics when
      sklearn is installed, for assisted generation with a confidence threshold;
    * wandb (~1.4 s): trl's base trainer imports it when installed, to put the URL of an active
      wandb run on the model card; Soup runs without --wandb, so there is no run and the URL is
      None either way.

    While exactly those two modules are imported, the availability check they consult answers
    False; it is restored right after, so everything imported later sees the real answer. The
    packages stay installed and importable; nothing is stubbed in sys.modules.
    """
    import sys

    import transformers
    import transformers.utils as transformers_utils

    if "transformers.generation.candidate_generator" not in sys.modules:
        real = transformers_utils.is_sklearn_available
        transformers_utils.is_sklearn_available = lambda: False
        try:
            import transformers.generation.candidate_generator  # noqa: F401
        finally:
            transformers_utils.is_sklearn_available = real
    if "trl.trainer.base_trainer" not in sys.modules:
        transformers.__dict__["is_wandb_available"] = lambda: False  # what trl imports by name
        try:
            import trl.trainer.base_trainer  # noqa: F401
        finally:
            transformers.__dict__.pop("is_wandb_available", None)  # lazy attribute again
