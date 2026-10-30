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
