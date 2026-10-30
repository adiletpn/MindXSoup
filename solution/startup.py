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
