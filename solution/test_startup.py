"""The cached packages_distributions() returns what the real one returns, as independent copies."""

import importlib.metadata


def test_cached_mapping_matches_and_is_a_copy():
    real = importlib.metadata.packages_distributions
    try:
        from solution.startup import cache_package_distributions

        cache_package_distributions()
        cached = importlib.metadata.packages_distributions
        if cached is real:  # transformers not importable here: nothing to cache
            return
        expected = {k: sorted(v) for k, v in real().items()}
        first = cached()
        assert {k: sorted(v) for k, v in first.items()} == expected
        first.clear()
        next(iter(cached().values())).append("changed")
        assert {k: sorted(v) for k, v in cached().items()} == expected
    finally:
        importlib.metadata.packages_distributions = real
