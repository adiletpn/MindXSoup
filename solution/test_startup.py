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


def test_in_memory_map_gives_the_same_rows_and_leaves_random_alone():
    import random

    import pytest

    datasets = pytest.importorskip("datasets")
    import numpy as np
    import torch

    from solution.startup import skip_in_memory_map_hashing

    rows = [{"text": f"row {i}", "n": i} for i in range(50)]
    ref = datasets.Dataset.from_list(rows).map(lambda r: {"n2": r["n"] * 2}).filter(
        lambda r: r["n"] % 3 != 0)
    skip_in_memory_map_hashing()
    states = random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state()
    got = datasets.Dataset.from_list(rows).map(lambda r: {"n2": r["n"] * 2}).filter(
        lambda r: r["n"] % 3 != 0)
    assert got.to_list() == ref.to_list()
    assert random.getstate() == states[0]
    assert (np.random.get_state()[1] == states[1]).all()
    assert torch.equal(torch.get_rng_state(), states[2])


def test_optional_import_checks_are_restored():
    import transformers
    import transformers.utils as transformers_utils

    from solution.startup import skip_unused_optional_imports

    real = transformers_utils.is_sklearn_available
    skip_unused_optional_imports()
    assert transformers_utils.is_sklearn_available is real
    assert "is_wandb_available" not in transformers.__dict__ or \
        transformers.is_wandb_available.__module__.startswith("transformers")
    import trl.trainer.base_trainer as base_trainer

    assert base_trainer.is_wandb_available() in (True, False)


def test_dynamo_flag_is_restored():
    import os

    from solution.startup import import_bitsandbytes_without_compile_wrappers

    before = os.environ.get("TORCHDYNAMO_DISABLE")
    import_bitsandbytes_without_compile_wrappers()
    assert os.environ.get("TORCHDYNAMO_DISABLE") == before


def test_gpu_dtype_conversion_is_bit_identical():
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    torch.manual_seed(0)
    values = torch.cat([torch.randn(1 << 20) * scale for scale in (1e-8, 1e-5, 1e-2, 1, 1e3, 1e5)])
    values = torch.cat([values, torch.tensor([0.0, -0.0, 65504.0, 65520.0, 7e4, -7e4,
                                              5.96e-8, 2.98e-8, float("inf"), float("nan")])])
    source = values.to(torch.bfloat16)
    cpu = source.to(torch.float16)
    gpu = source.cuda().to(torch.float16).cpu()
    # every number converts to the same bits; NaN stays NaN (its payload bits differ between
    # CPU and GPU, and model weights hold no NaN)
    nan = torch.isnan(cpu)
    assert torch.equal(nan, torch.isnan(gpu))
    assert torch.equal(cpu[~nan].view(torch.int16), gpu[~nan].view(torch.int16))
