"""MindX solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every feature is one patch with its own switch, so
any of them can be turned off in one line; every patch falls back to the original code on inputs
it does not handle (CPU tensors, fake modules in Soup's tests, unexpected dtypes).
"""

from .fused_ce import patch_qwen2_causal_lm, patch_trl_metrics
from .rmsnorm import patch_qwen2_rmsnorm
from .swiglu import patch_qwen2_mlp

FEATURES = {
    "example_swiglu": True,  # organizers' example, kept running; does not count as ours
    "rmsnorm": True,         # ours: fused RMSNorm forward/backward
    "fused_ce": True,        # ours: head + cross-entropy + metrics without full logits
}


def apply():
    if FEATURES["example_swiglu"]:
        patch_qwen2_mlp()
    if FEATURES["rmsnorm"]:
        patch_qwen2_rmsnorm()
    if FEATURES["fused_ce"]:
        patch_qwen2_causal_lm()
        patch_trl_metrics()
