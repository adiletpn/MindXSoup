"""MindX solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every feature is one patch with its own switch, so
any of them can be turned off in one line; every patch falls back to the original code on inputs
it does not handle (CPU tensors, fake modules in Soup's tests, unexpected dtypes).
"""

from . import warmup
from .attention import compile_qkv_rope, patch_qwen2_attention_fp16
from .embedding import patch_fp16_embedding
from .fused_ce import patch_qwen2_causal_lm, patch_trl_metrics
from .mlp import patch_qwen2_mlp_fp16
from .nf4 import patch_bnb_nf4
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_rope
from .swiglu import patch_qwen2_mlp

FEATURES = {
    "example_swiglu": True,  # organizers' example, kept running; does not count as ours
    "mlp_fp16": True,        # MLP keeps fp16 activations (uses the example's SwiGLU kernel)
    "rmsnorm": True,         # ours: fused RMSNorm forward/backward
    "fused_ce": True,        # ours: head + cross-entropy + metrics without full logits
    "nf4": True,             # ours: NF4 weight unpacking with fused double-dequant of absmax
    "rope": True,            # ours (teammate): rotary embedding for q and k in one launch each way
    "fp16_embedding": True,  # tied embedding / lm_head stored in fp16 (same values), -445 MiB
    "attention_fp16": True,  # ours (teammate): fp16 q/k/v, one kernel for LoRA sum + RoPE + cast
    "warmup": True,          # compile the kernels above in a background thread during model load
}


def apply():
    jobs = []
    if FEATURES["example_swiglu"]:
        patch_qwen2_mlp()
    if FEATURES["mlp_fp16"]:
        patch_qwen2_mlp_fp16()
    if FEATURES["example_swiglu"] or FEATURES["mlp_fp16"]:
        jobs.append(warmup.compile_example_swiglu)
    if FEATURES["rmsnorm"]:
        patch_qwen2_rmsnorm()
        jobs.append(warmup.compile_rmsnorm)
    if FEATURES["fused_ce"]:
        patch_qwen2_causal_lm()
        patch_trl_metrics()
        jobs.append(warmup.compile_head_ce)
    if FEATURES["nf4"]:
        patch_bnb_nf4()
        jobs.append(warmup.compile_nf4)
    if FEATURES["rope"]:
        patch_rope()
        jobs.append(warmup.compile_rope)
    if FEATURES["attention_fp16"]:
        patch_qwen2_attention_fp16()
        jobs.append(compile_qkv_rope)
    if FEATURES["fp16_embedding"]:
        patch_fp16_embedding()  # after fused_ce: wraps whichever forward is in place
    # wait() runs in the fused head's forward; without it the thread just finishes on its own
    if FEATURES["warmup"]:
        warmup.start(jobs)
