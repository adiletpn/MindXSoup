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
from .linear4bit import patch_bnb_linear4bit
from .mlp import patch_qwen2_mlp_fp16
from .nf4 import patch_bnb_nf4
from .recompute import patch_qwen2_layer_recompute
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_rope
from .startup import (cache_package_distributions, convert_dtype_on_gpu_at_load,
                      import_bitsandbytes_without_compile_wrappers, skip_in_memory_map_hashing,
                      skip_unused_optional_imports)
from .stream import patch_stream_layers
from .swiglu import patch_qwen2_mlp
from .unpad import patch_unpad

FEATURES = {
    "startup_cache": True,   # CPU startup: no unused compile wrappers/sklearn/wandb, no dill hashing, one metadata scan
    "example_swiglu": True,  # organizers' example, kept running; does not count as ours
    "mlp_fp16": True,        # MLP keeps fp16 activations (uses the example's SwiGLU kernel)
    "rmsnorm": True,         # ours: fused RMSNorm forward/backward
    "fused_ce": True,        # ours: head + cross-entropy + metrics without full logits
    "nf4": True,             # ours: NF4 weight unpacking with fused double-dequant of absmax
    "nf4_direct": True,      # Linear4bit: our unpack + F.linear directly, no bnb dispatch layers
    "rope": True,            # ours (teammate): rotary embedding for q and k in one launch each way
    "fp16_embedding": True,  # tied embedding / lm_head stored in fp16 (same values), -445 MiB
    "attention_fp16": True,  # ours (teammate): fp16 q/k/v, one kernel for LoRA sum + RoPE + cast
    "unpad": True,           # pad tokens never enter the decoder; attention per row
    "recompute": True,       # rerun decoder layers in backward, keep only their inputs (budget in recompute.py)
    "stream_weights": True,  # decoder 4-bit weights in pinned host memory, one layer on the GPU at a time
    "warmup": True,          # compile the kernels above in a background thread during model load
}


def apply():
    jobs = []
    if FEATURES["startup_cache"]:
        import_bitsandbytes_without_compile_wrappers()  # first: peft and trl import it too
        cache_package_distributions()  # before trl, imported further down
        skip_in_memory_map_hashing()
        skip_unused_optional_imports()  # before the patches below import transformers' models
        convert_dtype_on_gpu_at_load()
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
        if FEATURES["nf4_direct"]:
            patch_bnb_linear4bit()
    if FEATURES["rope"]:
        patch_rope()
        jobs.append(warmup.compile_rope)
    if FEATURES["attention_fp16"]:
        patch_qwen2_attention_fp16()
        jobs.append(compile_qkv_rope)
    if FEATURES["unpad"]:
        patch_unpad()  # used by the fused head's forward
    if FEATURES["recompute"]:
        patch_qwen2_layer_recompute()
    if FEATURES["stream_weights"] and FEATURES["nf4"] and FEATURES["nf4_direct"]:
        patch_stream_layers()  # set up by the fused head's forward, before the first step
    if FEATURES["fp16_embedding"]:
        patch_fp16_embedding()  # after fused_ce: wraps whichever forward is in place
    # wait() runs in the fused head's forward; without it the thread just finishes on its own
    if FEATURES["warmup"]:
        warmup.start(jobs)
