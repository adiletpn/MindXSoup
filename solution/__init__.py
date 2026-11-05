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
from .mlp_recompute import patch_qwen2_mlp_gateup_recompute
from .nf4 import patch_bnb_nf4
from .recompute import patch_qwen2_layer_recompute
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_rope
from .startup import (cache_package_distributions, convert_dtype_on_gpu_at_load,
                      import_bitsandbytes_without_compile_wrappers,
                      shrink_cuda_stack, skip_unused_optional_imports)
from .head_stream import patch_head_stream
from .stream import patch_stream_layers
from .swiglu import patch_qwen2_mlp
from .unpad import patch_unpad

FEATURES = {
    "startup_cache": True,   # CPU startup: no unused compile wrappers/sklearn/wandb, one metadata scan
    "example_swiglu": True,  # organizers' example, kept running; does not count as ours
    "mlp_fp16": True,        # MLP keeps fp16 activations (uses the example's SwiGLU kernel)
    "mlp_gateup_recompute": True,  # layers not rerun keep the MLP input, redo gate/up in backward
    "rmsnorm": True,         # ours: fused RMSNorm forward/backward
    "fused_ce": True,        # ours: head + cross-entropy + metrics without full logits
    "nf4": True,             # ours: NF4 weight unpacking with fused double-dequant of absmax
    "nf4_direct": True,      # Linear4bit: our unpack + F.linear directly, no bnb dispatch layers
    "nf4_cache_fp16": False,  # each NF4 weight unpacked once and kept in fp16 (speed profile)
    "rope": True,            # ours (teammate): rotary embedding for q and k in one launch each way
    "fp16_embedding": True,  # tied embedding / lm_head stored in fp16 (same values), -445 MiB
    "head_stream": True,     # ours: that fp16 table in pinned host memory, head by vocab slices
    "attention_fp16": True,  # ours (teammate): fp16 q/k/v, one kernel for LoRA sum + RoPE + cast
    "unpad": True,           # pad tokens never enter the decoder; attention per row
    "recompute": True,       # rerun decoder layers in backward, keep only their inputs (budget in recompute.py)
    "recompute_fp16_inputs": True,  # rerun layers keep their input as scaled fp16 rows, not fp32
    "recompute_host_inputs": True,  # ... and those fp16 rows wait for backward in pinned host memory
    "offload_activations": False,  # instead of rerunning, layers' saved tensors wait in pinned host memory
    "stream_weights": True,  # decoder 4-bit weights in pinned host memory, one layer on the GPU at a time
    "warmup": True,          # compile the kernels above in a background thread during model load
}


def _warmup_jobs():
    """Our kernels in the order a training step first launches them."""
    jobs = []
    if FEATURES["rmsnorm"]:
        jobs.append(warmup.compile_rmsnorm)
    if FEATURES["nf4"]:
        jobs.append(warmup.compile_nf4)
    if FEATURES["unpad"]:
        jobs.append(warmup.compile_lora_dropout)
    if FEATURES["attention_fp16"]:
        jobs.append(compile_qkv_rope)
    elif FEATURES["rope"]:  # with the fp16 attention path, q/k are rotated by qkv_rope_kernel
        jobs.append(warmup.compile_rope)
    if FEATURES["mlp_fp16"]:  # the fp16 MLP calls our silu_mul; the example's MLP is replaced
        jobs.append(warmup.compile_silu_mul)
    elif FEATURES["example_swiglu"]:
        jobs.append(warmup.compile_example_swiglu)
    if FEATURES["fused_ce"]:
        jobs.append(warmup.compile_head_stream if FEATURES["head_stream"] and
                    FEATURES["fp16_embedding"] else warmup.compile_head_ce)
    return jobs


def apply():
    if FEATURES["startup_cache"]:
        import_bitsandbytes_without_compile_wrappers()  # first: peft and trl import it too
    # compile in the background from here on, while the rest of apply() imports and patches and
    # Soup loads the model; wait() runs in the fused head's forward before the first step
    if FEATURES["warmup"]:
        warmup.start(_warmup_jobs())
    if FEATURES["startup_cache"]:
        cache_package_distributions()  # before trl, imported further down
        skip_unused_optional_imports()  # before the patches below import transformers' models
        shrink_cuda_stack()  # takes effect on the first training step (fused_ce.py)
        convert_dtype_on_gpu_at_load()
    if FEATURES["example_swiglu"]:
        patch_qwen2_mlp()
    if FEATURES["mlp_fp16"]:
        patch_qwen2_mlp_fp16()
        if FEATURES["mlp_gateup_recompute"]:
            patch_qwen2_mlp_gateup_recompute()  # wraps the fp16 forward above
    if FEATURES["rmsnorm"]:
        patch_qwen2_rmsnorm()
    if FEATURES["fused_ce"]:
        patch_qwen2_causal_lm()
        patch_trl_metrics()
    if FEATURES["nf4"]:
        patch_bnb_nf4()
        if FEATURES["nf4_direct"]:
            patch_bnb_linear4bit(cache_weights=FEATURES["nf4_cache_fp16"])
    if FEATURES["rope"]:
        patch_rope()
    if FEATURES["attention_fp16"]:
        patch_qwen2_attention_fp16()
    if FEATURES["unpad"]:
        patch_unpad()  # used by the fused head's forward
    if FEATURES["recompute"]:
        patch_qwen2_layer_recompute(fp16=FEATURES["recompute_fp16_inputs"],
                                    host=FEATURES["recompute_host_inputs"],
                                    offload_activations=FEATURES["offload_activations"])
    if FEATURES["stream_weights"] and FEATURES["nf4"] and FEATURES["nf4_direct"]:
        patch_stream_layers()  # set up by the fused head's forward, before the first step
    if FEATURES["fp16_embedding"]:
        patch_fp16_embedding()  # after fused_ce: wraps whichever forward is in place
        if FEATURES["head_stream"] and FEATURES["fused_ce"]:
            patch_head_stream()
