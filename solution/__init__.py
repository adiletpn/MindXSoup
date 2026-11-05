"""MindX solution for the Soup Kernel Challenge.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every feature is one patch with its own switch, so
any of them can be turned off in one line; every patch falls back to the original code on inputs
it does not handle (CPU tensors, fake modules in Soup's tests, unexpected dtypes).
"""

import os

from . import direct, dropout, head_stream, layer_input, recompute, stream, warmup
from .adam_host import patch_adam_host
from .attention import compile_qkv_rope, patch_qwen2_attention_fp16
from .embedding import patch_fp16_embedding
from .fused_ce import patch_qwen2_causal_lm, patch_trl_metrics
from .linear4bit import patch_bnb_linear4bit
from .mlp import patch_qwen2_mlp_fp16
from .mlp_recompute import patch_qwen2_mlp_gateup_recompute
from .mlp_tail import patch_mlp_tail
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
    "adam_host": True,       # AdamW moments in pinned host memory between optimizer steps
    "cublas_workspace": None,  # e.g. ":4096:1"; A/B on one T4: -8 MiB, +4.2 s, so the default stays
    "direct_nograd": True,   # no Function.apply wrappers when no graph is recorded (direct.py)
    "fused_layer_input": True,  # a rerun layer's input packed and restored in one kernel each way
    "fused_head_acc": True,  # the head's gradient accumulation in one kernel per slice
    "lean_draw": True,       # dropout draws from fresh ones instead of a kept 12 MiB buffer
    "mlp_gateup_recompute": True,  # layers not rerun keep the MLP input, redo gate/up in backward
    "mlp_tail": True,        # a rerun MLP skips down_proj and SwiGLU forward: their values are never read
    "rmsnorm": True,         # ours: fused RMSNorm forward/backward
    "fused_ce": True,        # ours: head + cross-entropy + metrics without full logits
    "nf4": True,             # ours: NF4 weight unpacking with fused double-dequant of absmax
    "nf4_direct": True,      # Linear4bit: our unpack + F.linear directly, no bnb dispatch layers
    "nf4_cache_fp16": False,  # each NF4 weight unpacked once and kept in fp16 (speed profile)
    "rope": True,            # ours (teammate): rotary embedding for q and k in one launch each way
    "fp16_embedding": True,  # tied embedding / lm_head stored in fp16 (same values), -445 MiB
    "head_stream": True,     # ours: that fp16 table in pinned host memory, head by vocab slices
    "head_slice": 4096,      # vocabulary rows per head slice (a multiple of the kernel's 4096)
    "attention_fp16": True,  # ours (teammate): fp16 q/k/v, one kernel for LoRA sum + RoPE + cast
    "unpad": True,           # pad tokens never enter the decoder; attention per row
    "recompute": True,       # rerun decoder layers in backward, keep only their inputs (budget in recompute.py)
    "recompute_fp16_inputs": True,  # rerun layers keep their input as scaled fp16 rows, not fp32
    "recompute_host_inputs": True,  # ... and those fp16 rows wait for backward in pinned host memory
    "recompute_group_tokens": 512,  # ... and rerun layers run their rows in groups of this many tokens
    "recompute_no_rerun": False,  # the layer hook only gives the streamer its hints, nothing reruns
    "offload_activations": False,  # instead of rerunning, layers' saved tensors wait in pinned host memory
    "layer_unpack": False,   # all 7 weights of the running layer unpacked by one launch, kept until the next layer
    "stream_weights": True,  # decoder 4-bit weights in pinned host memory, one layer on the GPU at a time
    "warmup": True,          # compile the kernels above in a background thread during model load
}

# Measured configurations. Only one runs; "rules_*" follow the challenge rules, "nolimits_*" are
# experiments outside them (see README) and do not count as a submission.
PROFILE = "rules_fast"
_SPEED = {"nf4_cache_fp16": True, "lean_draw": False, "adam_host": False, "cublas_workspace": None, "mlp_gateup_recompute": False, "head_stream": False, "recompute": False,
          "recompute_fp16_inputs": False, "recompute_host_inputs": False,
          "stream_weights": False}
PROFILES = {"rules_memory": {}, "rules_time": _SPEED,
            "rules_balanced": {"recompute_group_tokens": 1024},
            # the submission: all weights of a layer unpacked by one launch, no groups of rows
            "rules_fast": {"recompute_group_tokens": 0, "layer_unpack": True},
            "rules_fast_groups": {"recompute_group_tokens": 1024, "layer_unpack": True},
            # at most 600 MiB, as fast as that allows
            "rules_t600": {"recompute_group_tokens": 0, "head_slice": 8192},
            # no layer reruns (activations stay on the GPU); weights and table on the host
            "rules_mid": {"mlp_gateup_recompute": False, "recompute_fp16_inputs": False,
                          "recompute_host_inputs": False, "recompute_group_tokens": 0,
                          "recompute_no_rerun": True, "lean_draw": False}}
FEATURES.update(PROFILES[PROFILE])


def _warmup_jobs():
    """Our kernels in the order a training step first launches them."""
    jobs = []
    if FEATURES["rmsnorm"]:
        jobs.append(warmup.compile_rmsnorm)
    if FEATURES["nf4"]:
        jobs.append(warmup.compile_nf4)
        if FEATURES["layer_unpack"]:
            jobs.append(warmup.compile_nf4_layer)
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
    if FEATURES["recompute"] and FEATURES["recompute_fp16_inputs"]:
        jobs.append(warmup.compile_layer_input)
    if FEATURES["fused_ce"]:
        jobs.append(warmup.compile_head_stream if FEATURES["head_stream"] and
                    FEATURES["fp16_embedding"] else warmup.compile_head_ce)
    return jobs


def apply():
    dropout.KEEP_ONES = not FEATURES["lean_draw"]
    direct.enabled = FEATURES["direct_nograd"]
    stream.LAYER_UNPACK = FEATURES["layer_unpack"]
    layer_input.enabled = FEATURES["fused_layer_input"]
    head_stream.FUSED_ACC = FEATURES["fused_head_acc"]
    if FEATURES["cublas_workspace"]:  # read when the first matmul creates the cuBLAS handle
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", FEATURES["cublas_workspace"])
    if FEATURES["adam_host"]:
        patch_adam_host()
    head_stream.SLICE = FEATURES["head_slice"]
    if FEATURES["startup_cache"]:
        import_bitsandbytes_without_compile_wrappers()  # first: peft and trl import it too
    # compile in the background from here on, while the rest of apply() imports and patches and
    # Soup loads the model; wait() runs in the fused head's forward before the first step
    warmup.KEPT_MASKS = bool(FEATURES["recompute"] and FEATURES["recompute_group_tokens"])
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
        if FEATURES["mlp_tail"]:
            patch_mlp_tail()  # only inside a rerun; the first pass keeps its full forward
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
        if FEATURES["recompute_no_rerun"]:
            recompute.BUDGET_TOKENS = 1 << 30
        patch_qwen2_layer_recompute(fp16=FEATURES["recompute_fp16_inputs"],
                                    host=FEATURES["recompute_host_inputs"],
                                    offload_activations=FEATURES["offload_activations"],
                                    group_tokens=FEATURES["recompute_group_tokens"])
    if FEATURES["stream_weights"] and FEATURES["nf4"] and FEATURES["nf4_direct"]:
        patch_stream_layers()  # set up by the fused head's forward, before the first step
    if FEATURES["fp16_embedding"]:
        patch_fp16_embedding()  # after fused_ce: wraps whichever forward is in place
        if FEATURES["head_stream"] and FEATURES["fused_ce"]:
            patch_head_stream()
