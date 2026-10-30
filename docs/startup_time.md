# Startup timing observations ? 2026-10-30

Source: memory diagnostic version 1 on Kaggle T4. These are cold diagnostic process timings, not a decomposition of the official Soup training subprocess. Imports are sequential; later imports may already be cached by earlier imports. Model loading includes first-download and CUDA initialization costs.

| Stage | Seconds |
|---|---:|
| import torch | 5.057 |
| import transformers | 9.724 |
| import trl | 2.237 |
| import peft | 13.557 |
| import bitsandbytes | 0.000 |
| import soup_cli | 0.000 |
| load_model | 30.172 |
| prepare_and_lora | 0.139 |

Do not sum these stages and compare directly with the organizer estimate of ~20 seconds. The official harness warms downloads first. Official wall-minus-train-loop observations below include imports, model loading, setup, compilation and saving.

## Official warmed benchmark, seed 1

| Run | Wall s | Loop s | Outside loop s | apply s (included, not additional) |
|---|---:|---:|---:|---:|
| base_1 | 85.02 | 57.24 | 27.78 | None |
| base_2 | 85.49 | 59.08 | 26.41 | None |
| solution_1 | 73.53 | 47.16 | 26.37 | 9.03 |
| solution_2 | 73.08 | 46.56 | 26.52 | 8.93 |

The residual is about 26-28 s in these runs. It is not all import time; the available logs do not provide timestamps sufficient to attribute every second. Solution apply() takes about 9 s, some of which may be imports that the base performs elsewhere. No claimed optimization is inferred from subtraction alone.

## Round 2: current solution startup (2026-10-30)

Solution commit `5782a18`; Kaggle T4: https://www.kaggle.com/code/ansargabit/ansargabit-startup2 (version 2). All worker processes exited zero. Each worker has a separate temporary TRITON_CACHE_DIR and a fresh Python process. Raw importtime output and JSON are in the notebook output.

### Import-only profile

`python -X importtime -c "import solution"`, top 15 by self time (cumulative values overlap and must not be summed):

| Import | Self s | Cumulative s |
|---|---:|---:|
| torch | 2.228357 | 4.617031 |
| torch._C | 0.894293 | 1.000240 |
| triton._C.libtriton | 0.266011 | 0.266635 |
| torch._prims | 0.149391 | 0.153046 |
| torch._meta_registrations | 0.099564 | 0.362168 |
| triton.backends | 0.090428 | 0.103382 |
| solution.attention | 0.057051 | 0.574760 |
| torch._refs | 0.046100 | 0.070686 |
| torch.utils._debug_mode | 0.046095 | 0.046095 |
| torch._decomp.decompositions | 0.036862 | 0.189907 |
| triton.language.core | 0.031170 | 0.031170 |
| torchgen.model | 0.019345 | 0.021619 |
| numpy._core._multiarray_umath | 0.019055 | 0.023199 |
| triton.language.standard | 0.016756 | 0.016756 |
| torch.utils._pytree | 0.014063 | 0.045925 |

The import-only profile is cold and is not the harness apply timer. `harness/_train.py` imports torch, initializes CUDA, imports transformers and installs TritonWatch (imports Triton) before starting its timer around `import solution; solution.apply()`. Thus torch/transformers/Triton cold import costs are already paid by both baseline and solution. PEFT and bitsandbytes are required for NF4 + LoRA model setup, and TRL is required by Soup SFT training later anyway. Their lazy submodule imports are not pure solution overhead; timing their displacement requires a paired complete training trace. Import-only does not execute patch functions, so it misses their lazy imports.

### Patch timing in apply order

The patch worker pre-imports torch, transformers and triton.runtime.jit and initializes CUDA. Times include lazy imports caused by each patch. Warmup is kept enabled; its completion is measured separately.

| Patch / boundary | Seconds |
|---|---:|
| patch_qwen2_mlp | 9.244382 |
| patch_qwen2_mlp_fp16 | 0.000006 |
| patch_qwen2_rmsnorm | 0.000004 |
| patch_qwen2_causal_lm | 0.000006 |
| patch_trl_metrics | 8.269901 |
| patch_bnb_nf4 | 0.000012 |
| patch_rope | 0.000007 |
| patch_qwen2_attention_fp16 | 0.000006 |
| patch_unpad | 0.000018 |
| patch_qwen2_layer_recompute | 0.000005 |
| patch_fp16_embedding | 0.000016 |
| apply_total | 17.516343 |
| apply_plus_warmup_join | 23.683214 |

The first Qwen MLP patch pays the Qwen modeling import; patch_trl_metrics pays the SFTTrainer/Trainer import. Other patches are microseconds once dependencies are loaded. This run measured 17.516 s for apply itself, not the older approximately 9 s observation: host/caches and the solution revision differ. Do not infer a regression or an additional 17.5 s penalty versus baseline from this standalone diagnostic. Background warmup join completes at 23.683 s; this diagnostic has no concurrent model loading, so it does not measure overlap benefit.

### First versus second kernel launch

Inputs are preallocated; CUDA is initialized and synchronized before timing. Batch 4, T=512, hidden=1536, q/kv heads=12/2, dim=128, intermediate=8960, vocabulary=151936. RMSNorm and RoPE inputs are fp32. Head CE uses the real 512-row chunk of fp16 logits, NF4 unpacks a 8960x1536 matrix, and SwiGLU uses fp16 gate/up as in the current MLP path. All rows are synchronized wall seconds, not CUDA-event kernel-only times.

| Kernel | First s | Second s |
|---|---:|---:|
| rmsnorm_fwd_kernel | 1.864728 | 0.000184 |
| rmsnorm_bwd_kernel | 0.333215 | 0.000219 |
| rope_kernel | 0.406442 | 0.000349 |
| head_ce_rows_kernel | 0.494018 | 0.002149 |
| nf4_unpack_kernel | 0.381047 | 0.000373 |
| example_swiglu_fwd_kernel | 0.310748 | 0.000513 |
| example_swiglu_bwd_kernel | 0.085085 | 0.000823 |

The first launch includes compilation, module loading, launch and execution; the first RMSNorm additionally pays lazy Triton driver initialization. These numbers are not isolated compiler CPU time. The second head-CE call reuses the in-place buffer (this is timing, not a correctness test). The example kernels do not count as our own in the competition.

### Three-thread experiment

Three independent jobs: RMSNorm forward, RoPE forward and head CE. Fresh process and separate empty cache for every trial; allocated tensors and CUDA initialization excluded. ThreadPoolExecutor uses three Python threads, one job per thread, and propagates exceptions. Trial order: serial, threaded, threaded, serial, serial, threaded.

| Trial | Serial s | Three threads s |
|---|---:|---:|
| 1 | 2.782820 | 2.927691 |
| 2 | 2.776133 | 2.904569 |
| 3 | 2.755615 | 2.829598 |

Medians: serial 2.776133 s, threaded 2.904569 s; serial/threaded = 0.956x. No compilation speedup was observed; threading was about 4.6% slower. This does not prove compiler internals never overlap, and does not rule out the existing single background warmup overlapping model I/O. Do not budget a 3x compilation speedup from three threads.

Reproduce: `CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/startup_probe.py`.
