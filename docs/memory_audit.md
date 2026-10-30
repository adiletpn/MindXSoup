# Person A memory audit ? 2026-10-30

Source: team checkout `5a2ae0a`, unpatched model, Kaggle Tesla T4.
Run: https://www.kaggle.com/code/ansargabit/mindx-person-a-memory-profile (version 1).

## Findings from the requested 4 x 512 diagnostic

- Model setup used 1,553.7 MiB; forward raised allocation to 13,206.4 MiB.
- The largest saved-tensor group was MLP: 5,880 MiB across MLP and SiLU, including three 70 MiB fp32 tensors per layer.
- Saved fp32 tensors accounted for 10,080.3 MiB, versus 788.7 MiB in fp16 (unique data-pointer attribution).
- Returned logits were fp16 (593.5 MiB), while the saved loss-related fp32 tensor was 1,187 MiB. This is larger than the logits alone.
- Backward failed trying to allocate another 1.16 GiB. The recorded 14,406.5 MiB allocated peak is a **partial failed-run peak**, not a successful benchmark.

This supports the activation-memory hypothesis. These totals are not additive to the model footprint: saved tensors can alias parameters/storage. The audit omits Soup metrics, rewind, and optimizer state; the official benchmark uses real batches with dynamic sequence lengths. A separate 384-token synthetic diagnostic completed successfully (below); it is not the 512-token result.

# Baseline saved-tensor memory audit

Status: failed
Synthetic batch 4 x 512; unpatched NF4 + LoRA; MiB = bytes / 2^20.
This excludes Soup trainer metrics, rewind and optimizer state; use the harness for official peaks.

Error: OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 14.56 GiB of which 248.81 MiB is free. Including non-PyTorch memory, this process has 14.32 GiB memory in use. Of the allocated memory 14.06 GiB is allocated by PyTorch, and 135.61 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)

- after_model_setup: 1553.705 MiB
- before_forward: 1553.752 MiB
- after_forward: 13206.409 MiB
- peak_allocated: 14406.519 MiB
- peak_reserved: 14652.000 MiB

Save events: 894; unique data pointers: 645.
Unique backing storages: 11030.969 MiB.
Pointer totals use first-save attribution and can include frozen weights; they are not extra allocations.
Views may share storage. Storage totals avoid counting those views twice; neither total equals peak VRAM.

Saved tensors by module group

| Group | MiB |
|---|---:|
| MLP / Qwen2MLP | 3920.000 |
| attention / Qwen2Attention | 2340.500 |
| MLP / SiLUActivation | 1960.000 |
| LoraModel | 1187.016 |
| Qwen2RMSNorm | 672.766 |
| lm_head | 445.125 |
| LoRA A/B | 343.562 |
| Dropout | 162.000 |

Saved tensors by dtype

| Group | MiB |
|---|---:|
| torch.float32 | 10080.266 |
| torch.float16 | 788.688 |
| torch.bool | 162.000 |
| torch.int64 | 0.016 |

Saved tensors by layer

| Group | MiB |
|---|---:|
| outside layers | 1644.154 |
| 1 | 336.301 |
| 2 | 336.301 |
| 3 | 336.301 |
| 4 | 336.301 |
| 5 | 336.301 |
| 6 | 336.301 |
| 7 | 336.301 |
| 8 | 336.301 |
| 9 | 336.301 |
| 10 | 336.301 |
| 11 | 336.301 |
| 12 | 336.301 |
| 13 | 336.301 |
| 14 | 336.301 |
| 15 | 336.301 |
| 16 | 336.301 |
| 17 | 336.301 |
| 18 | 336.301 |
| 19 | 336.301 |
| 20 | 336.301 |
| 21 | 336.301 |
| 22 | 336.301 |
| 23 | 336.301 |
| 24 | 336.301 |
| 25 | 336.301 |
| 26 | 336.301 |
| 27 | 336.301 |
| 0 | 306.693 |

Parameters by dtype

| Group | MiB |
|---|---:|
| torch.float32 | 899.115 |
| torch.uint8 | 624.750 |

Parameters by class

| Group | MiB |
|---|---:|
| Parameter | 899.115 |
| Params4bit | 624.750 |

Layer 0: 306.693 MiB; x 28 = 8587.414 MiB (extrapolation only).

Largest saved tensors

| Module | dtype | Shape | MiB |
|---|---|---|---:|
| base_model | torch.float32 | [2048, 151936] | 1187.000 |
| base_model.model.lm_head | torch.float16 | [1536, 151936] | 445.125 |
| base_model.model.model.layers.0.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.0.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.0.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.1.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.1.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.1.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.2.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.2.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.2.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.3.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.3.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.3.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.4.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.4.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.4.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.5.mlp.act_fn | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.5.mlp | torch.float32 | [4, 512, 8960] | 70.000 |
| base_model.model.model.layers.5.mlp | torch.float32 | [4, 512, 8960] | 70.000 |

Logits: {"dtype": "torch.float16", "shape": [4, 512, 151936], "mib": 593.5}

Startup timings (s; model load can include download on a cold cache):

- import torch: 5.057
- import transformers: 9.724
- import trl: 2.237
- import peft: 13.557
- import bitsandbytes: 0.000
- import soup_cli: 0.000
- load_model: 30.172
- prepare_and_lora: 0.139

## Complete supplementary 4 x 384 diagnostic

Run version 2. Uses the harness allocator setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; version 1 used the default allocator. Sequence length and allocator differ, so the runs are not a controlled comparison.

# Baseline saved-tensor memory audit

Status: complete
Synthetic batch 4 x 384; unpatched NF4 + LoRA; MiB = bytes / 2^20.
This excludes Soup trainer metrics, rewind and optimizer state; use the harness for official peaks.

- after_model_setup: 1544.143 MiB
- before_forward: 1544.178 MiB
- after_forward: 10131.597 MiB
- after_backward: 2014.756 MiB
- peak_allocated: 11912.085 MiB
- peak_reserved: 12014.000 MiB

Save events: 894; unique data pointers: 645.
Unique backing storages: 8133.605 MiB.
Pointer totals use first-save attribution and can include frozen weights; they are not extra allocations.
Views may share storage. Storage totals avoid counting those views twice; neither total equals peak VRAM.

Saved tensors by module group

| Group | MiB |
|---|---:|
| MLP / Qwen2MLP | 2940.000 |
| attention / Qwen2Attention | 1503.375 |
| MLP / SiLUActivation | 1470.000 |
| LoraModel | 890.262 |
| Qwen2RMSNorm | 504.656 |
| lm_head | 445.125 |
| LoRA A/B | 258.688 |
| Dropout | 121.500 |

Saved tensors by dtype

| Group | MiB |
|---|---:|
| torch.float32 | 7308.281 |
| torch.float16 | 703.812 |
| torch.bool | 121.500 |
| torch.int64 | 0.012 |

Saved tensors by layer

| Group | MiB |
|---|---:|
| outside layers | 1344.398 |
| 1 | 243.266 |
| 2 | 243.266 |
| 3 | 243.266 |
| 4 | 243.266 |
| 5 | 243.266 |
| 6 | 243.266 |
| 7 | 243.266 |
| 8 | 243.266 |
| 9 | 243.266 |
| 10 | 243.266 |
| 11 | 243.266 |
| 12 | 243.266 |
| 13 | 243.266 |
| 14 | 243.266 |
| 15 | 243.266 |
| 16 | 243.266 |
| 17 | 243.266 |
| 18 | 243.266 |
| 19 | 243.266 |
| 20 | 243.266 |
| 21 | 243.266 |
| 22 | 243.266 |
| 23 | 243.266 |
| 24 | 243.266 |
| 25 | 243.266 |
| 26 | 243.266 |
| 27 | 243.266 |
| 0 | 221.035 |

Parameters by dtype

| Group | MiB |
|---|---:|
| torch.float32 | 899.115 |
| torch.uint8 | 624.750 |

Parameters by class

| Group | MiB |
|---|---:|
| Parameter | 899.115 |
| Params4bit | 624.750 |

Layer 0: 221.035 MiB; x 28 = 6188.984 MiB (extrapolation only).

Largest saved tensors

| Module | dtype | Shape | MiB |
|---|---|---|---:|
| base_model | torch.float32 | [1536, 151936] | 890.250 |
| base_model.model.lm_head | torch.float16 | [1536, 151936] | 445.125 |
| base_model.model.model.layers.0.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.0.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.0.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.1.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.1.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.1.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.2.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.2.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.2.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.3.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.3.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.3.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.4.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.4.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.4.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.5.mlp.act_fn | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.5.mlp | torch.float32 | [4, 384, 8960] | 52.500 |
| base_model.model.model.layers.5.mlp | torch.float32 | [4, 384, 8960] | 52.500 |

Logits: {"dtype": "torch.float16", "shape": [4, 384, 151936], "mib": 445.125}

Startup timings (s; model load can include download on a cold cache):

- import torch: 5.691
- import transformers: 10.686
- import trl: 2.425
- import peft: 15.160
- import bitsandbytes: 0.000
- import soup_cli: 0.000
- load_model: 23.887
- prepare_and_lora: 0.179

## Reproduction and interpretation

Run in the pinned Kaggle T4 environment after installing the task dependencies:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/memory_audit.py --out /kaggle/working/out/audit_512
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tools/memory_audit.py --seq-length 384 --out /kaggle/working/out/audit_384
```

The first command is expected to reproduce the documented OOM on this environment; the script preserves its partial JSON and Markdown report and exits nonzero. Run the second command separately. For a controlled comparison, use identical allocator settings in both runs.

Module names identify the active module at the first save, not necessarily the tensor's producer. A nested pre/post-hook stack restores the parent after child calls. Shared tensors are attributed once; the layer-0 extrapolation is illustrative because frozen-input behavior differs from later layers. Saved-tensor hooks cannot enumerate arbitrary tensors stored directly on custom Python contexts, and parameter byte sums omit auxiliary quantization buffers and CUDA context memory. Tables explain major contributors, not a complete additive accounting of peak memory.
