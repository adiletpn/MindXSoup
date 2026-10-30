# Solution memory audit - 2026-10-30

Measured solution commit: `5782a18`. All FEATURES enabled, including fp16 embedding, attention, unpadding, adaptive recomputation and background warmup.
Run: https://www.kaggle.com/code/ansargabit/ansargabit-audit2 (version 1). All three processes completed forward and backward on Tesla T4.

The current main is newer than the 6,341 MiB result in the task prompt. These are synthetic batch-4 diagnostics, not official Soup peaks: no optimizer state, trainer metrics or rewind, and allocated memory excludes the CUDA context. All processes used `CUDA_VISIBLE_DEVICES=0` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Allocated memory (MiB)

| Stage | Base T=384 | Solution T=384 | Solution T=512 |
|---|---:|---:|---:|
| after_model_setup | 1544.143 | 1099.018 | 1099.018 |
| before_forward | 1544.178 | 1099.053 | 1099.065 |
| after_forward | 10131.597 | 3706.182 | 4050.184 |
| after_backward | 2014.756 | 1125.520 | 1125.535 |
| peak_allocated | 11912.085 | 3869.568 | 4217.173 |
| peak_reserved | 12014.000 | 3958.000 | 4322.000 |

At equal length 384, allocated peak improves by 3.078x. T=512 is the requested primary solution run; do not use it as the denominator for the 384-token baseline.

## Saved tensors by group (MiB)

| Group | Base T=384 | Solution T=384 | Solution T=512 |
|---|---:|---:|---:|
| Attention | 1503.375 | 973.500 | 1178.000 |
| Dropout | 121.500 | 81.000 | 84.000 |
| Head / CE | 1335.387 | 9.000 | 12.000 |
| LoRA | 258.688 | 166.359 | 171.828 |
| MLP | 4410.000 | 945.000 | 980.000 |
| Other | 0.000 | 90.000 | 168.000 |
| RMSNorm | 504.656 | 333.434 | 348.396 |

## Saved tensors by dtype (MiB)

| Group | Base T=384 | Solution T=384 | Solution T=512 |
|---|---:|---:|---:|
| torch.bool | 121.500 | 81.000 | 84.000 |
| torch.float16 | 703.812 | 1111.359 | 1151.828 |
| torch.float32 | 7308.281 | 1405.934 | 1706.396 |
| torch.int64 | 0.012 | 0.000 | 0.000 |

Forward save events only, deduplicated by device and data pointer with first-save attribution. Views can share backing storage; saved tensors can include frozen parameters. These totals are not additive to peak memory. The active module is not necessarily the producer. The outer LoraModel saves the loss/CE gradient, classified here as Head / CE. Checkpointed layers hide their internal temporary tensors from the forward save list; the measured peak still includes their execution.

## Remaining contributors at T=512

- MLP: 980 MiB, fp16 gate/up tensors in non-checkpointed layers; more recomputation can reduce storage at a compute cost.
- Attention: 1,178 MiB, SDPA state and attention intermediates; inspect the saved shapes before proposing further reduction, and validate numerical tolerances.
- RMSNorm: 348.396 MiB, fp32 hidden inputs plus small weights/statistics; recomputation or safe storage changes may help, but naive fp16 conversion risks overflow.
- LoRA: 171.828 MiB, low-rank matrices and saved inputs; fusion/recomputation may help, while rank, targets and trained values must stay unchanged.
- Dropout: 84 MiB, saved dropout masks; do not remove dropout or change RNG behavior.
- Head / CE: 12 MiB, the precomputed hidden-state gradient; full logits are absent. Further reduction needs gradient recomputation or justified precision changes.
- Other: 168 MiB, decoder checkpoint boundary inputs; reducing boundaries can save memory at additional recompute cost.

## Parameters: tied embedding / head and norms

| Storage | Baseline | Solution |
|---|---:|---:|
| embed_tokens / tied lm_head | 890.250 MiB fp32 | 445.125 MiB fp16 |
| RMSNorm weights | 0.333984 MiB fp32 | 0.333984 MiB fp32 |

The head and embedding share a parameter: count it once. Both setup and post-forward snapshots confirm these dtypes. The existing embedding patch has already saved exactly 445.125 MiB. Converting all norm weights to fp16 would save only 0.166992 MiB and needs numerical validation. Parameter sums omit quantization auxiliary buffers.

## Top 20 saved tensors, solution T=512

| Module | dtype | Shape | MiB |
|---|---|---|---:|
| base_model.model.model.layers.14.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.14.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.15.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.15.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.16.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.16.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.17.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.17.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.18.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.18.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.19.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.19.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.20.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.20.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.21.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.21.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.22.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.22.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.23.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |
| base_model.model.model.layers.23.mlp | torch.float16 | [1, 2048, 8960] | 35.000 |

## Reproduce

```bash
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python tools/memory_audit.py --apply-solution --seq-length 512 --out /kaggle/working/out/solution512
python tools/memory_audit.py --seq-length 384 --out /kaggle/working/out/base384
python tools/memory_audit.py --apply-solution --seq-length 384 --out /kaggle/working/out/solution384
```

The audit uses synthetic IDs and needs no prepared dataset. The Kaggle runner prepares task data as part of its generic setup. Raw JSON and per-run Markdown are attached to the Kaggle run output.
