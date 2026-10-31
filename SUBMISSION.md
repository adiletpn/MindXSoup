# Submission: MindX

## Links
- Repository, final commit: https://github.com/BAITC-Hacks/hack-3e361ee8-mindx/commit/<FINAL_SHA> _(set at submission, 31.10)_
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): <KAGGLE_NOTEBOOK_URL>?scriptVersionId=<VERSION> _(set at submission)_
- Demo video (up to 3 minutes): <VIDEO_URL> _(set at submission)_
- Presentation: <PRESENTATION_URL> _(set at submission)_

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Adilet | captain; core solution, kernels, integration, Kaggle runner | adiletpn | adiletzholdasbek | via the platform |
| Dilya | RoPE kernel, fp16 attention path, per-row SDPA fix, wide NF4 kernel, one-kernel LoRA dropout, speed/memory ladder, final-candidate runs | meeexcv-lang | aqniet | |
| Ansar | profiles, memory audits (peak owners), gate/up recompute, budget A/B, submission notebook | ansargabit99-boop | ansargabit | |
| | presentation, video | aqnietqaysarbay-ctrl | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_unpack_wide_kernel` | `solution/nf4.py` | bitsandbytes' two-kernel NF4 dequantization (8-bit absmax, then 4-bit values) | 7.8% |
| `lora_dropout_kernel` | `solution/dropout.py` | gather of the real rows of the base's dropout mask, scale, fp16 cast (forward and backward) | 1.6% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` forward and backward | 1.6% |
| `qkv_rope_kernel` | `solution/attention.py` | LoRA add and scaling, RoPE and the fp16 casts of q/k/v, forward and backward | 0.8% |
| `head_slice_kernel` | `solution/head_stream.py` | online softmax / CE / entropy / argmax and the hidden-state gradient over vocabulary slices of the host table | 0.5% |
| `head_ce_rows_kernel`, `rope_kernel` | `solution/fused_ce.py`, `solution/rope.py` | fallbacks (head with the table on the GPU; RoPE outside the fp16 attention path) | — |
| **total** | | | **13.1%** in the final run (12.0–17.3% across runs) |

Ideas taken from other work, with links (we wrote the code ourselves): Liger Kernel
(arXiv:2410.10989), Cut Cross-Entropy (arXiv:2411.09009), RoFormer (arXiv:2104.09864), QLoRA
(arXiv:2305.14314), the Triton "Layer Normalization" tutorial, padding-free attention as in
FlashAttention varlen (arXiv:2205.14135), activation checkpointing (arXiv:1604.06174), online
softmax (arXiv:1805.02867), ZeRO-Offload (arXiv:2101.06840). Details and
the rule each technique relies on: README, "What we did" and "Within the rules".

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
Final code, `harness/run_task.py --repeats 2` (commit `73402c2`, 31.10 22:00; the submitted
commit differs only in documents). The notebook above re-runs it.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 81.1 | 60.4 | speedup 1.34× |
| Peak GPU memory, MiB | 14 345 | 505 | base/ours 28.4× |
| Eval loss (public rows) | 2.04964 | 2.04964 | 0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (every candidate run on 31.10; seeds 1, 7, 1234 earlier)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (fp32 copies of fp16 values, the logits tensor, padding, CPU
overhead), how the kernels work, results against the base (speed, memory, quality), limits and next
steps.

## Limitations
Every decoder layer is recomputed in backward (one extra forward per step): we traded that time for
memory because the score is logarithmic in both ratios (1 077 → 505 MiB cost about 1.44× → 1.34×).
The step is bound by cuBLAS matmuls and NF4 unpacking: cutting ~1 300 small launches per step did
not change the time. Next steps: activations in host memory instead of recomputation, a fused NF4
unpack + matmul. Own tokenization (−1.8 s) was removed after the organizers' ruling. Tuned for this
task (Qwen2, SDPA, T4 fp16, NF4 with double quantization, right padding); anything else takes the
original code.
