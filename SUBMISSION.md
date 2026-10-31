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
| Adilet | captain; core solution, kernels, integration | adiletpn | adiletzholdasbek | via the platform |
| Dilya | RoPE kernel, fp16 attention path, per-row SDPA fix, seed checks | meeexcv-lang | | |
| Ansar | profiles, memory audits, startup probe, submission notebook | ansargabit99-boop | ansargabit | |
| | presentation, video | aqnietqaysarbay-ctrl | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_unpack_kernel` | `solution/nf4.py` | bitsandbytes' two-kernel NF4 dequantization (8-bit absmax, then 4-bit values) | 8.2–8.8% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` forward and backward | 1.8% |
| `qkv_rope_kernel` | `solution/attention.py` | LoRA add, RoPE and the fp16 casts of q/k/v, forward and backward | 1.0% |
| `head_ce_rows_kernel` | `solution/fused_ce.py` | log-softmax / NLL / gradient / entropy / argmax over the 151 936 logits | 0.6% |
| `rope_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (fallback when the fp16 attention path does not apply) | — |
| **total** | | | **11.6%** (lowest of 2 runs) |

Ideas taken from other work, with links (we wrote the code ourselves): Liger Kernel
(arXiv:2410.10989), Cut Cross-Entropy (arXiv:2411.09009), RoFormer (arXiv:2104.09864), QLoRA
(arXiv:2305.14314), the Triton "Layer Normalization" tutorial, padding-free attention as in
FlashAttention varlen (arXiv:2205.14135), activation checkpointing (arXiv:1604.06174). Details and
the rule each technique relies on: README, "What we did" and "Within the rules".

## Results on Kaggle T4 (from `runs/result.json` and `runs/checks.json`)
Measured run of the current `main` (`harness/run_task.py --repeats 2`, 31.10 05:30); the final
numbers from the notebook above replace these at submission.

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 83.5 | 58.4 | speedup 1.43× |
| Peak GPU memory, MiB | 14 345 | 1 209 | base/ours 11.87× |
| Eval loss (public rows) | 2.04966 | 2.04966 | +0.000% vs base |

- [x] All 10 checks of `harness/check_rules.py` passed (every run since 30.10, seeds 1, 7, 1234)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Presentation covers
Which operations we sped up and why (fp32 copies of fp16 values, the logits tensor, padding, CPU
overhead), how the kernels work, results against the base (speed, memory, quality), limits and next
steps.

## Limitations
Every decoder layer is recomputed in backward (one extra forward per step, about +6 s): we traded
that time for memory (3.56 → 1.74 GB) because the score rewards the larger relative gain; with a
1 024-token budget the same code runs at 1.78× / 4.03×. The tied embedding / `lm_head` (445 MiB)
still sits on the GPU. The speed-up depends on the padding in the batches (43% of the positions
here) and on PCIe overlapping the weight copies. Tuned for this task (Qwen2, SDPA, T4 fp16, NF4 with
double quantization, right padding); anything else takes the original code.
