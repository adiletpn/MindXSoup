# MindX: Soup Kernel Challenge

Team MindX's solution for the Soup Kernel Challenge (Decentrathon 6.0). It makes the same
`soup train` run (Qwen2.5-1.5B-Instruct, 4-bit NF4 + LoRA, 64 steps on a Kaggle T4) faster and
lighter, and keeps the outputs: the same adapter quality, metrics and `rewind.jsonl`. All of it
lives in `solution/` and is switched on by `solution.apply()`.

| Kaggle T4, `harness/run_task.py --repeats 2` | Base Soup | MindX | Ratio |
|---|---|---|---|
| Time of `soup train` | 86.2 s | **49.1 s** | **1.76× faster** |
| Peak GPU memory (nvidia-smi) | 14 345 MiB | **3 561 MiB** | **4.03× less** |
| Eval loss on the public rows | 2.04964 | 2.04965 | +0.000% |
| Official checks (`harness/check_rules.py`) | | **10 / 10 PASS** | |

Our own Triton kernels take 11.4% of the training time (check 4 needs 5%). Across Kaggle sessions
and seeds the speed-up ranges from 1.60× to 1.76× and the memory ratio from 3.94× to 4.06×;
every run passes all 10 checks (see [Results](#results)).

**Public Kaggle notebook:** _link added at submission_ · Organizers' task description:
[docs/CHALLENGE.md](docs/CHALLENGE.md)

## What we did

The base run spends its memory and time on three things: activations kept in fp32 although they
came out of fp16 matmuls, a 151 936-wide logits tensor (1.2 GB in fp32), and pad tokens (43% of
the positions of a batch on this data). Each change below removes one of them without changing
what is computed for the real tokens. Every change is one switch in `solution/__init__.py`
(`FEATURES`) and falls back to the original code on any input it does not handle (CPU tensors,
Soup's fake test modules, other dtypes or shapes).

| # | Change | Where | Effect (approx.) |
|---|---|---|---|
| 1 | **Fused head + cross-entropy, no logits.** The head matmul runs in 512-row chunks into one fp16 buffer; our kernel takes each row once and produces the loss, its gradient (written in place), the entropy, the argmax and the per-token loss. TRL's `entropy`, `num_tokens`, `mean_token_accuracy` and Soup's per-row loss for `rewind.jsonl` (through Soup's `ChunkedStep` record) come from these values. | `fused_ce.py` | −3 to −4 GB |
| 2 | **fp16 storage of fp16 values.** `prepare_model_for_kbit_training` keeps the residual stream in fp32, so the base stores fp32 copies of fp16 matmul outputs. The MLP runs on one fp16 input cast and keeps `gate`/`up` in fp16; attention casts its input once and one kernel adds the LoRA output, applies RoPE and writes fp16 q, k, v; the frozen tied embedding / `lm_head` stays in fp16 (it is never upcast). Same values, fewer casts. | `mlp.py`, `attention.py`, `embedding.py` | −3 GB, −10% step |
| 3 | **Pad tokens never enter the model.** The real tokens of the 4 rows are packed into one sequence (position ids restart per row); attention runs per row, causal, on k/v repeated to the 12 query heads (the fp16 fused SDPA kernel, as in the base); targets are shifted inside each row before packing; per-token values are scattered back to `[B, T]` for TRL and Soup. LoRA dropout draws its mask on the full padded shape, exactly as the base, and keeps the real rows, so the random masks are the same. | `unpad.py` | −40% step, −1 GB |
| 4 | **Adaptive recomputation.** A batch of N real tokens reruns the first ⌈28 · (1 − 1024 / N)⌉ decoder layers in backward (`torch.utils.checkpoint` in our patch; Soup's `gradient_checkpointing` stays off). Batches up to 1024 tokens rerun nothing; the longest ones rerun half of the layers. The peak is set by the longest batches, so this caps it at little time cost. | `recompute.py` | −0.9 GB |
| 5 | **NF4 unpacking kernel.** bitsandbytes dequantizes the double-quantized NF4 weights in two kernels (8-bit absmax, then the 4-bit values); ours does it in one pass, used by the forward fallback and by every backward. | `nf4.py` | 8% of the step in our kernel |
| 6 | **RMSNorm forward/backward kernel.** One launch each way instead of 5–6 PyTorch ops; keeps the input and `rstd`. | `rmsnorm.py` | fewer launches |
| 7 | **Startup.** Triton compiles our kernels in a background thread while Soup loads the model (first step 4.6 → 0.8 s); `importlib.metadata.packages_distributions()` (2.3 s) is computed once instead of twice; `datasets` gets a random fingerprint for `map`/`filter` on in-memory datasets instead of hashing the tokenizer with dill; the unused `sklearn` and `wandb` imports are skipped (≈3 s). All allowed by the organizers in the participants' chat (30–31 Oct), all inside the measured run. | `warmup.py`, `startup.py` | ≈ −8 s |

### Our kernels

Share of the training time from the harness (CUDA events on our launches, steps 3–64, run of
31.10 01:10, seed 7):

| Kernel (`@triton.jit`) | File | Replaces | Share |
|---|---|---|---|
| `nf4_unpack_kernel` | `solution/nf4.py` | bitsandbytes `dequantize_blockwise` + `dequantize_4bit` | 8.2% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` (pow, mean, rsqrt, mul, cast; autograd of each) | 1.7% |
| `qkv_rope_kernel` | `solution/attention.py` | LoRA add, RoPE and three fp16 casts of q/k/v, forward and backward | 0.8% |
| `head_ce_rows_kernel` | `solution/fused_ce.py` | log-softmax, NLL, its gradient, TRL's entropy and argmax over 151 936 logits | 0.8% |
| `rope_kernel` | `solution/rope.py` | `apply_rotary_pos_emb` (used when the fp16 attention path does not apply) | — |

The organizers' example `example_swiglu_*` (`solution/swiglu.py`) is kept and runs inside our fp16
MLP; it does not count as ours and is not counted above. No `@triton.autotune`; every size that
changes between batches is in `do_not_specialize`, so a new length never recompiles. GEMMs stay in
cuBLAS (Triton's `tl.dot` does not use the T4's tensor cores).

### Why the outputs stay the same

- Everything that is fp16 in the base is fp16 here and everything that is fp32 stays fp32; we only
  drop fp32 *copies* of fp16 numbers and the pad positions, which a causal mask never lets a real
  token see.
- The random draws are the base's: LoRA dropout masks are generated on the padded shape, and
  recomputation restores the generator state; warm-up uses zeros only.
- Tests (`python -m pytest solution -q`, 68 on a T4) compare every kernel and patch with PyTorch,
  forward and backward, including a full Qwen2 training step padded vs. unpadded, with and without
  recomputation, and bit-identical results for the fp16 embedding.

## How to run

**Kaggle (the jury's way).** Open the public notebook above, or import
`templates/kaggle_notebook.ipynb` into Kaggle (Accelerator **GPU T4 x2**, Internet **on**), put
this repository in the first cell (`COMMIT` = the final commit) and **Run All** (10–15 minutes). It
trains the base and our solution, scores both adapters and prints the 10 checks. For a private
repository add a Kaggle secret `GITHUB_TOKEN` with read access.

**Linux with a CUDA GPU** (only T4 numbers are comparable):

```bash
pip install "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" "trl==0.29.1" \
    "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest
python task/prepare_data.py                  # data/train.jsonl, data/eval_public.jsonl
python -m pytest solution -q                 # our kernels and patches vs PyTorch
python harness/run_task.py --repeats 2       # base and solution -> runs/result.json
python harness/check_rules.py                # the 10 checks -> runs/checks.json
python harness/score.py runs/result.json     # speed-up and memory ratio
```

**From a laptop without a GPU** (how we worked): `tools/kaggle_run.py` packs the working tree into
a Kaggle script, runs it on a T4 and downloads the logs (`kaggle_out/`); it needs the Kaggle CLI
and `~/.kaggle/kaggle.json`.

```bash
python tools/kaggle_run.py tests                          # pytest solution
python tools/kaggle_run.py full --repeats 2 --seed 7      # the jury's measurement
```

To see what one change gives, set its entry in `FEATURES` (`solution/__init__.py`) to `False`.

## Results

All on Kaggle T4, `--repeats 2` unless noted, full log in [docs/results.md](docs/results.md).
Kaggle machines differ by up to ±10% between sessions, so compare within a row; the effect of a
single change was measured A/B on one machine where stated.

| Step (cumulative) | Time base → ours, s | Peak, MiB | Speed-up | Memory | Checks |
|---|---|---|---|---|---|
| RMSNorm + fused CE + NF4 + fp16 MLP (1 repeat) | 81.4 → 74.5 | 6 343 | 1.09× | 2.26× | 10/10 |
| + RoPE kernel, background compile | 83.2 → 67.7 | 6 341 | 1.23× | 2.26× | 10/10 |
| + fp16 tied embedding | 92.1 → 75.0 | 5 461 | 1.23× | 2.63× | 10/10 |
| + fp16 attention | 85.5 → 70.1 | 5 441 | 1.22× | 2.64× | 10/10 |
| + unpadding | 80.9 → 57.0 | 5 487 | 1.42× | 2.61× | 10/10 |
| + adaptive recomputation | 88.3 → 58.6 | 4 569 | 1.51× | 3.14× | 10/10 |
| + startup savings | 83.8 → 53.2 | 4 569 | 1.58× | 3.14× | 10/10 |
| + per-row SDPA on repeated k/v | 79.9 → 49.9 | 3 561 | 1.60× | 4.03× | 10/10 |
| **final `main`, seed 1** | 86.2 → 49.1 | 3 561 | **1.76×** | **4.03×** | **10/10** |
| final `main`, seed 7 | 85.9 → 48.8 | 3 539 | 1.76× | 4.06× | 10/10 |
| final, seed 1234 | 84.6 → 51.3 | 3 639 | 1.65× | 3.94× | 10/10 |

Same-machine A/B: the per-row SDPA fix alone gives −6.3 s and −1 000 MiB; the fp16 attention path
−7% of the step time. Eval loss stays within ±0.003% of the base in every run.

Where the time goes now: cuBLAS matmuls are the largest part (63% of the GPU time in our last
step profile), our kernels 11%, the rest attention, casts and small ops. Where the memory goes at T = 512 (synthetic batch,
[docs/memory_audit_solution.md](docs/memory_audit_solution.md)): attention 1.2 GB, MLP 1.0 GB,
RMSNorm inputs 0.35 GB, weights 1.1 GB.

## Limitations

- Tuned for this task: Qwen2 with SDPA attention, a T4 with fp16 autocast, NF4 with double
  quantization, right padding. Anything else takes the original code (correct, not faster); new
  sizes only compile again.
- The step is now bound by cuBLAS matmuls; further speed needs fewer FLOPs, not faster
  elementwise code.
- The memory peak depends on the longest batch; recomputation caps it by a token budget (1024),
  which trades time for memory. A lower budget saves more memory and costs more time.
- `torch.utils.checkpoint` doubles the forward of the recomputed layers of long batches.
- On other data the speed-up depends on how much padding the batches have (43% here).

## Sources of ideas

We wrote all code ourselves; these works gave the ideas:

- Liger Kernel, Hsu et al., [arXiv:2410.10989](https://arxiv.org/abs/2410.10989): fused linear +
  cross-entropy computed in chunks.
- Cut Cross-Entropy, Wijmans et al. (Apple), [arXiv:2411.09009](https://arxiv.org/abs/2411.09009):
  never materializing the full logits.
- RoFormer, Su et al., [arXiv:2104.09864](https://arxiv.org/abs/2104.09864): rotary position
  embedding.
- QLoRA, Dettmers et al., [arXiv:2305.14314](https://arxiv.org/abs/2305.14314): NF4 and double
  quantization.
- Triton tutorial "Layer Normalization": one program per row for a norm and its backward.
- Padding-free / packed training with per-sequence attention (as in FlashAttention varlen),
  [arXiv:2205.14135](https://arxiv.org/abs/2205.14135).
- Activation checkpointing, Chen et al., [arXiv:1604.06174](https://arxiv.org/abs/1604.06174).
- The organizers' Q&A and participants' chat (startup savings, background compilation, fp16
  storage of frozen weights), summarized in [docs/QA_NOTES.md](docs/QA_NOTES.md).

## Team

| | GitHub | Did |
|---|---|---|
| Adilet | adiletpn | architecture, fused head + CE, fp16 MLP and embedding, NF4, RMSNorm, unpadding, recomputation, startup, Kaggle runner |
| Dilya | meeexcv-lang | RoPE kernel, fp16 attention path (`qkv_rope_kernel`), per-row SDPA fix, seed checks |
| Ansar | ansargabit99-boop | baseline profile, memory audits of base and solution, startup probe, submission notebook |
| | aqnietqaysarbay-ctrl | presentation, video |

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The test files in `tests/` come from Soup v0.77.0. Data:
[AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2);
model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
