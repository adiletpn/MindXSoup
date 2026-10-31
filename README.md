# MindX: Soup Kernel Challenge

Team MindX's solution for the Soup Kernel Challenge (Decentrathon 6.0). It makes the same
`soup train` run (Qwen2.5-1.5B-Instruct, 4-bit NF4 + LoRA, 64 steps on a Kaggle T4) faster and
lighter, and keeps the outputs: the same adapter quality, metrics and `rewind.jsonl`. All of it
lives in `solution/` and is switched on by `solution.apply()`.

| Kaggle T4, `harness/run_task.py --repeats 2` | Base Soup | MindX | Ratio |
|---|---|---|---|
| Time of `soup train` | 81.1 s | **60.4 s** | **1.34× faster** |
| Peak GPU memory (nvidia-smi, whole run) | 14 345 MiB | **505 MiB** | **28.4× less** |
| Eval loss on the public rows | 2.04964 | 2.04964 | **0.000%** |
| Official checks (`harness/check_rules.py`) | | **10 / 10 PASS** | |

Measured run of the final code (commit `73402c2`, 31.10 22:00; later commits change only documents).
Our own Triton kernels take 13.1% of the training time (check 4 needs 5%).
The eval loss stays within ±0.005% of the base on every seed we ran (1, 7, 1234). We traded speed
for memory on purpose: the score is logarithmic in both ratios, and 1 077 → 505 MiB cost us about
1.44× → 1.34×; with recomputation off the code runs about 6 s faster at 4.1 GB, see
[Results](#results).

**Public Kaggle notebook:** https://www.kaggle.com/code/ansargabit/mindx-soup-kernel-challenge (COMMIT = `d0ef96f`) · Organizers' task description:
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
| 1 | **Fused head + cross-entropy, no logits.** The head matmul runs in 256-row chunks into one fp16 buffer; our kernel takes each row once and produces the loss, its gradient (written in place), the entropy, the argmax and the per-token loss. TRL's `entropy`, `num_tokens`, `mean_token_accuracy` and Soup's per-row loss for `rewind.jsonl` (through Soup's `ChunkedStep` record) come from these values. | `fused_ce.py` | −3 to −4 GB |
| 2 | **fp16 storage of fp16 values.** `prepare_model_for_kbit_training` keeps the residual stream in fp32, so the base stores fp32 copies of fp16 matmul outputs. The MLP runs on one fp16 input cast and keeps `gate`/`up` in fp16; attention casts its input once and one kernel adds the LoRA output, applies RoPE and writes fp16 q, k, v; the frozen tied embedding / `lm_head` stays in fp16 (it is never upcast). Same values, fewer casts. | `mlp.py`, `attention.py`, `embedding.py` | −3 GB, −10% step |
| 3 | **Pad tokens never enter the model.** The real tokens of the 4 rows are packed into one sequence (position ids restart per row); attention runs per row, causal, on k/v repeated to the 12 query heads (the fp16 fused SDPA kernel, as in the base); targets are shifted inside each row before packing; per-token values are scattered back to `[B, T]` for TRL and Soup. LoRA dropout draws its mask on the full padded shape, exactly as the base, and keeps the real rows, so the random masks are the same. | `unpad.py` | −40% step, −1 GB |
| 4 | **Recomputation.** Each decoder layer keeps only its input for backward and reruns its forward there (`torch.utils.checkpoint` in our patch; Soup's `gradient_checkpointing` stays off). A token budget sets how many layers a batch reruns (⌈28 · (1 − budget / N)⌉ of N real tokens); we use 0, every layer, which scored best (1 739 vs 3 561 MiB for +6 s). | `recompute.py` | −1.8 GB, +6 s |
| 5 | **Decoder weights live in pinned host memory.** Right after bitsandbytes quantizes a decoder weight while transformers loads the model, its packed NF4 bytes move to pinned host memory; the GPU keeps a 1-byte placeholder of the same shape. Two GPU slots hold the layers in use: while layer i runs, a side CUDA stream copies the next layer of the pass into the other slot (12.4 GB/s on the T4; a layer's 24 MiB take 2 ms, hidden behind its 4 ms of matmuls). Every 4-bit matmul asks for its bytes right before use, so correctness never depends on the prefetch. The loader's `caching_allocator_warmup` (a whole-model GPU reservation that only speeds up loading) is skipped. | `stream.py` | −0.53 GB, ~0 s |
| 6 | **NF4 unpacking kernel and a direct 4-bit linear.** bitsandbytes dequantizes the double-quantized NF4 weights in two kernels (8-bit absmax, then the 4-bit values); ours does it in one pass. For fp16 inputs our `Linear4bit` path runs exactly bitsandbytes' own ops (unpack, `F.linear`; backward unpack, matmul) without its Python dispatch layers, bit-identical. | `nf4.py`, `linear4bit.py` | 8% of the step in our kernel |
| 7 | **RMSNorm forward/backward kernel.** One launch each way instead of 5–6 PyTorch ops; keeps the input and `rstd`. | `rmsnorm.py` | fewer launches |
| 8 | **The tied embedding / `lm_head` table in pinned host memory.** The 445 MiB fp16 table never sits on the GPU: while transformers loads the checkpoint it is converted on the CPU (the same conversion as before) straight into pinned host memory, and the model gets a 1-element GPU placeholder of its shape. Before that change the table sat on the GPU from loading to the first step, and that set the whole run's peak (673 → 505 MiB). The embedding lookup gathers the batch's rows on the host; the head walks the vocabulary in slices of 4 096 rows, each copied over PCIe into one of two GPU buffers while the previous one is used. Our `head_slice_kernel` carries each row's online-softmax state (running max, sum of exp, label logit, first argmax) from slice to slice in fp32 and, in the same walk, turns the slice into `exp(x − m)` so the hidden-state gradient is accumulated with FlashAttention-style rescaling: one pass over the table per step. Slice boundaries are multiples of the old kernel's block, so loss, entropy and argmax match it. | `head_stream.py` | −445 MiB |
| 9 | **Cheaper recompute boundaries.** A rerun layer keeps its fp32 input as fp16 rows scaled by one fp32 factor per row (no overflow possible), and that copy waits for backward in pinned host memory. Layers that are not rerun keep the MLP's fp16 input instead of `gate` and `up` and redo those two matmuls in backward (same bytes, same kernels, no random draws). The token budget is 0: every layer reruns (59.2 s / 675 MiB against 58.3 s / 957 MiB at 256, same machine). | `recompute.py`, `mlp_recompute.py` | −300 to −400 MiB, +2 s |
| 12 | **Fewer launches around attention.** Per-row attention splits q/k/v with `split` (its backward is one `cat`; slicing gave a zero-filled full-size gradient and an add per row and tensor, ~1 000 launches per step) and transposes all rows' outputs with one copy after the `cat`. peft's LoRA scaling (2.0, a power of two, so exact in fp16) is applied inside `qkv_rope_kernel` instead of a separate multiply. Same bits (tested). | `unpad.py`, `attention.py` | −1 300 launches per step |
| 10 | **LoRA dropout in one kernel.** The mask is drawn by PyTorch on the full padded shape exactly as in the base (same random numbers); our `lora_dropout_kernel` applies it to the real rows and writes the fp16 input of `lora_A` in one pass. | `dropout.py`, `unpad.py` | fewer launches |
| 11 | **Startup.** Triton compiles our kernels in a background thread while Soup loads the model (first step 4.6 → 0.8 s); `importlib.metadata.packages_distributions()` (2.3 s) is computed once instead of twice; the unused `sklearn` and `wandb` imports are skipped (≈3 s); `TORCHDYNAMO_DISABLE` is set only while bitsandbytes is imported, so it does not build `torch.compile` wrappers for its CPU backend (1.2 s); checkpoint tensors are converted bf16 → fp16 on the GPU instead of the CPU (same values); the compile thread runs at `nice 19` so it does not take the CPU from training; the CUDA per-thread stack reservation is set to 256 B. All allowed by the organizers in the participants' chat (30–31 Oct), all inside the measured run. | `warmup.py`, `startup.py` | ≈ −9 s |

### Our kernels

Share of the training time from the harness (CUDA events on our launches, steps 3–64; per kernel
from the profiled run of `1d3b11e`, 12.0% in total; 13.1% in the final run):

| Kernel (`@triton.jit`) | File | Replaces | Share |
|---|---|---|---|
| `nf4_unpack_wide_kernel` | `solution/nf4.py` | bitsandbytes `dequantize_blockwise` + `dequantize_4bit` | 7.8% |
| `head_slice_kernel` | `solution/head_stream.py` | online softmax / CE / entropy / argmax and the gradient over vocabulary slices of the host table | 0.5% |
| `lora_dropout_kernel` | `solution/dropout.py` | dropout of the LoRA input on the real rows + fp16 cast | 1.6% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` (pow, mean, rsqrt, mul, cast; autograd of each) | 1.6% |
| `qkv_rope_kernel` | `solution/attention.py` | LoRA add, RoPE and three fp16 casts of q/k/v, forward and backward | 0.8% |
| `head_ce_rows_kernel` | `solution/fused_ce.py` | log-softmax, NLL, its gradient, TRL's entropy and argmax over 151 936 logits (used when the table is on the GPU) | — |
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
- Weight streaming moves the same bytes; loading, the dtype conversion and the 4-bit linear path
  give bit-identical tensors (tested).
- Tests (`python -m pytest solution -q`, 135 on a T4) compare every kernel and patch with PyTorch,
  forward and backward, including a full Qwen2 training step padded vs. unpadded, with and without
  recomputation, streamed vs. resident 4-bit weights with constant eviction, a 4-bit model loaded
  with and without the load-time offload, and bit-identical results for the fp16 embedding.

### Within the rules

Every change is our own code in `solution/`; no Soup or TRL switch is changed. Where a rule or an
organizer's answer covers a technique, it is cited here (rules: [docs/CHALLENGE.md](docs/CHALLENGE.md),
"Rules"; organizers' answers: [docs/QA_NOTES.md](docs/QA_NOTES.md)).

| Technique | Why it is allowed |
|---|---|
| Own Triton kernels, autograd functions, patches of transformers / peft / bitsandbytes / Soup | Rules, "Allowed"; QA 3, 15, 24, 34, 40 |
| Head + loss without full logits; TRL metrics and the rewind per-row loss computed by our kernel and handed to Soup's `ChunkedStep` record | QA 5, 12, 13, 35; Soup's `stream_loss_chunk` switch stays off |
| **Unpadding.** The real tokens of one batch are concatenated; attention runs separately per row (causal, position ids restart per row), so no row sees another; rows, batches, their order and the token count (`attention_mask` sum, collator untouched) are unchanged. Soup's / TRL's `packing` and `padding_free` switches stay off. | Rules: "Removing padding inside a batch in your own code is fine; mixing rows across batches is not"; QA 8 |
| Recomputation of decoder layers with `torch.utils.checkpoint` in our patch | Rules, "Allowed: recomputation … in your own code (its time counts)"; QA 25; Soup's `gradient_checkpointing` stays off |
| fp16 storage of activations that are fp16 numbers | Case brief, section 9 hints; QA 6, 32 |
| Frozen tied embedding / `lm_head` kept in fp16 (never upcast; same values and compute dtypes) | QA 7, 17, 33, 42 |
| Triton compilation in a thread of the measured process (no helper process, empty cache) | QA 18, 41, 46 |
| Startup: one `packages_distributions()` scan, unused `sklearn`/`wandb` imports skipped, `TORCHDYNAMO_DISABLE` only while bitsandbytes is imported | QA 37, 45, 49; nothing stubbed in `sys.modules`, nothing reused between runs |
| Decoder weights in pinned host memory, streamed per layer; loader warm-up reservation skipped | Rules, "Allowed: … offloading in your own code (its time counts)"; QA 47 (CPU offloading of frozen weights, copies and syncs in the measured time) |
| bf16 → fp16 conversion of checkpoint tensors on the GPU at load | Same values (tested); QA 37 (startup work may be optimized inside the run) |
| LoRA dropout | Same rate, same random masks as the base (drawn on the padded shape); QA 4, 60 |
| Tied embedding / `lm_head` table in pinned host memory (put there at load), head over vocabulary slices with online softmax | QA 47 (own CPU offloading of the frozen table, head streamed in chunks), QA 70 (slice size, buffers, prefetch timing; only the summation order changes), QA 62 |
| Recompute inputs kept as row-scaled fp16 in pinned host memory; MLP gate/up redone in backward | QA 6, 32 (compact storage of tensors kept for backward), QA 25, 67 (own recomputation), QA 47, 52 (own offloading) |
| CUDA per-thread stack reservation set to 256 B | QA 59, 74 |
| Compile thread at `nice 19` | QA 68 |
| **Removed** after rulings: own SFT tokenization (`Dataset.from_list` instead of Soup's `map`), explicit `new_fingerprint` | QA 57, 68, 69 |

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
single change was measured A/B on one machine where stated. Time is the harness's median over
the repeats, the peak its maximum (nvidia-smi over the whole process).

### Final solution vs. base

The final commit `73402c2`, `harness/run_task.py --repeats 2`, each run on its own Kaggle T4:

| Run | Base, s | MindX, s | Speed-up | Peak base → MindX, MiB | Memory | Eval loss base / MindX | Checks, tests |
|---|---|---|---|---|---|---|---|
| seed 1 (`mindx-final500b`) | 79.75 / 82.35 | **60.30 / 60.50** | **1.34×** | 14 345 → **505** | **28.4×** | 2.04964 / 2.04964 (0.000%) | 10/10, own kernels 13.1%, 135 tests |
| seed 1, another machine (`dilya-f500`) | 92.49 / 85.56 | 64.05 / 63.70 | 1.39× | 14 345 → 505 | 28.4× | 2.04967 / 2.04965 (−0.001%) | 10/10, own kernels 12.9%, 135 tests |
| seed 7 (`dilya-f500-s7`) | 79.77 / 85.69 | 62.62 / 64.58 | 1.30× | 14 351 → 507 | 28.3× | 2.05115 / 2.05117 (+0.001%) | 10/10, own kernels 12.1%, 135 tests |

The same code gives 1.30–1.39× on different machines: the base itself took 79.8–92.5 s. The peak is
505–507 MiB in every run, and PyTorch's own peak is 343 MiB.

### How we got here

| Step (cumulative) | Time base → ours, s | Peak, MiB | Eval loss base / ours | Speed-up | Memory | Checks |
|---|---|---|---|---|---|---|
| RMSNorm + fused CE + NF4 + fp16 MLP (1 repeat) | 81.4 → 74.5 | 6 343 | 2.04964 / 2.0497 | 1.09× | 2.26× | 10/10 |
| + RoPE kernel, background compile | 83.2 → 67.7 | 6 341 | 2.04965 / 2.0496 | 1.23× | 2.26× | 10/10 |
| + fp16 tied embedding | 92.1 → 75.0 | 5 461 | 2.04965 / 2.04965 | 1.23× | 2.63× | 10/10 |
| + fp16 attention | 85.5 → 70.1 | 5 441 | 2.04965 / 2.04967 | 1.22× | 2.64× | 10/10 |
| + unpadding | 80.9 → 57.0 | 5 487 | 2.04968 / 2.04967 | 1.42× | 2.61× | 10/10 |
| + adaptive recomputation (budget 1 024) | 88.3 → 58.6 | 4 569 | 2.04962 / 2.04967 | 1.51× | 3.14× | 10/10 |
| + startup savings | 83.8 → 53.2 | 4 569 | 2.04968 / 2.04967 | 1.58× | 3.14× | 10/10 |
| + per-row SDPA on repeated k/v | 79.9 → 49.9 | 3 561 | 2.04966 / 2.04971 | 1.60× | 4.03× | 10/10 |
| same, seed 7 | 85.9 → 48.8 | 3 539 | 2.05114 / 2.05118 | 1.76× | 4.06× | 10/10 |
| same, seed 1234 | 84.6 → 51.3 | 3 639 | — (−0.001%) | 1.65× | 3.94× | 10/10 |
| + bitsandbytes imported without compile wrappers | 87.9 → 49.3 | 3 561 | 2.04961 / 2.04966 | 1.78× | 4.03× | 10/10 |
| + recomputation of every layer (budget 0) | 83.5 → 55.3 | 1 739 | 2.04968 / 2.04971 | 1.51× | 8.25× | 10/10 |
| + decoder weights streamed from host memory | 83.2 → 55.8 | 1 413 | 2.04966 / 2.04967 | 1.49× | 10.15× | 10/10 |
| + offload at load, GPU dtype conversion, head chunk 256 | 83.5 → 58.4 | 1 209 | 2.04966 / 2.04966 | 1.43× | 11.87× | 10/10 |
| budget 512, backward prefetch hints, one-kernel LoRA dropout | 88.3 → 50.8 | 2 067 | — (−0.001%) | 1.61× | 6.94× | 10/10 |
| budget 256, wide NF4 kernel | 86.6 → 54.8 | 1 603 | — | 1.58× | 8.95× | 10/10 |
| + embedding / head table in host memory, fp16 recompute inputs (`a0dfdee`) | 89.3 → 58.4 | 1 077 | 2.04965 / 2.04967 | 1.53× | 13.32× | 10/10 |
| − own tokenization (organizers' ruling), warm-up thread at nice 19 (`f5da29b`) | 79.6 → 55.2 | 1 077 | — (+0.002%) | 1.44× | 13.32× | 10/10 |
| + gate/up recompute, recompute inputs in host memory, head slices of 4 096 (`196bd9a`) | 92.3 → 65.6 | 757 | 2.04963 / 2.04966 | 1.41× | 18.95× | 10/10 |
| + recompute budget 0, fewer attention launches, LoRA scale in the kernel (`7df91c3`) | 86.3 → 62.0 | 673 | 2.04966 / 2.04964 | 1.39× | 21.3× | 10/10 |
| **+ the table loaded straight into host memory (`73402c2`, final)** | **81.1 → 60.4** | **505** | **2.04964 / 2.04964** | **1.34×** | **28.4×** | **10/10** |
| same, seed 7 (`d0ef96f`) | 84.8 → 65.2 | 507 | 2.05115 / 2.05121 | 1.30× | 28.3× | 10/10 |

Same-machine A/B: the per-row SDPA fix alone gives −6.3 s and −1 000 MiB; the fp16 attention path
−7% of the step time. Eval loss stays within ±0.005% of the base in every run.

### Speed against memory, measured on one machine

Each rung adds one change to the previous one (`1d3b11e`, one run each, `aqniet/dilya-ladder`).
Points by the score formula with the base at 86.6 s / 14 345 MiB and a best team assumed at
41 s / 600 MiB:

| Rung | Time, s | Peak, MiB | Median step, s | Points (35 + 30) |
|---|---|---|---|---|
| `1d3b11e` | 55.0 | 1 077 | 0.510 | ≈ 45.7 |
| + recompute budget 512 | 54.7 | 1 561 | 0.528 | ≈ 42.5 |
| + decoder weights kept on the GPU | 53.4 | 2 127 | 0.532 | ≈ 40.7 |
| + embedding / head table kept on the GPU | 53.3 | 2 527 | 0.538 | ≈ 39.1 |
| + no recomputation | 48.6 | 4 149 | 0.465 | ≈ 38.8 |

Every rung wins less on speed than it loses on memory, so we went the other way: on the final code
budget 0 instead of 256 costs ≈ +1 s for −282 MiB (59.2 s / 675 MiB vs 58.3 s / 957 MiB).

### Where the time and the memory go now

**Time.** Profile of 16 steps of `1d3b11e`: cuBLAS matmuls ≈ 0.19 s per step, NF4 unpacking,
attention and small ops ≈ 0.15 s, host-to-device copies ≈ 0.14 s on a side stream (overlapped).
Removing ~1 300 small launches per step did not change the time measurably, so the step is bound by
GPU work, not by Python. An earlier, finer GPU profile ([docs/runtime_profile.md](docs/runtime_profile.md),
`7384b69`, **measured before PR #11**, so before the host-memory table and the final recomputation):
matmuls 43% of the summed GPU time, host-to-device copies 23%, NF4 unpacking 9%, attention 5%,
elementwise ops, SwiGLU, casts and dropout ≈ 4% each. About 19 s of each run is outside the
training loop; 2.4 s of it is the metadata scan in `import transformers`, before `apply()` runs.

**Memory.** A replay of the CUDA allocator's history (`tools/peak_memory.py`) puts PyTorch's peak
(343 MiB) in the backward of a rerun decoder layer on the longest batch: that layer's gate/up and
their gradients (≈ 120 MiB at 1 430 tokens), the gradient of `down_proj`'s input, the two 22 MiB
decoder weight slots, two 12 MiB head slices, the rerun layer's attention and dropout tensors,
20 MiB of NF4 quantization state and 25 MiB of LoRA weights with their AdamW state. The tied table
and every layer's saved input live in host memory. nvidia-smi adds the CUDA context and the
allocator's cache: 505 MiB in total.

## Limitations

- **Only this setup is faster.** Tuned for Qwen2 with SDPA attention on a T4 with fp16 autocast,
  NF4 with double quantization, right padding. Anything else (bf16 GPUs, other models or
  quantization) takes the original code: correct, not faster. New sizes only compile again.
- **Weight and table streaming depend on PCIe.** The decoder weights and the vocabulary table live
  in pinned host memory and cross PCIe every step, hidden behind compute on the T4 (12.4 GB/s). On a
  slower bus, or for very short batches whose compute is shorter than a layer's copy, the step
  waits for its weights.
- **We chose memory over speed.** Every decoder layer reruns its forward in backward (budget 0):
  the final run is 1.34–1.39× faster than the base, while our fastest configurations reached
  1.78× (49.3 s at 3.5 GB) and 48.6 s at 4.1 GB.
- **Kaggle time varies by ±10%** between machines and sessions (the same base: 79.8–92.5 s), so the
  speed-up depends on the machine the jury gets; single changes were measured A/B on one machine.
- The step is bound by cuBLAS matmuls and memory passes of small kernels; see [What next](#what-next).
- On other data the speed-up depends on how much padding the batches have (43% here).
- Removed after an organizers' ruling: tokenizing the SFT rows in our own loop (same tokens,
  −1.8 s) is not allowed, because the dataset fingerprint then differs from Soup's `map`.

**Tried and not taken** (numbers on a T4):

| Idea | Result | Why not |
|---|---|---|
| Recomputation budget 512 instead of 256 | −3.2 s, +484 MiB | ≈ −0.5 points by the score formula |
| Recomputation off | 48.6 s, 4 149 MiB | memory ×3.9 for −6 s |
| Weight / table streaming off | −1.7 s, +1 450 MiB | memory cost far larger |
| int8 tensor-core matmuls for backward | ≤ 1 s per run at best | too little for the accuracy change |
| int8 forward | first-step loss off by > 1% on 10 of 64 batches | breaks check 6 |
| Gated MLP in one kernel | 1 s slower than cuBLAS + SwiGLU | slower |
| Head over the vocabulary in two walks | race between the copy stream and the second walk (fixed by an event; replaced by one walk) | — |
| Varlen attention over the packed batch | 10/10, 55.5 s vs 54.8 s for per-row SDPA | no gain |
| MLP of a rerun layer backpropagated in pieces of 512 tokens | 10/10 but 675 MiB (the peak is elsewhere) and 1.25× | slower, no gain |
| Activations kept in host memory instead of recomputation | estimated −2…4 s | not built: needs async copies both ways; ran out of time |
| Recomputation budget 256 instead of 0 (final code) | −0.9 s, +282 MiB | memory |
| CUDA graphs for the decoder step | 0% at ~1 000 tokens per batch | no gain |
| fp16 copies of the LoRA weights cached per step (fewer casts) | ≈ 0.2 s per run (8–12 weight casts per layer-step, 0.15–0.20% of device time) | too small |
| Loading the model in a background thread | ≤ 1 s | within Kaggle's noise |
| Fused q/k/v projection; `.item()` syncs deferred after backward; backward prefetch hints | no gain in A/B on one machine | — |

## What next

Not built; ordered by the gain we expect from our own measurements on the T4.

| Idea | Expected | Basis and risk |
|---|---|---|
| Fused NF4 unpack + matmul on the T4's tensor cores (inline-PTX `mma`, QA 21) | −2…3 s | NF4 unpacking is ≈ 9% of the GPU time and every weight is unpacked three times per step (forward, rerun, backward); pays only if the fused matmul keeps cuBLAS speed |
| Saved activations in pinned host memory with asynchronous copies both ways, instead of rerunning layers | −2…4 s, same peak | a rerun layer costs one extra forward; a layer's activations at ~1 400 tokens are tens of MiB each way over PCIe, so only part of them hides behind compute |
| Fewer, larger elementwise kernels: residual add + RMSNorm + casts; the LoRA branch (dropout + A + B) in one or two | −1…3 s | ≈ 16% of the GPU time is small elementwise work; removing ~1 300 launches alone changed nothing measurable, so the gain is in memory passes, not launch count |
| Startup: model weights read in a thread, lazier imports, cuBLAS initialized in the background | −1…2 s | loading in a thread measured ≤ 1 s; 2.4 s of metadata scan happen before `apply()` and cannot move |
| int8 matmuls for the input gradient with a careful scale (QA 59) | ≤ 1 s | measured in a bench; check 6 leaves little room for error |
| A rerun layer's backward on the batch in two halves (QA 64, dropout masks drawn in the original order) | −100…150 MiB | halves the largest working set at the peak; costs launches |
| SwiGLU backward writing its gradients over gate and up (branch `swiglu-inplace`) | 505 → 485 MiB, time unchanged (A/B on one T4, 2 runs each) | built and tested (10/10 not run); −4% is below the score's 5% threshold, so not taken before the deadline |
| Vocabulary table in int8 with a per-row scale (QA 62) | −30…50 MiB of slice buffers | head and loss values change slightly; checks 5–7 decide |

Below ~350 MiB is out of reach for this stack: the CUDA context with cuBLAS and the other
libraries alone takes ~250 MiB.

**What we learned.** A profile shows where the time goes, not what a change will save. Several
ideas that a profile made look promising (CUDA graphs, removing ~1 300 small launches per step,
the MLP backpropagated in pieces) gained nothing in a same-machine A/B; every change in the final
solution was kept only after one. Kaggle machines differ by ±10%, so a number from another
session is not evidence.

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
- Online softmax, Milakov & Gimelshein, [arXiv:1805.02867](https://arxiv.org/abs/1805.02867), and
  FlashAttention's rescaled accumulation: the head over vocabulary slices in one walk.
- ZeRO-Offload, Ren et al., [arXiv:2101.06840](https://arxiv.org/abs/2101.06840): keeping frozen
  weights in host memory and streaming them to the GPU.
- The organizers' Q&A and participants' chat (startup savings, background compilation, fp16
  storage of frozen weights), summarized in [docs/QA_NOTES.md](docs/QA_NOTES.md).

## Team

| | GitHub | Did |
|---|---|---|
| Adilet | adiletpn | architecture, fused head + CE, fp16 MLP and embedding, NF4 and the direct 4-bit linear, RMSNorm, unpadding, recomputation, weight streaming, startup, Kaggle runner |
| Dilya | meeexcv-lang | RoPE kernel, fp16 attention path (`qkv_rope_kernel`), per-row SDPA fix, wide NF4 unpack kernel, one-kernel LoRA dropout, speed/memory ladder, seed checks |
| Ansar | ansargabit99-boop | baseline profile, memory audits of base and solution (peak owners), gate/up recompute, recompute budget A/B, startup probe, submission notebook |
| Akniyet | aqnietqaysarbay-ctrl | presentation, video, submission audit |

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The test files in `tests/` come from Soup v0.77.0. Data:
[AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2);
model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
