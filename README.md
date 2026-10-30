# MindX: Soup Kernel Challenge

Team MindX's solution for the Soup Kernel Challenge (Decentrathon 6.0). It makes the same
`soup train` run (Qwen2.5-1.5B-Instruct, 4-bit NF4 + LoRA, 64 steps on a Kaggle T4) faster and
lighter, and keeps the outputs: the same adapter quality, metrics and `rewind.jsonl`. All of it
lives in `solution/` and is switched on by `solution.apply()`.

| Kaggle T4, `harness/run_task.py --repeats 2` | Base Soup | MindX | Ratio |
|---|---|---|---|
| Time of `soup train` | 83.5 s | **58.4 s** | **1.43× faster** |
| Peak GPU memory (nvidia-smi, whole run) | 14 345 MiB | **1 209 MiB** | **11.87× less** |
| Eval loss on the public rows | 2.04966 | 2.04966 | **+0.000%** |
| Official checks (`harness/check_rules.py`) | | **10 / 10 PASS** | |

Measured run of the `main` commit (31.10, 05:30). Our own Triton kernels take 11.6% of the training
time (check 4 needs 5%). The training is the base's to the last digit of the eval loss, on every
seed we ran (1, 7, 1234). We traded some speed for memory on purpose: the same code with the
recomputation budget at 1 024 tokens runs in 49 s at 3.56 GB (1.78×, 4.03×), see
[Results](#results).

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
| 1 | **Fused head + cross-entropy, no logits.** The head matmul runs in 256-row chunks into one fp16 buffer; our kernel takes each row once and produces the loss, its gradient (written in place), the entropy, the argmax and the per-token loss. TRL's `entropy`, `num_tokens`, `mean_token_accuracy` and Soup's per-row loss for `rewind.jsonl` (through Soup's `ChunkedStep` record) come from these values. | `fused_ce.py` | −3 to −4 GB |
| 2 | **fp16 storage of fp16 values.** `prepare_model_for_kbit_training` keeps the residual stream in fp32, so the base stores fp32 copies of fp16 matmul outputs. The MLP runs on one fp16 input cast and keeps `gate`/`up` in fp16; attention casts its input once and one kernel adds the LoRA output, applies RoPE and writes fp16 q, k, v; the frozen tied embedding / `lm_head` stays in fp16 (it is never upcast). Same values, fewer casts. | `mlp.py`, `attention.py`, `embedding.py` | −3 GB, −10% step |
| 3 | **Pad tokens never enter the model.** The real tokens of the 4 rows are packed into one sequence (position ids restart per row); attention runs per row, causal, on k/v repeated to the 12 query heads (the fp16 fused SDPA kernel, as in the base); targets are shifted inside each row before packing; per-token values are scattered back to `[B, T]` for TRL and Soup. LoRA dropout draws its mask on the full padded shape, exactly as the base, and keeps the real rows, so the random masks are the same. | `unpad.py` | −40% step, −1 GB |
| 4 | **Recomputation.** Each decoder layer keeps only its input for backward and reruns its forward there (`torch.utils.checkpoint` in our patch; Soup's `gradient_checkpointing` stays off). A token budget sets how many layers a batch reruns (⌈28 · (1 − budget / N)⌉ of N real tokens); we use 0, every layer, which scored best (1 739 vs 3 561 MiB for +6 s). | `recompute.py` | −1.8 GB, +6 s |
| 5 | **Decoder weights live in pinned host memory.** Right after bitsandbytes quantizes a decoder weight while transformers loads the model, its packed NF4 bytes move to pinned host memory; the GPU keeps a 1-byte placeholder of the same shape. A ring of 3 GPU slots holds the layers in use: while layer i runs, a side CUDA stream copies layer i ± 2 (12.4 GB/s on the T4; a layer's 24 MiB take 2 ms, hidden behind its 4 ms of matmuls). Every 4-bit matmul asks for its bytes right before use, so correctness never depends on the prefetch. The loader's `caching_allocator_warmup` (a whole-model GPU reservation that only speeds up loading) is skipped. | `stream.py` | −0.53 GB, ~0 s |
| 6 | **NF4 unpacking kernel and a direct 4-bit linear.** bitsandbytes dequantizes the double-quantized NF4 weights in two kernels (8-bit absmax, then the 4-bit values); ours does it in one pass. For fp16 inputs our `Linear4bit` path runs exactly bitsandbytes' own ops (unpack, `F.linear`; backward unpack, matmul) without its Python dispatch layers, bit-identical. | `nf4.py`, `linear4bit.py` | 8% of the step in our kernel |
| 7 | **RMSNorm forward/backward kernel.** One launch each way instead of 5–6 PyTorch ops; keeps the input and `rstd`. | `rmsnorm.py` | fewer launches |
| 8 | **Startup.** Triton compiles our kernels in a background thread while Soup loads the model (first step 4.6 → 0.8 s); `importlib.metadata.packages_distributions()` (2.3 s) is computed once instead of twice; `datasets` gets a random fingerprint for `map`/`filter` on in-memory datasets instead of hashing the tokenizer with dill; the unused `sklearn` and `wandb` imports are skipped (≈3 s); `TORCHDYNAMO_DISABLE` is set only while bitsandbytes is imported, so it does not build `torch.compile` wrappers for its CPU backend (1.2 s); checkpoint tensors are converted bf16 → fp16 on the GPU instead of the CPU (same values). All allowed by the organizers in the participants' chat (30–31 Oct), all inside the measured run. | `warmup.py`, `startup.py` | ≈ −9 s |

### Our kernels

Share of the training time from the harness (CUDA events on our launches, steps 3–64, measured
run of `main`, 31.10 05:30, lower of the two runs):

| Kernel (`@triton.jit`) | File | Replaces | Share |
|---|---|---|---|
| `nf4_unpack_kernel` | `solution/nf4.py` | bitsandbytes `dequantize_blockwise` + `dequantize_4bit` | 8.2–8.8% |
| `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | `Qwen2RMSNorm` (pow, mean, rsqrt, mul, cast; autograd of each) | 1.8% |
| `qkv_rope_kernel` | `solution/attention.py` | LoRA add, RoPE and three fp16 casts of q/k/v, forward and backward | 1.0% |
| `head_ce_rows_kernel` | `solution/fused_ce.py` | log-softmax, NLL, its gradient, TRL's entropy and argmax over 151 936 logits | 0.6% |
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
- Tests (`python -m pytest solution -q`, 97 on a T4) compare every kernel and patch with PyTorch,
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
| Startup: one `packages_distributions()` scan, random fingerprint for in-memory `map`/`filter`, unused `sklearn`/`wandb` imports skipped, `TORCHDYNAMO_DISABLE` only while bitsandbytes is imported | QA 37, 45, 49; nothing stubbed in `sys.modules`, nothing reused between runs |
| Decoder weights in pinned host memory, streamed per layer; loader warm-up reservation skipped | Rules, "Allowed: … offloading in your own code (its time counts)"; QA 47 (CPU offloading of frozen weights, copies and syncs in the measured time) |
| bf16 → fp16 conversion of checkpoint tensors on the GPU at load | Same values (tested); QA 37 (startup work may be optimized inside the run) |
| LoRA dropout | Same rate, same random masks as the base (drawn on the padded shape); QA 4 |

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
| **+ offload at load, GPU dtype conversion, head chunk 256 (`main`)** | 83.5 → 58.4 | **1 209** | **2.04966 / 2.04966** | **1.43×** | **11.87×** | **10/10** |

Same-machine A/B: the per-row SDPA fix alone gives −6.3 s and −1 000 MiB; the fp16 attention path
−7% of the step time. Eval loss stays within ±0.005% of the base in every run.

Where the time goes now: cuBLAS matmuls are the largest part of the GPU time, our kernels 11.6%,
the rest attention, casts and small ops; the GPU still waits for Python on Kaggle's slow CPU in
part of each step (thousands of small launches). Where the memory goes now: CUDA context ≈ 0.2 GB,
the fp16 tied embedding / `lm_head` 0.45 GB, recompute boundaries (the layer inputs, fp32) up to
0.3 GB, one layer's working set, three 24 MiB weight slots and a 78 MB head chunk.

## Limitations

- Tuned for this task: Qwen2 with SDPA attention, a T4 with fp16 autocast, NF4 with double
  quantization, right padding. Anything else takes the original code (correct, not faster); new
  sizes only compile again.
- The step is now bound by cuBLAS matmuls; further speed needs fewer FLOPs, not faster
  elementwise code.
- We rerun every decoder layer in backward: one extra forward per step (about +6 s). A budget of
  1 024 tokens gives 49 s at 3.56 GB instead; we chose memory because the score rewards the larger
  relative gain.
- Weight streaming relies on PCIe overlapping with compute; very short batches (fewer tokens than
  a layer's copy time covers) can wait for their weights.
- The tied embedding / `lm_head` (445 MiB) is still on the GPU; it is now about a third of our
  peak.
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
| Adilet | adiletpn | architecture, fused head + CE, fp16 MLP and embedding, NF4 and the direct 4-bit linear, RMSNorm, unpadding, recomputation, weight streaming, startup, Kaggle runner |
| Dilya | meeexcv-lang | RoPE kernel, fp16 attention path (`qkv_rope_kernel`), per-row SDPA fix, seed checks |
| Ansar | ansargabit99-boop | baseline profile, memory audits of base and solution, startup probe, submission notebook |
| | aqnietqaysarbay-ctrl | presentation, video |

## License

Apache-2.0 (`LICENSE`, `NOTICE`). The test files in `tests/` come from Soup v0.77.0. Data:
[AmanMussa/kazakh-instruction-v2](https://huggingface.co/datasets/AmanMussa/kazakh-instruction-v2);
model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
