# ENGINEERING BRIEF — Soup Kernel Challenge (written by Claude, for Claude)

Read this before touching `solution/`. It is the distilled result of reading the task PDF, the
harness, and the sources of soup-cli 0.77.0, trl 0.29.1, peft 0.20.0, transformers 5.16.1 and
bitsandbytes 0.50.2. Facts marked **[verified-src]** were read in source; **[estimate]** are my
numbers; **[verify-on-T4]** must be measured before relying on them.

---

## 0. Mission and definition of done

Make the fixed `soup train` run (Qwen2.5-1.5B-Instruct, NF4 + LoRA r16 on q/v, batch 4, ≤512 tok,
64 steps, Kaggle T4, fp16 AMP) faster and lighter through `solution.apply()` with **our own**
`@triton.jit` kernels, without changing what is trained.

Score = 20 (gate: all 10 checks) + 35·ln s/ln s_best + 30·ln m/ln m_best + 15 README.
Gains < 1.05 count as 1.0. Ties broken by memory.

**Done means:** all 10 checks PASS on a clean Kaggle "Run All" with `--repeats 2`, with margin
(see §4), public notebook pinned to the final commit, README + SUBMISSION.md complete.
Priority order is fixed: **gate > memory > speed > polish.** Never trade the gate for points.

---

## 1. Environment facts

- Kaggle T4 (sm_75, 15 360 MiB, ~320 GB/s, fp16 tensor cores, **no bf16**), one of two GPUs used
  (`CUDA_VISIBLE_DEVICES=0`). Python 3.13, torch 2.11, Triton 3.6.0.
- Pinned: soup-cli 0.77.0, transformers 5.16.1, peft 0.20.0, trl 0.29.1, bitsandbytes 0.50.2,
  accelerate 1.14.0.
- Allocator: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by harness; do not touch).
- **No local GPU** (captain has a Mac). Every GPU fact comes from Kaggle runs. Preferred loop:
  Kaggle API (`kaggle kernels push/status/output`) with the captain's `kaggle.json`.
- Triton on sm_75: `tl.dot` fp16 should lower to mma (Turing m16n8k8) but this is
  **[verify-on-T4]**; avoid depending on `tl.dot` speed until a microbenchmark confirms. Elementwise
  / reduction kernels are safe. Apple CCE requires Ampere — irrelevant (and banned anyway).

Qwen2.5-1.5B config **[from model card, confident]**: hidden 1536, 28 layers, 12 heads, 2 KV heads,
head_dim 128, intermediate 8960, vocab 151 936, tied embeddings, rms_eps 1e-6, rope_theta 1e6.
Tokens per step ≤ 4×512 = 2048 (dynamic padding to the longest row of the batch).

---

## 2. What the base run actually does (dtype trace) [verified-src]

1. Load: 4-bit NF4 double-quant, compute dtype fp16. On pre-Ampere Soup keeps fp16 compute.
2. `peft.prepare_model_for_kbit_training` (gradient checkpointing OFF for this config): **casts every
   non-`Params4bit` fp16/bf16 parameter to fp32** → embeddings, all RMSNorm weights, q/k/v biases
   and the tied `lm_head` are fp32. Embedding fp32 = 151 936×1536×4 B ≈ **0.93 GB**.
3. Therefore `hidden_states` (residual stream) is **fp32** everywhere.
4. `bnb.nn.Linear4bit.forward`: `inp_dtype = x.dtype; x = x.to(fp16); out = matmul_4bit(...)`
   `.to(inp_dtype)` → every 4-bit linear receives fp32, casts to fp16, returns **fp32**.
   (Also mutates the bias to fp16 on first call.)
5. `MatMul4Bit` (bnb 0.50.2) saves **only the packed weight** (no activation) because the base weight
   is frozen. For M=2048 it uses `dequant + F.linear` (custom gemm only for tiny M). Backward
   dequantizes again: `grad_A = grad_out @ dequant(W)`.
6. PEFT LoRA (q_proj, v_proj): `result = base(x).clone(); result += lora_B(lora_A(dropout(x))) * 2.0`
   under autocast. LoRA weights fp32 (trainable), dropout p=0.05 on the fp32 x.
7. `Qwen2RMSNorm.forward`: `h=x.float(); var=h.pow(2).mean(-1); h=h*rsqrt(var+eps);`
   `return weight * h.to(input_dtype)` → fp32 in, fp32 out; autograd keeps several fp32 copies.
8. MLP: `down(act(gate(x)) * up(x))` → gate/up outputs fp32 (from fp16 matmuls), silu + mul in fp32,
   down casts back to fp16. Saved for backward per layer ≈ gate, up, silu(gate), product:
   ~4 × (2048×8960×4 B = 73 MB) ≈ **290 MB/layer → ~8 GB over 28 layers** [estimate]. The example
   SwiGLU saves ~2 GB precisely by fusing these.
9. Attention: SDPA (mem-efficient backend on sm_75), RoPE on fp32 q/k.
10. Head + loss: under autocast `lm_head` casts the fp32 weight to fp16 (autocast weight cache
    ≈ 0.47 GB during forward), logits fp16 `[B,T,151936]`; `ForCausalLMLoss` upcasts to fp32 and
    keeps more full-size buffers at the backward peak. TRL then: `entropy_from_logits(outputs.logits)`
    (chunked), accuracy via `outputs.logits[..., :-1, :].contiguous()` (**another full copy**) +
    argmax, plus several `.item()` syncs per step. Soup's rewind computes per-row CE from the same
    logits. Logits-related peak ≈ **3–4 GB** [estimate].
11. Peak happens at the forward→backward turn: all activations + logits machinery alive.

Base numbers (organizers): wall 79–83 s (64 steps = 56–61 s), peak ~14 350 MiB, eval loss 2.050.
~20 s of the wall is outside the steps: imports, load+quantize, tokenization, trainer init, save.

---

## 3. How the harness measures (exploitable facts, legitimately) [verified-src]

- `_train.py` patches `triton.runtime.jit.JITFunction.run`: counts launches by kernel name; from
  step index 2 on (3rd step), kernels **whose source file is under `solution/`** are timed with
  CUDA events around the launch. Share = Σ own kernel ms / Σ wall of timed steps; **min over runs**.
  ⇒ Launch kernels via normal `kernel[grid](...)` (goes through `JITFunction.run`). Never call a
  cached `CompiledKernel` directly, never use CUDA graphs.
- Kernel names starting `example_` or with AST similarity ≥ 0.95 to the example are excluded;
  ≥ 0.8 is flagged for manual jury review. Inductor names (`triton_poi_` …) excluded.
- Wall time = whole subprocess incl. Python start, imports, `apply()`, model load, Triton JIT
  compilation (fresh `TRITON_CACHE_DIR` per run), saving. **Compile time is on the clock.**
- Memory = max over 100 ms nvidia-smi samples of the sum over all GPUs (includes CUDA context).
  Check 9 requires nvidia-smi peak ≥ 97% of `torch.cuda.max_memory_reserved` (so no big non-torch
  allocations).
- `apply()` runs **before** the model/trainer exist and also before Soup's 79 regression tests
  (`SOUP_APPLY_SOLUTION=1`), some of which use fake transformers/peft modules.
- Eval (`eval_adapter.py`) loads the adapter with plain transformers+peft, **without our code**.
  The adapter must be a standard PEFT LoRA with identical tensor names/shapes/dtypes.

---

## 4. The 10 checks and our safety margins

| # | Check | Limit | Our internal target | Main threat |
|---|---|---|---|---|
| 1 | all runs exit 0 | — | — | crash on unexpected shape/dtype → always fall back to original |
| 2 | no banned libs | list in common.py | never import anything outside torch/triton/transformers/peft/bitsandbytes/soup | — |
| 3 | own kernels launched in every run | — | ≥ 2 independent kernels always on | patch guard silently disabling everything |
| 4 | own kernels ≥ 5% of step time | 5% | **≥ 8%** | making everything fast shrinks our share |
| 5 | same outputs | tokens/step exact, entropy/accuracy/rewind ≤ 3%, metric names equal, adapter config/tensors equal, trainer args equal | ≤ 1% | removing logits without re-providing entropy/accuracy/rewind |
| 6 | loss curve | step 1 ≤ 1%, mean ≤ 3% | step 1 ≤ 0.3% | low-precision accumulation |
| 7 | eval loss | ≤ +0.5% | ≤ +0.1% | any numerical drift |
| 8 | not worse | time/mem ≤ +10% | faster & lighter | recompute too much |
| 9 | clean measurement | T4, idle, 1 GPU, allocator, peaks agree | don't allocate outside torch | — |
| 10 | 79 Soup tests pass with apply() | exactly 79 passed | run every iteration | unguarded patch |

Final verification must mirror the jury: `run_task.py --repeats 2` (B,S,S,B) and at least one extra
run with a different `--seed`.

---

## 5. Hard rules distilled (what the jury will read for)

- Allowed: our Triton kernels, autograd Functions, monkeypatches of transformers/peft/bnb/Soup,
  **recomputation or offloading in our own code** (time counts), removing pad tokens within a batch,
  ideas from others' kernels **with credit**, AI agents.
- Forbidden: banned libs; Soup switches (`gradient_checkpointing`, `activation_offloading`,
  `fast_lora`, `use_liger`, `use_cut_ce`, `packing`, `padding_free`, `stream_loss_chunk` …);
  changing model/data/batches/order/steps/LoRA (incl. dropout)/optimizer/schedule; new trainable
  params; changing Soup outputs; editing harness/tests/task; behaving differently under measurement
  or in tests; precompiled caches; copying kernel code.
- Former grey zones — **all answered YES by the organizers at the Q&A (2026-10-30, see
  docs/QA_NOTES.md)**, provided the checks pass:
  1. Passing per-token losses to rewind through Soup's `ChunkedStep` / `_soup_chunked_step`
     from our code (without enabling `stream_loss_chunk` in the config) — allowed.
  2. Custom autograd Functions + monkeypatching transformers/peft for fusion and activation
     storage — "the case is designed for this".
  3. Storing saved activations compactly (fp16, even int8) and restoring them in backward — allowed.
  4. fp16 copies of frozen fp32 weights, including the frozen tied embedding used by lm_head —
     allowed.
  5. Fused / chunked loss without materializing full logits — allowed, "the main memory reserve";
     metrics must be computed at the same positions as the base.
  6. Removing pad tokens in our own forward, also in MLP and norms — allowed; `num_tokens` is the
     attention_mask sum and stays equal as long as the collator and attention_mask are untouched.
- Q&A also confirmed: share denominator = total time of steps 3+, worst run counts; speed/memory
  use the mean of the two runs per side; jury image = Python 3.13, torch 2.11, CUDA 12.8,
  Triton 3.6.0; only the listed checks are automatic, plus the jury reads the code. Video and
  presentation are not scored but still part of the submission.
- Q&A part 2: Soup internals may be used as long as our code computes the values and no built-in
  mode is switched on; frozen embedding in fp16 OK (~445 MB); **a background compile thread
  inside apply() is explicitly allowed** (e.g. JIT-compile our kernels while the model loads);
  organizers expect strong teams at s 1.2-1.5 and m 1.5-2.

---

## 6. Strategy (ordered) and kernel specs

### Phase 0 — gate (target: today)
Ship small, robust, independent kernels so checks 3/4/10 pass early, keep example SwiGLU on.

**K1 RMSNorm fwd/bwd** (`solution/rmsnorm.py`, names `rmsnorm_fwd_kernel`, `rmsnorm_bwd_kernel`).
- One program per row, `BLOCK = next_pow2(H)` (2048), `num_warps=8`, fp32 accumulation.
- Forward exactly as HF: `y = w * (x * rstd).to(input_dtype)`; output dtype =
  `torch.result_type(w, x)`. Save `x` and per-row `rstd` (fp32) only.
- Backward: `wdy = w*dy; dx = rstd*(wdy - x̂·mean(wdy·x̂))`; compute `dw` only if
  `w.requires_grad` (it is False here — norms are frozen).
- `do_not_specialize` the row count/strides. Guard: CUDA, contiguous last dim, dtype ∈ {fp16,fp32}.
- Expected: speed + memory (−2..3 fp32 copies per norm × 57 norms ≈ −1 GB) and ~2–3% share.

### Phase 1 — memory core
**K2 Fused head + cross-entropy + metrics** (`solution/fused_ce.py`).
- Patch `Qwen2ForCausalLM.forward`: when `labels` is given and training, run the decoder, then our
  op on final hidden states; return output with `loss` and `logits=None`.
- Per chunk of rows (e.g. 256–512 tokens): `logits16 = h16 @ W16ᵀ` (cuBLAS; W16 = one fp16 copy of
  the tied weight, built once), then one Triton kernel per row over the vocab with **online
  softmax** producing: logsumexp, loss at label, entropy `-(p·logp).sum`, argmax, and writing
  `dlogits = (softmax − onehot)·(1/N_label)` in place over the logits buffer; then
  `dh = dlogits @ W16` accumulated into a `[N, H]` fp32 buffer. Gradient computed in forward,
  scaled by `grad_output` in backward (idea: Liger FLCE — credit it; also cite Apple CCE).
- Numerics must mirror base: logits are fp16 values; loss in fp32; entropy in the precision TRL
  uses (fp16 logits in); rewind row loss = CE in logits dtype then `.float()`; accuracy from argmax
  over fp16 logits of positions `[:, :-1]` vs labels `[:, 1:]`. Denominator = number of labelled
  tokens (or `num_items_in_batch` when passed).
- Metrics plumbing: patch `trl.SFTTrainer.compute_loss` (Soup's RewindTrainer calls
  `super().compute_loss`, resolved at call time): when our hook produced a step, call
  `transformers.Trainer.compute_loss`, log `entropy` (masked by attention_mask), `num_tokens`
  (add `attention_mask.sum()` to `_total_train_tokens`), `mean_token_accuracy` exactly like TRL,
  and set `self._soup_chunked_step = ChunkedStep(token_entropy, token_loss, correct, labelled)` so
  rewind reads per-token losses (`row_losses_from_token_losses`).
- Expected: −3..4 GB peak, also removes the 7% softmax time and the extra `.contiguous()` copy.

**K3 MLP with fp16 saved tensors** (`solution/mlp.py`).
- Own autograd over the whole MLP: `x16 = x.half()` once; `g16,u16 = bnb.matmul_4bit(x16, W)`
  (skip the `.to(fp32)` round trip), activation kernel (ours, fp16 in/out, fp32 math), `down` on
  fp16, return `.float()` only at the residual. Save `g16,u16` (2×36 MB) or, behind a flag, only
  `x16` (6 MB) and recompute `g,u` in backward (+2 NF4 matmuls/layer: measure cost first).
- Bit-exactness argument: base also rounds `silu(g)*u` to fp16 before `down`, and bnb backward
  works in fp16. Our kernel must not be structurally similar to the example SwiGLU (AST check):
  fuse something extra (e.g. the fp32→fp16 handling / both directions in one variant).
- Expected: −4..6 GB.

### Phase 2 — speed / separation
- **K4 RoPE fwd/bwd** in fp32 (cos/sin per position; backward = rotation by −θ).
- **Attention block** custom autograd (q/k/v + LoRA + RoPE + SDPA + o_proj) storing fp16.
- **Unpadding**: run all per-token ops on packed `[N_real, H]`; attention per row with SDPA
  `is_causal=True` on its true length (exact; right padding). Potentially −30..45% tokens.
  Must keep `num_tokens` (attention_mask sum) and all metrics identical.
- Embedding in fp16 (grey zone #4) only if measured lossless on metrics.
- Startup seconds: profile the ~20 s outside steps; only legal speedups that keep outputs.

### Explicitly not doing
NF4 fused dequant-matmul (big effort, tl.dot on sm_75 unverified), caching dequantized weights
(costs ~2.6 GB), full-layer checkpointing (kills speed points), autotune.

---

## 7. Compile-time budget
Every distinct (kernel, constexpr, dtype, specialization) costs a JIT compile on Kaggle's slow CPU
(~1–3 s each **[verify-on-T4]**). Rules: ≤ 8 kernels total; one dtype path where possible; fixed
BLOCK/num_warps; `do_not_specialize` every size/stride that varies by batch; no autotune; no
`tl.dot` variants per shape. Measure `apply_s` and first-step time in `metrics.json`.

---

## 8. Engineering rules for every patch
1. Class-level monkeypatch, idempotent (`if getattr(cls, "_mindx_patched", False): return`).
2. Fallback to the original on anything unexpected (not CUDA, wrong dtype, fake module, eval mode
   where needed). Never raise from a patch.
3. One flag per feature in `solution/__init__.py` so any feature can be switched off in one line.
4. Each kernel has `solution/test_<name>.py`: fwd + bwd vs PyTorch on fp16/fp32, odd shapes,
   tolerance fp16 2e-3 / fp32 1e-4.
5. Run order per change: `pytest solution` → `run_task.py --only solution --profile` → full
   `run_task.py` + `check_rules.py` + `score.py`. Log every run in `docs/results.md`.
6. README credits: Liger Kernel FLCE (arXiv 2410.10989), Apple Cut Cross-Entropy
   (arXiv 2411.09009), Unsloth (general idea of patching via apply()), Triton tutorials (layer norm).

---

## 9. Open questions for organizers (ask, don't block on)
1. May our code set Soup's internal `_soup_chunked_step` / use `ChunkedStep` for rewind/metrics?
2. Recompute inside our own autograd backward = allowed "recomputation in your own code"?
3. fp16 storage of saved tensors produced by fp16 matmuls — OK?
4. Unpadding with per-row SDPA — OK?

## 10. Team constraints
Team of 4; captain + Claude own the core. Teammates must commit their own parts (jury checks
contribution): Kaggle runner/profiling, RMSNorm/RoPE kernels + tests, README/presentation/video.
Prompts live in `docs/tasks/`. Deliverables: repo, public Kaggle notebook, ≤3 min video,
presentation. Internal deadline 22:00 on Oct 31; freeze code by 20:00.
