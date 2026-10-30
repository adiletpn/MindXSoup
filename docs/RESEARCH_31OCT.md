# Research for 31.10 — where the remaining points are (written 31.10 ~03:30)

Read this first on resume. Numbers are Kaggle T4, official harness, `--repeats 2` unless noted.

## 0. Where we are

| Variant | time | peak | s | m | checks |
|---|---|---|---|---|---|
| main (`ac8eca8`+startup) budget 1024 | 49.3 s (base 87.9) | 3 561 MiB | 1.78× | 4.03× | 10/10 |
| budget 0 (every layer recomputed) | 55.3 s (base 83.5) | **1 739 MiB** | 1.51× | **8.25×** | 10/10 |
| + direct NF4 linear (`linear4bit.py`) | running (`mindx-b0-direct`) | | | | |

Competition (chat, unverified): many at ~2 GB / 47–50 s; one dashboard **v3 27 s / 856 MiB / eval −0.90%**,
**v4 19 s / 613 MiB / eval −1.35%** (base 85.0 s / 14 343 MiB — same harness). Their eval-loss shift
says their training differs from the base (low-precision matmuls? QA 50 now says int8 matmuls of
frozen weights are allowed if all checks pass).

## 1. Score model

`35·ln s / ln s_best + 30·ln m / ln m_best`. With the dashboard as best (s 4.47, m 23.4):
1 unit of ln s = 23.4 points, 1 unit of ln m = 9.5 points. Our levers:

| lever | Δ ln m | Δ ln s | Δ points |
|---|---|---|---|
| layers' NF4 weights streamed from host (−0.6 GB of 1.74) | +0.43 | −0.03? | **+3.4** |
| + tied embedding/head streamed (−0.45 GB more) | +0.52 | −0.04? | **+4.0** |
| int8 matmuls (−4…5 s of ~52) | 0 | +0.09 | +2.1 |
| GPU-idle removal (fusions, −5 s) | 0 | +0.10 | +2.3 |

If the dashboard is not admitted and the best is ≈ 2 GB / 37 s, our ~0.75 GB would be the best
memory: all 30 memory points. Memory first, speed second; tie-break is memory too.

## 2. Memory floor and what the 1 739 MiB contains (budget 0)

nvidia-smi peak = CUDA context + cuBLAS workspace (~0.3–0.45 GB) + NF4 layers 625 MiB (+ absmax
≈ 40 MiB) + tied embedding fp16 445 MiB + LoRA/Adam (~30 MiB) + recompute boundaries
(28 × N × 1536 × 4 B = 344 MiB at N = 2048) + one layer's live activations (~150 MiB at N = 2048)
+ head chunk 155 MiB + allocator slack. torch max_allocated was 1 551 MiB.

Floor with everything streamed: context 0.35 + boundaries 0.34 + one layer 0.15 + head chunk
0.08 (CHUNK 256) + two layer weight buffers 0.05 ≈ **0.95 GB at N = 2048**; the boundaries can go
to fp16 (organizers allow compact storage; the residual stream stays fp32 in compute) → ≈ 0.78 GB.

**Load-time trap.** nvidia-smi peak is over the whole process, including model loading: at the end
of `from_pretrained` all NF4 weights (0.63 GB) and the fp16 embedding (0.45 GB) are on the GPU
(+ context) ≈ 1.4 GB. Streaming only during training cannot go below that. To get under it the
weights must leave the GPU **as they are quantized** (patch `Params4bit` quantization / the loader
to move packed bytes to pinned host memory right away) and the embedding must go to host when it is
materialized. Check the bf16→fp16 conversion of the embedding (core_model_loading
`_materialize_copy`: `.to(device, dtype)` from CPU converts on the CPU — 5.9 s of py-spy samples,
≈2–3 s real) — converting on the GPU would be faster but adds a 0.9 GB transient: do it only for
tensors that stay on the GPU, keep the embedding on the host.

## 3. Layer weight streaming — design (Adilet)

- Keep each Linear4bit's packed bytes (`weight.data`, uint8 [(N·K)/2, 1]) and `quant_state.absmax`
  (uint8) in **pinned host memory**; `state2.absmax`, `code`, `offset` are tiny — stay on GPU.
- Placeholder on the GPU so peft/Soup still see a CUDA uint8 weight of the right shape:
  `torch.empty(1, dtype=uint8, device=cuda).expand(n, 1)` (1 byte of storage).
- A ring of 3 per-layer GPU buffers (≈ 24 MiB each). Prefetch layer i+1 on a side stream with
  `non_blocking` H2D + event while layer i computes; before layer i+1 runs, the compute stream
  waits for the event and the 7 modules' `weight.data` point into the buffer.
- Order with budget 0: forward 0..27 (prefetch i+1), then backward 27..0, where each layer first
  re-runs its forward (checkpoint) then its backward → prefetch i−1 when layer i's recompute starts
  (hook in `recompute.py`'s `run`). Per step 2 × 0.63 GB over PCIe.
- Our direct path (`linear4bit.py`) reads `self.weight.data` at call time and saves it on `ctx`
  → it must see the real buffer; any other reader (bnb fused kernel for few tokens, MatMul4Bit
  backward) too. Swap `.data` in a decoder-layer pre-forward hook; never reuse a buffer while a
  saved view of it may still be read (ring of 3 is enough with full recompute).
- Feasibility depends on §6 bench: PCIe GB/s and copy/compute overlap; one layer's GEMMs at
  N ≈ 830 vs 24 MiB / bandwidth.

## 4. Tied embedding / head streaming — design (Dilya, task 0 in her round-3 prompt)

One pass over the vocabulary with online softmax (FlashAttention-style accumulator for dH), pinned
host weight, double-buffered vocab chunks, input lookup by CPU gather of the batch rows.

## 5. Speed levers

1. **GPU idle ≈ 33 % of a step** (budget 1024 profile: GPU busy 307 ms of a 456 ms mean step;
   4 558 kernels per step, 1 892 shorter than 20 µs). Python/launch overhead on Kaggle's slow CPU.
   - direct NF4 linear (done, bit-identical, in the run above);
   - LoRA dropout path = 9 kernels per call × 56 calls (ones fill, fused_dropout, ne, gather,
     masked_fill + mul, bitwise_not, masked_fill in backward) → one Triton kernel each way after
     the full-shape `F.dropout` (same random numbers): ≈ −280 launches/step;
   - residual add fused into RMSNorm (Dilya task 2);
   - per-row SDPA loop (4 rows × 28 layers × fwd/recompute/bwd) — consider nested-jagged SDPA.
2. **int8 matmuls** (QA 50 allows): needs `torch._int_mm` speed on sm75 (§6). Weight int8 per
   output row from the NF4 unpack (write int8 instead of fp16: half the bytes), activations
   per-token dynamic int8, int32 accumulate, rescale in the epilogue (our kernel). Backward in fp16.
   Watch check 6 (first-step loss ≤ 1 %) and eval loss on two seeds. Only after §3.
3. **Startup**: done — one metadata scan, no dill hashing, no unused sklearn/wandb, no bnb CPU
   compile wrappers (apply 4.9 → 3.7 s). Left: model loading conversion (§2).

## 6. Bench results (Kaggle T4, `mindx-research`, 31.10 03:40)

| what | result |
|---|---|
| H2D pinned (24 / 96 / 445 MiB) | **12.3–12.4 GB/s** (2.0 / 8.1 / 37.7 ms) |
| H2D pageable | 6.7–6.9 GB/s → always pin |
| copy on a side stream during GEMMs | GEMMs 9.72 ms + copy 8.14 ms → **both 9.41 ms: the copy is fully hidden** |
| one decoder layer's 7 forward GEMMs | **3.86 ms at N = 830**, 8.80 ms at N = 2048 |
| one layer's NF4 bytes over PCIe (≈ 24 MiB) | ≈ 2.0 ms → hidden behind its own forward GEMMs (and ≈ 3× more compute in recompute + backward) |
| head weight over PCIe (445 MiB) | 37.7 ms per pass → one pass per step ≈ +2.4 s if not overlapped; overlap with the last layers' backward |
| fp16 GEMM, our shapes | 15–38 TFLOPS |
| int8 `torch._int_mm(x8, w8.t())` (B column-major) | ×2.38 (1536²), ×1.28 (N 256), ×1.39 (N 8960), ×1.41 (K 8960), ×1.07–1.53 at N = 2048 |
| int8 with B row-major | slower than fp16 (×0.35–1.12): keep weights as `[N, K]` and pass `.t()` |

Conclusions: layer streaming is nearly free (copy < compute, overlap works) → do it. int8 gives
≈ ×1.4 on the big GEMMs before quantization overhead → maybe −10…15 % of matmul time (≈ −2…3 s per
run) at a precision risk → after memory work, if time remains.

## 7. Order of work 31.10 and safety

1. Merge the direct NF4 path + budget 0 if `mindx-b0-direct` is 10/10.
2. Layer streaming (training only) → measure; then load-time offload → measure.
3. Dilya: head/embedding streaming. Ansar: notebook, profile, small-op list.
4. int8 only if 2–3 are in by 14:00.
5. **17:00 freeze.** 18:00 final `--repeats 2` (+ seed 7), notebook rerun, README numbers,
   SUBMISSION.md, video. Nothing merges without its own 10/10 run.
