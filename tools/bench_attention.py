"""Profile one Qwen2DecoderLayer fwd+bwd with the original attention and with the fp16 path.

    python tools/bench_attention.py

The layer is built as in training (NF4 projections, LoRA on q/v, fp32 input, fp16 autocast,
batch 4, T=512 with right padding) with solution.apply() in place, so RMSNorm, MLP, NF4 and RoPE
run our kernels in both variants; only Qwen2Attention.forward differs. Prints time per fwd+bwd
(CUDA events), activation memory kept by the forward, and the torch.profiler table of each variant.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

BATCH, N_TOKENS, LAYERS, WARMUP, ITERS = 4, 512, 28, 5, 20


def build_layer():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    from solution.test_attention import qwen2_config, quantize_attention, to_nf4

    torch.manual_seed(0)
    layer = Qwen2DecoderLayer(qwen2_config(), layer_idx=0)
    quantize_attention(layer.self_attn)
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, to_nf4(getattr(layer.mlp, name)))
    return layer.cuda().train()  # norms stay fp32, as after prepare_model_for_kbit_training


def make_inputs():
    from solution.test_attention import padding_mask
    from solution.test_rope import model_like_cos_sin

    hidden = torch.randn(BATCH, N_TOKENS, 1536, device="cuda", requires_grad=True)
    cos, sin = model_like_cos_sin(1, N_TOKENS, torch.float32)
    mask = padding_mask(BATCH, N_TOKENS, [N_TOKENS - 60 * i for i in range(BATCH)])
    grad = torch.randn(BATCH, N_TOKENS, 1536, device="cuda")
    return hidden, (cos, sin), mask, grad


def step(layer, hidden, pos, mask, grad):
    with torch.autocast("cuda", dtype=torch.float16):
        out = layer(hidden, attention_mask=mask, position_embeddings=pos)
    out = out[0] if isinstance(out, tuple) else out
    out.backward(grad)
    return out


def measure(layer, inputs, label):
    hidden, pos, mask, grad = inputs
    for _ in range(WARMUP):
        step(layer, hidden, pos, mask, grad)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        step(layer, hidden, pos, mask, grad)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / ITERS

    # activation memory: what the forward keeps for backward, and the peak of one fwd+bwd
    hidden.grad = None
    for p in layer.parameters():
        p.grad = None
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        out = layer(hidden, attention_mask=mask, position_embeddings=pos)
    out = out[0] if isinstance(out, tuple) else out
    kept = torch.cuda.memory_allocated() - base
    out.backward(grad)
    peak = torch.cuda.max_memory_allocated() - base
    del out

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            step(layer, hidden, pos, mask, grad)
        torch.cuda.synchronize()
    averages = prof.key_averages()
    sort_key = "self_device_time_total" if hasattr(averages[0], "self_device_time_total") \
        else "self_cuda_time_total"
    print(f"\n===== {label}: {ms:.2f} ms per layer fwd+bwd, forward keeps {kept / 2**20:.0f} MiB, "
          f"fwd+bwd peak +{peak / 2**20:.0f} MiB =====")
    print(averages.table(sort_by=sort_key, row_limit=25, max_name_column_width=70))
    return ms, kept, peak


def main():
    import solution
    from transformers.models.qwen2 import modeling_qwen2

    from solution import attention, warmup

    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    solution.apply()
    warmup.wait()
    cls = modeling_qwen2.Qwen2Attention
    attention.patch_qwen2_attention_fp16()  # no-op if apply() already did it
    patched = cls.forward
    original = patched._mindx_original
    layer = build_layer()
    inputs = make_inputs()

    results = {}
    for label, forward in (("original attention", original), ("fp16 attention", patched)):
        cls.forward = forward
        torch.manual_seed(0)
        results[label] = measure(layer, inputs, label)

    (ms0, kept0, peak0), (ms1, kept1, peak1) = results.values()
    print("\n== Summary, one decoder layer, B=4 T=512 ==")
    print(f"{'':20} {'fwd+bwd, ms':>12} {'kept by fwd, MiB':>17} {'peak, MiB':>10}")
    for label, (ms, kept, peak) in results.items():
        print(f"{label:20} {ms:12.2f} {kept / 2**20:17.0f} {peak / 2**20:10.0f}")
    print(f"saved per layer: {ms0 - ms1:.2f} ms ({100 * (ms0 - ms1) / ms0:.1f}%), "
          f"per step of {LAYERS} layers: {LAYERS * (ms0 - ms1):.0f} ms; activations kept "
          f"{(kept0 - kept1) / 2**20:.0f} MiB less per layer, {LAYERS * (kept0 - kept1) / 2**20:.0f} "
          f"MiB per model")


if __name__ == "__main__":
    main()
