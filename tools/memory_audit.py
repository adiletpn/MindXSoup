"""Audit one unpatched synthetic Qwen NF4/LoRA forward/backward on Kaggle T4.

This is a diagnostic, not the official Soup benchmark: it omits trainer metrics,
rewind, and optimizer state. Only metadata is retained by the recording hooks.
"""

import argparse
from collections import defaultdict
import importlib
import importlib.metadata
import json
from pathlib import Path
import re
import time


def grouped(rows, key, size="bytes"):
    totals = defaultdict(int)
    for row in rows:
        totals[str(row[key])] += row[size]
    return dict(sorted(totals.items(), key=lambda item: -item[1]))


def module_group(name, cls):
    if "lora_A" in name or "lora_B" in name:
        return "LoRA A/B"
    if cls == "Dropout":
        return "Dropout"
    if "lm_head" in name:
        return "lm_head"
    if cls == "Embedding":
        return "Embedding"
    if "RMSNorm" in cls:
        return cls
    if ".mlp" in name:
        return "MLP / " + cls
    if ".self_attn" in name:
        return "attention / " + cls
    return cls


def render(data):
    lines = ["# Baseline saved-tensor memory audit", "", f"Status: {data['status']}",
             f"Synthetic batch 4 x {data.get('seq_length', 512)}; unpatched NF4 + LoRA; MiB = bytes / 2^20.",
             "This excludes Soup trainer metrics, rewind and optimizer state; use the harness for official peaks.", ""]
    if data.get("error"):
        lines += [f"Error: {data['error']}", ""]
    for name, value in data.get("memory_mib", {}).items():
        lines.append(f"- {name}: {value:.3f} MiB")
    records = data.get("saved_tensors", [])
    unique = {}
    for row in records:
        unique.setdefault((row["device"], row["data_ptr"]), row)
    rows = list(unique.values())
    storage = {}
    for row in records:
        storage.setdefault((row["device"], row["storage_ptr"]), row["storage_bytes"])
    lines += ["", f"Save events: {len(records)}; unique data pointers: {len(rows)}.",
              f"Unique backing storages: {sum(storage.values()) / 2**20:.3f} MiB.",
              "Pointer totals use first-save attribution and can include frozen weights; they are not extra allocations.",
              "Views may share storage. Storage totals avoid counting those views twice; neither total equals peak VRAM."]
    tables = [("Saved tensors by module group", grouped(rows, "group")),
              ("Saved tensors by dtype", grouped(rows, "dtype")),
              ("Saved tensors by layer", grouped(rows, "layer")),
              ("Parameters by dtype", grouped(data.get("parameters", []), "dtype")),
              ("Parameters by class", grouped(data.get("parameters", []), "class"))]
    for title, values in tables:
        lines += ["", title, "", "| Group | MiB |", "|---|---:|"]
        for key, value in values.items():
            lines.append(f"| {key} | {value / 2**20:.3f} |")
    layer0 = sum(r["bytes"] for r in rows if r["layer"] == "0")
    lines += ["", f"Layer 0: {layer0 / 2**20:.3f} MiB; x {data.get('layers', 28)} = "
              f"{layer0 * data.get('layers', 28) / 2**20:.3f} MiB (extrapolation only).",
              "", "Largest saved tensors", "", "| Module | dtype | Shape | MiB |", "|---|---|---|---:|"]
    for row in sorted(rows, key=lambda r: -r['bytes'])[:20]:
        lines.append(f"| {row['module']} | {row['dtype']} | {row['shape']} | {row['bytes'] / 2**20:.3f} |")
    if "logits" in data:
        lines += ["", "Logits: " + json.dumps(data["logits"])]
    lines += ["", "Startup timings (s; model load can include download on a cold cache):", ""]
    lines += [f"- {name}: {value:.3f}" for name, value in data["startup_s"].items()]
    return "\n".join(lines) + "\n"


def audit(data):
    def timed(name, fn):
        start = time.perf_counter()
        result = fn()
        data["startup_s"][name] = time.perf_counter() - start
        return result

    for module in ("torch", "transformers", "trl", "peft", "bitsandbytes", "soup_cli"):
        timed("import " + module, lambda m=module: importlib.import_module(m))
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; run this audit on Kaggle T4")
    data["gpu"] = torch.cuda.get_device_name(0)
    if "T4" not in data["gpu"]:
        raise RuntimeError("Expected T4, got " + data["gpu"])
    data["versions"] = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "peft", "trl", "bitsandbytes", "soup-cli")}
    torch.manual_seed(1)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    def load():
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
            quantization_config=q, device_map={"": 0}, dtype=torch.float16,
            attn_implementation="sdpa")
        torch.cuda.synchronize()
        return model
    model = timed("load_model", load)
    def prepare():
        prepared = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        prepared = get_peft_model(prepared, LoraConfig(r=16, lora_alpha=32,
            lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        torch.cuda.synchronize()
        return prepared
    model = timed("prepare_and_lora", prepare)
    model.train()
    data["layers"] = model.config.num_hidden_layers
    data["parameters"] = [{"name": n, "dtype": str(p.dtype), "class": type(p).__name__,
                           "bytes": p.numel() * p.element_size()} for n, p in model.named_parameters()]
    memory = data["memory_mib"]
    memory["after_model_setup"] = torch.cuda.memory_allocated() / 2**20
    ids = torch.randint(0, model.config.vocab_size, (4, data['seq_length']), device="cuda")
    labels, mask = ids.clone(), torch.ones_like(ids)
    stack, handles = [], []
    for name, module in model.named_modules():
        def enter(mod, args, n=name):
            stack.append((n or "<model>", type(mod).__name__))
        def leave(mod, args, output):
            stack.pop()
        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(leave, always_call=True))

    def pack(tensor):
        name, cls = stack[-1] if stack else ("<outside module>", "unknown")
        layer = re.search(r"\.layers\.(\d+)(?:\.|$)", name)
        storage = tensor.untyped_storage()
        data["saved_tensors"].append({"module": name, "class": cls,
            "group": module_group(name, cls), "layer": layer.group(1) if layer else "outside layers",
            "dtype": str(tensor.dtype), "shape": list(tensor.shape), "device": str(tensor.device),
            "bytes": tensor.numel() * tensor.element_size(), "data_ptr": tensor.data_ptr(),
            "storage_ptr": storage.data_ptr(), "storage_bytes": storage.nbytes()})
        return tensor

    torch.cuda.synchronize()
    memory["before_forward"] = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            with torch.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=ids, labels=labels, attention_mask=mask, use_cache=False)
            torch.cuda.synchronize()
            memory["after_forward"] = torch.cuda.memory_allocated() / 2**20
            logits = outputs.logits
            data["logits"] = {"dtype": str(logits.dtype), "shape": list(logits.shape),
                              "mib": logits.numel() * logits.element_size() / 2**20}
            data["loss"] = outputs.loss.item()
            outputs.loss.backward()
            torch.cuda.synchronize()
        memory["after_backward"] = torch.cuda.memory_allocated() / 2**20
    finally:
        memory["peak_allocated"] = torch.cuda.max_memory_allocated() / 2**20
        memory["peak_reserved"] = torch.cuda.max_memory_reserved() / 2**20
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/out"))
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Synthetic diagnostic length; shorter runs are not the requested 512-token audit")
    args = parser.parse_args()
    if not 1 < args.seq_length <= 512:
        parser.error("--seq-length must be between 2 and 512")
    args.out.mkdir(parents=True, exist_ok=True)
    data = {"status": "running", "seq_length": args.seq_length,
            "startup_s": {}, "memory_mib": {}, "saved_tensors": []}
    try:
        audit(data)
        data["status"] = "complete"
    except Exception as exc:
        data.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.out / "memory_audit.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        report = render(data)
        (args.out / "memory_audit.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
