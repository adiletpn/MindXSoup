"""Keep the frozen token embedding (tied to lm_head in Qwen2.5-1.5B) in fp16 instead of fp32.

Soup loads the checkpoint in fp16 on a T4 and `prepare_model_for_kbit_training` then upcasts every
non-quantized parameter to fp32, so the 151936 x 1536 embedding takes 890 MiB although each value
is an fp16 number. Stored back in fp16 the values are the same bits: the lookup is cast to fp32 on
the way out, so the residual stream sees exactly what it saw before, and the head matmul already
ran on an fp16 copy of the weight (autocast in the base, our fused head here). Saves 445 MiB of
resident weights and the fp16 copy the head made every step.

peft's prepare_model_for_kbit_training is wrapped so the fp16 embedding simply skips its fp32
upcast (no fp32 copy is ever made); a model prepared any other way is converted on its first
training forward, and only when the round trip is exact.
"""

import torch

DONE_ATTR = "_mindx_fp16_embedding"


def _to_fp32(_module, _inputs, output):
    return output.float()


def _head_input_to_weight_dtype(module, inputs):
    # Outside autocast an fp32 hidden state cannot meet the fp16 weight; the base would have run
    # this matmul in fp32, so give back fp32 logits (computed from the same fp16 weight values).
    if torch.is_autocast_enabled() or inputs[0].dtype == module.weight.dtype:
        return None
    return (inputs[0].to(module.weight.dtype), *inputs[1:])


def _head_output_to_fp32(module, inputs, output):
    if torch.is_autocast_enabled() or output.dtype != torch.float16:
        return None
    return output.float()


def _tied_parts(causal_lm):
    """(embedding, head or None) when the input embedding is a plain frozen-able nn.Embedding."""
    get_input = getattr(causal_lm, "get_input_embeddings", None)
    get_output = getattr(causal_lm, "get_output_embeddings", None)
    if not isinstance(causal_lm, torch.nn.Module) or get_input is None or get_output is None:
        return None
    embedding, head = get_input(), get_output()
    if not isinstance(embedding, torch.nn.Embedding):
        return None
    shares = isinstance(head, torch.nn.Linear) and head.weight is embedding.weight
    return embedding, (head if shares else None)


def _install_hooks(causal_lm, embedding, head):
    setattr(causal_lm, DONE_ATTR, True)
    embedding.register_forward_hook(_to_fp32)
    if head is not None:
        head.register_forward_pre_hook(_head_input_to_weight_dtype)
        head.register_forward_hook(_head_output_to_fp32)


def shrink_embedding(causal_lm):
    """Store the frozen fp32 input embedding (and the head that shares it) in fp16; once."""
    if getattr(causal_lm, DONE_ATTR, False):
        return
    setattr(causal_lm, DONE_ATTR, True)
    parts = _tied_parts(causal_lm)
    if parts is None:
        return
    embedding, head = parts
    weight = embedding.weight
    if not (weight.is_cuda and weight.dtype == torch.float32 and not weight.requires_grad):
        return
    half = weight.detach().to(torch.float16)
    if not torch.equal(half.float(), weight.detach()):
        return  # not fp16 numbers: keep fp32
    weight.data = half
    _install_hooks(causal_lm, embedding, head)


def _prepare_keeping_fp16(original):
    """prepare_model_for_kbit_training that leaves an fp16 CUDA embedding in fp16.

    peft upcasts every fp16 parameter to fp32; for the embedding that briefly holds both copies
    (445 + 890 MiB) on the GPU, only for us to cast it back. Hiding the weight from that loop
    gives the same end state as shrink_embedding without the detour."""

    def prepare_model_for_kbit_training(model, *args, **kwargs):
        parts = _tied_parts(model)
        weight = parts[0].weight if parts is not None else None
        if weight is None or not weight.is_cuda or weight.dtype != torch.float16 \
                or getattr(model, DONE_ATTR, False):
            return original(model, *args, **kwargs)
        kept = weight.data
        weight.data = torch.empty(0, dtype=torch.float32, device=kept.device)
        try:
            result = original(model, *args, **kwargs)
        finally:
            weight.data = kept
        _install_hooks(model, *parts)
        return result

    prepare_model_for_kbit_training._mindx_patched = True
    return prepare_model_for_kbit_training


def patch_fp16_embedding():
    import peft
    import peft.utils
    import peft.utils.other
    from transformers.models.qwen2 import modeling_qwen2

    if not getattr(peft.prepare_model_for_kbit_training, "_mindx_patched", False):
        patched_prepare = _prepare_keeping_fp16(peft.prepare_model_for_kbit_training)
        for module in (peft, peft.utils, peft.utils.other):
            if hasattr(module, "prepare_model_for_kbit_training"):
                module.prepare_model_for_kbit_training = patched_prepare

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls, "_mindx_embedding_patched", False):
        return
    forward = cls.forward

    def patched(self, *args, **kwargs):
        if self.training and isinstance(self, torch.nn.Module):
            shrink_embedding(self)  # in case the model was prepared some other way
        return forward(self, *args, **kwargs)

    cls.forward = patched
    cls._mindx_embedding_patched = True
