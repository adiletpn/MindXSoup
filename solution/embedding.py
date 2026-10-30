"""Keep the frozen token embedding (tied to lm_head in Qwen2.5-1.5B) in fp16 instead of fp32.

Soup loads the checkpoint in fp16 on a T4 and `prepare_model_for_kbit_training` then upcasts every
non-quantized parameter to fp32, so the 151936 x 1536 embedding takes 890 MiB although each value
is an fp16 number. Stored back in fp16 the values are the same bits: the lookup is cast to fp32 on
the way out, so the residual stream sees exactly what it saw before, and the head matmul already
ran on an fp16 copy of the weight (autocast in the base, our fused head here). Saves 445 MiB of
resident weights and the fp16 copy the head made every step.

The conversion happens on the first training forward, after Soup has prepared the model, and only
when the round trip is exact; otherwise nothing changes.
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


def shrink_embedding(causal_lm):
    """Store the frozen fp32 input embedding (and the head that shares it) in fp16; once."""
    if getattr(causal_lm, DONE_ATTR, False):
        return
    setattr(causal_lm, DONE_ATTR, True)
    get_input = getattr(causal_lm, "get_input_embeddings", None)
    get_output = getattr(causal_lm, "get_output_embeddings", None)
    if get_input is None or get_output is None:
        return
    embedding, head = get_input(), get_output()
    if not isinstance(embedding, torch.nn.Embedding):
        return
    weight = embedding.weight
    if not (weight.is_cuda and weight.dtype == torch.float32 and not weight.requires_grad):
        return
    half = weight.detach().to(torch.float16)
    if not torch.equal(half.float(), weight.detach()):
        return  # not fp16 numbers: keep fp32
    weight.data = half
    embedding.register_forward_hook(_to_fp32)
    if isinstance(head, torch.nn.Linear) and head.weight is weight:
        head.register_forward_pre_hook(_head_input_to_weight_dtype)
        head.register_forward_hook(_head_output_to_fp32)


def patch_fp16_embedding():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls, "_mindx_embedding_patched", False):
        return
    forward = cls.forward

    def patched(self, *args, **kwargs):
        if self.training and isinstance(self, torch.nn.Module):
            shrink_embedding(self)
        return forward(self, *args, **kwargs)

    cls.forward = patched
    cls._mindx_embedding_patched = True
