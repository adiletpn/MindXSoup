"""NF4 weight unpacking for bitsandbytes 4-bit layers, in one Triton kernel.

Every frozen 4-bit linear is unpacked to a dense fp16 matrix twice per step: before the forward
matmul (bitsandbytes' dequant + F.linear path, used for more than 1536 tokens) and before the
backward matmul for the input gradient. With double quantization that takes two CUDA kernels and
an fp32 temporary per call: first the 8-bit absmax values are expanded with their own code book and
scale, then the 4-bit indices are looked up in the NF4 table and scaled.

Here one program unpacks a run of packed bytes and reconstructs the absmax of each 64-value block
on the fly (8-bit code -> code book -> second-level scale + offset), so the fp32 absmax temporary
and the second launch disappear. The math is bitsandbytes' own: value = fp16(NF4[q] * absmax) with
absmax = code[q8] * absmax2 + offset in fp32, and the high nibble of each byte is the first value.

Hooked where bitsandbytes looks the functions up at call time: the CUDA dequant + linear fallback
of `gemm_4bit` (forward) and `bitsandbytes.functional.dequantize_4bit` (MatMul4Bit backward).
Anything else (fp4, other dtypes, shapes or block sizes) takes bitsandbytes' path.
Idea: the NF4 format from QLoRA (arXiv 2305.14314); the code is ours.
"""

import torch
import triton
import triton.language as tl

BYTES_PER_PROGRAM = 1024


@triton.jit
def nf4_unpack_kernel(packed_ptr, out_ptr, nf4_ptr, absmax_ptr, code_ptr, absmax2_ptr, offset_ptr,
                      n_bytes, NESTED: tl.constexpr, BYTES: tl.constexpr):
    start = tl.program_id(0) * BYTES
    byte_idx = start + tl.arange(0, BYTES)
    valid = byte_idx < n_bytes
    packed = tl.load(packed_ptr + byte_idx, mask=valid, other=0).to(tl.int32)
    # two values per byte, 64 values per absmax block -> 32 bytes per block
    block = byte_idx // 32
    if NESTED:
        q8 = tl.load(absmax_ptr + block, mask=valid, other=0).to(tl.int32)
        scale = tl.load(code_ptr + q8, mask=valid, other=0.0) \
            * tl.load(absmax2_ptr + block // 256, mask=valid, other=0.0)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + block, mask=valid, other=0.0)
    high = tl.load(nf4_ptr + (packed >> 4)) * scale
    low = tl.load(nf4_ptr + (packed & 15)) * scale
    values = tl.interleave(high, low).to(out_ptr.dtype.element_ty)
    out_idx = 2 * start + tl.arange(0, 2 * BYTES)
    tl.store(out_ptr + out_idx, values, mask=out_idx < 2 * n_bytes)


_NF4_TABLE = {}


def _nf4_table(device):
    table = _NF4_TABLE.get(device)
    if table is None:
        from bitsandbytes.functional import get_4bit_type

        table = get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
        _NF4_TABLE[device] = table
    return table


def unpack_nf4(packed, shape, absmax, blocksize, dtype, absmax_8bit=None, code=None,
               offset=None):
    """Dense `shape` matrix of `dtype` from packed NF4 bytes. With `absmax_8bit` the block scales
    are double-quantized: `absmax` is then the second-level scale (one per 256 blocks)."""
    n_bytes = packed.numel()
    out = torch.empty(shape, dtype=dtype, device=packed.device)
    nested = absmax_8bit is not None
    grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
    nf4_unpack_kernel[grid](
        packed, out, _nf4_table(packed.device),
        absmax_8bit if nested else absmax, code if nested else absmax,
        absmax if nested else absmax, offset if nested else absmax,
        n_bytes, NESTED=nested, BYTES=BYTES_PER_PROGRAM, num_warps=4)
    return out


def _supported(packed, shape, blocksize, quant_type, dtype, nested_blocksize=256):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return (quant_type == "nf4" and blocksize == 64 and nested_blocksize == 256
            and dtype in (torch.float16, torch.float32) and packed.is_cuda
            and packed.dtype == torch.uint8 and packed.is_contiguous()
            and packed.numel() * 2 == numel)


def patch_bnb_nf4():
    import bitsandbytes.functional as bnb_functional
    from bitsandbytes.backends.cuda import ops as bnb_cuda_ops

    if getattr(bnb_functional, "_mindx_patched", False):
        return
    original_dequantize = bnb_functional.dequantize_4bit
    original_fallback = bnb_cuda_ops._dequant_linear_fallback

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        state = quant_state
        if (state is None or out is not None or A.dim() != 2 or A.shape[0] == 1
                or not _supported(A, state.shape, state.blocksize, state.quant_type,
                                  state.dtype,
                                  state.state2.blocksize if state.nested else 256)):
            return original_dequantize(A, quant_state=quant_state, absmax=absmax, out=out,
                                       blocksize=blocksize, quant_type=quant_type)
        if state.nested:
            return unpack_nf4(A, state.shape, state.state2.absmax, state.blocksize, state.dtype,
                              absmax_8bit=state.absmax, code=state.state2.code,
                              offset=state.offset)
        return unpack_nf4(A, state.shape, state.absmax.float(), state.blocksize, state.dtype)

    def dequant_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                                absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if not _supported(B, shapeB, blocksize, quant_type, A.dtype) or \
                (absmax_8bit is None and absmax.dtype != torch.float32):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                     absmax_8bit, absmax_code, absmax_offset)
        weight = unpack_nf4(B, shapeB, absmax, blocksize, A.dtype, absmax_8bit=absmax_8bit,
                            code=absmax_code, offset=absmax_offset)
        return torch.nn.functional.linear(A, weight, bias)

    bnb_functional.dequantize_4bit = dequantize_4bit
    bnb_cuda_ops._dequant_linear_fallback = dequant_linear_fallback
    bnb_functional._mindx_patched = True
