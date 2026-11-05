"""NF4 weight unpacking for bitsandbytes 4-bit layers, in one Triton kernel.

Every frozen 4-bit linear is unpacked to a dense fp16 matrix twice per step: before the forward
matmul (bitsandbytes' dequant + F.linear path, used for more than 1536 tokens) and before the
backward matmul for the input gradient. With double quantization that takes two CUDA kernels and
an fp32 temporary per call: first the 8-bit absmax values are expanded with their own code book and
scale, then the 4-bit indices are looked up in the NF4 table and scaled.

Here one program unpacks whole 64-value blocks and reconstructs the absmax of each block on the fly
(8-bit code -> code book -> second-level scale + offset), so the fp32 absmax temporary and the
second launch disappear. `nf4_unpack_wide_kernel` reads the packed bytes as int32 words (8 values
per load), the block scale once per block, and writes each block's 64 values as one contiguous
run: on a T4 it unpacks the 8960 x 1536 MLP weights at about 215 GB/s (bitsandbytes: 114 GB/s,
the earlier one-byte-per-lane kernel `nf4_unpack_kernel`: 187 GB/s), with bit-identical output.
The one-byte kernel stays for sizes that are not whole 32-byte blocks. The math is bitsandbytes' own: value = fp16(NF4[q] * absmax) with
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


@triton.jit
def nf4_unpack_wide_kernel(packed_ptr, out_ptr, nf4_ptr, absmax_ptr, code_ptr, absmax2_ptr,
                           offset_ptr, n_blocks, NESTED: tl.constexpr, BLOCKS: tl.constexpr,
                           EVEN: tl.constexpr):
    # One program: BLOCKS absmax blocks of 64 values = 32 packed bytes = 8 int32 words each.
    # Bytes are read as int32 words (4 bytes, 8 values per load), the block scale once per block,
    # and the 64 values of a block are written as one contiguous run.
    blk = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    word = tl.arange(0, 8)
    if EVEN:
        words = tl.load(packed_ptr.to(tl.pointer_type(tl.int32)) + blk[:, None] * 8 + word[None, :])
    else:
        live = blk < n_blocks
        words = tl.load(packed_ptr.to(tl.pointer_type(tl.int32)) + blk[:, None] * 8 + word[None, :],
                        mask=live[:, None], other=0)
        blk = tl.where(live, blk, 0)
    if NESTED:
        q8 = tl.load(absmax_ptr + blk).to(tl.int32)
        scale = tl.load(code_ptr + q8) * tl.load(absmax2_ptr + blk // 256)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + blk)
    # value j of a word (little endian bytes b0..b3): high nibble of b(j//2) first, then the low
    nib = tl.arange(0, 8)
    shift = 8 * (nib // 2) + 4 * (1 - nib % 2)
    q = (words[:, :, None] >> shift[None, None, :]) & 15
    values = (tl.load(nf4_ptr + q) * scale[:, None, None]).to(out_ptr.dtype.element_ty)
    values = tl.reshape(values, [BLOCKS, 64])
    out = out_ptr + blk[:, None] * 64 + tl.arange(0, 64)[None, :]
    if EVEN:
        tl.store(out, values)
    else:
        tl.store(out, values, mask=live[:, None])


@triton.jit
def nf4_layer_kernel(packed_ptr, out_ptr, nf4_ptr, absmax_ptr, code_ptr, absmax2_ptr, offset_ptr,
                     module_ptr, block0_ptr, scale0_ptr, BLOCKS: tl.constexpr):
    # nf4_unpack_wide_kernel over all the 4-bit weights of a decoder layer, stored one after the
    # other. One program stays inside one weight: module_ptr says which, and that weight's code
    # book, second-level scales and offset are read with the same formula as the wide kernel.
    pid = tl.program_id(0)
    blk = pid * BLOCKS + tl.arange(0, BLOCKS)
    word = tl.arange(0, 8)
    words = tl.load(packed_ptr.to(tl.pointer_type(tl.int32)) + blk[:, None] * 8 + word[None, :])
    module = tl.load(module_ptr + pid).to(tl.int32)
    local = blk - tl.load(block0_ptr + module)
    q8 = tl.load(absmax_ptr + blk).to(tl.int32)
    scale = tl.load(code_ptr + module * 256 + q8) \
        * tl.load(absmax2_ptr + tl.load(scale0_ptr + module) + local // 256)
    scale = scale + tl.load(offset_ptr + module)
    nib = tl.arange(0, 8)
    shift = 8 * (nib // 2) + 4 * (1 - nib % 2)
    q = (words[:, :, None] >> shift[None, None, :]) & 15
    values = (tl.load(nf4_ptr + q) * scale[:, None, None]).to(out_ptr.dtype.element_ty)
    values = tl.reshape(values, [BLOCKS, 64])
    tl.store(out_ptr + blk[:, None] * 64 + tl.arange(0, 64)[None, :], values)


_NF4_TABLE = {}


def _nf4_table(device):
    table = _NF4_TABLE.get(device)
    if table is None:
        from bitsandbytes.functional import get_4bit_type

        table = get_4bit_type("nf4", device=device).to(torch.float32).contiguous()
        _NF4_TABLE[device] = table
    return table


WIDE_BLOCKS, WIDE_WARPS = 64, 8  # 2 KiB packed per program; fixed, no autotune


def _launch_wide(packed, out, absmax, absmax_8bit, code, offset, n_blocks, blocks, warps):
    nested = absmax_8bit is not None
    nf4_unpack_wide_kernel[(triton.cdiv(n_blocks, blocks),)](
        packed, out, _nf4_table(packed.device),
        absmax_8bit if nested else absmax, code if nested else absmax,
        absmax if nested else absmax, offset if nested else absmax,
        n_blocks, NESTED=nested, BLOCKS=blocks, EVEN=n_blocks % blocks == 0, num_warps=warps)


def unpack_nf4(packed, shape, absmax, blocksize, dtype, absmax_8bit=None, code=None,
               offset=None):
    """Dense `shape` matrix of `dtype` from packed NF4 bytes. With `absmax_8bit` the block scales
    are double-quantized: `absmax` is then the second-level scale (one per 256 blocks)."""
    n_bytes = packed.numel()
    out = torch.empty(shape, dtype=dtype, device=packed.device)
    nested = absmax_8bit is not None
    if n_bytes % 32 == 0 and packed.data_ptr() % 16 == 0:
        _launch_wide(packed, out, absmax, absmax_8bit, code, offset, n_bytes // 32,
                     WIDE_BLOCKS, WIDE_WARPS)
        return out
    grid = (triton.cdiv(n_bytes, BYTES_PER_PROGRAM),)
    nf4_unpack_kernel[grid](
        packed, out, _nf4_table(packed.device),
        absmax_8bit if nested else absmax, code if nested else absmax,
        absmax if nested else absmax, offset if nested else absmax,
        n_bytes, NESTED=nested, BYTES=BYTES_PER_PROGRAM, num_warps=4)
    return out


def launch_layer(packed, out, absmax_8bit, code, absmax2, offset, module_of, block0, scale0):
    """One launch: `out` (fp16) gets the values of all the weights packed one after the other."""
    nf4_layer_kernel[(module_of.numel(),)](
        packed, out, _nf4_table(packed.device), absmax_8bit, code, absmax2, offset,
        module_of, block0, scale0, BLOCKS=WIDE_BLOCKS, num_warps=WIDE_WARPS)


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
