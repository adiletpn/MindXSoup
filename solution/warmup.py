"""Compile the Triton kernels in a background thread while Soup loads the model.

The cache is empty in the measured run, so every kernel pays its compile (about 2.5 s each on
Kaggle) on its first launch, inside the first training steps. `start()` launches each kernel once
on tiny tensors with the dtypes, layouts and specialized sizes of the real training calls, in a
thread of this process, while the main thread imports and loads the 4-bit model; `wait()` joins
it before the first forward. A dummy that does not match a real call costs nothing but time: the
real call then compiles its own variant as before.

The dummies only use zeros and ones, never the random generators, so dropout masks and every
other random draw of the run stay as they were.
"""

import os
import sys
import threading

import torch

# Qwen2.5-1.5B-Instruct: other models just compile on first use.
HIDDEN, Q_HEADS, KV_HEADS, HEAD_DIM, INTERMEDIATE, VOCAB = 1536, 12, 2, 128, 8960, 151936
RMS_EPS = 1e-6

_thread = None
KEPT_MASKS = False  # set by apply(): dropout masks kept per group are in use


def compile_rmsnorm():
    from .rmsnorm import rmsnorm

    x = torch.zeros(2, HIDDEN, device="cuda", requires_grad=True)
    weight = torch.ones(HIDDEN, device="cuda")  # frozen, as after prepare_model_for_kbit_training
    rmsnorm(x, weight, RMS_EPS).backward(torch.zeros(2, HIDDEN, device="cuda"))


def compile_rope():
    from .rope import apply_rope

    tokens = 16
    q_in = torch.zeros(1, tokens, Q_HEADS * HEAD_DIM, device="cuda", requires_grad=True)
    k_in = torch.zeros(1, tokens, KV_HEADS * HEAD_DIM, device="cuda", requires_grad=True)
    # the projections' outputs viewed as [B, heads, T, D], as in Qwen2Attention.forward
    q = q_in.view(1, tokens, Q_HEADS, HEAD_DIM).transpose(1, 2)
    k = k_in.view(1, tokens, KV_HEADS, HEAD_DIM).transpose(1, 2)
    cos = torch.ones(1, tokens, HEAD_DIM, device="cuda")
    sin = torch.zeros(1, tokens, HEAD_DIM, device="cuda")
    q_out, k_out = apply_rope(q, k, cos, sin)
    torch.autograd.backward((q_out, k_out), (torch.zeros_like(q_out), torch.zeros_like(k_out)))


def compile_head_stream():
    from .head_stream import SLICE, head_acc_kernel, launch_slice

    logits = torch.zeros(1, SLICE, dtype=torch.float16, device="cuda")
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    state = [torch.zeros(1, device="cuda") for _ in range(4)]
    state.insert(4, torch.zeros(1, dtype=torch.int32, device="cuda"))
    state += [torch.zeros(1, device="cuda") for _ in range(3)]
    for grad, acc in ((False, True), (False, False), (True, False)):  # one walk; two walks
        launch_slice(logits, target, state, 0, 1.0, grad, acc)
    head_acc_kernel[(1,)](torch.zeros(1, HIDDEN, device="cuda"), torch.ones(1, device="cuda"),
                          torch.zeros(1, HIDDEN, dtype=torch.float16, device="cuda"), HIDDEN,
                          num_warps=8)


def compile_head_ce():
    from .fused_ce import launch_rows

    logits = torch.zeros(1, VOCAB, dtype=torch.float16, device="cuda")
    target = torch.zeros(1, dtype=torch.int32, device="cuda")
    loss, entropy = (torch.empty(1, device="cuda") for _ in range(2))
    argmax = torch.empty(1, dtype=torch.int32, device="cuda")
    launch_rows(logits, target, loss, entropy, argmax, with_grad=True)


def compile_layer_input():
    from .layer_input import pack, restore

    half, scale = pack(torch.zeros(1, 2, HIDDEN, device="cuda"))
    restore(half, scale)


def compile_nf4():
    from .nf4 import BYTES_PER_PROGRAM, unpack_nf4

    n_bytes = 2 * BYTES_PER_PROGRAM
    packed = torch.zeros(n_bytes, dtype=torch.uint8, device="cuda")
    absmax_8bit = torch.zeros(n_bytes // 32, dtype=torch.uint8, device="cuda")
    code = torch.zeros(256, device="cuda")
    absmax2 = torch.ones(1, device="cuda")
    offset = torch.zeros((), device="cuda")
    # compute dtype fp16 (training) with double-quantized absmax
    unpack_nf4(packed, (64, 2 * n_bytes // 64), absmax2, 64, torch.float16,
               absmax_8bit=absmax_8bit, code=code, offset=offset)


def compile_nf4_layer():
    from .nf4 import WIDE_BLOCKS, launch_layer

    n_bytes = WIDE_BLOCKS * 32
    launch_layer(torch.zeros(n_bytes, dtype=torch.uint8, device="cuda"),
                 torch.empty(2 * n_bytes, dtype=torch.float16, device="cuda"),
                 torch.zeros(WIDE_BLOCKS, dtype=torch.uint8, device="cuda"),
                 torch.zeros(256, device="cuda"), torch.ones(1, device="cuda"),
                 torch.zeros(1, device="cuda"),
                 torch.zeros(1, dtype=torch.int8, device="cuda"),
                 torch.zeros(1, dtype=torch.int32, device="cuda"),
                 torch.zeros(1, dtype=torch.int32, device="cuda"))


def compile_lora_dropout():
    from .dropout import KeptDropout, PackedDropout

    tokens = 16
    x = torch.zeros(1, tokens, HIDDEN, device="cuda", requires_grad=True)
    drawn = torch.ones(1, tokens, HIDDEN, device="cuda")
    index = torch.arange(tokens, device="cuda")
    # training: fp32 input, fp16 output for lora_A, fp16 gradient back
    out = PackedDropout.apply(x, drawn, index, 1.0 / 0.95, torch.float16)
    out.backward(torch.zeros(1, tokens, HIDDEN, dtype=torch.float16, device="cuda"))
    if not KEPT_MASKS:
        return
    # groups of rows (recompute.py, microbatch.py): the mask already kept, both ways
    keep = torch.ones(tokens, HIDDEN, dtype=torch.int8, device="cuda")
    out = KeptDropout.apply(x, keep, 1.0 / 0.95, torch.float16)
    out.backward(torch.zeros(1, tokens, HIDDEN, dtype=torch.float16, device="cuda"))


def compile_silu_mul():
    from .silu_mul import silu_mul

    gate = torch.zeros(2, INTERMEDIATE, dtype=torch.float16, device="cuda", requires_grad=True)
    up = torch.zeros(2, INTERMEDIATE, dtype=torch.float16, device="cuda", requires_grad=True)
    silu_mul(gate * 1, up * 1).backward(torch.zeros(2, INTERMEDIATE, dtype=torch.float16,
                                                    device="cuda"))


def compile_example_swiglu():
    from .swiglu import swiglu

    gate = torch.zeros(INTERMEDIATE, dtype=torch.float16, device="cuda", requires_grad=True)
    up = torch.zeros(INTERMEDIATE, dtype=torch.float16, device="cuda", requires_grad=True)
    swiglu(gate, up).backward(torch.zeros(INTERMEDIATE, dtype=torch.float16, device="cuda"))


def _compile_all(jobs):
    try:  # organizers (31.10): lowering this thread's OS priority is allowed; Linux: per thread
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 19)
    except (AttributeError, OSError):
        pass
    for job in jobs:
        try:
            job()
        except Exception as exc:  # noqa: BLE001 - the real call compiles it instead
            print(f"[mindx] warm-up of {job.__name__} skipped: {exc!r}", file=sys.stderr)


def start(jobs):
    """Compile `jobs` (functions above) in a daemon thread; once per process."""
    global _thread
    if _thread is not None or not jobs or not torch.cuda.is_available():
        return
    _thread = threading.Thread(target=_compile_all, args=(jobs,), name="mindx-triton-warmup",
                               daemon=True)
    _thread.start()


def wait():
    """Block until the warm-up is done; free after the first call."""
    thread = _thread
    if thread is not None and thread.is_alive():
        thread.join()
