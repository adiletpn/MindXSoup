"""silu_mul against PyTorch and the organizers' example, forward and backward, and its in-place
backward inside an MLP (same gradients, no extra [N, 8960] buffers).

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def reference(gate, up, grad):
    """fp32 math on the fp16 inputs, rounded to fp16 like the kernel."""
    g, u, d = gate.float(), up.float(), grad.float()
    s = torch.sigmoid(g)
    out = (g * s * u).half()
    d_gate = (d * u * s * (1 + g * (1 - s))).half()
    d_up = (d * g * s).half()
    return out, d_gate, d_up


@pytest.mark.parametrize("shape", [(1, 8960), (37, 8960), (1, 1430, 8960), (5, 1000), (3, 2049)])
def test_matches_reference(shape):
    from solution.silu_mul import silu_mul

    torch.manual_seed(0)
    gate = (torch.randn(shape, device="cuda") * 3).half().requires_grad_(True)
    up = (torch.randn(shape, device="cuda") * 3).half().requires_grad_(True)
    grad = torch.randn(shape, device="cuda").half()
    want_out, want_gate, want_up = reference(gate.detach(), up.detach(), grad)
    # inputs that are not leaves, as in the MLP (their memory becomes the gradients)
    g, u = gate * 1, up * 1
    out = silu_mul(g, u)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, want_out, atol=1e-3, rtol=1e-3)
    out.backward(grad)
    torch.testing.assert_close(gate.grad, want_gate, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(up.grad, want_up, atol=1e-3, rtol=1e-3)


def test_matches_example():
    from solution.silu_mul import silu_mul
    from solution.swiglu import swiglu

    torch.manual_seed(1)
    shape = (1, 777, 8960)
    gate = (torch.randn(shape, device="cuda") * 2).half()
    up = (torch.randn(shape, device="cuda") * 2).half()
    grad = torch.randn(shape, device="cuda").half()
    results = []
    for fn in (swiglu, silu_mul):
        g = gate.clone().requires_grad_(True)
        u = up.clone().requires_grad_(True)
        out = fn(g * 1, u * 1)
        out.backward(grad)
        results.append((out.detach(), g.grad, u.grad))
    (out_a, g_a, u_a), (out_b, g_b, u_b) = results
    assert torch.equal(out_a, out_b)  # forward: the same fp32 products, bit for bit
    # backward: the same derivative written in another order, at most one fp16 ulp apart
    torch.testing.assert_close(g_b, g_a, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(u_b, u_a, atol=1e-3, rtol=1e-3)
    assert (g_b != g_a).float().mean() < 0.01


def test_backward_writes_over_gate_and_up():
    """The MLP's gradient flows through the projections as before, and backward allocates no
    [N, 8960] buffer of its own."""
    from solution.silu_mul import silu_mul

    torch.manual_seed(2)
    n, hidden, inter = 1430, 1536, 8960
    w_gate = (torch.randn(inter, hidden, device="cuda") / hidden ** 0.5).half()
    w_up = (torch.randn(inter, hidden, device="cuda") / hidden ** 0.5).half()
    w_down = (torch.randn(hidden, inter, device="cuda") / inter ** 0.5).half()
    x = torch.randn(n, hidden, device="cuda").half()
    grad = torch.randn(n, hidden, device="cuda").half()

    def mlp(act, x):
        x = x.detach().requires_grad_(True)
        h = act(torch.nn.functional.linear(x, w_gate), torch.nn.functional.linear(x, w_up))
        out = torch.nn.functional.linear(h, w_down)
        return x, out

    def torch_act(g, u):
        return (torch.nn.functional.silu(g.float()) * u.float()).half()

    x_ref, out_ref = mlp(torch_act, x)
    out_ref.backward(grad)
    x_new, out_new = mlp(silu_mul, x)
    torch.testing.assert_close(out_new, out_ref, atol=2e-2, rtol=2e-2)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out_new.backward(grad)
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - before
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=5e-2, rtol=5e-2)
    one = n * inter * 2
    # d(h) from down_proj is the only new [N, 8960] tensor; d(gate) and d(up) reuse gate and up
    assert extra < 1.5 * one, extra / one
