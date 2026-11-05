"""AdamW with its moments on the host between steps follows the same trajectory.

Run on a GPU: `python -m pytest solution -q`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def trajectory(moments=None):
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(16, 4, device="cuda")) for _ in range(3)]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    if moments is not None:
        optimizer.register_step_pre_hook(moments.before_step)
        optimizer.register_step_post_hook(moments.after_step)
    for _ in range(5):
        for p in params:
            p.grad = p.detach() * 0.5 + 1.0
        optimizer.step()
    return [p.detach().clone() for p in params], optimizer


def test_same_values_and_host_between_steps():
    from solution.adam_host import HostMoments

    moments = HostMoments()
    ref, _ = trajectory()
    got, optimizer = trajectory(moments)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))
    for state in optimizer.state.values():
        assert not state["exp_avg"].is_cuda and not state["exp_avg_sq"].is_cuda
    optimizer.state_dict()  # a checkpoint reads them as they are
