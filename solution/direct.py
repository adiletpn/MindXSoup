"""Autograd functions called directly when no gradient is being recorded.

A rerun layer's first pass runs under no_grad, and `Function.apply` there only adds Python around
`Function.forward`: thousands of calls per training step, while the GPU waits for its next launch
(a timeline of the step: 29% of it nothing runs on the GPU). Without a graph there is nothing for
the wrapper to do, so `run` calls the forward with a context that ignores what backward would need.
The computation is the same.
"""

import torch

enabled = False  # set by apply()


class _NoBackward:
    """What a Function's forward sees of its context when there will be no backward."""

    needs_input_grad = ()

    def __setattr__(self, _name, _value):
        pass  # shared by all calls: must not keep anything alive

    def save_for_backward(self, *_tensors):
        pass

    def mark_non_differentiable(self, *_tensors):
        pass


_context = _NoBackward()


def run(function, *args):
    if enabled and not torch.is_grad_enabled():
        return function.forward(_context, *args)
    return function.apply(*args)
