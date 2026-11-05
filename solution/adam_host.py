"""AdamW's state in pinned host memory between optimizer steps.

The optimizer's moments (2 x the LoRA weights, 16.6 MiB here) are read only inside
optimizer.step(), while the GPU peak is reached in the decoder's backward. A pre-step hook copies
them to the GPU as one block, a post-step hook copies them back, the same values, and the trainer
still sees them in optimizer.state. Organizers (QA 61): allowed.
"""

import torch

_KEYS = ("exp_avg", "exp_avg_sq")


def _moments(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state.get(param)
            if state:
                for key in _KEYS:
                    if key in state:
                        yield state, key


class HostMoments:
    def __init__(self):
        self.host = None
        self.device = None

    def before_step(self, optimizer, *_args, **_kwargs):
        if self.host is None:
            return  # the first step creates the moments on the GPU
        block = self.host.to(self.device, non_blocking=True)
        start = 0
        for state, key in _moments(optimizer):
            tensor = state[key]
            state[key] = block[start:start + tensor.numel()].view(tensor.shape)
            start += tensor.numel()

    def after_step(self, optimizer, *_args, **_kwargs):
        moments = list(_moments(optimizer))
        if not moments or any(not state[key].is_cuda for state, key in moments):
            return
        tensors = [state[key] for state, key in moments]
        total = sum(t.numel() for t in tensors)
        if self.host is None:
            self.host = torch.empty(total, dtype=tensors[0].dtype, pin_memory=True)
            self.device = tensors[0].device
        # blocking: the trainer may read optimizer.state right after this step (a checkpoint)
        self.host.copy_(torch.cat([t.reshape(-1) for t in tensors]))
        start = 0
        for (state, key), tensor in zip(moments, tensors):
            state[key] = self.host[start:start + tensor.numel()].view(tensor.shape)
            start += tensor.numel()
