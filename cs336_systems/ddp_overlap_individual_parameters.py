from __future__ import annotations

import torch
import torch.distributed as dist


class DDPOverlapIndividualParameters(torch.nn.Module):
    """Minimal DDP wrapper that overlaps backward compute with gradient all-reduce.

    For each trainable parameter, we install a post-accumulate-grad hook that launches
    an asynchronous all-reduce as soon as that parameter's gradient is ready.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP.")

        self.module = module
        self.world_size = dist.get_world_size()
        self._handles: list[dist.Work] = []

        # Keep all ranks on the same initial weights.
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Launch async gradient all-reduce as soon as each gradient is available.
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))

    def _make_post_accumulate_hook(self, param: torch.nn.Parameter):
        def _hook(_unused: torch.Tensor) -> None:
            if param.grad is None:
                return
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._handles.append(handle)

        return _hook

    def forward(self, *inputs, **kwargs):
        if self._handles:
            raise RuntimeError(
                "finish_gradient_synchronization() from the previous step was not called. "
                "Please call it before optimizer.step()."
            )
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Ensure all all-reduce operations are queued/completed before optimizer.step().
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        # Convert summed gradients to mean gradients.
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size