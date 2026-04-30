from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _Bucket:
    params: list[torch.nn.Parameter]
    pending: int
    launched: bool = False
    flat_grad: torch.Tensor | None = None
    handle: dist.Work | None = None


class DDPOverlapBucketed(torch.nn.Module):
    """DDP wrapper with bucketed, overlapped gradient synchronization."""

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP.")

        self.module = module
        self.world_size = dist.get_world_size()
        self._handles: list[dist.Work] = []
        self._bucket_size_bytes = int(bucket_size_mb * 1024 * 1024) if bucket_size_mb is not None else 0
        self._buckets: list[_Bucket] = []
        self._param_to_bucket: dict[int, _Bucket] = {}

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        self._build_buckets()
        self._register_hooks()
        self.start_train_batch()

    def _build_buckets(self) -> None:
        params = [p for p in self.module.parameters() if p.requires_grad]
        params.reverse()

        cur_params: list[torch.nn.Parameter] = []
        cur_size = 0
        cur_dtype: torch.dtype | None = None
        cur_device: torch.device | None = None

        def _flush() -> None:
            nonlocal cur_params, cur_size, cur_dtype, cur_device
            if not cur_params:
                return
            bucket = _Bucket(params=list(cur_params), pending=len(cur_params))
            self._buckets.append(bucket)
            for p in cur_params:
                self._param_to_bucket[id(p)] = bucket
            cur_params = []
            cur_size = 0
            cur_dtype = None
            cur_device = None

        for param in params:
            p_size = param.numel() * param.element_size()
            dtype_mismatch = cur_dtype is not None and param.dtype != cur_dtype
            device_mismatch = cur_device is not None and param.device != cur_device
            if (
                cur_params
                and (
                    (self._bucket_size_bytes > 0 and cur_size + p_size > self._bucket_size_bytes)
                    or dtype_mismatch
                    or device_mismatch
                )
            ):
                _flush()
            cur_params.append(param)
            cur_size += p_size
            cur_dtype = param.dtype
            cur_device = param.device
        _flush()

    def _register_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))

    def _make_post_accumulate_hook(self, param: torch.nn.Parameter):
        def _hook(_unused: torch.Tensor) -> None:
            bucket = self._param_to_bucket.get(id(param))
            if bucket is None or bucket.launched:
                return
            bucket.pending -= 1
            if bucket.pending == 0:
                flat_chunks = []
                for p in bucket.params:
                    if p.grad is None:
                        flat_chunks.append(torch.zeros_like(p).contiguous().view(-1))
                    else:
                        flat_chunks.append(p.grad.contiguous().view(-1))
                bucket.flat_grad = torch.cat(flat_chunks)
                bucket.handle = dist.all_reduce(bucket.flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                bucket.launched = True
                self._handles.append(bucket.handle)

        return _hook

    def start_train_batch(self) -> None:
        self._handles.clear()
        for bucket in self._buckets:
            bucket.pending = len(bucket.params)
            bucket.launched = False
            bucket.flat_grad = None
            bucket.handle = None

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        for bucket in self._buckets:
            if bucket.flat_grad is None:
                continue
            bucket.flat_grad /= self.world_size
            offset = 0
            for param in bucket.params:
                n = param.numel()
                grad_view = bucket.flat_grad[offset : offset + n].view_as(param)
                param.grad.copy_(grad_view)
                offset += n