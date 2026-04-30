from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class _Bucket:
    """
    训练过程中 _Bucket 的状态机：

    初始状态：
        pending = len(params)   # 等待所有参数梯度就绪
        launched = False
        flat_grad = None
        handle = None
            ↓
    每个参数梯度就绪时：
        pending -= 1
            ↓
    pending == 0 时：
        flat_grad = cat(所有参数梯度)
        handle = dist.all_reduce(flat_grad, async_op=True)
        launched = True
            ↓
    finish_gradient_synchronization() 时：
        handle.wait()
        flat_grad /= world_size
        写回各参数的 .grad
    """
    params: list[torch.nn.Parameter]        # 这个 bucket 包含哪些参数
    pending: int                            # 还有多少参数的梯度未就绪
    launched: bool = False                  # 是否已经发出 all_reduce
    flat_grad: torch.Tensor | None = None   # 这个 bucket 的梯度
    handle: dist.Work | None = None         # 异步通信的句柄


class DDPOverlapBucketed(torch.nn.Module):
    """DDP wrapper with bucketed, overlapped gradient synchronization."""

    def __init__(self, module: torch.nn.Module, bucket_size_mb: float | None):
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed process group must be initialized before constructing DDP.")

        self.module = module
        self.world_size = dist.get_world_size()
        self._handles: list[dist.Work] = []
        # bucket_size_mb=None 或 0 时，所有参数放一个 bucket
        self._bucket_size_bytes = int(bucket_size_mb * 1024 * 1024) if bucket_size_mb is not None else 0
        self._buckets: list[_Bucket] = []
        self._param_to_bucket: dict[int, _Bucket] = {}

        # Step 1：广播 rank 0 的初始权重，保证所有 rank 起点相同
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Step 2：把参数分配到 bucket
        self._build_buckets()
        # Step 3：给每个参数注册 hook
        self._register_hooks()
        # Step 4：初始化 bucket 状态
        self.start_train_batch()

    def _build_buckets(self) -> None:
        params = [p for p in self.module.parameters() if p.requires_grad]
        # backward 是从 loss 往输入方向传播的，越靠近 loss 的层梯度越先算好。
        # model.parameters() 返回的顺序是正向（从输入到输出），反转后得到的顺序和梯度就绪顺序对齐，使得 bucket 能尽早凑满、尽早发起通信。
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

        # 分 bucket 逻辑
        for param in params:
            p_size = param.numel() * param.element_size()
            # 触发新 bucket 的三个条件
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
                # 把当前积累的参数打包成一个 bucket
                _flush()
            cur_params.append(param)
            cur_size += p_size
            cur_dtype = param.dtype
            cur_device = param.device
        # 处理最后一组
        _flush()

    def _register_hooks(self) -> None:
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            param.register_post_accumulate_grad_hook(self._make_post_accumulate_hook(param))

    def _make_post_accumulate_hook(self, param: torch.nn.Parameter):
        def _hook(_unused: torch.Tensor) -> None:
            # 找到这个参数属于哪个 bucket
            bucket = self._param_to_bucket.get(id(param))
            # launched=True 说明 bucket 已经发出通信，不需要再处理
            if bucket is None or bucket.launched:
                return
            # 这个参数的梯度就绪了，pending 减 1
            bucket.pending -= 1
            if bucket.pending == 0:
                # 把所有参数的梯度拍平拼接成一个大 tensor
                flat_chunks = []
                for p in bucket.params:
                    if p.grad is None:
                        flat_chunks.append(torch.zeros_like(p).contiguous().view(-1))
                    else:
                        flat_chunks.append(p.grad.contiguous().view(-1))
                bucket.flat_grad = torch.cat(flat_chunks)
                # 发起异步 all_reduce
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
        if any(bucket.launched for bucket in self._buckets):
            raise RuntimeError(
                "start_train_batch() was not called for the current step. "
                "Please call it after optimizer.step() and before the next forward pass."
            )
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Step 1：对没有凑齐的 bucket 也补发 all-reduce；unused 参数用 0 梯度占位。
        for bucket in self._buckets:
            if bucket.launched:
                continue
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

        # Step 2：等待所有异步通信排入 GPU 队列
        for handle in self._handles:
            handle.wait()

        # Step 3：对每个 bucket 的梯度做后处理
        for bucket in self._buckets:
            if bucket.flat_grad is None:
                continue
            bucket.flat_grad /= self.world_size
            # 把拍平的梯度切回原始形状，写回各参数的 .grad
            offset = 0
            for param in bucket.params:
                n = param.numel()
                grad_view = bucket.flat_grad[offset : offset + n].view_as(param)
                # ← in-place 写回，不改变 param.grad 的内存地址
                if param.grad is None:
                    if torch.count_nonzero(grad_view).item() != 0:
                        param.grad = grad_view.clone()
                else:
                    param.grad.copy_(grad_view)
                offset += n
        self._handles.clear()