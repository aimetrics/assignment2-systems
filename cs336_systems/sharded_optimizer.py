from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Optimizer wrapper that shards optimizer state across distributed ranks."""

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """Initializes the sharded state optimizer. 
        Args:
            params: a collection of parameters to be optimized (or parameter
                groups, in case the user wants to use different hyperparameters, such as learning rates, for different parts of the model); 
                these parameters will be sharded across all the ranks.
            optimizer_cls: the type of optimizer to be wrapped (e.g., optim.AdamW).
            kwargs: any remaining keyword arguments are forwarded to the constructor of the optimizer_cls.
        """
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs  # 例如 lr=1e-3, weight_decay=0.1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        # 全局参数计数器，决定每个参数分给哪个 rank
        self._next_param_index = 0
        # 初始化锁：防止 add_param_group 在 super().__init__ 期间提前创建 _local_optimizer
        self._initialized = False
        # super().__init__ 调用期间积累的本地参数组
        self._pending_local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: Optimizer | None = None

        # NOTE: super().__init__ 会触发 add_param_group
        super().__init__(params, defaults={})
        #  ↓ 内部对每个参数组调用
        #  self.add_param_group(param_group)
        #       ↓
        #       _build_local_group()         决定哪些参数属于本 rank
        #       ↓
        #       _initialized = False         此时还没有 _local_optimizer
        #       ↓
        #       append to _pending_local_param_groups   先积累

        # super().__init__ 返回后，一次性创建
        # 为什么需要 _pending_local_param_groups：
        # 问题：super().__init__ 过程中会调用 add_param_group
        #     但此时 _local_optimizer 还不存在
        # 解决：_initialized=False 时先积累到 _pending_local_param_groups
        #     super().__init__ 完成后，一次性创建 _local_optimizer
        #     之后的 add_param_group 直接操作 _local_optimizer
        if self._pending_local_param_groups:
            self._local_optimizer = self.optimizer_cls(self._pending_local_param_groups, **self.optimizer_kwargs)

        self._initialized = True

    def _build_local_group(self, param_group: dict[str, Any]) -> dict[str, Any] | None:
        # 核心分片逻辑
        local_params = []
        for param in param_group["params"]:
            if self._next_param_index % self.world_size == self.rank:
                local_params.append(param)
            self._next_param_index += 1

        if not local_params:
            return None

        # 保留原 group 的所有超参（lr, weight_decay 等），只替换 params
        local_group = {k: v for k, v in param_group.items() if k != "params"}
        local_group["params"] = local_params
        return local_group

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        # Step 1：基类管理完整参数组（所有 rank 都有完整视图）
        super().add_param_group(param_group)
        # Step 2：提取本 rank 负责的参数.拿到刚刚被 PyTorch 基类 Optimizer 标准化并加入的最后一个参数组。
        canonical_group = self.param_groups[-1]
        local_group = self._build_local_group(canonical_group)

        if local_group is None:
            return

        # Step 3：如果还没初始化，先积累到 _pending_local_param_groups
        if not self._initialized:
            self._pending_local_param_groups.append(local_group)
            return

        # Step 4：如果已经初始化，直接添加到 _local_optimizer
        if self._local_optimizer is None:
            self._local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self._local_optimizer.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        loss = None
        # Step 1：只更新本 rank 负责的参数
        if self._local_optimizer is not None:
            loss = self._local_optimizer.step(closure=closure, **kwargs)
        elif closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Step 2：广播更新后的参数，同步所有 rank
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            param_index = 0
            for group in self.param_groups:
                for param in group["params"]:
                    owner_rank = param_index % self.world_size
                    dist.broadcast(param.data, src=owner_rank)
                    param_index += 1

        return loss