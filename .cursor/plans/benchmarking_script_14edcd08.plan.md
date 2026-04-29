---
name: Benchmarking Script
overview: 全新创建 benchmarking.py，专注实现 Problem (benchmarking_script) 的全部要求：端到端 Transformer 模型初始化、随机数据生成、带预热的计时循环，以及子任务 (b)/(c) 的扫描模式。
todos:
  - id: create-file
    content: 创建全新的 benchmarking.py，包含 MODEL_CONFIGS、imports、核心计时逻辑和 CLI
    status: completed
  - id: verify-run
    content: 验证脚本在 CPU/GPU 环境下均可正常运行，无报错
    status: in_progress
isProject: false
---

# Benchmarking Script 实现计划

## 目标文件
新建 [`benchmarking.py`](benchmarking.py)（从零创建，仅覆盖 Problem benchmarking_script）

---

## 脚本结构（约 130 行）

### 模型配置表（对应 PDF Table 1）

```python
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
ROPE_THETA = 10_000.0
```

### 核心计时函数 `time_model()`

```
1. 用 BasicsTransformerLM(**cfg) 初始化模型，移到 device
2. 随机生成 (BATCH_SIZE, context_length) 的整型 token IDs
3. 预热循环（不计时）：
     for _ in range(warmup_steps):
         logits = model(x)
         if not forward_only: logits.sum().backward()
         if cuda: torch.cuda.synchronize()
         optimizer.zero_grad()   # 清空梯度，防止内存累积
4. 计时循环：
     times = []
     for _ in range(num_steps):
         t0 = timeit.default_timer()
         logits = model(x)
         if not forward_only: logits.sum().backward()
         if cuda: torch.cuda.synchronize()
         optimizer.zero_grad()
         times.append(timeit.default_timer() - t0)
5. 返回 mean_ms, std_ms（用 statistics.mean / statistics.stdev）
```

> 注：`optimizer.zero_grad()` 不计入 timing（在 synchronize 之后调用），仅用于在多步测量时防止梯度无限累积。

### CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model_size` | `all` | `small/medium/large/xl/2.7B/all` |
| `--context_length` | `128` | 序列长度（多个值用空格分隔） |
| `--warmup_steps` | `5` | 预热轮次 |
| `--num_steps` | `10` | 计时轮次 |
| `--forward_only` | flag | 仅前向 pass（默认含反向） |
| `--compare_warmup` | flag | 子任务 (c)：对比 warmup=0/1/2/5 |
| `--device` | `cuda` | 运行设备 |

### 运行模式

**单次运行**（指定具体 model_size）：
```bash
uv run python benchmarking.py --model_size small --context_length 128 --warmup_steps 5 --num_steps 10
```

**全量扫描**（子任务 b，--model_size all）：
```bash
uv run python benchmarking.py --model_size all --context_length 128 --warmup_steps 5 --num_steps 10
uv run python benchmarking.py --model_size all --context_length 128 --warmup_steps 5 --num_steps 10 --forward_only
```

**预热对比**（子任务 c，--compare_warmup）：
```bash
uv run python benchmarking.py --model_size small --context_length 128 --compare_warmup
```

### 输出示例

```
======================================================
 End-to-End Transformer Benchmark
 mode=fwd+bwd  warmup=5  steps=10  context=128
======================================================
 size      d_model  layers   mean(ms)    std(ms)
 --------- -------  ------  ---------  --------
 small         768      12     xxx.xx      x.xx
 medium       1024      24     xxx.xx      x.xx
 ...
```

---

## 关键实现细节

- `torch.cuda.synchronize()` 在每步 backward/forward 之后、`timeit` 停表之前调用
- 用 `timeit.default_timer()` 而非 `time.time()`（更高精度）
- 零梯度（`model.zero_grad()`）在计时区间外，仅为防止内存溢出
- 如果没有 CUDA，自动降级到 CPU（适合本地调试）
