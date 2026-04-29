# torch 2.8.0 在 Apple Silicon Mac 上安装失败的排查与修复

## 问题现象

在 Apple M4 Pro 上运行 `uv run pytest` 时报错：

```
error: Distribution `torch==2.8.0 @ registry+...` can't be installed because it
doesn't have a source distribution or wheel for the current platform

hint: You're on macOS (`macosx_15_0_x86_64`), but `torch` (v2.8.0) only has
wheels for the following platforms: manylinux_2_28_aarch64, manylinux_2_28_x86_64,
macosx_11_0_arm64, win_amd64
```

uv 认为当前平台是 `x86_64`，但 torch 2.8.0 只提供 `macosx_11_0_arm64` 的 wheel，导致无法安装。

## 根因排查

### 第一步：确认 Python 报告的架构

```bash
.venv/bin/python3 -c "import platform; print(platform.machine())"
# 输出：x86_64
```

机器是 M4 Pro（arm64），但 Python 却报告 `x86_64`，说明 venv 使用的是通过 **Rosetta 2** 运行的 x86_64 Python。

### 第二步：定位问题 Python 的路径

运行 `arch -arm64 uv sync` 后，uv 输出：

```
Using CPython 3.13.2 interpreter at: /usr/local/opt/python@3.13/bin/python3.13
```

`/usr/local/opt/` 是 **Intel Homebrew** 的安装前缀。Apple Silicon Mac 上 Homebrew 的原生 arm64 路径是 `/opt/homebrew/`。uv 选中了 Intel Homebrew 下的 x86_64 Python binary，导致整个 venv 以 x86_64 架构创建，平台标识变成 `macosx_15_0_x86_64`，从而无法匹配 torch 的 `macosx_11_0_arm64` wheel。

## 解决方案

让 uv 自行下载并管理一个原生 arm64 Python，绕过系统中的 x86_64 Python：

```bash
# 1. 安装 uv managed arm64 Python 3.12
arch -arm64 uv python install 3.12

# 2. 用该 Python 重建 venv 并同步依赖
arch -arm64 uv sync --python 3.12
```

验证架构是否正确：

```bash
file .venv/bin/python3
# 应输出：Mach-O 64-bit executable arm64
```

之后 `uv run pytest` 即可正常解析并安装 `torch-2.8.0-...-macosx_11_0_arm64.whl`。
