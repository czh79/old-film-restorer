# 🎞️ 老片 AI 修复器 · Old Film Restorer

> 单文件、零 `basicsr` 依赖的 Real-ESRGAN 推理器，专为老电影整片修复设计：分块断点续跑、去黄调色、30 秒 PK 对比。

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![No basicsr](https://img.shields.io/badge/deps-zero%20basicsr-success)]()

**中文** | [English](#-english)

---

## 为什么不用现成的 Real-ESRGAN 仓库

官方 `Real-ESRGAN` 依赖 `basicsr`，而 `basicsr` 对 `torchvision` 版本极其敏感——装一次能跑，换台机器就 `ImportError`，再加上 Windows 上的编译问题，劝退无数人。

这个脚本把 **RRDBNet 网络结构原样实现在单文件里**，只依赖 `torch + numpy + opencv`。权重照常加载官方 `*.pth`，不需要改动任何模型文件。

## 核心能力

| 能力 | 说明 |
|---|---|
| **纯 PyTorch 实现** | `ResidualDenseBlock_5C` / `RRDB` / `FlexibleRRDBNet` 全部手写，零 `basicsr`、零 `realesrgan` 包 |
| **分块断点续跑** | 每 1000 帧存一个独立切片（约 40 秒），断电只丢当前块，重跑自动跳过已完成部分 |
| **老片去黄** | 电影胶片偏黄/偏绿校正滤镜，`remove_yellow=True` 一键开 |
| **自适应分辨率** | 按长边基准（默认 1920）等比缩放，不拉伸变形 |
| **双模式** | AI 超分（需 GPU + 权重）／轻量胶片调校（纯 CPU，带实时进度与速度） |
| **30 秒 PK 测试** | 先跑前 30 秒对比两套方案，再决定整片用哪个，避免几小时白跑 |

## 两种模式怎么选

```
模式 1 · AI 深度超分    需要 GPU + *.pth 权重
  └─ 画质质变，适合 480p/720p 老片 → 1080p
  └─ 耗时：约 2–4 倍片长（RTX 3060 实测）

模式 2 · 轻量胶片调校    纯 CPU，无需模型
  └─ 降噪 + CAS 锐化 + 去黄，实时进度
  └─ 耗时：接近 1:1，适合"只想去黄提亮"的场景
```

不确定用哪个？选 **3（30 秒 PK 测试）**，两边各跑 30 秒，直接看效果。

## 快速开始

```bash
git clone https://github.com/czh79/old-film-restorer.git
cd old-film-restorer
pip install -r requirements.txt
```

**准备权重**（仅模式 1 需要）

下载任意 Real-ESRGAN 官方权重，例如 `RealESRGAN_x4plus.pth`，放到脚本同目录即可——脚本会自动 glob `*ESRGAN*.pth`：

```bash
# 从官方 Release 下载（约 64MB）
curl -L -o RealESRGAN_x4plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

**运行**

```bash
python restore.py
```

菜单选模式 → 选影片（支持 `all` 批量）→ 输入长边分辨率 → 开跑。

输出文件名固定为 `<原名>_enhanced.mp4`。

## 断点续跑是怎么做的

```
影片.mp4
  ├─ chunk_0000.mp4   (帧 0     - 1000)
  ├─ chunk_0001.mp4   (帧 1000  - 2000)
  ├─ ...
  └─ chunk_NNNN.mp4   (尾部不足 1000 帧)

每块处理前用 verify_chunk_validity() 校验帧数，
已存在且帧数正确的块直接跳过 → 最后无损拼接
```

中途 Ctrl+C 或断电，重跑同一条命令就会从断点继续，已经在跑的大片不用重来。

## 性能参考

| 硬件 | 片长 | 模式 | 耗时 |
|---|---|---|---|
| RTX 3060 12G | 90 min | AI 超分 4x → 1080p | ~3.5 h |
| RTX 3060 12G | 90 min | 轻量调校 | ~1.6 h |
| i7-12700 (CPU) | 90 min | 轻量调校 | ~2.1 h |

CPU 跑模式 1 会非常慢（10 倍以上），建议先用模式 3 看效果再决定。

---

## 🇬🇧 English

> A single-file Real-ESRGAN inference tool for restoring old films. Zero `basicsr` dependency, chunked resumable processing, yellow-cast correction, and a 30-second A/B test mode.

The official Real-ESRGAN stack pulls in `basicsr`, which is notoriously fragile against `torchvision` versions — it works on the machine it was installed on and breaks everywhere else. This project reimplements the `RRDBNet` architecture directly with plain PyTorch, so you only need `torch + numpy + opencv`, and the official `*.pth` weights load unchanged.

### Highlights

- **Zero `basicsr`** — `ResidualDenseBlock_5C`, `RRDB`, `FlexibleRRDBNet` are implemented inline.
- **Resumable by design** — video is split into 1000-frame chunks; each is validated by frame count before being skipped, so a power cut costs you one chunk instead of the whole run.
- **Age-corrected colour** — a film-stock yellow/green cast removal filter.
- **Two modes** — AI upscale (GPU + weights) or a CPU-only light grain/noise/sharpness pass with live progress.
- **A/B test first** — render the first 30 seconds with both pipelines before committing hours to a full run.

### Usage

```bash
pip install -r requirements.txt
# place any *ESRGAN*.pth next to restore.py
python restore.py
```

Output: `<name>_enhanced.mp4`.

---

<div align="center">

帮你把一柜子老碟救回来的话，**点个 Star** ⭐

</div>
