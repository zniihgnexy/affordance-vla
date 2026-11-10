# VLA branch V0：A6000 服务器搭建记录

**日期：2026-11-10｜状态：进行中（环境和数据在装，GPU 探测还没提交）｜对应 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 第 7 节 V0**

服务器是共享的 `estar`，从项目负责人的 Mac 上用 `ssh estar` 连接。所有查询和执行都走 SSH。本文记录在服务器上建了什么、怎么复现，以及提交 GPU 探测之前需要项目负责人决定的事。

## 1. 服务器现状（2026-11-10 晚，只读检查）

| 项目 | 情况 |
| --- | --- |
| 硬件 | x86-64，8 × RTX A6000（每张 48GB），144 核，1TB 内存，驱动 580 / CUDA 13.0，没有 `nvcc` |
| 可用的卡 | 只能用 0、1、4、5、6、7；2 和 3 不能碰 |
| 当时的占用 | 卡 0、4 被另一位用户的 vLLM 占着（43GB、38GB）；卡 1、5、6、7 各跑着两个 `vid_tokenizer` 队列作业（每张约 12GB，利用率 99–100%） |
| 存储 | `/home` 已用 97%，只剩 117GB；`/data` 剩 3.5TB。所以本分支的东西全部放 `/data` |
| 网络 | huggingface.co 0.05–0.1 MB/s 且不稳定；hf-mirror.com 约 0.3 MB/s；modelscope.cn 约 25 MB/s；pypi.org 很慢，阿里云镜像可用；GitHub 能连但慢。**HTTP/1.1 连接被限速到约 0.8 MB/s，HTTP/2 约 20 MB/s**，而 pip 只用 HTTP/1.1，所以大文件要先用 curl 下载 |

## 2. 在服务器上建了什么

工作目录是 `/data/xinyizheng/affordance3d_vla/`：

| 路径 | 内容 |
| --- | --- |
| `repos/starVLA` | starVLA，分支 `starVLA_dev`，commit `13d8ad4`（浅克隆） |
| `repos/LIBERO` | LIBERO 官方仓库（浅克隆，640MB） |
| `data/libero_lerobot/` | LIBERO 四个 suite 的 LeRobot 格式数据（`IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot`），starVLA 的官方训练数据 |
| `models/Qwen3-VL-2B-Instruct` | Qwen3-VL-2B 权重（4.26GB），从 ModelScope 下载 |
| `cache/` | conda、pip、HF 的缓存，避免写到 `/home` |
| `scripts/` | 从本仓库 `scripts/remote/` 拷过去的脚本 |
| `logs/`、`runs/` | 安装、下载日志和实验输出 |

conda 环境是前缀环境 `/data/xinyizheng/envs/starvla`：Python 3.10，torch 2.6.0（CUDA 12.4 版），其余按 starVLA 的 `requirements.txt` 固定版本。装好后的完整版本清单在 `logs/starvla_env_freeze.txt`。

**复现方法**（都不占 GPU）：

```bash
W=/data/xinyizheng/affordance3d_vla
mkdir -p $W/{repos,data,models,logs,runs,cache,scripts}
git clone --depth 1 -b starVLA_dev https://github.com/starVLA/starVLA.git $W/repos/starVLA
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git $W/repos/LIBERO
# 把本仓库 scripts/remote/ 下的脚本拷到 $W/scripts/ 之后：
nohup bash $W/scripts/setup_starvla_env.sh > $W/logs/setup_starvla_env.log 2>&1 &
nohup bash $W/scripts/download_assets.sh   > $W/logs/download_assets.log   2>&1 &
```

## 3. `scripts/remote/` 里的脚本

| 脚本 | 用途 | 占 GPU |
| --- | --- | --- |
| `setup_starvla_env.sh` | 建 conda 前缀环境，装 starVLA 依赖 | 否 |
| `download_assets.sh` | 下载模型（ModelScope）和 LIBERO 数据（hf-mirror，自动重试），可断点续传 | 否 |
| `smoke_starvla_cpu.py` | 在 CPU 上建模型、取一个训练 batch，打印各模块参数量。用来在占用 GPU 之前查出路径、配置和依赖问题 | 否 |
| `probe_starvla.sh` | V0 探测：用 starVLA 原版训练器在一张卡上各跑几十步（batch 16 / 8 / 4），记录显存峰值和每步时间 | 是，只能通过队列启动 |
| `submit_queue_job.sh` | 往共享 GPU 队列里放一个作业 | 间接 |

## 4. GPU 队列是怎么工作的

队列属于 `vid_tokenizer` 项目，运行器是 `/home/xinyizheng/vid_tokenizer/scripts/queue_runner.py`（常驻进程）。

- **提交：** 往 `/home/xinyizheng/vid_tokenizer/queue/pending/` 放一个 bash 文件，文件名 `<三位优先级>_<名字>.sh`。
- **文件头：** `#Q minmb=24000 share=2 cards=1 5`，分别是这张卡至少要空出多少 MiB、一张卡上最多同时跑几个队列作业、限定哪些卡。
- **调度：** 每 15 秒检查一次。按文件名的字典序取作业，哪张卡空闲显存够、作业数没到 `share`，就在那张卡上启动，并设置 `CUDA_VISIBLE_DEVICES`。
- **一个作业只有一张卡。**
- **结果：** 作业文件移到 `queue/done/` 或 `queue/failed/`，日志在 `logs/queue/<作业名>.log`，状态在 `queue/status.tsv`。

## 5. 提交探测之前发现的两个问题

1. **需要整张卡的作业会一直排不上。** 队列里现在有 240 个 `vid_tokenizer` 作业在等，都是 `share=2`、`minmb=24000`。一张卡每空出一个位置，运行器就立刻补上一个这样的作业，所以卡上永远有两个作业。探测如果要求独占一张卡（`share=1`），就永远等不到。
2. **队列不支持多卡作业。** 正式训练 starVLA 需要 4 张以上的卡用 DeepSpeed 一起跑好几天，这和现在"一个作业一张卡"的队列不兼容，也会和 `vid_tokenizer` 的实验抢卡。

**探测的两种做法：**

| 做法 | 怎么做 | 能得到什么 | 对 `vid_tokenizer` 的影响 |
| --- | --- | --- | --- |
| A：共享一张卡 | 提交时设 `share=3 minmb=30000`，和两个现有作业挤在一张卡上 | 显存结论（上限约 36GB）；速度会被同卡作业拖慢，只能当下限 | 那张卡上的两个作业变慢约 1 小时 |
| B：占住一张卡 | 同时提交一个不用 GPU 的占位作业和探测作业，都限定同一张卡、都设 `share=2`。占位作业先占一个位置，另一个现有作业跑完后探测启动，这时卡是空的 | 干净的显存和速度数字 | 那张卡先半空约 40 分钟，再被探测占用约 1 小时 |

## 6. 待项目负责人决定

1. 探测用做法 A 还是 B，优先级排在 `vid_tokenizer` 作业之前还是之后。
2. 正式训练时两个项目怎么分卡。例如等 `vid_tokenizer` 这轮实验跑完，或者固定划出几张卡给 VLA 训练。

## 7. 当前进度

- [x] 只读检查服务器、读懂队列
- [x] 克隆 starVLA 和 LIBERO
- [ ] conda 环境安装（进行中）
- [ ] 模型和 LIBERO 数据下载（进行中）
- [ ] CPU 冒烟测试，得到参数量和显存下限估计
- [ ] 通过队列提交 GPU 探测（等上面第 6 节的决定）
