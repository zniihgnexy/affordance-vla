# VLA branch V0：A6000 服务器搭建记录

**日期：2026-11-10｜状态：训练环境、仿真环境、数据、模型已就绪；官方权重的推理探测和闭环 rollout 已通过队列跑完（第 9 节）；QwenPI-2B 的训练显存探测还没做，backbone 配置暂时搁置｜对应 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 第 7 节 V0**

服务器是共享的 `estar`，从项目负责人的 Mac 上用 `ssh estar` 连接。所有查询和执行都走 SSH。本文记录在服务器上建了什么、怎么复现，以及提交 GPU 探测之前需要项目负责人决定的事。

## 1. 服务器现状（2026-11-10 晚，只读检查）

| 项目 | 情况 |
| --- | --- |
| 硬件 | x86-64，8 × RTX A6000（每张 48GB），144 核，1TB 内存，驱动 580 / CUDA 13.0，没有 `nvcc` |
| 可用的卡 | 原规定只能用 0、1、4、5、6、7，2 和 3 不能碰。2026-11-10 项目负责人更新为：优先用 0–3 号卡里的空卡，卡上有别的作业但显存够也可以放（见第 7 节）。队列运行器默认的可用卡和 `submit_queue_job.sh` 接受的卡仍是 0、1、4、5、6、7 |
| 当时的占用 | 卡 0、4 被另一位用户的 vLLM 占着（43GB、38GB）；卡 1、5、6、7 各跑着两个 `vid_tokenizer` 队列作业（每张约 12GB，利用率 99–100%） |
| 存储 | `/home` 已用 97%，只剩 117GB，只放代码等小文件；NAS（`/mnt/nas/xinyizheng/`，NFS）剩 7.9TB，放数据和模型；本地盘 `/data` 剩 3.5TB，只放 conda 环境 |
| 网络 | huggingface.co 0.05–0.1 MB/s 且不稳定；hf-mirror.com 约 0.3 MB/s；modelscope.cn 约 25 MB/s；pypi.org 很慢，阿里云镜像可用；GitHub 能连但慢。**HTTP/1.1 连接被限速到约 0.8 MB/s，HTTP/2 约 20 MB/s**，而 pip 只用 HTTP/1.1，所以大文件要先用 curl 下载 |

## 2. 在服务器上建了什么

**2026-11-10 晚起，工作以服务器为准。** 项目负责人的 Mac 空间不够，所以代码的工作副本、数据和模型都在服务器上，改动在服务器上提交并 push 到 GitHub。约定是：`/home` 只放代码等小文件，大的数据和模型放 NAS。

| 位置 | 内容 |
| --- | --- |
| `~/affordance-vla` | 本仓库的工作副本，`origin` 是 `git@github.com:zniihgnexy/affordance-vla.git`（服务器的 SSH 密钥已绑定 GitHub 账号）。其中 `data`、`runs`、`report/assets` 是指向 NAS 的符号链接，不进 git |
| `/mnt/nas/xinyizheng/affordance_vla/` | NAS 上的大文件目录，下面分 `models`、`data`、`repos`、`cache`、`runs`、`logs` |
| `/data/xinyizheng/envs/starvla` | conda 前缀环境，留在本地盘（导入快；conda 环境不能直接搬，换位置要重建） |
| `/data/xinyizheng/envs/libero` | LIBERO 仿真环境（MuJoCo 3.2.3、robosuite 1.4.0），只用 CPU，软件渲染。和 starVLA 环境分开，因为 LIBERO 需要较旧的 numpy、gym 和 torch |

NAS 目录里的内容：

| 路径 | 内容 |
| --- | --- |
| `repos/starVLA` | starVLA，分支 `starVLA_dev`，commit `13d8ad4`（浅克隆），以可编辑方式装进环境 |
| `repos/LIBERO` | LIBERO 官方仓库（浅克隆，640MB） |
| `data/libero_lerobot/` | LIBERO 四个 suite 的 LeRobot 格式数据，也就是 starVLA 官方训练用的 `IPEC-COMMUNITY/libero_*_no_noops_1.0.0_lerobot`。从 ModelScope 上的同名转存（`adamzhuo/...`）下载：抽样文件与 HF 原版逐字节一致，每个文件都按 ModelScope 列表里的 sha256 校验 |
| `data/libero/` | LIBERO 原始 HDF5 的一个任务文件（汇报 notebook 用） |
| `models/Qwen3-VL-2B-Instruct` | Qwen3-VL-2B 权重（4.26GB），从 ModelScope 下载 |
| `models/Qwen3-VL-PI-LIBERO-4in1` | starVLA 官方的 QwenPI LIBERO 权重（16.4GB，底座是 Qwen3-VL-4B），从 ModelScope 的官方镜像 `starVLA/...` 下载并逐文件校验 sha256 |
| `models/Qwen3-VL-4B-Instruct` | 上面这份权重建模型时要用的 Qwen3-VL-4B 底座（8.9GB） |
| `cache/libero_config` | LIBERO 的路径配置（没有它，LIBERO 第一次导入时会停下来等输入） |
| `runs/vla_pipeline_demo` | 流程 demo 的探测结果和 22 条逐步日志（约 0.8GB）。`runs/vla_pipeline_demo_prev` 是第一遍的结果，留作对照 |
| `cache/wheelhouse` | 建环境时预先下载的大安装包（torch、CUDA 库、flash-attn） |
| `runs/`、`logs/` | 实验输出和安装、下载日志 |

环境：Python 3.10，torch 2.6.0（CUDA 12.4 版），flash-attn 2.7.4.post1，其余按 starVLA 的 `requirements.txt` 固定版本。完整版本清单存在本仓库的 `scripts/remote/starvla_env_freeze.txt`。

最初这些大文件放在 `/data/xinyizheng/affordance3d_vla/`，2026-11-10 晚按上面的约定搬到了 NAS，原目录已删除。

**复现方法**（都不占 GPU）：

```bash
git clone git@github.com:zniihgnexy/affordance-vla.git ~/affordance-vla
W=/mnt/nas/xinyizheng/affordance_vla
mkdir -p $W/{models,data,repos,cache,runs,logs}
git clone --depth 1 -b starVLA_dev https://github.com/starVLA/starVLA.git $W/repos/starVLA
git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git $W/repos/LIBERO
cd ~/affordance-vla
ln -s $W/data data && ln -s $W/runs runs
nohup bash scripts/remote/setup_starvla_env.sh > $W/logs/setup_starvla_env.log 2>&1 &
nohup bash scripts/remote/download_assets.sh   > $W/logs/download_assets.log   2>&1 &
```

## 3. `scripts/remote/` 里的脚本

| 脚本 | 用途 | 占 GPU |
| --- | --- | --- |
| `setup_starvla_env.sh` | 建 conda 前缀环境，装 starVLA 依赖 | 否 |
| `download_assets.sh` | 下载模型和 LIBERO 数据（都走 ModelScope），可断点续传 | 否 |
| `fetch_modelscope_dataset.py` | 用 curl 并行下载一个 ModelScope 数据集，逐个校验 sha256；有文件缺失或不一致就以非零状态退出 | 否 |
| `smoke_starvla_cpu.py` | 在 CPU 上建模型、取一个训练 batch，打印各模块参数量。用来在占用 GPU 之前查出路径、配置和依赖问题 | 否 |
| `probe_starvla.sh` | V0 探测：用 starVLA 原版训练器在一张卡上各跑几十步（batch 16 / 8 / 4），记录显存峰值和每步时间 | 是，只能通过队列启动 |
| `submit_queue_job.sh` | 往共享 GPU 队列里放一个作业 | 间接 |
| `fetch_modelscope_model.sh` | 用 curl 下载一个 ModelScope 模型仓库，按大小和 sha256 逐文件校验，可断点续传 | 否 |
| `setup_libero_env.sh`、`libero_env.sh`、`requirements_libero_env.txt` | 建 LIBERO 仿真环境；`libero_env.sh` 设置软件渲染和路径变量，运行仿真前先 `source` 它 | 否 |
| `rollout_libero_logged.py` | 在仿真里跑闭环 rollout 或回放一条示范，逐步记录末端、物体位置、接触、可见像素和目标判定 | 否（只连策略服务） |
| `probe_vla_demo.py` | 加载官方权重，测模型的 token 数、去噪过程、开环预测和训练损失 | 是，只能通过队列启动 |
| `run_vla_demo_gpu.sh` | 流程 demo 的 GPU 作业：先跑 `probe_vla_demo.py`，再起 starVLA 的策略服务并行跑 21 条 rollout | 是，只能通过队列启动 |

## 4. GPU 队列是怎么工作的

队列属于 `vid_tokenizer` 项目，运行器是 `/home/xinyizheng/vid_tokenizer/scripts/queue_runner.py`（常驻进程）。

- **提交：** 往 `/home/xinyizheng/vid_tokenizer/queue/pending/` 放一个 bash 文件，文件名 `<三位优先级>_<名字>.sh`。
- **文件头：** `#Q minmb=24000 share=2 cards=1 5`，分别是这张卡至少要空出多少 MiB、一张卡上最多同时跑几个队列作业、限定哪些卡。
- **调度：** 每 15 秒检查一次。按文件名的字典序取作业，哪张卡空闲显存够、作业数没到 `share`，就在那张卡上启动，并设置 `CUDA_VISIBLE_DEVICES`。
- **一个作业只有一张卡。**
- **结果：** 作业文件移到 `queue/done/` 或 `queue/failed/`，日志在 `logs/queue/<作业名>.log`，状态在 `queue/status.tsv`。

## 5. CPU 冒烟测试的结果（不占 GPU）

用 `smoke_starvla_cpu.py` 在 CPU 上按官方 LIBERO 配置建模型、取一个训练 batch，结果通过：

| 项目 | 结果 |
| --- | --- |
| 模型 | starVLA QwenPI + Qwen3-VL-2B，建模型用时 23 秒 |
| 参数量 | 共 38.0 亿：VLM 21.3 亿（55.9%），动作头 16.8 亿（44.1%） |
| 一个训练样本 | 两张图像、一条指令、8 步 × 7 维的动作块；没有本体状态输入 |
| 数据 | 四个 suite 都能读；batch 为 2 时每个 epoch 202,938 个 batch |

**最重要的结论：全量微调在单张 A6000 上放不下。** 按 fp32 的权重、梯度和 AdamW 两个动量各占 4 字节算，38.0 亿参数需要 56.7 GiB，这还没算激活，而一张卡只有 48GB。

粗略估计（未实测）：用 DeepSpeed ZeRO-2 分到 N 张卡上时，每张卡要放完整的权重（14.2 GiB）、反向传播时的完整梯度（14.2 GiB）和 1/N 的优化器状态（28.3 GiB ÷ N），再加激活。

| 卡数 | 每张卡的固定占用（不含激活） | 判断 |
| --- | --- | --- |
| 1 | 56.7 GiB | 放不下 |
| 2 | 约 42.5 GiB | 基本放不下 |
| 4 | 约 35.4 GiB | 可能可以，batch 要小 |
| 6 | 约 33.1 GiB | 可能可以 |

所以 `probe_starvla.sh` 现在的单卡写法对这个配置一定会爆显存，探测方案要跟着下面的决定改。

## 6. 提交探测之前发现的两个问题

1. **需要整张卡的作业会一直排不上。** 队列里现在有 240 个 `vid_tokenizer` 作业在等，都是 `share=2`、`minmb=24000`。一张卡每空出一个位置，运行器就立刻补上一个这样的作业，所以卡上永远有两个作业。探测如果要求独占一张卡（`share=1`），就永远等不到。
2. **队列不支持多卡作业。** 第 5 节说明 QwenPI-2B 全量微调至少要 4 张卡一起跑，这和现在"一个作业一张卡"的队列不兼容，也会和 `vid_tokenizer` 的实验抢卡。

**单卡探测的两种做法**（只适用于单卡放得下的配置，例如换成更小的动作头，或者冻结一部分参数）：

| 做法 | 怎么做 | 能得到什么 | 对 `vid_tokenizer` 的影响 |
| --- | --- | --- | --- |
| A：共享一张卡 | 提交时设 `share=3 minmb=30000`，和两个现有作业挤在一张卡上 | 显存结论（上限约 36GB）；速度会被同卡作业拖慢，只能当下限 | 那张卡上的两个作业变慢约 1 小时 |
| B：占住一张卡 | 同时提交一个不用 GPU 的占位作业和探测作业，都限定同一张卡、都设 `share=2`。占位作业先占一个位置，另一个现有作业跑完后探测启动，这时卡是空的 | 干净的显存和速度数字 | 那张卡先半空约 40 分钟，再被探测占用约 1 小时 |

## 7. 待项目负责人决定

**2026-11-10 的答复：**

1. **GPU 怎么用：** 优先用 0–3 号卡里的空卡；卡上有别的作业但显存够，也可以放进去；只要能跑就排队跑。仍然都走队列。
2. **正式训练怎么分卡：** 两个项目分时间段使用，一个时间段内 8 张卡都可用，具体由项目负责人到时逐个安排。
3. **backbone 配置：** 先搁置，等 VLA 模型部分讨论之后再定。

下面是当时提出的原问题，留作背景。

1. **backbone 配置。** QwenPI-2B 全量微调只能多卡跑。可选的方向：
   - 保持 QwenPI-2B 全量微调，探测和训练都用 4 张以上的卡；
   - 换成更小的动作头（starVLA 的 QwenGR00T，R3 估计动作头约 3 亿参数），单卡可能放得下；
   - 冻结一部分参数。这会削弱"训练了 VLA"的说法。
2. **多卡怎么拿。** 队列一个作业只给一张卡。多卡探测和正式训练需要临时或长期划出几张卡，例如等 `vid_tokenizer` 这轮实验跑完，或者固定划出几张卡给 VLA。
3. **单卡探测**（如果选了单卡放得下的配置）用做法 A 还是 B，优先级排在 `vid_tokenizer` 作业之前还是之后。

## 8. 当前进度

- [x] 只读检查服务器、读懂队列
- [x] 克隆 starVLA 和 LIBERO
- [x] conda 环境安装（含 flash-attn）
- [x] 模型和 LIBERO 数据下载，逐文件校验
- [x] CPU 冒烟测试：模型和数据都能建，得到参数量和显存下限
- [x] LIBERO 仿真环境（CPU 软件渲染），渲染出第一帧
- [x] 官方权重的推理探测和 21 条闭环 rollout，都通过队列跑在 1 号卡上（第 9 节）
- [ ] QwenPI-2B 的训练显存探测（backbone 配置搁置，暂不做）

## 9. 流程 demo 的实测数字（2026-11-10）

结果和图都在 [流程 demo notebook](../../report/2026-11-10_vla_pipeline_demo.ipynb) 里，这里只记以后排作业会用到的数字。

| 项目 | 实测 |
| --- | --- |
| 官方权重 `Qwen3-VL-PI-LIBERO-4in1` | 77.9 亿参数（VLM 44.4 亿，动作头 33.5 亿）；bf16 推理占约 14.7 GiB，峰值 14.8 GiB；从 NAS 加载约 50 秒 |
| 一次策略调用 | 约 1.3 秒，其中模型部分约 0.55–0.6 秒（VLM 340–380 毫秒，4 步去噪每步约 52 毫秒）；当时这张卡上还有两个别的作业 |
| 仿真 | 软件渲染每步约 0.34 秒（两个相机的 RGB 和实例分割，256×256）；当时服务器 CPU 负载在 50–120 之间 |
| 队列作业 | `minmb=22000 share=3 cards=0 1`，提交后立刻在 1 号卡上启动（卡上原有两个 `vid_tokenizer` 作业）；探测加 21 条 rollout 共 15 分钟。整个作业跑了两遍，第二遍用的是提交进仓库的最终脚本 |
| 并行 | 9 个仿真进程共用一个策略服务。服务一次只处理一个请求，每个动作块要等约 9 秒，所以这次的总时间主要花在排队等策略上 |
| 示范回放 | 这个任务的前 6 条示范里，5 条回放后能再次达到目标（去掉开头 10 步静止则是 4 条）；其中 4 条的末端路径与原始记录相差约 10–15 毫米，另一条偏差明显更大。做标签时要按回放是否成功筛选 |
| 装环境 | 服务器上 pip 只有约 0.3 MB/s；conda 默认求解器在 conda-forge 上要算很久，要加 `--solver libmamba`；robosuite 不限定 opencv 版本，pip 会逐个版本试，要固定 `opencv-python==4.11.0.86` |
