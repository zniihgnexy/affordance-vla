# 第 0 步 ②：RoboCasa365 在服务器上的搭建记录

2026-11-26（服务器时间）。任务：AFFO-10。分支 `agent/vla-engineer/affo-10`（服务器 `~/wt-affo-10`）。
数字都来自实际跑出来的运行；推断单独标「推断」。输出在 NAS：`$W=/mnt/nas/xinyizheng/affordance_vla`。

## 1. 四个问题的答案

| # | 问题 | 答案 |
| --- | --- | --- |
| 1 | 装得上、渲染得出吗？ | **能。** 新环境 `/data/xinyizheng/envs/robocasa365`（Python 3.11），CPU 软件渲染（OSMesa），不开 GPU。 |
| 2 | 示范能重放出深度和分割吗？ | **能。** 3 任务 × 5 条，从保存的状态重放，渲染 RGB、深度、实例分割。RGB 和原视频同帧 PSNR 34.2–36.2 dB（错开一帧只有 26.5–29.1 dB）。 |
| 3 | starVLA 的训练和评测能跑通吗？ | **QwenOFT（starVLA 示例自带的配置）能；QwenPI 在一张 48 GB 的卡上放不下。** QwenOFT：冻结 VLM 微调 300 步，显存峰值 11.4 GiB，每步 0.445 s（批大小 8）；评测 ⟦EVAL_EP⟧。QwenPI：冻结 VLM 批大小 8、批大小 1，全量微调批大小 1，都显存不足。**没有公开的 RoboCasa365 权重**。 |
| 4 | 三件事 | 见第 6 节：留出实例划分能直接用且排行榜用它；组合任务的逐帧子任务标注和文档一致；mug / cup / bowl 有 27 / 22 / 22 个实例，网格都能导出表面点。 |

## 2. 版本与安装

- 脚本：`envs/server/setup_robocasa365_env.sh`（可重跑，已完成的步骤跳过）、`envs/server/robocasa365_env.sh`（激活）。
- robocasa `7bfc60c`（v1.0.1，2026-11-18 的 main）、robosuite `5ce6643`（1.5.2）、MuJoCo 3.3.1、numba 0.61.2、torch 2.7.1+cu126（lerobot 拉进来的，只用来读数据，不用 GPU）。完整冻结表：`$W/logs/robocasa365_env_freeze.txt`。
- 代码用 codeload 的压缩包装（服务器上 `git clone` 会卡住）；资产 6 个 Box 链接共约 11 GB 的 zip，解压后 22 GB，在 `$W/repos/robocasa365/robocasa/models/assets`。
- 数据根目录写进 `macros_private.py`：`$W/data/robocasa365`。starVLA 评测/训练代码：`$W/repos/starVLA_rc365`（commit `99764f5` 的压缩包，不是 git 仓库，一个文件没改，只在 `examples/simBenchmarks/AFFO10_rc365/` 下加了三任务混合的注册文件）。
- 遇到的问题：
  1. 资产的 `fixtures` 和 `fixtures_lightwheel` 两个 zip 共用顶层目录 `fixtures/`，仓库自带一部分 `fixtures/`，原来的"目录存在就跳过"让第二个 zip 没解压，场景构建报 `Window044/model.xml` 找不到。已改成每个 zip 检查自己独有的子目录。
  2. numba 0.61.2 / llvmlite 0.44 在本机（Xeon 8352V）编译物体摆放采样器（`placement_samplers.py`）时，LLVM 优化阶段段错误。`NUMBA_OPT=0` 解决，写在 `robocasa365_env.sh` 里。
  3. 评测客户端还要 `tyro websockets msgpack msgpack-numpy`，已加进安装脚本。

## 3. 渲染（CPU 软件渲染，256 × 256）

实测（3 任务 × 5 条，`$W/runs/robocasa365_replay/20261125_3tasks_5eps/*_replay.json`；同时有 3 个重放进程在跑）：

| 项目 | 秒 |
| --- | --- |
| 渲染一路 RGB | 0.103–0.114 |
| 一路 RGB + 深度 + 分割 | 0.278–0.318 |
| `env.step`（不渲染） | 0.054–0.061 |
| 重建一条示范的场景（`reset_to`） | 16.5–18.8 |
| 创建环境 | 7.5–7.7 |

每步 3 路相机的仿真，约 0.05 + 3 × 0.1 ≈ 0.35 s（推断，由上面三个数相加）。评测里实测见第 5 节。太慢的话可选的办法（推断，未试）：并行更多进程；评测时只渲染模型用到的相机；GPU 渲染（EGL）——按要求没有改用。

## 4. 数据与重放

- 下载：`scripts/data/fetch_robocasa365.sh`，先打印大小再下载（curl，可续传）。3 个目标任务：CoffeeSetupMug 477 MB（抓放 mug）、OpenDrawer 479 MB、TurnOffStove 423 MB（旋钮），共 1.38 GB，约 1 分钟（23 MB/s）。另为第 6 节下了 2 个组合任务：PrepareCoffee 943 MB、StackBowlsCabinet 742 MB。
- **和调研记录不一致的地方：** ① LeRobot 版本是 **v2.1**，不是 v3.0（`meta/info.json: codebase_version`）；② 每任务 **514–515 条**；③ 50 个目标任务的 Box 压缩包合计 **61.6 GB**（atomic 9.2 + composite 52.4；用 HTTP Content-Range 量了全部 50 个），不是 188 GB——HF 数据卡写的 188 GB 我没有对上来源，**推断**是别的口径（例如解压后或含别的内容），未核实。
- `extras/episode_XXXXXX/`：`model.xml.gz`、`states.npz`（每步一个展平的 MuJoCo 状态）、`ep_meta.json`。`dataset_meta.json` 记录示范是用 robocasa 0.5.1 / MuJoCo 3.3.1 / robosuite 1.5.2 录的，我们装的是 robocasa v1.0.1。
- 重放：`scripts/data/replay_robocasa365.py`，共享代码 `affordance_vla/sim/robocasa365.py`。

| 任务 | 条数 / 帧数 | RGB 同帧 PSNR 均值 | 错开 ±1 帧（较高者） | 最差一帧 |
| --- | --- | --- | --- | --- |
| CoffeeSetupMug | 5 / 174–251 | 34.2 dB | 26.5 dB | 29.7 dB |
| OpenDrawer | 5 / 208–433 | 35.5 dB | 29.1 dB | 31.8 dB |
| TurnOffStove | 5 / 152–225 | 36.2 dB | 28.4 dB | 32.2 dB |

每 5 帧比一次，3 路相机取平均。结论：**重放和原示范一致**（对齐，差在 h264 压缩和渲染细节；差分图只剩边缘）。PSNR 有限不是 ∞，是因为视频有压缩，没有拿到原始未压缩帧。图：`report/media/weekly_2026_11_23/robocasa365/replay_video_vs_rerender.png`。
开环动作重放（从第一个状态起施加记录的动作）：**不是逐位一致**——整个状态向量与记录的误差范数，第一步 0.05–0.67，中间 0.05–0.16，最后一步 0.05–0.81，最大 0.14–5.9；没有一条的所有步都精确相等。**推断**：库版本不同（录制时 robocasa 0.5.1，现在 v1.0.1）和接触的混沌放大；没有检查回放后任务是否成功。要用的话建议按状态重放（上表），不要靠动作重放。

## 5. starVLA 流程

起点：starVLA 的 `examples/simBenchmarks/Robocasa_365`（PandaOmron，12 维动作，3 路相机，状态 16 维）。训练在 `starvla` 环境（torch 2.6.0、deepspeed 0.16.9，和该示例的 requirements 一致），评测客户端在 `robocasa365` 环境。命令和配置：`configs/rc365/`、`scripts/rc365/`，登记在 `EXPERIMENTS.md` 第 6 节。

**公开权重：没有。** HF 上 StarVLA 名下 45 个模型里，带 Robocasa 的只有 `Qwen3-VL-OFT-Robocasa`（模型卡：24 个 RoboCasa-GR1 桌面任务）和 `Qwen3-VL-GR00T-Robocasa-gr1`，是另一个基准（示例 README 自己也这么说）。365 示例的 README 说本地快照的权重和日志没有放出。所以没有"先评测公开权重"这一步。

**训练**（3 任务混合：CoffeeSetupMug + OpenDrawer + TurnOffStove，每任务 ~514 条，单卡 A6000 48 GB，DeepSpeed ZeRO-2，bf16）：

| 运行 | 可训练参数 | 批大小 | 结果 | 显存峰值（`torch.cuda.max_memory_allocated`） |
| --- | --- | --- | --- | --- |
| QwenOFT，冻结 VLM | 65.6 M / 4.50 B | 8 | 跑完 300 步，约 2.6 min，每步模型 0.445 s（含取数约 0.52 s）；L1 损失 1.24（第 10 步）→ 0.22（150）→ 0.11（300） | **11.42 GiB** |
| QwenPI，冻结 VLM | 3.35 B / 7.79 B | 8 | 第一次前向就显存不足（`qwenpi_rc365_3tasks_frozen`） | 46.41 GiB 时失败 |
| QwenPI，冻结 VLM | 3.35 B / 7.79 B | 1 | 显存不足（`qwenpi_rc365_frozen_b1`；再申请 6.25 GiB 失败） | 41.46 GiB 时失败 |
| QwenPI，全量 | 7.79 B / 7.79 B | 1 | 显存不足（`qwenpi_rc365_full_b1`；再申请 12.5 GiB 失败） | 37.30 GiB 时失败 |
| QwenOFT，全量 | 4.50 B / 4.50 B | ⟦FULL_BS⟧ | ⟦FULL_OFT⟧ | ⟦FULL_OFT_MEM⟧ |

QwenPI 放不下的原因：它的动作头有 3.35 B 参数（每个 VLM 层配一个 2560 宽的块，starVLA 的 QwenPI 设计如此），加上 VLM 共 7.79 B；单卡 AdamW 在 fp32 下仅权重、梯度和两个动量就需要约 116 GiB（`smoke_starvla_cpu.py` 的估算，不含激活）。按要求没有想办法绕。**建议（推断）：** 第 1 步要用 QwenPI，只能多卡（ZeRO-2/3 切分）或像 LIBERO 那一阶段一样冻结全部权重，只训练 LoRA 和我们的新模块；或者改用 QwenOFT（头是小 MLP）。需要项目负责人决定用哪个骨架。

**评测**（`scripts/rc365/run_eval_gpu.sh`：一个策略服务器占一张卡，每个任务一个 CPU 客户端，3 个客户端同时跑；回合长度用 RoboCasa v1.0.1 注册表里每任务的：CoffeeSetupMug 600、OpenDrawer 750、TurnOffStove 750，不是示例默认的 500）：被评测的是上面 300 步的 QwenOFT，所以成功率只是流程检查。

⟦EVAL_TABLE⟧

评测客户端和训练的不一致（示例 README 自己警告过，我没有改 starVLA）：训练读 3 路相机，评测桥只送 `robot0_agentview_left` 一路，状态无条件构造；评测环境不带种子（`Args.seed` 没传给 `gym.make`）。所以这个成功率不能当成模型的能力。

## 6. 三件事

**a. 留出物体实例的划分能直接用吗；排行榜的 50 个任务启用了吗？** 能用，启用了。
- 规则在 `robocasa/models/objects/kitchen_object_utils.py`（`sample_kitchen_object`）：每个类别、每个物体库里，前 `max(n − 5, ceil(n / 2))` 个实例是 `pretrain`，其余是 `target`。注意文档字符串写"最后 4 个"，代码是最后 5 个（实例少于 10 个时是后一半）。
- `split="target"` 同时设 `obj_instance_split="target"` 和 10 个固定厨房（`robocasa/utils/env_utils.py:87`）。
- 官方评测协议用的就是它：`docs/benchmarking/benchmarking_overview.md` 第 28 行"target 划分是 10 个不相交的目标厨房和不相交的目标物体"；`foundation_model_learning.md` 的 50 任务评测命令都是 `--split target`；starVLA 的评测客户端也是 `gym.make(..., split="target")`。
- 下载的示范自己也是这样录的：`extras/dataset_meta.json` 的 `env_kwargs.obj_instance_split = "target"`。
- 所以**训练的示范里出现的就是 target 实例**，评测的也是 target 实例——目标集不是"训练见过的实例 vs 没见过的实例"。要做留出物体的泛化，需要用 `pretrain` 划分的实例训练、`target` 的评测，或者自己在 target 内部再划分；这是后面设计要定的事。

**b. 组合任务的逐帧子任务标注和文档一致吗？** 一致。PrepareCoffee 3 条、StackBowlsCabinet 2 条，每帧有 `annotation.human.subtask`（语言）、`annotation.human.subtask_name`（技能名，如 CoffeeSetupMug）、`annotation.human.subtask_stage`（pick / place / execute / done）、`subtask_idx`（段序号），字符串存在 `meta/tasks.jsonl`。PrepareCoffee 每条 4 段：pick the mug → place under the dispenser → press start → done；StackBowlsCabinet 5 段。同任务各条的段序、语言完全相同，只有边界帧不同（PrepareCoffee ep0：0–205、206–466、467–703、704–744）。原子任务（OpenDrawer 等）没有这些列。

**c. mug / bowl / cup 的实例数，网格能导出表面点吗？** 能。`affordance_vla/sim/robocasa365.py::category_instances / surface_points`（读 `visual/*.obj`，按 `model.xml` 的缩放和参考旋转变到物体坐标系，按面积加权采样）。

| 类别 | 物体库：实例数（留出） | 合计（留出） | 导出表面点 | 包围盒（x, y, z，中位数 / 最大，m） |
| --- | --- | --- | --- | --- |
| mug | objaverse 17（5）、aigen 10（5） | 27（10） | 27 / 27 | 0.076 × 0.107 × 0.083 / 0.105 × 0.135 × 0.097 |
| cup | objaverse 4（2）、aigen 18（5） | 22（7） | 22 / 22 | 0.064 × 0.063 × 0.070 / 0.086 × 0.086 × 0.115 |
| bowl | objaverse 18（5）、aigen 4（2） | 22（7） | 22 / 22 | 0.122 × 0.122 × 0.049 / 0.139 × 0.139 × 0.072 |

另外：glass_cup 24（lightwheel）、coffee_cup 18 + 6。注意：把手朝向没有统一（图 `category_surface_points.png`：mug_4 的把手朝 −y，mug_1、mug_5 朝 +y），类别坐标系需要像 Pipeline v3 那样把把手转到 +x，这一步没做。

## 7. GPU 用量

提交 6 个作业（都在 GPU 6，优先级 164，一次一张）：4 个训练 + 1 个评测 ⟦GPU_N⟧；占卡时间合计约 ⟦GPU_H⟧ 小时，上限 12 小时。

## 8. 没验证的事和建议

- 没有评测"训练到可用"的模型；评测只量速度。没有做完整 50 任务评测，没有下整个数据集。
- 动作开环重放的任务成功与否没查。
- 没有核对 starVLA 的评测桥传 1 路相机是否让成功率偏低（README 已警告）。
- 建议：① 第 1 步用 QwenOFT 骨架，或冻结 QwenPI 全部权重后训练 LoRA；② 评测前先对齐相机数和状态；③ 评测慢（见上），每个设置的回合数要预算：每回合约 ⟦EP_MIN⟧ 分钟（3 个客户端并行时）。

## 9. 复现

```bash
bash envs/server/setup_robocasa365_env.sh
bash scripts/data/fetch_robocasa365.sh                   # 3 个任务；--dry-run 只看大小
source envs/server/robocasa365_env.sh
"$RC365_PYTHON" scripts/data/replay_robocasa365.py --task CoffeeSetupMug OpenDrawer TurnOffStove --episodes 5 \
    --out $W/runs/robocasa365_replay/<run>
"$RC365_PYTHON" scripts/report/make_rc365_figures.py --replay $W/runs/robocasa365_replay/<run> \
    --out report/media/weekly_2026_11_23/robocasa365
bash scripts/server/submit_queue_job.sh 164 rc365_oft 30000 1 "cards=6 7" -- \
    env RUN=<run> CONFIG=$PWD/configs/rc365/qwenoft_rc365_3tasks.yaml bash $PWD/scripts/rc365/run_train_gpu.sh
bash scripts/server/submit_queue_job.sh 164 rc365_eval 20000 1 "cards=6 7" -- \
    env CKPT=$W/runs/robocasa365_starvla/<run>/checkpoints/steps_300_pytorch_model.pt \
    bash $PWD/scripts/rc365/run_eval_gpu.sh
```

运行目录：`$W/runs/robocasa365_replay/20261125_3tasks_5eps`、`$W/runs/robocasa365_starvla/{qwenoft_rc365_frozen,qwenpi_rc365_3tasks_frozen,qwenpi_rc365_frozen_b1,qwenpi_rc365_full_b1}`、`$W/runs/robocasa365_eval/eval_qwenoft_frozen_300`；日志 `$W/logs/affo10_*`。
