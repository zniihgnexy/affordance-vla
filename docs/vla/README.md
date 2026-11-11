# VLA 方向文档

本仓库（`affordance-vla`）的研究方向：改变 VLA 的训练方式。训练时让 VLA 预测"接触点–接近方向–路径点"计划，约束动作与计划一致，并加入 episode 内的轻量 affordance 状态，目标是在杂乱的桌面操作仿真中提高成功率、减少抓错物体、方向错误和碰撞。必须训练 VLA，这是和记忆方向（环境记忆系统）的区别。

**仓库拆分（2026-11-10）：** 本仓库由 [affordance3D](https://github.com/zniihgnexy/affordance3D) 的 `vla` 分支带着完整提交历史独立出来；记忆方向（`mem`）留在 affordance3D。拆分之前写的文档里：

- "`vla` 分支"指的就是本仓库；
- "`mem` 分支"指 affordance3D 仓库里的记忆方向；
- "`main` 上的文档"指 `docs/vla/` 以外、从 affordance3D 带过来的早期文档，是两个方向共享的 brainstorming 基础，本仓库不再维护。

## 当前状态（2026-11-11）

- **工作方式：** 以服务器上的 `~/affordance-vla` 为准，改动在服务器上提交并 push；大文件在 NAS 上，布局见 [V0 服务器搭建记录](2026-11-10_VLA_branch_V0_服务器搭建记录.md) 第 2 节。
- **阶段：** V0（核实与搭建）已完成，第一个诊断实验已做完。
  - 检索 R1–R3 和数据集汇报已完成；服务器上的训练环境、仿真环境、模型和数据都已就绪。
  - 已用 starVLA 官方开源权重在一个 LIBERO 任务上把整条流程跑通（见下面的流程 demo）。
  - **计划跟随实验（2026-11-11）：** 在冻结的官方权重上只训练一小部分参数，把真值计划交给动作头，看它跟不跟。结论见[实验记录](2026-11-11_VLA_branch_计划跟随实验.md)第 1 节。
  - 完整的方法（训练整个 VLA、由模型预测计划）还没有训练。
- **已决定（2026-11-10，项目负责人）：**
  1. GPU 作业优先放 0–3 号卡里的空卡；卡上有别的作业但显存够也可以放；只要能跑就排队跑，仍然都走队列。2026-11-11 更新：目前空的卡都可以用。
  2. 正式训练时两个项目分时间段使用服务器，一个时间段内 8 张卡都可用，具体安排到时由项目负责人逐个定。
  3. QwenPI-2B 全量微调单卡放不下的问题先搁置，等 VLA 模型部分讨论之后再定。
- **已同意（2026-11-11，项目负责人）：** 主线是"换过的计划加一致性损失"，先做计划跟随实验。
- **下一步：** 等项目负责人决定[实验记录](2026-11-11_VLA_branch_计划跟随实验.md) 1.6 节的 4 件事：一致性损失在方法里的位置、计划头实验、搬运和释放阶段要不要加换过的计划、以后每个条件跑多少条。

## 入口

1. [Pipeline v2（2026-11-10）](2026-11-10_VLA_branch_Pipeline_v2.md)：现行设计，包括方法、backbone 和 benchmark 选型、实验配置、阶段计划和后续检索清单。
2. [检索结果：backbone、benchmark 与近邻（2026-11-10）](2026-11-10_VLA_branch_检索结果_backbone_benchmark_近邻.md)：三个检索 agent 的原始报告。
3. [R1 近邻全文核实（2026-11-10）](2026-11-10_VLA_branch_R1_近邻全文核实.md)：六篇最接近的工作逐篇核对，以及 ObjDream、Ledger 等新发现的近邻。
4. [R2 LIBERO 变体协议（2026-11-10）](2026-11-10_VLA_branch_R2_LIBERO变体协议.md)：三个 LIBERO 变体的训练与评测协议、公开数据的内容，以及哪些标签可以直接读取、哪些需要回放。
5. [R3 A6000 训练成本（2026-11-10）](2026-11-10_VLA_branch_R3_A6000训练成本.md)：starVLA 和 π0.5 在 8 卡 A6000 上的显存、时间和总 GPU 小时估计。
6. [V0 服务器搭建记录（2026-11-10）](2026-11-10_VLA_branch_V0_服务器搭建记录.md)：A6000 服务器上的目录、环境、数据和脚本，GPU 队列的用法，以及提交探测前要决定的事。
7. [检索清单与 Pipeline v1（2026-11-10）](2026-11-10_VLA_branch_检索清单与Pipeline_v1.md)：已被取代，保留作历史记录。
8. [计划跟随实验（2026-11-11）](2026-11-11_VLA_branch_计划跟随实验.md)：第一个训练实验的结论、设置、做法和复现步骤。

后续检索结果、选型和实验记录都放在本目录，文件名以日期开头。

## 汇报 notebook

- [计划跟随实验（2026-11-11）](../../report/2026-11-11_plan_following_experiment.ipynb)：把真值计划交给官方权重的动作头，看它是否跟随。
  - **内容：** 示范的状态回放；计划输入；换过的计划和目标路径；训练曲线；开环指标；闭环结果（三个模型，各 5 条标准、18 条目标挪动、30 条换物体）；分阶段统计；指令的抖动；结论和要决定的事。
  - **哪些是真的：** 示范、开源模型、训练、开环数字、闭环 rollout。
  - **哪些是 ORACLE：** 计划本身。训练时来自示范，闭环时来自仿真真值。
  - **主要结果：** 只把计划当输入会被忽略；训练时加入换过的计划之后，夹爪 18 次里 18 次到达挪动后的抓取点，换物体时 30 次里 27–30 次到达、成功次数从 6 次升到 19–20 次；改写动作标签和一致性损失效果相当。
  - **重新运行：** 见[实验记录](2026-11-11_VLA_branch_计划跟随实验.md)第 4 节。
- [流程 demo（2026-11-10）](../../report/2026-11-10_vla_pipeline_demo.ipynb)：用 LIBERO-Object 的一个任务和 starVLA 官方权重 `Qwen3-VL-PI-LIBERO-4in1`，一步一步走完整条流程。
  - **内容：** 训练样本；模型内部（token、隐状态、4 步去噪）；开环预测和训练损失；闭环 rollout（标准、换指令、挪动目标，共 21 条）；CAW 标签；计划–动作一致性度量；episode 状态。
  - **哪些是真的：** 数据、开源模型、它的开环和闭环动作、仿真日志，以及由日志算出的所有数字。
  - **哪些还是占位（notebook 里标为 ORACLE）：** CAW 计划、一致性度量里的"计划"、episode 状态。它们都由示范回放和仿真真值算出，还没有模型预测它们。
  - **主要结果：** 按训练时的指令 5 次全部成功；指令换成场景里别的物体，10 次里 8 次拿错；目标挪动 4–16 厘米，6 次里只成功 1 次。整组 21 条跑了两遍，20 条的结果相同。
  - **重新运行（都在服务器上）：**
    1. `scripts/remote/setup_libero_env.sh`，建 LIBERO 仿真环境；
    2. `scripts/remote/fetch_modelscope_model.sh`，下载官方权重和 Qwen3-VL-4B；
    3. 按 `scripts/remote/run_vla_demo_gpu.sh` 文件头的说明，先在 CPU 上回放一条示范，再把这个脚本提交到 GPU 队列；
    4. 在 `report/` 下运行 `scripts/build_vla_pipeline_demo.py`，再执行 notebook。
  - **对应代码：** `affordance3d/` 下的 `plan_consistency.py`（一致性度量）、`episode_state.py`（episode 状态）、`rollout_outcome.py`（结果分类）、`rollout_log.py`（读日志）、`libero_camera.py`（相机投影）；`report/` 下的 `vla_pipeline_demo_helpers.py` 和 `vla_pipeline_demo_labels.py`（图和表）。
- [数据集与仿真环境汇报（2026-11-10）](../../report/2026-11-10_vla_datasets_sim_report.ipynb)：各 benchmark 的能力对照表和官方 teaser 图，LIBERO-Plus 及两个诊断集的说明，一个 LIBERO-Object 任务的细节（视频、末端路径、CAW 标签、动作积分检查），以及 RoboTwin 2.0 Hard。
- 重新运行前先准备：
  1. `python -m pip install h5py==3.16.0`，在 `affordance3d` 环境里装（已写入 `requirements.txt`）；
  2. `bash scripts/download_vla_report_images.sh`，下载官方图片；
  3. `bash scripts/download_libero_demo.sh`，下载约 664MB 的任务文件和它的 BDDL，存到本仓库的 `data/libero/` 下（不进 git）。
- 对应代码：`affordance3d/libero_io.py`（读取 LIBERO 文件）、`affordance3d/caw_labels.py`（从末端路径和夹爪指令生成 CAW 标签，检查动作积分）、`report/vla_report_helpers.py`（表格和图）。
