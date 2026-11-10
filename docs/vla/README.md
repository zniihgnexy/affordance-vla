# VLA 方向文档

本仓库（`affordance-vla`）的研究方向：改变 VLA 的训练方式。训练时让 VLA 预测"接触点–接近方向–路径点"计划，约束动作与计划一致，并加入 episode 内的轻量 affordance 状态，目标是在杂乱的桌面操作仿真中提高成功率、减少抓错物体、方向错误和碰撞。必须训练 VLA，这是和记忆方向（环境记忆系统）的区别。

**仓库拆分（2026-11-10）：** 本仓库由 [affordance3D](https://github.com/zniihgnexy/affordance3D) 的 `vla` 分支带着完整提交历史独立出来；记忆方向（`mem`）留在 affordance3D。拆分之前写的文档里：

- "`vla` 分支"指的就是本仓库；
- "`mem` 分支"指 affordance3D 仓库里的记忆方向；
- "`main` 上的文档"指 `docs/vla/` 以外、从 affordance3D 带过来的早期文档，是两个方向共享的 brainstorming 基础，本仓库不再维护。

## 入口

1. [Pipeline v2（2026-11-10）](2026-11-10_VLA_branch_Pipeline_v2.md)：现行设计，包括方法、backbone 和 benchmark 选型、实验配置、阶段计划和后续检索清单。
2. [检索结果：backbone、benchmark 与近邻（2026-11-10）](2026-11-10_VLA_branch_检索结果_backbone_benchmark_近邻.md)：三个检索 agent 的原始报告。
3. [R1 近邻全文核实（2026-11-10）](2026-11-10_VLA_branch_R1_近邻全文核实.md)：六篇最接近的工作逐篇核对，以及 ObjDream、Ledger 等新发现的近邻。
4. [R2 LIBERO 变体协议（2026-11-10）](2026-11-10_VLA_branch_R2_LIBERO变体协议.md)：三个 LIBERO 变体的训练与评测协议、公开数据的内容，以及哪些标签可以直接读取、哪些需要回放。
5. [R3 A6000 训练成本（2026-11-10）](2026-11-10_VLA_branch_R3_A6000训练成本.md)：starVLA 和 π0.5 在 8 卡 A6000 上的显存、时间和总 GPU 小时估计。
6. [V0 服务器搭建记录（2026-11-10）](2026-11-10_VLA_branch_V0_服务器搭建记录.md)：A6000 服务器上的目录、环境、数据和脚本，GPU 队列的用法，以及提交探测前要决定的事。
7. [检索清单与 Pipeline v1（2026-11-10）](2026-11-10_VLA_branch_检索清单与Pipeline_v1.md)：已被取代，保留作历史记录。

后续检索结果、选型和实验记录都放在本目录，文件名以日期开头。

## 汇报 notebook

- [数据集与仿真环境汇报（2026-11-10）](../../report/2026-11-10_vla_datasets_sim_report.ipynb)：各 benchmark 的能力对照表和官方 teaser 图，LIBERO-Plus 及两个诊断集的说明，一个 LIBERO-Object 任务的细节（视频、末端路径、CAW 标签、动作积分检查），以及 RoboTwin 2.0 Hard。
- 重新运行前先准备：
  1. `python -m pip install h5py==3.16.0`，在 `affordance3d` 环境里装（已写入 `requirements.txt`）；
  2. `bash scripts/download_vla_report_images.sh`，下载官方图片；
  3. `bash scripts/download_libero_demo.sh`，下载约 664MB 的任务文件和它的 BDDL，存到本仓库的 `data/libero/` 下（不进 git）。
- 对应代码：`affordance3d/libero_io.py`（读取 LIBERO 文件）、`affordance3d/caw_labels.py`（从末端路径和夹爪指令生成 CAW 标签，检查动作积分）、`report/vla_report_helpers.py`（表格和图）。
