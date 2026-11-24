# 数据集与实验范围调研（第一轮宽筛）

AFFO-4 · 2026-11-24 · VLA Scout。只读摘要、项目主页、数据卡，未下载、未运行任何东西。

**问题：** 有没有数据集（或两个的组合），既有足够环境信息做理解，动作标签又精细到能训练 VLA。

**标记：** [仓] = 沿用仓库已有调研（出处见文末 R1–R3）；其余为本轮新查。本轮新查的条目都只读了项目页、数据卡或排行榜，均属「只读摘要」级。「未查到」= 本轮没找到，不是没有。

## 1. 对比表（11 个候选）

### 室内仿真操作 benchmark（带示范）

| # | 名称 | 场景 | 环境信息 | 动作标签 | 规模 | 同类实例数 | 闭环评测 | 可得性 | 和现有代码距离 | 记忆方向 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **RoboCasa365**（2026，robocasa.ai）[仓 R2；数字已核对] | 房间级厨房，含移动操作 | 第三人称+腕部 [仓]；深度/分割未核实；语言指令；无问答标注 | 有；PandaOmron 移动臂；控制频率未查到 | 365 任务，2500+ 厨房，2200+ 小时（600+ 人类示范 + 1600+ 合成）| 3200+ 物体 [仓]；每类实例数未查到 | 有仿真。排行榜 50 任务：π0.5 16.9，π0 14.8，GR00T N1.5 23.9，最高 VLA 62.6（`robocasa.ai/leaderboard`）| MIT/CC BY 4.0 [仓]；MuJoCo，无头未核实 [仓] | LeRobot [仓]；starVLA 官方支持的是 RoboCasa-GR1 [仓 R2]，RoboCasa365 未查到 | 可用（厨房多视角）|
| 2 | **BEHAVIOR-1K 2025 挑战赛**（OmniGibson，behavior.stanford.edu）| 房间级整屋，全身移动操作 | 头部 720² + 腕部 480²，RGB+深度+实例分割；物体状态真值；技能和子任务标注、语言标注 | 有；R1Pro 双臂移动机器人，30 fps；动作空间类型未查到 | 10,000 条，1200+ 小时，50 任务，平均 6.6 分钟/条 | 未查到 | 有仿真和排行榜：最好 π0.5 改版 q-score 0.26（成功率 0.12）；starVLA 提交 0.002（公开验证集）| 数据 ~1.5 TB，HF 公开，可按任务/相机分块下；许可未查到；Isaac Sim，A6000 未查到 | LeRobot v2.1（定制）；starVLA 未查到 | 可用（深度+分割+位姿可重建点云）|
| 3 | **MS-HAB**（ICLR 2025，ManiSkill3）| ReplicaCAD 公寓，移动操作 | RGB-D 128²+状态；事件标注 | 有；Fetch，关节位置增量 | 84 套公寓；18K+18K+8K 条 | YCB 物体 | 有仿真；只有 RL/IL 基线，无 VLA 数字 | 公开；ManiSkill3 需 Vulkan [仓 R2]；许可未查到 | 非 LeRobot；starVLA 未查到 | 可用 |
| 4 | **LIBERO / PRO / Plus** [仓 R2] | 桌面 | 腕部+第三人称；深度/分割需 flag；语言 | 有；末端；Panda | 50 示范 × 130 任务 | 每场景 2–8 个可动物体 | 有；LIBERO-PRO 最佳均值 69.8% | MIT；MuJoCo EGL 可无头 | LeRobot/RLDS；starVLA 已支持（当前主线）| 有限 |

### 带场景信息的真实机器人数据

| # | 名称 | 场景 | 环境信息 | 动作标签 | 规模 | 同类实例数 | 闭环评测 | 可得性 | 和现有代码距离 | 记忆方向 |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | **DROID**（2024，droid-dataset.github.io）| 564 个真实场景（厨房、办公室、酒店房间等），固定臂，非移动 | 2 路外部 + 1 路腕部，ZED 立体；原始版含立体视频，RLDS 版只有 RGB；语言（75k 条带 3 条标注）；部分校准位姿 | 有；Franka，15 Hz，关节速度+末端位姿 | 76k 条，350 小时，86 任务 | 未查到 | **有**：IsaacLab-Arena 厨房 42 任务，π0.5 平均 38.3%（100 回合/环境）；`sim-evals` 零样本 | CC-BY 4.0；RLDS 1.7 TB，原始 5.6–8.7 TB；Isaac Sim | RLDS，HF 有 LeRobot 镜像；starVLA 未查到 | 可用（多视角+校准）|
| 6 | **AgiBot World**（2025，arXiv 2503.06669）| 106 个复刻场景（家庭、零售、工业等），移动双臂 | 8 路相机（含前置 RGB-D）30 Hz；语言+关键帧子步骤标注 | 有；G1 双臂，30 fps；动作类型未查到 | 1,001,552 条，2976 小时，217 任务 | 3000+ 物体，每类未查到 | 仿真评测未查到；GO-1 基线数字未查到 | CC BY-NC-SA；Beta 48.1 TB，Alpha ~8.5 TB；HF 需登记 | LeRobot v2 | 可用 |

### 第一人称或人类视频（理解/affordance 来源）

| # | 名称 | 场景 | 环境信息 | 动作标签 | 规模 | 同类实例数 | 闭环评测 | 可得性 | 和现有代码距离 | 记忆方向 |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | **HD-EPIC**（CVPR 2025，hd-epic.github.io）| 9 个真实厨房，第一人称 | Aria 眼镜；**场景数字孪生（404 个固定装置）**、相机位姿、注视、物体移动、3D 掩码；VQA 26.6K 题 | 无机器人动作；59K 细粒度人类动作标签 | 41 小时，69 道菜 | 未查到 | 无仿真；VQA 基准（Gemini Pro 37.0%）| 许可、大小未查到 | 无 | **直接可用**（带 3D 的长视频）|
| 8 | **EgoDex**（Apple 2025，ml-egodex）| 桌面，家居用品 | 第一人称 1080p 30 Hz；相机位姿；语言（LLM 自动生成，可能有误）| 无机器人动作；3D 手和上半身位姿（68 关节）| 829 小时，194 任务 | 未查到 | 无；有手轨迹预测基准 | CC-BY-NC-ND；~1.7 TB，直链下载 | HDF5+MP4，无 | 部分（只有相机位姿，无场景 3D）|
| 9 | **EPIC-KITCHENS + VISOR** [仓 R1，只读摘要] | 真实厨房，第一人称 | 动作段；VISOR 手/物体掩码；EPIC-Fields 重建位姿 | 无 | 未查到 | 未查到 | 无 | 未查到 | 无 | 可用（仓库已定为时间记忆）|
| 10 | **HOI4D** [仓 R1，只读摘要] | 室内桌面，第一人称 | RGB-D；手/物体位姿、分割、网格、场景点云 | 无机器人动作；手/物体位姿 | 未查到 | 有类别内多实例（具体数未查到）| 无 | 未查到 | 无 | 可用 |
| 11 | **SceneFun3D**（`scenefun3d.github.io`）[仓 R1，只读摘要] | 重建的室内场景 | RGB-D、位姿、3D 功能元素（把手/开关/旋钮）、语言任务、运动参数 | 无 | 未查到 | 未查到 | 无 | 未查到 | 无 | **直接可用**（仓库的首选记忆数据）|

## 2. 结论

1. **前三名（兼顾环境理解和精细动作）：** ① BEHAVIOR-1K：整屋、RGB-D+实例分割+物体状态+子任务标注，最接近；缺口是极难（最好 q-score 0.26，starVLA 提交近 0）、1.5 TB、Isaac Sim 要核对 A6000，且没有问答类理解标注。② RoboCasa365：2500 厨房、2200 小时、LeRobot、有含 VLA 的排行榜、MuJoCo；缺口是只有厨房，深度/分割和 starVLA 支持未核实，同样没有理解标注。③ DROID：真实 564 场景、三路相机、有 IsaacLab 零样本闭环评测；缺口是固定臂、RLDS 版无深度、仿真评测是另一套合成场景，不是数据本身的环境。
2. **一个够不够：** 本轮没找到一个数据集同时有问答/物体级理解标注和精细动作。BEHAVIOR-1K 是最接近的单一数据集（推断：它的分割和物体状态真值可以自动生成理解标注，但要自建）。
3. **最合适的两个组合（推断）：** (a) **RoboCasa365 + HD-EPIC**：动作和闭环评测在仿真厨房，场景理解和 affordance 提取用 HD-EPIC 的数字孪生视频，同为厨房，域接近；(b) **DROID + EgoDex/EPIC**：全部真实，闭环只能靠 IsaacLab-Arena 的零样本评测。
4. **只有真实数据、没有（数据对应）仿真的候选，单列不排除：** AgiBot World（仿真评测未查到）、EgoDex、HD-EPIC、EPIC+VISOR、HOI4D、SceneFun3D（后四个无机器人动作，只能做理解/affordance 来源）。
5. 第二轮建议深核：BEHAVIOR-1K 的许可与 A6000 可行性、RoboCasa365 的相机/深度字段和 starVLA 适配、DROID 的相机位姿覆盖率、HD-EPIC 的许可与大小。

## 3. 来源

- R1 `docs/research/Pipeline_Datasets_and_Simulation_Guide_2026-11-05.md`（SceneFun3D、EPIC+VISOR、HOI4D、LIBERO 条目）
- R2 `docs/vla/2026-11-10_VLA_branch_检索结果_backbone_benchmark_近邻.md`（benchmark 表：LIBERO 族、RoboCasa、ManiSkill3、starVLA 支持）
- R3 `docs/research/数据与仿真Agent_需求交接清单_2026-11-02.md`、`docs/data-and-simulation.md`、`docs/vla/2026-11-10_VLA_branch_R2_LIBERO变体协议.md` 已浏览，表中事实未取自它们（R3 的需求是 placement 记忆，非 VLA 数据集）
- RoboCasa365：https://robocasa.ai/ ，排行榜 https://robocasa.ai/leaderboard.html （更新 2026-11-23）
- BEHAVIOR：https://behavior.stanford.edu/challenge/archive/2025/index.html ，/dataset.html ，/leaderboard.html ；数据 https://huggingface.co/datasets/behavior-1k/2025-challenge-demos
- MS-HAB：https://arth-shukla.github.io/mshab/ ，https://github.com/arth-shukla/mshab/ ，ICLR 2025
- DROID：https://droid-dataset.github.io/droid/the-droid-dataset ，https://github.com/droid-dataset/droid_policy_learning ，https://www.roboticscenter.ai/datasets/droid （第三方汇总，规模和许可待第二轮用官方页复核）；仿真评测 https://isaac-sim.github.io/IsaacLab-Arena/main/pages/example_workflows/kitchen_bench_catalog.html ，https://github.com/arhanjain/sim-evals
- AgiBot World：arXiv 2503.06669，https://huggingface.co/agibot-world ，https://www.kineticblocks.com/open-data/agibot-world （第三方索引，许可和大小待复核）
- EgoDex：https://machinelearning.apple.com/research/egodex-learning-dexterous-manipulation ，https://github.com/apple/ml-egodex
- HD-EPIC：https://hd-epic.github.io/ ，CVPR 2025 论文页，https://github.com/hd-epic/hd-epic-annotations
