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

---

## 4. 第二轮深核（2026-11-24）

标记：**全文** = 读了论文正文；**官方文档** = 项目文档、数据卡、仓库 README；**未查到**。RoboCasa365 论文只读到正文和附录 B–D、G.2（附录 E、F 的任务清单与统计表未读到）。第三方汇总站的数字标「第三方」。

### 4.1 核对一：HD-EPIC 两处出入

arXiv 2502.04144 摘要与表 2 写 Gemini Pro 37.6%、413 个固定装置；CVPR 开放获取页写 37.0%、404 个；项目主页写每个厨房平均 44.9 个固定装置，arXiv 正文写 45.9。2026 年排行榜复现的是 37.57，和 arXiv 一致。**本文引用 arXiv 当前版（37.6 / 413）；CVPR 页可能是较早稿，版本历史未查到（推断）。** 来源：arxiv.org/abs/2502.04144、CVPR 2025 页、hd-epic.github.io/site。

### 4.2 核对二：“同类物体的实例数”（第 6 列）

| 候选 | 实例数 | 来源 |
|---|---|---|
| RoboCasa | 2,509 个物体，153 类；365 版另加 57 类；mug 17 Objaverse + 10 AI 生成 = 27；bowl 18+4 = 22；cup 4+18 = 22；coffee_cup 18+6；glass_cup 24（Lightwheel）| 官方文档 Objects 页；论文 3.1（全文）|
| RoboCasa 的实例划分 | 代码有 `obj_instance_split`：`pretrain` 取除最后 4 个以外的实例（或前一半，取大者），`target`/`test` 取其余。**能直接做“没见过的 mug”测试**；365 排行榜是否启用该划分未查到 | `robocasa/environments/kitchen/kitchen.py`（官方文档）；RSS24 原版论文评测只用未见实例 |
| 固定装置 | 456 个实例，12 类，每类 20–50 个 | 论文 3.1（全文）|
| BEHAVIOR-1K、MS-HAB、DROID、AgiBot | 每类实例数仍未查到（AgiBot 只有 3000+ 物体总数）| — |

### 4.3 RoboCasa365（主候选）

| # | 问题 | 结论 | 标注 |
|---|---|---|---|
| 1 | 任务构成 | 365 = 65 原子 + 300 组合；220 个需要移动操作，145 个不需要。技能：抓放、开关门、开关抽屉、旋钮、拉杆、按键、插入、导航、滑架、开关盖。每个技能各多少任务未查到（附录 E.1 未读到）。排行榜 50 任务 = Atomic-Seen 18 + Composite-Seen 16 + Composite-Unseen 16，具体清单未读到。非抓取交互占比未查到 | 全文；官方文档 |
| 2 | 物体 | 见 4.2。网格来源：Objaverse、Lightwheel、Luma.ai 生成，许可逐源未查到；论文 CC BY 4.0，HF 目标集 apache-2.0 | 官方文档 |
| 3 | 示范字段 | LeRobot v3.0：3 路 RGB（左/右第三人称 + 腕部），256×256，20 fps；本体状态在 parquet；**无深度、无分割**。`extras/` 里有每条的 MuJoCo 模型 `model.xml.gz`、`states.npz`（可重放）、`ep_meta.json`（布局、固定装置、物体）。重放后重渲染深度/分割在 MuJoCo 上可行（推断，未试）。动作：OSC 控制，机械臂 7 维（平移 3 + 旋转 3 + 夹爪 1）+ 底盘/躯干 5 维 = 12 维，20 Hz。组合任务的目标集带逐帧子任务标注（子任务序号、技能名、阶段、语言）| 官方文档；全文 B.2；HF 数据卡 |
| 4 | 训练协议和成本 | 预训练人类集：300 任务 × 100 条 = 30k 条，**482 小时**（文档）；论文正文写 404 小时、摘要写 600+，三处不一致，以文档为准。MimicGen 集 1615 小时。目标集 50 任务 × 500 条 = 25k 条，**193 小时**，HF 数据卡 25,307 条、14.96M 帧、**188 GB**。预训练集 GB 未查到。可按任务下载（`--tasks`）、按划分下载。官方基线：DP 批量 192 训 250k 步，π0 批量 64 训 75k 步，GR00T 批量 128 训 120k 步；卡数和时长未查到。较小的官方协议：原 RoboCasa 的 24 个原子任务（见 6）| 官方文档；全文 3.4 |
| 5 | 评测成本 | 每任务 30 回合，步数上限随任务不同（v1.0.1 起全部放宽 1.5 倍）。仿真 20 Hz，约实时，CPU 物理 + GPU 渲染，可多环境并行。EGL/软件渲染要求未查到 | 全文 G.2、B.1；官方文档 |
| 6 | 和现有代码的距离 | **starVLA 已支持 RoboCasa365（PandaOmron）**：2026-09-05 更新，50 任务本地快照，最好 StarVLA-PI 40.64%（1016/2500），非官方排行榜提交；另有 GR1 桌面版（24 任务，Fourier GR1，Qwen3-VL-4B：OFT 48.8、GR00T 47.8、π 43.9、FAST 39.0，50 回合/任务）。仓库文档里 “RoboCasa-GR1 39.0–48.8” 是 GR1 桌面版，和 365 不能比 | starVLA README 与 docs（官方文档）|
| 6b | 其他论文的 RoboCasa 协议 | AffordanceVLA 62.5→65.0：原 RoboCasa 24 个原子任务，每任务 300 条示范（全文）。CoRAct 75.5：RoboCasa Atomic，含 OOD 物体/纹理/光照/场景五种设置，自述正文与表差 0.1（项目页）。GIFT 61.4、Pri4R 的协议这一轮未核对（未查到）。三者之间、以及和 365 排行榜之间，数字不可直接比 | 论文/项目页 |
| 7 | 理解类标注 | 没有问答、物体定位、场景图。可导出的真值：`ep_meta.json` 的物体和固定装置，`states.npz` 重放得到位姿；组合任务有子任务标注 | 官方文档 |
| 8 | 合成数据 | MimicGen：60 个原子任务，每任务以 100 条人类示范为种子生成 ~10k 条；论文说生成质量参差。换物体实例时如何搬运示范，论文没展开，原 MimicGen 的做法是把种子示范的子任务轨迹变换到新物体位姿的坐标系下（推断，MimicGen 原文未读）。MimicGen 不能生成移动操作 | 全文；RSS24 原版 |

### 4.4 BEHAVIOR-1K（只核可行性）

1. **许可：** 代码 MIT；数据包只许非商业学术研究，加密，只能在 OmniGibson 内使用，**禁止提取数据或逆向——网格不能导出**。机器人资产未加密。来源：`BEHAVIOR-1K/setup.ps1` 许可文本、官方文档 Important Concepts。
2. **硬件：** Ubuntu 22.04+，RAM 32 GB+，**RTX 2070+，显存 8 GB+**；基于 Isaac Sim，安装脚本有无头选项。Isaac Sim 具体版本未查到；A6000 实测未查到。第一名在 RTX 4090 上推理、评测。官方文档、全文。
3. **最小子集：** 10,000 条 / 50 任务 = 每任务约 200 条，全集 ~1.5 TB，平均 ~30 GB/任务（推断）；可按任务、相机、模态部分下载。分任务成功率未查到，只有总分：第一名 q-score 0.26（成功率 0.12），第二 0.25。训练算力：第一名 8×H200 多任务训练 15 天，再按任务组微调约每组一周，一个月约 2 个 epoch，总预算约 1.3 万美元。评测：500 回合单机约 10 天，他们用 20 张 4090 并行不到 2 天。全文 arXiv 2512.06951。
4. **动作空间：** 23 维连续（底盘速度 3 + 躯干 4 + 左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1），30 fps，关节位置。全文。
5. **理解标注：** 每条有语言标注；技能/子任务标注共 270,600 个技能、31 种；物体状态真值、实例分割。基于它的问答或场景图基准未查到。官方文档。
6. **结论：** 在 8×A6000（单卡 48 GB）上，**全协议不可行**：赢家的训练量是 8×H200 一个月，全评测是几十张卡的几天。仅微调 1–3 个任务并评测少量回合大概可行（推断，无实测）。starVLA 有 BEHAVIOR 数据与流水线（2025-12-25），其提交的公开验证 q-score 为 0.002，可行性证据弱。

### 4.5 HD-EPIC

1. **许可和大小：** CC BY-NC 4.0；总 2.3 TB（VRS 1.9 TB、SLAM+注视 349 GB、视频 115.5 GB、音频 27 GB、手掩码 1.95 GB、数字孪生 1.35 GB）；标注约 73 MB。可按类型、按参与者（1–9）、按视频 ID 下载，**只下标注加视频约 120 GB**。
2. **标注：** 手掩码 7.7M（含自动预测）；物体移动 19.9K 条轨迹，带 2D 框、3D 位置、所属固定装置；数字孪生为每个厨房一份 Blender 文件；VQA 26,650 题、30 种题型、7 大类（食谱、食材、营养、细粒度动作、3D 感知、物体运动、注视）。有手腕和手掌位置（Aria MPS），**没有接触点或抓取级标注**（未查到）。
3. **和 RoboCasa365 的重叠物体：** 每厨房平均约 12 个柜子、8 个抽屉、5 个电器；mug、柜门、抽屉、旋钮在叙述里的出现次数未统计（未查到）。
4. **VQA 已发表数字：** Gemini Pro 37.6；Qwen2.5-VL-7B 33.5、32B 36.9（第三方排行榜，微调版 7B 41.6）；Ricoh 技术报告零样本 Qwen2.5-VL-7B 28.66、32B–72B 32.14；CVPR 2026 挑战赛 Qwen3-VL-8B 加推理流水线 67.22（第三方，推理流水线，非微调的基础模型数字）。人类 90.3。
5. 两处出入见 4.1。
6. **EPIC-KITCHENS + VISOR 替代：** EPIC-100：100 小时，45 个厨房，741.2 GiB，Non-Commercial Government Licence；VISOR：27.1 万人工掩码、36 小时，密集掩码 9.9M（另一页写 14.9M，不一致），257 类物体，67K 手-物体关系，商用需邮件申请。官方页。**没有 3D 数字孪生。**

### 4.6 结论（八行）

- **RoboCasa365：** 能用。风险：物体实例划分要自己配（`obj_instance_split`），无深度和分割要重放重渲染，365 的 starVLA 结果只有本地快照。成本：目标集 188 GB，预训练集 GB 未查到；基线 π0 批量 64 × 75k 步；评测 30 回合/任务。
- **BEHAVIOR-1K：** 只作参照。风险：数据加密不能导出网格，8×A6000 跑不动全协议，最好成绩 q-score 0.26。成本：1.5 TB，8×H200 训练一个月，全评测 500 回合单机约 10 天。
- **HD-EPIC：** 能用作理解和 affordance 来源。风险：没有抓取级标注，非商用，和 RoboCasa 的物体重叠未统计。成本：只下视频加标注约 120 GB。
- **RoboCasa365 上最小的一组实验（推断）：** 目标集里取 6–8 个原子任务（含抓放、开抽屉或柜门、旋钮），每任务 500 条示范，用 starVLA-QwenPI 微调，评测用 `obj_instance_split` 留出的 mug/bowl/cup 实例，每任务 30 回合。对照基线用 starVLA 的 50 任务本地快照（StarVLA-PI 40.64%，非官方）和官方 π0.5 的目标集结果；官方 π0.5 的多任务数字（Atomic-Seen 39.6）是预训练场景协议，不能直接比。

### 4.7 第二轮来源

- RoboCasa365 论文 arXiv 2603.04356；文档 robocasa.ai/docs（Datasets、Multi-Task Learning、Objects、Installation）；HF `lerobot/robocasa_target_human_unified`；`robocasa/environments/kitchen/kitchen.py`；RSS24 原版 robocasa.ai/assets/robocasa_rss24.pdf
- starVLA：github.com/starvla/starvla；starvla.github.io/docs/benchmarks/robocasa；HF `StarVLA/Qwen3-VL-OFT-Robocasa`
- AffordanceVLA arXiv 2606.06155 表 4；CoRAct onlyxingxingzyx.github.io/CoRAct
- BEHAVIOR：behavior.stanford.edu 安装页与 Important Concepts；`StanfordVL/BEHAVIOR-1K/setup.ps1`；arXiv 2512.06951；`IliaLarchenko/behavior-1k-solution`；HF `StarVLA/BEHAVIOR-1K`
- HD-EPIC：arxiv.org/abs/2502.04144；hd-epic.github.io/site；`hd-epic/hd-epic-downloader`；data.bris readme；HF 镜像 `WitneyWW/HD-EPIC`；sota2.com 排行榜（第三方）；hd-epic.github.io/assets/reports/2025/HD-EPIC-R3.pdf；arXiv 2605.24500
- EPIC：data.bris.ac.uk EPIC-KITCHENS-100；epic-kitchens.github.io/VISOR
