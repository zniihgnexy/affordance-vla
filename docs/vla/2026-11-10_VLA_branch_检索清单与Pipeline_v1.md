# VLA branch：检索清单与 Pipeline v1

**日期：2026-11-10｜分支：`vla`｜状态：已被 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 取代，保留作历史记录**

> v1 把贡献放在 memory 模块上，主 benchmark 选的是记忆类 benchmark，和 `mem` 分支重叠，也没有把 VLA 训练当成核心。项目负责人在同一天要求改正，现行设计以 v2 为准。v1 的检索清单 S9（投稿会议选择）仍然有效。

这是 VLA branch 的起点文档，回答四个问题：为什么转向、要建什么、先查什么、查完按什么流程做。`main` 上的已有文档是两个方向共享的 brainstorming 基础，本文不修改[项目总控](../项目总控.md)。另一个方向在 `mem` 分支，做 SceneFun3D 类场景中的人类动作记忆，不涉及机器人。两个方向靠分支名区分。

标 **［我的决定］** 的条目是起草时由我选定的默认值，检索后可以改；其余内容已在 2026-11-10 的对话中与项目负责人确认。

## 1. 为什么转向

之前的路线是：先从环境视频（SceneFun3D、EPIC-KITCHENS）建记忆，以后再接 VLA。问题在于这些视频没有机器人动作和指令，也没有闭环评测。这样做出来的记忆既接不进 VLA，也证明不了它对动作有帮助。

VLA branch 把顺序反过来：

1. 先选 VLA 本来就在用的数据和 benchmark，要求有动作、有指令、能闭环评测；
2. 在这些数据上找出"VLA 缺了记忆就会做错"的情形；
3. 围绕这些情形设计记忆，让记忆以 VLA 能用的形式参与动作生成。

## 2. 要建的系统

- **场景：** 桌面操作，桌上物体多，摆放杂乱。
- **4D：** 三维场景加上时间。affordance 不是一张静态图，物体被移动或被挡住时，它也会跟着变。
- **Memory：** 随时间累积、更新的 4D affordance memory，包括被挡住的物体和已经被动过的物体。
- **粗动作预测：** 单独训练的模块。输入是 memory 读出的 affordance、指令和图像，输出"大致怎么动"，包括接触点、接近方向和若干路径点，不是关节命令。
- **VLA：** affordance map、粗动作预测和 vision tower 的视觉特征一起送入 VLA，由 VLA 生成实际执行的动作。
- **目的：** affordance 让动作贴合场景中物体的真实状态，粗动作预测提供前瞻信息，两者结合以减少抓错物体、方向不对、碰撞这类错误。

### 运行时数据流（每个控制步）

```text
相机 RGB(-D) o_t + 本体状态 q_t + 指令 ℓ
        │
 (A) 当前帧 affordance 感知       A_t = f_aff(o_t, ℓ)
        │   oracle 版本：仿真真值；方法版本：模型预测
        │
 (M) 4D affordance memory 更新    M_t = U(M_{t-1}, A_t, 相机位姿, 可见性, 已执行动作)
        │   每个记忆单元：3D 位置或物体部位、affordance 特征、最后观察时间、
        │   状态（可见 / 被挡住 / 已被移动 / 从未观察）、来源帧
        │
 (P) 粗动作预测                    τ̂_t = g(M_t 读出, o_t, ℓ, q_t)
        │   接触点、接近方向、K 个末端路径点（机器人基座坐标）
        │
 (V) VLA                            a_{t:t+H} = π(vision_tower(o_t), ℓ, q_t, enc(M_t), enc(τ̂_t))
        │
  执行动作块 → 得到新观察 → 回到 (A)
```

符号只用来说明接口。具体网络、K、H 和编码方式由检索和实验决定。

## 3. 目标、非目标与反目标

### 目标：仿真闭环对比

主对比有三组：

| 配置 | 内容 |
| --- | --- |
| C0 | 同一个 VLA，不加任何东西 |
| C1 | 加当前帧 affordance，不用 memory |
| C4 | 加 4D affordance memory 和粗动作预测（完整方案） |

**成功标准：** C4 的成功率高于 C0 和 C1，抓错物体、方向不对、碰撞更少，而且优势主要出现在需要记忆的情形里（物体被挡住、被移动之后）。

**补充对照［我的决定］：**

- C2：有 memory，没有粗动作预测；C3：有粗动作预测，没有 memory。两组消融用来区分各部分的贡献。
- oracle 行：使用 affordance 真值或粗动作真值，作为上界单独报告。
- 强对照：同一 backbone 上的通用 memory VLA（例如 RoboMME 中的变体），以及 AFI 式的测试时 affordance 干预。只赢 C0 不足以说明问题。

**评测范围［我的决定］：**

- 主结论只看需要物体或空间记忆的任务，也就是物体被挡住、被移动之后的情形。计数、时序、程序类记忆任务照常报告，但不纳入主结论。
- 主评测使用现有 benchmark 中符合条件的任务子集，以便与已发表的基线对照。如果符合的任务太少，撑不起统计比较，就在 ManiSkill 中另建一组遮挡和移动变体，作为第二套单独报告，不和已发表的基线分数混在一起。
- 学习得到的 affordance 模型可以用主 benchmark 训练集的真值标签训练；测试集的标签只用于评价。

**非 oracle 配置可以读哪些仿真信息（2026-11-10 确认）：**

- **可以用：** 深度、相机内外参、机器人本体状态。第一版还可以用仿真给出的、当前可见物体的分割和物体编号；如果选定的 backbone 或 benchmark 自带这些信号，就直接用。
- **不可以用：** 被挡住物体的位姿、未来状态、测试集标签。
- **另报一行：** 不用仿真分割，改用 SAM 2 一类的跟踪模块。论文中要把对仿真分割的依赖写成一条限制。

### 投稿方向与必做范围（2026-11-10）

- **目标会议：** 首选 ICCV 2027 或 NeurIPS 2027，时间比较灵活。如果 VLA 部分做得完整，也考虑机器人会议；但实验只有仿真，要挑接受纯仿真工作的会议和 track（见 S9）。
- **必做［我的决定］：** C0–C4、oracle 行、"不用仿真分割"那一行，以及至少一个强对照（同一 backbone 上的 memory VLA）。
- **建议做：** 第二套 benchmark（LIBERO-Pro 或 LIBERO-Occ）和 AFI 式对照。顶会审稿通常希望看到不止一个 benchmark。
- **可选：** S7 真实数据集部分。

**粗略时间表［我的决定，粗估］：**

| 时间 | 阶段 |
| --- | --- |
| 2026-11 底 | V0 |
| 2026-12 | V1、V2 |
| 2027-01 | V3 |
| 2027-02 至 2027-04 | V4、V5 |

ICCV 的投稿截止通常在 3 月初，按这个节奏会很紧；NeurIPS 通常在 5 月中，更现实。具体日期以官网为准。

### 非目标

- 真实机器人；
- 从头训练 VLA；
- 人类视频、SceneFun3D、SMPL-X（属于 `mem` 分支）；
- 移动操作和导航；
- 视频世界模型生成。

### 反目标：技术上能跑通，但仍算失败

- 只在 oracle affordance 下有效，却被写成方法有效。
- 测试任务在当前帧就能看到全部信息，memory 的价值无从体现；或者只赢 C0、赢不了 C1。
- 完整方案比对照多用了数据、训练步数或参数，却没有做预算匹配。
- 非 oracle 配置读到了仿真真值，例如被挡住物体的位姿或未来状态。
- 粗动作预测做成了第二个完整 policy，和 VLA 的职责重叠；或者离线指标好看，闭环却没有提升。
- 把 AFI、A4A、AffordanceVLA 等已有工作的机制当作本项目的新贡献。

## 4. 约束

- **设备（2026-11-10 确认）：**
  - 远程 8 卡 A6000 服务器，随时可用，是 V1–V5 的主力。
  - NVIDIA 5060 机器，用于仿真和可视化。
  - Isambard-AI 正在申请。批下来后有 1,000 小时，每次 4 块 GPU，约 4,000 GPU 小时，作为扩展算力。据我了解它是 GH200（ARM/aarch64 CPU），仿真器渲染和 openpi 能否在上面运行要单独核实，所以不放在关键路径上。
  - Mac（M1，16GB）用于阅读、轻量检查和查看结果。
  - 按 openpi 的说明（需核实），π0.5 的 LoRA 微调需要约 22.5GB 显存，全量微调需要 70GB 以上。A6000 每卡 48GB，所以默认用 LoRA；全量微调需要多卡分片。
- **机器人：** 不碰真实机器人。真实数据集只用来离线提取标签或做预训练。
- **环境与呈现：** 用 conda 加 `envs/legacy/requirements_mac_e0.txt`，环境由项目负责人自己创建。结果用预先执行好的分步 notebook 呈现。
- **可复用代码：** `affordance_vla/geometry.py`（反投影、投影、可见性）、`affordance_vla/legacy/memory.py`（`VoxelMemory`、`fuse`、`coverage`）、`affordance_vla/legacy/features.py`（DINOv2）、`affordance_vla/legacy/viz.py`，以及 E0 式的数据检查流程。

## 5. 已知的直接近邻（初步，需检索核实）

来源级别：**S** = 本轮检索摘要（未读全文）；**R** = `main` 上 11-02 VLA 调研已记录；**K** = 我已有的知识（需核对版本和链接）。

| 工作 | 和本方案重合的地方 | 初步差异（待核实） | 级别 |
| --- | --- | --- | --- |
| [AFI：Affordance Field Intervention](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Affordance_Field_Intervention_Enabling_VLAs_to_Escape_Memory_Traps_in_CVPR_2026_paper.html)，CVPR 2026 | 3D 空间 affordance field 生成路径点，用来锚定 π0/π0.5 的动作；在 LIBERO-Pro 和真机上有提升 | 测试时插件，不训练 VLA。它说的 "memory trap" 指 VLA 背诵训练轨迹，不是场景记忆。用 GPT-4o 解析指令。需查清它的 "recent high-affordance regions" 有多少时间维度。**最需要先读** | S |
| [A4A](https://arxiv.org/abs/2609.05892)，2026-09 | "action-oriented 4D affordance" 指与语言相关的、交互相关 3D 点的未来轨迹，接近本方案的粗动作预测；用于多种 VLA 的预训练 | 从人类示范学来；摘要中没有提到 memory 和杂乱场景；需查运行时是否作为输入 | S |
| [AffordanceVLA](https://arxiv.org/abs/2606.06155)，2026（10 月有 v2） | Which2Act / Where2Act（2D affordance map）/ How2Act（3D），用三个专家的 MoT 生成动作，LIBERO 上有结果 | 主要是当前帧，未见时序 memory | S、R |
| AffordVLA（arXiv 2605.17517） | 用功能 teacher 蒸馏到 VLA 特征 | 没有时序 memory | R |
| [VLAff](https://arxiv.org/abs/2608.05215)，2026-08 | affordance token 预测视觉 affordance、抓取 affordance 和轨迹 affordance | 偏 affordance 模型，从第一人称人类视频学；接不接 VLA 需要查 | S |
| [AffordTrajDP](https://arxiv.org/abs/2608.01603)，2026-09 | 从示范库检索 affordance，沿预测位姿传播成随时间变化的 affordance 轨迹，再条件化 diffusion policy | 这里的 memory 是示范库，不是当前场景的记忆 | S |
| [Mem-World](https://arxiv.org/abs/2606.18960)，2026-06 | 4D 腕部视角记忆，锚定在随时间演化的表面元素上，针对遮挡 | 是 action-conditioned 世界模型，不是 affordance，也不是 VLA 输入 | S |
| MolmoAct、HAMSTER | 先生成感知或 2D 轨迹计划，再生成动作 | 没有跨时间的 affordance memory | K |
| SOMA、GeoMem-VLA、MemoryVLA、MEM、NativeMEM、EvoMem-VLA | memory 接入 VLA | 不以 affordance 为记忆内容，或没有粗动作计划 | R |
| [RoboMME](https://arxiv.org/abs/2603.04639) 中的 14 个 memory 变体 | 在 π0.5 上系统比较符号、感知和循环三类记忆 | 是 benchmark 的对照组，不是 affordance memory；正好可以当强对照 | S |

**初步判断：** "affordance → 路径点 → VLA"和"memory → VLA"各自都已有直接先例。本轮还没检索到这样的工作：把能处理遮挡和物体移动的 4D affordance memory 与粗动作预测一起作为 VLA 输入，并在需要记忆的杂乱桌面任务上验证。这是候选空白，但只基于本轮 5 次检索的摘要，不能当作已确认。

## 6. 检索清单

优先级：P0 决定能否开工，P1 决定怎么做或投到哪里，P2 用于辅助或写论文。检索词用英文，适合 arXiv、Google Scholar 和 Semantic Scholar。每组最后都要对已知起点做一次**正向引用检索**（谁引用了它），以补上 2026 年 10–11 月的新工作。

### S1（P0）需要记忆的桌面操作 benchmark → 决定 D1

- **要回答：** 哪些 benchmark 的桌面任务确实依赖记忆，同时物体多、有遮挡、有物体被移动？是否提供深度、相机内外参、分割和物体位姿（4D memory 和标签都需要）？是否已有 VLA 基线分数和 checkpoint？能否在无图形界面的环境下运行？
- **检索词：** `memory-dependent robotic manipulation benchmark` · `VLA memory benchmark occlusion object permanence` · `cluttered tabletop manipulation benchmark vision-language-action` · `LIBERO occlusion` · `ManiSkill memory tasks VLA` · `long-horizon manipulation benchmark memory subtasks`
- **已知起点：**
  - [RoboMME](https://arxiv.org/abs/2603.04639)（ICML 2026 oral）：ManiSkill，Franka；16 个任务，分为时间、空间、物体、程序四类记忆；1,600 条示范；在 π0.5 上做了 14 个 memory 变体（S）
  - [MIKASA-Robo-VLA](https://arxiv.org/abs/2610.00604)（2026-09）：90 个语言条件任务，22,500 条 oracle 轨迹，10 类记忆；π0.5 参考基线在 14 个任务上平均成功率约 0.211（S）
  - [RoboMemArena](https://arxiv.org/abs/2605.10921)：26 个任务，平均超过 1,000 步，68.9% 的子任务依赖记忆，包括多物体遮挡和计数（S）
  - [MEMOBench](https://arxiv.org/abs/2609.07047)：过程级记忆评测（S）
  - RMBench：两个来源给出的日期不一致，需核实（S）
  - LIBERO-Occ（2026-06）和 "Recovering the View"（2026-09）：遮挡与主动视角，来自 awesome list，需核实（S）
  - 更早的工作：MIKASA-Robo、MemoryBench（SAM2Act，基于 RLBench）、LIBERO-Long、LIBERO-Pro、LIBERO-Plus（K）
  - 偏杂乱、但记忆不是重点：RoboTwin 2.0、VLABench、RoboCasa、Colosseum、ManiSkill3 自定义场景（K）
- **要提取：** 按本节末尾的对比表模板填写。

### S2（P0）直接重合检查 → 决定定位与贡献表述

- **要回答：** 第 5 节每项工作：affordance 来自当前帧还是累积而来？是否跨时间更新，如何处理遮挡和物体移动？粗动作是学出来的，还是规则或 LLM 生成的？在 VLA 的哪个位置注入？用了哪个 benchmark？代码是否开源？
- **检索词：** `affordance field VLA waypoints` · `4D affordance vision-language-action` · `affordance memory VLA occlusion` · `trajectory prior conditioning VLA` · `plan-then-act VLA visual trace` · `object permanence memory affordance manipulation policy`
- **已知起点：** 第 5 节全部条目，加上 AFI、A4A、RoboMME 的正向引用。
- **产出：** 一张近邻对比表，再用一段话说明"本方案和最接近的三篇工作分别差在哪里"。

### S3（P0）VLA backbone 和可接入点 → 决定 D2、D5

- **要回答：**
  - 哪个 backbone 有开放权重、微调代码，以及在 D1 候选 benchmark 上的 checkpoint？
  - 加额外输入有几种方式：画在图像上、加 prefix token、作为动作专家的条件。各自需要改多少代码？
  - 在 GH200 上做 LoRA 或全量微调需要多少显存和时间？推理延迟是多少？
- **检索词：** `openpi fine-tune custom observation keys` · `pi0.5 LoRA fine-tuning memory` · `OpenVLA-OFT extra input tokens` · `ManiSkill openpi` · `LeRobot SmolVLA ManiSkill` · `VLA aarch64 GH200`
- **已知起点：** π0 / π0.5（openpi；RoboMME 和 AFI 都用它们）、OpenVLA-OFT、SmolVLA（LeRobot）、GR00T N1.x（R）

### S4（P1）从仿真数据自动生成 affordance 和粗动作标签 → 决定 D3、D4

- **要回答：** 已有工作如何从示范中得到接触点和 affordance（夹爪闭合时刻、与物体的接触、分割）？如何得到粗轨迹标签（末端路径降采样、关键位姿）？用 2D 轨迹还是 3D 路径点？
- **检索词：** `affordance labels from robot demonstrations contact point` · `keyframe extraction gripper state change manipulation` · `visual trace annotation robot dataset` · `end-effector trajectory projected 2D trace VLA` · `unsupervised affordance distillation manipulation`
- **已知起点：** RoboPoint、A0、HAMSTER 和 MolmoAct 的轨迹标注方式、RT-Trajectory、LLARVA、Magma（trace-of-mark）、PerAct 式关键帧（K）；A4A 的点轨迹；[UAD](https://arxiv.org/abs/2506.09284)（S）

### S5（P1）4D affordance memory 的表示与更新 → 决定 D6

- **要回答：** 随时间变化的场景 affordance 用什么表示（体素、点、高斯、物体槽、场景图）？怎样区分"被挡住"和"被移动"（物体恒存性）？跟踪怎么做？能否跟上控制频率？
- **检索词：** `dynamic scene memory robot manipulation object permanence` · `4D scene representation manipulation occlusion` · `object-centric memory tracking occluded objects robot` · `spatial affordance field update over time`
- **已知起点：** [Mem-World](https://arxiv.org/abs/2606.18960)（S）；SOMA、ChronoGraph（R）；D3Fields、ManiGaussian、SAM 2、CoTracker、FoundationPose（K）；`main` 分支的 `VoxelMemory`

### S6（P1）中间表示注入 VLA 的方式 → 决定 D5

- **要回答：** 三种注入方式（画在图像上、加 token、作为动作专家条件）在少量数据下哪种更有效？新增输入会不会削弱 VLA 原有的能力？
- **检索词：** `visual prompting trajectory VLA` · `conditioning action expert on waypoints` · `VLA additional modality injection fine-tuning`
- **已知起点：** HAMSTER、TraceVLA、RT-Trajectory、MolmoAct、ECoT、CoT-VLA、DreamVLA、π0.5 的子任务预测（K）；AFI 的路径点锚定；[Hermite 曲线轨迹先验](https://arxiv.org/abs/2608.01265)（S，方法未核实）

### S7（P2）真实 VLA 数据集 → 离线标签和预训练

- **要回答：** 哪些真实数据集带深度和相机标定、物体较多，适合离线提取 affordance 和轨迹标签？许可和体积如何？
- **检索词：** `robot manipulation dataset depth camera calibration cluttered` · `DROID depth extrinsics` · `large-scale real robot dataset RGB-D`
- **已知起点：** Open X-Embodiment、DROID、BridgeData V2（R）；AgiBot World、RoboMIND、LeRobot 社区数据集（K）

### S8（P2）评测协议与错误分类 → 决定指标

- **要回答：** 已有论文怎样给失败分类（抓错物体、抓取失败、碰撞、超时）？"需要记忆的子集"怎么定义？每个配置需要多少次试验、多少个随机种子，差异才算可信？
- **已知起点：** RoboMME、MIKASA-Robo-VLA、RoboMemArena 的失败分析；LIBERO-Pro / LIBERO-Plus 的扰动协议。

### S9（P1）投稿会议选择 → 决定投稿目标和必做范围

- **要回答：** 哪些会议和 track 接受只有仿真实验的 VLA 或记忆工作？2027 年各会议的截止日期、页数限制，以及是否要求真实机器人实验？
- **候选：**
  - 视觉或机器学习方向：ICCV 2027、NeurIPS 2027（主会；如果自建了任务集，也可以考虑 Datasets & Benchmarks track）、ICLR 2028。
  - 机器人方向：CoRL 2027、RSS 2027、ICRA 2028、IROS 2027、RA-L（滚动投稿）。
- **我的理解（需核实）：** CoRL 和 RSS 的审稿一般期望有真实机器人实验，纯仿真工作比较吃亏；ICRA、IROS、RA-L 对纯仿真更宽容；NeurIPS、ICLR 上纯仿真的 VLA 工作很常见。可以参考同类工作的发表处：RoboMME 在 ICML，AFI 在 CVPR。
- **检索词：** `CoRL 2027 call for papers` · `ICRA 2028 submission deadline` · `IROS 2027 call for papers` · `NeurIPS 2027 datasets and benchmarks track` · `ICCV 2027 submission deadline` · `simulation-only robot learning paper acceptance CoRL`

### 对比表模板（S1、S7 共用）

| 字段 | 说明 |
| --- | --- |
| 平台 / 仿真器 | ManiSkill、robosuite、CoppeliaSim 等 |
| 机器人与动作空间 | 末端位姿还是关节角，绝对还是相对，控制频率 |
| 相机 | 第三人称和腕部，数量和分辨率 |
| 4D 所需信息 | 深度、内外参、分割、物体位姿：直接提供，还是需要回放渲染 |
| 杂乱程度 | 每个场景的物体数量，是否有干扰物 |
| 记忆依赖 | 遮挡、移动、计数等类型，以及所占比例 |
| 规模 | 示范数量、平均步数、下载体积 |
| 闭环协议 | 试验次数、随机种子、已有基线分数 |
| 现成 checkpoint | π0.5、OpenVLA-OFT、SmolVLA 等 |
| 运行条件 | Mac 能否运行；Isambard（aarch64、无显示器）能否运行 |
| 许可 | 数据和代码 |

## 7. 新 pipeline（VLA branch v1）

| 阶段 | 内容 | 设备 | 产出 | 进入下一阶段的条件 |
| --- | --- | --- | --- | --- |
| V0 检索与选型 | 完成 S1–S3（P0）和 S9，确定 D1–D3；在 A6000 服务器上试跑候选仿真器的无显示器渲染和 backbone 推理；估算 V2–V5 需要的 GPU 小时 | Mac、A6000 服务器 | benchmark 对比表、近邻对比表、会议选择表、选型说明、渲染检查结果、算力估算 | 选定 1 个主 benchmark 和 1 个 backbone：有记忆依赖任务，能取得深度、分割和位姿，在 A6000 上能渲染，算力估算在 8 卡 A6000 的可用范围内 |
| V1 数据检查（对应 E0） | 下载少量示范；回放或渲染深度和分割，验证相机参数、坐标系和投影；自动生成 affordance 和粗动作标签并可视化 | A6000 服务器或 5060 机器；Mac 上看 notebook | 分步 notebook | 投影和标签通过目视与数值检查 |
| V2 基线复现（C0） | 用选定的 backbone 在主 benchmark 上微调并闭环评测，与论文报告值对照 | A6000 服务器 | 基线分数、失败分类 | 分数与报告值在合理范围内 |
| V3 memory 与粗动作预测（离线） | 先用 oracle affordance 建 4D memory（沿用 `VoxelMemory` 的思路，加入时间和状态），再换成预测的 affordance；训练粗动作预测模块 g；另做一个不用仿真分割、用跟踪模块的版本 | A6000 服务器 | memory 状态准确率、接触点误差、路径点误差 | 在需要记忆的片段上，memory 读出明显优于当前帧 |
| V4 注入 VLA | 先用最便宜的方式（画在图像上），再比较 token 注入和动作专家条件；LoRA 微调 | A6000 服务器（Isambard 批下来后用来扩展） | 训练好的 C1–C4 | — |
| V5 闭环对比 | C0–C4、oracle 行、"不用仿真分割"行和强对照；数据相同、训练步数相同、参数匹配 | A6000 服务器（Isambard 批下来后用来扩展） | 总成功率、记忆子集成功率、错误分类、延迟 | 达到第 3 节的成功标准 |

**检索后的继续或调整规则［我的决定］：** 如果 S2 发现已有工作完整覆盖本方案（随时间更新的 affordance memory 加粗动作预测，一起作为 VLA 输入），就暂停 V1 及之后的阶段，整理两三个备选方向（收窄贡献，或换一个切入角度），交给项目负责人决定。部分重合时继续推进，但要在近邻对比表里写清楚差异。

### 检索后要定的六个决定（默认值都是［我的决定］）

| 决定 | 默认值与理由 | 备选 |
| --- | --- | --- |
| D1 主 benchmark | 基于 ManiSkill 的记忆 benchmark（RoboMME 或 MIKASA-Robo-VLA）。理由：任务确实依赖记忆，是 Franka 桌面场景，有 π0.5 基线，仿真能提供深度、分割和位姿。LIBERO-Pro 或 LIBERO-Occ 作为第二套，便于和 AFI、AffordanceVLA 对齐 | 杂乱程度不够时，在 ManiSkill 里加干扰物，或换 RoboTwin 2.0 |
| D2 backbone | π0.5（openpi）。RoboMME 和 AFI 都用 π0 / π0.5，可以直接比较；在 A6000 上用 LoRA 微调 | 算力不够时用 SmolVLA；OpenVLA-OFT |
| D3 affordance 表示 | 物体部位级的 3D 点或体素加 affordance 分数；先用仿真真值做 oracle，再换成预测 | 每个相机一张 2D 热图 |
| D4 粗动作表示 | 接触点、接近方向和 K 个 3D 末端路径点，也可以投影成 2D 轨迹 | 关键位姿序列；只用 2D 轨迹 |
| D5 注入方式 | 先画在图像上（不改模型结构），再试额外 token 和动作专家条件 | — |
| D6 memory 形式 | 显式的体素或物体槽记忆，带时间戳和四种状态；可学习的隐式记忆放到之后 | 物体级场景图；隐式 token 记忆 |

## 8. 留给执行者决定的事

具体网络结构、K、H、训练超参数；标签自动生成的阈值和规则；每个配置的试验次数和随机种子数（参考 S8 的结论）；notebook 的拆分方式。

## 9. 与 `main` 已有材料的关系

- 11-02 的 VLA 调研（memory VLA、affordance VLA 列表）直接作为 S2 的起点。
- 11-05 的环境记忆设计中"affordance 是记忆的一层，保留观察有效性和来源"的思路继续沿用，但数据不再用 SceneFun3D。
- `affordance_vla/` 的几何与体素记忆代码在 V1、V3 中复用；SceneFun3D 加载器在本分支不用。

## 10. 本轮检索记录

2026-11-10 共做 5 次网页检索，只读了检索摘要，没有读论文全文：

1. `memory-dependent robot manipulation benchmark VLA occlusion 2026 arXiv`
2. `affordance map trajectory plan conditioned vision-language-action model 2026 arXiv`
3. `4D affordance memory robot manipulation cluttered tabletop dynamic scene 2026`
4. `RoboMME benchmark memory robotic generalist policies simulator ManiSkill tasks π0.5`
5. `"Affordance Field Intervention" VLA memory traps CVPR 2026`

数字和日期都来自摘要或二手聚合页，使用前要在 arXiv 或会议页面核对。标 K 的条目来自我已有的知识，需补链接、核对版本。
