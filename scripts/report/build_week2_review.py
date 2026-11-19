"""Build report/2026-11-20_week2_review.ipynb, the review of week 2 (16-22 Nov 2026; cells only, execute it after).

Usage on the server (starvla env, from report/):
    python ../scripts/report/build_week2_review.py 2026-11-20_week2_review.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-20_week2_review.ipynb
Reads finished runs only: runs/libero_pro/{released,predicted_g1_hold_s10,full_*_s9_seed*}, the replayed
demonstrations, and the folders of the jobs started this week (their progress is read when the notebook runs).
Writes its videos to report/media/week2/ (tracked). Nothing runs on a GPU. The text is in Chinese for the
project lead.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# VLA 方向周报 · 第二周：2026-11-16 至 11-22

**仓库：** `affordance-vla` · **写于：** 2026-11-20（周四；本周还没结束，进行中的任务在第 4 节实时读取）

**这份 notebook 的顺序按优先级：**

1. 真实 demo 的图像和视频；
2. 现行 pipeline（v3）的三张图，以及逐块的详细解释；
3. 本周的实验结果（E0）；
4. 进行中的任务；
5. 时间线和决定；
6. 下一步。

**说明：**

- **数字从哪来：** 表格和图都是从服务器上已经跑完的结果里读出来的，执行 notebook 时现算。
- **ORACLE：** 指仿真里的真值；"真值计划"是上限，不是方法的结果。
- **状态标记：** 【已实现】表示代码在跑、有结果；【v3 计划】表示设计已确认，还没实现。
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.sim.rollout_log import load_rollout_log
from report import grasp_orientation_helpers as G
from report import vla_pipeline_demo_helpers as D
from report import week2_helpers as K

set_matplotlib_formats("jpeg")
RUNS = ROOT / "runs"
PRO = RUNS / "libero_pro"
MEDIA = ROOT / "report" / "media" / "week2"   # tracked in git
MEDIA.mkdir(parents=True, exist_ok=True)
FIGURES = ROOT / "report" / "media" / "pipeline_overview"
OURS = PRO / "predicted_g1_hold_s10"   # grid plan head, plans kept, action head with the labels of 15 Nov
RELEASED = PRO / "released"            # the released checkpoint, no plan
""")

md("""
## 0 · 一页总结

**本周主线：**

- **定下 pipeline v3。** 核心是一张带方向的交互 affordance map，它要给出三样东西：
  - 认对哪个物体；
  - 在哪里接触；
  - 夹爪朝哪个方向。

  计划从这张 map 解码，动作头被训练成服从计划。11-19 确认，同一天晚上修订成以交互 map 为核心。
- **第一个实验 E0 做完了。** LIBERO-Object 上，失败的抓取不是朝向错了，而是位置偏了：夹爪离物体还远就闭合了。按事先约定的规则：
  - 方向部分改到有把手物体的 LIBERO-10 上验证；
  - LIBERO-Object 上先做 map 的位置部分。

**进行中：**

- E1：语义诊断，不训练；
- L：交互 map 的标签，示范正在重新回放，补记物体朝向；
- 为方向部分准备 LIBERO-10：下载示范、回放，并跑官方权重的基线。

**其他：**

- 仓库按研究代码的要求整理过一遍（`AGENTS.md`）；
- Mac 上可以在 MuJoCo 里打开场景、回放服务器上的测试；
- 读了 SemanticVLA；
- 给记忆系统写了一份"带方向的交互 affordance map"总结页。

**现在最好的数字**（第一周的结果，本周没有新的闭环训练）：LIBERO-PRO Object，按论文协议（每个任务 50 个起始状态，3 个种子）。

| | 位置扰动 | 任务扰动 |
| --- | --- | --- |
| 官方权重 | 0.000 | 0.090 |
| 我们：预测的计划（不用真值） | 0.660 | 0.681 |
| 真值计划（ORACLE，上限） | 0.920 | 0.753 |
""")

md("""
## 1 · 真实 demo：图像和视频

下面所有画面都来自服务器上的真实闭环日志，没有任何合成。"我们"指现在最好的配置：

- 图像格子计划头 `g1`；
- 计划保持到夹爪动作完成；
- 用 15 日标签训练的动作头 `s10`；
- 每个任务 20 个起始状态，种子 7。

### 1.1 训练时和测试时的场景

只在标准 LIBERO-Object 上训练。测试用 LIBERO-PRO，零样本：

- **位置扰动：** 物体挪了位置；
- **任务扰动：** 同一个场景，指令点名的是另一个物体。

橙色轮廓是指令点名的物体，由日志里的分割图画出。
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for axis, (name, title) in zip(axes, [("scene_train", "training: LIBERO-Object (salad dressing named)"),
                                       ("scene_swap", "test: position perturbation (it moved)"),
                                       ("scene_task", "test: task perturbation (tomato sauce named)")]):
    axis.imshow(plt.imread(FIGURES / f"{name}.png"))
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
fig.tight_layout()
""")

md("""
### 1.2 视频：同一个位置扰动场景，官方权重和我们

沙拉酱被挪了位置。官方权重伸向沙拉酱在训练时常在的地方；我们的计划头在画面里找到它的新位置，动作头跟着计划走。
""")

code("""
TASK = 2   # the salad dressing task
pair = [load_rollout_log(RELEASED / "libero_object_swap" / f"task{TASK}_ep0.npz"),
        load_rollout_log(OURS / "libero_object_swap" / f"task{TASK}_ep0.npz")]
for log, name in zip(pair, ["released", "ours"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released checkpoint", "ours: map plan kept + s10"], MEDIA / "swap_released_vs_ours.mp4")
Video(str(clip), embed=True, width=760)
""")

md("""
### 1.3 视频：任务扰动，同一个场景点名番茄酱

场景和训练时的沙拉酱任务一样，但指令要的是番茄酱。
""")

code("""
pair = [load_rollout_log(RELEASED / "libero_object_task" / f"task{TASK}_ep0.npz"),
        load_rollout_log(OURS / "libero_object_task" / f"task{TASK}_ep0.npz")]
for log, name in zip(pair, ["released", "ours"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released checkpoint", "ours: map plan kept + s10"], MEDIA / "task_released_vs_ours.mp4")
Video(str(clip), embed=True, width=760)
""")

md("""
### 1.4 方法内部：计划画在画面上

**怎么看：**

- **左边是智能体视角。** 白色细线是 VLM 的 8 × 8 个图像格子；橙色格子和十字是这一步交给动作头的计划，也就是计划头预测的接触点，保持到夹爪完成抓取。
- **右边是腕部视角。**

下面的图是视频里的几个关键时刻：第一次查询、夹爪开始闭合、0.6 秒后、最后一步。
""")

code("""
log = load_rollout_log(OURS / "libero_object_swap" / f"task{TASK}_ep0.npz")
clip = K.plan_video(log, MEDIA / "ours_plan_overlay.mp4")
display(Video(str(clip), embed=True, width=760))
K.plan_strip(log);
""")

md("""
### 1.5 一个失败的例子：位置偏了（E0 的发现）

这是一个失败的回合：夹爪第一次在点名物体附近闭合，却没有抓住。从腕部视角能看到，手指合在物体上方或旁边，没有包住它。E0 统计的是同一件事：失败时，夹爪闭合时离物体中心的中位数是 4.7 cm，成功时是 2.0 cm（第 3 节）。
""")

code("""
failed = K.first_failure(OURS, ["libero_object_swap", "libero_object_task"])
if failed is None:
    print("no failed rollout with images and a missed first grasp in this run")
else:
    print(failed.path.parent.name, failed.path.name, "-", failed.meta["instruction"])
    clip = K.plan_video(failed, MEDIA / "ours_failure.mp4")
    display(Video(str(clip), embed=True, width=760))
    K.plan_strip(failed);
""")

md("""
## 2 · 现行 pipeline：v3，逐块解释

完整设计在 [`docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md`](../docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md)。图里紫色虚线框是 v3 新增、还没实现的部分，其余部分现在都在跑。

### 2.1 测试时：从画面到动作
""")

code("""
Image(filename=str(FIGURES / "v3_model.png"), width=1000)
""")

md("""
**B1 输入【已实现】**

- 智能体视角和腕部视角各一张 224×224 的图；
- 指令文本；
- 夹爪当前的位置（来自仿真，用来算"离目标还有多远"）。

**B2 VLM【已实现，冻结】**

- Qwen3-VL-4B，36 层，宽度 2560。
- 每张图被切成 8×8 = 64 个格子 token，每个格子约 28×28 像素。格子 token 排在前面，指令 token 跟在后面。
- 它输出每一层所有 token 的隐藏状态，交互 map 头和动作头都读这些状态。
- 诊断和消融时 VLM 冻结；主对比时全量微调。

**B3 交互 map 头【第一版已实现；v3 扩展】**

- **已实现的第一版（8×8 格子头）：**
  - 36 层隐藏状态按可学的权重加权平均，投影到 512 维；
  - 前面加一个汇总 token，过 2 层双向 Transformer。在 VLM 里格子 token 排在指令前面，看不到指令；在这里，格子能"看到"指令；
  - 每个格子输出交互分数、格内偏移（x、y）和深度；
  - 全局输出下一个夹爪事件（抓 / 放 / 无）和接近方向。
- **v3 计划的扩展：**
  - 分辨率更细：先用 VLM 视觉编码器在合并之前的特征，每边 16 块，不加新模型；
  - 每个点再输出方向场，也就是夹爪闭合方向。用 cos2θ、sin2θ 表示，转 180° 算同一方向。

**B4 语义先验【v3 计划；E1 正在测】**

- 取指令 token 和 64 个格子 token，算余弦相似度，在指令方向上合并成一张 8×8 的语义图。合并用取平均还是取最大值，由 E1 不训练、直接比较后选定。
- 语义图乘一个从 0 开始学的系数，加到交互分数上。系数从 0 开始，所以刚开始训练时不会变差。
- **为什么指令 token 有用：** 它排在图像后面，隐藏状态已经"看过"画面，含有"指的是哪个物体"的信息。

**B5 解码【位置已实现；朝向 v3】**

1. 取分数最高的格子：像素位置 = (格子编号 + 格内偏移) × 28，加上深度，用相机内外参换算成世界坐标下的 3D 接触点。
2. 【v3】接近方向作为夹爪的 z 轴；峰值处的闭合方向从图像抬到 3D 后，作为 y 轴。

**B6 计划向量【10 维已实现；v3 为 16 维】**

- **现在的 10 个数：**
  - reach / 10 cm，3 个数，粗略的距离；
  - tanh(reach / 2 cm)，3 个数，最后几厘米的精细距离；
  - 接近方向，3 个数；
  - 事件码，1 个数：抓 +1，放 −1，无 0。

  其中 reach = 接触点 − 夹爪当前位置。
- **v3 加 6 个数：** 目标朝向相对当前夹爪朝向的旋转，用 6D 表示。

**B7 计划保持【已实现】**

- 预测出的接触点保持不动，直到夹爪执行完这次事件：抓的计划要连续 5 步发出"闭合"，放的计划要连续 5 步发出"张开"。之后再预测新计划。
- 保持期间，每次查询都会按夹爪的新位置重算计划向量。

**B8 计划编码器【已实现】**

- 3 层 MLP，最后一层初始化为 0。
- 输出加到动作头的每个动作 token 上。官方动作头里动作 token 之间不互相注意，放在旁边的额外 token 读不到，所以只能直接加进去。

**B9 动作头【已实现】**

- starVLA 的 DiT：8 个动作 token，用交叉注意力读 VLM 的隐藏状态。
- 从噪声出发，用 flow matching 走 4 步，得到 8 步 × 7 维的动作：Δx, Δy, Δz, Δroll, Δpitch, Δyaw, 夹爪。
- 训练的部分：DiT 的 LoRA（rank 16）和 B8。

**B10 执行【已实现】**

- robosuite 的 OSC 控制器把末端位姿增量换成关节力矩，逆运动学由它负责。
- 20 Hz 下 8 步是 0.4 秒。执行完拿新画面，回到 B1。

### 2.2 训练
""")

code("""
Image(filename=str(FIGURES / "v3_training.png"), width=1000)
""")

md("""
**T1 交互 map 头的训练**

- **数据：** `dataset10`，即 LIBERO-Object 10 个任务的示范回放，每一帧都是一个样本。
- **已实现的标签：**
  - 夹爪指令从"张开"切到"闭合"（或者反过来）的那一刻，末端位置就是接触点；
  - 投影到画面，得到格子、偏移、深度；
  - 再加上事件类别和接近方向。
- **已实现的损失：** 交叉熵（格子）+ 10 × Huber（偏移）+ 10 × Huber（深度）+ 交叉熵（事件）+ 0.5 ×（1 − cos）（接近方向）。
- **v3 的标签：**
  - **L1：** 抓取那一刻的夹爪位姿，加上物体分割图，得到"接触区域 + 闭合方向"的逐像素标签；
  - **L2：** 同一类物体的所有抓取换到物体自己的坐标系里叠加，得到这类物体的标准 map，再按物体位姿投影到每一帧、每个物体上；
  - 物体位姿只用来生成标签和做评测，测试时不输入模型。为此，本周给日志加了物体朝向，正在重新回放示范。
- **v3 的损失：** 交互分数对"多个可行抓法"组成的标签分布算交叉熵；方向场只在接触区域内，对 cos2θ、sin2θ 算 L2。

**T2 动作头的训练：反事实监督，本工作的核心【已实现】**

- 每一步 8 帧 × 2 次噪声。画面不变，把计划改成下面三种之一，动作标签按挪过的计划改写：
  - **挪位置：** 18 cm 以内；
  - **换成另一个物体：** 前段以起点为中心旋转缩放，末段用该物体自己示范里的接近方式；
  - **换释放点。**
- 用 flow matching 损失只更新计划编码器和 DiT 的 LoRA。
- 【v3】加上"转朝向"，动作的旋转一起改写。E0 之后，这一项改到 LIBERO-10 上做。

### 2.3 泛化和诊断
""")

code("""
Image(filename=str(FIGURES / "v3_generalization.png"), width=1000)
""")

md("""
**V1 离线首帧检查**（便宜，每次改了 map 头都先做）：在 LIBERO-PRO 每个起始状态的第一帧上看 map 的输出。

- **认对物体率：** 峰值落在指令点名的物体分割区域里的比例；
- **位置误差：** 沿视线方向和垂直视线方向分开算；
- **方向误差：** 单位是度，按 180° 对称取较小的。

**V2 闭环：**

- 位置扰动、任务扰动，加测语言扰动（`libero_object_lan`）；
- 朝向的主要测试在 LIBERO-10（有杯子、摩卡壶的把手）；
- 开发阶段每个任务 20 个起始状态，正式报告用 50 个 × 3 个种子。

**V3 失败归类：** 每个失败的回合归到一个要改的模块。

| 失败 | 要改的模块 |
| --- | --- |
| 认错物体 | 语义先验 |
| 位置偏 | 交互分数、深度、更细的特征 |
| 方向错 | 方向场 + 交互标签 |
| 没跟上计划 | 动作头 + 反事实标签 |
| 抓到后失败 | 释放阶段的标签 |

**实验顺序和通过标准：** 见 v3 文档第 5 节。E0、E1、L 同时做，然后依次是：

- E2 语义先验；
- E3 朝向上界（改到 LIBERO-10）；
- E4 交互 map 头；
- E5 用 map 解码出的计划做闭环；
- E6 基线：B、D、E、F 四个；
- E7 推广。
""")

md("""
## 3 · 本周结果：E0，失败的抓取是不是朝向错了

**问题：** 失败抓取的夹爪朝向，是不是比抓住的更偏离示范？

**事先约定的通过线：** 失败比抓住多偏至少 10°。

**数据：** 论文协议的全部闭环，两种配置 × 3 个种子，每种配置 3000 个回合。

完整的 notebook：[`2026-11-20_grasp_orientation.ipynb`](2026-11-20_grasp_orientation.ipynb)。
""")

code("""
demos = G.demo_grasps(sorted((RUNS / "plan_following" / "demos").glob("*.npz")))
reference = G.demo_rotations(demos)
configs = {"图像格子计划（g1 + s9）": "full_g1_hold_s9_seed{}", "真值计划（s9，ORACLE）": "full_oracle_s9_seed{}"}
for name, pattern in configs.items():
    attempts = pd.concat([G.rollout_attempts(PRO / pattern.format(seed), ["libero_object_swap", "libero_object_task"],
                                             reference) for seed in (7, 8, 9)], ignore_index=True)
    gap, passed = G.verdict(attempts)
    print(f"{name}：失败比抓住多偏 {gap:.1f}° → {'通过' if passed else '不通过'}（通过线 10°）")
    display(G.group_table(attempts)[["episodes", "angle_median", "yaw_median", "tilt_median"]])
    display(G.distance_table(attempts))
""")

md("""
**结论：**

- **E0 不通过。** 失败的抓取和示范的朝向几乎一样（只多偏 1–2°）；示范本身也几乎只有一种抓法，都是从上往下。
- **差别在位置。** 夹爪闭合时离物体中心的距离：抓住且成功时中位数约 2 cm，没抓住时约 4.7 cm。这和 1.5 节的画面一致。
- **按约定的规则：**
  - 方向部分在 LIBERO-Object 上暂停，改到 LIBERO-10 去验证；
  - LIBERO-Object 上先做交互 map 的位置部分：分数、深度、更细的特征。
""")

md("""
## 4 · 进行中的任务（执行时实时读取）

- **L：** LIBERO-Object 示范重新回放，补记物体朝向。只用 CPU。
- **LIBERO-10：** 先下载示范的状态和动作；等 Object 的回放结束后再回放，这样服务器上同时跑的仿真器不超过约 50 个。
- **LIBERO-10 基线：** 官方权重在 `libero_10_swap` 和 `libero_10_task` 上，每个任务 20 个起始状态，走 GPU 队列。
- **E1：** 语义诊断，走 GPU 队列，不训练。
""")

code("""
display(K.progress_table(RUNS.resolve().parent))   # runs/ links into the NAS work folder
probe = K.probe_table(RUNS / "semantic_probe" / "e1" / "summary.json")
if probe is None:
    print("E1 还没出结果。")
else:
    display(probe)
""")

md("""
## 5 · 本周时间线和决定

| 日期 | 做了什么 | 提交 |
| --- | --- | --- |
| 11-16 晚 | 第一周周报；项目负责人暂停实验，做深度调研 | `ddf3ff2`、`c75460f` |
| 11-17 | 仓库按研究代码要求整理：按领域分包，每个文件不超过 500 行，实验索引写在 `EXPERIMENTS.md`；Mac 上的 MuJoCo 本机仿真（打开场景、脚本抓取、回放服务器上的测试） | `b1b09ad` 至 `f593260` |
| 11-18 | 逐个问答理解 pipeline。发现 affordance 只给接触点、没有朝向，提出"计划里加夹爪朝向"的方案，画了第一版图 | `5d6bb35` 里的 `pose_plan_pipeline.png` |
| 11-19 | 读 SemanticVLA；**确认 v3 的 6 条**，画三张图。晚上**修订：以带方向的交互 affordance map 为核心**（5 条） | `5d6bb35`、`1245ebb` |
| 11-20 | **E0 完成**（不通过，失败的原因是位置）；日志加物体朝向（逐字节核对过）；L 回放、LIBERO-10 下载和基线、E1 开始；为记忆系统写方向性 map 总结页 | `ce7b263`、`2d447e0` |

**本周的决定**（原文在 [`docs/vla/README.md`](../docs/vla/README.md)）：

- **11-19，6 条：**
  1. v3 取代 v2 的方法部分；
  2. 按 E0–E7 的顺序和通过标准做实验；
  3. 基线定为 B、D、E、F 四个；
  4. 测试加语言扰动；
  5. LIBERO-Spatial 挪到 E7；
  6. 恢复实验。
- **11-19 晚，修订 5 条：**
  1. 以交互 map 为核心；
  2. 标签用 L1 + L2，物体位姿只用来生成标签和评测；
  3. 第一版做 2D map；
  4. 朝向的主要测试加 LIBERO-10；
  5. 读物体朝向放在第二阶段。

  另有两条细节：语义图的合并方式由 E1 选定；"转朝向"先设 ±45°。
""")

md("""
## 6 · 下一步

1. **E1 出结果后：** 选定语义图的合并方式和层，做 E2（在 8×8 头上加语义先验，训练计划头 + 闭环）。
2. **L 回放完成后：** 写 L1、L2 的标签代码，检查同一类物体的接触区域和方向是否一致。
3. **LIBERO-10：** 回放和基线完成后，看官方权重在有把手物体上的失败是不是和朝向有关，再做 E3（真值朝向的上界）。
4. **E4：** 更细分辨率的交互 map 头，带方向场。

**需要项目负责人决定的事：** 暂时没有。E1 和 LIBERO-10 的结果出来后再汇报。
""")

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, OUT)
print("wrote", OUT)
