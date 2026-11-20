"""Build report/2026-11-16_weekly_review.ipynb, the review of the London week 16-22 Nov 2026 (cells only; execute it).

Usage on the server (starvla env, from report/):
    python ../scripts/report/build_weekly_2026_11_16.py 2026-11-16_weekly_review.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-16_weekly_review.ipynb
Reads finished runs only: runs/libero_pro/{released,predicted_g1_hold_s10,full_*_s9_seed*}, the replayed
demonstrations, and the folders of the jobs started this week (their progress is read when the notebook runs).
Writes its videos to report/media/weekly_2026_11_16/ (tracked). Nothing runs on a GPU. The text is in Chinese for the
project lead. Sections 2 and 6 come from weekly_2026_11_16_sections.py; only the latest version of every figure and
block is kept, and section 2.1 records how the pipeline changed.
"""
import sys
from pathlib import Path

import nbformat as nbf
from weekly_2026_11_16_sections import pipeline_cells, plan_cells

OUT = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# VLA 方向周报 · 第二周：2026-11-16 至 11-22（伦敦时间）

**仓库：** `affordance-vla` · **更新于：** 2026-11-20（周五，伦敦时间；本周还没结束，进行中的任务在第 4 节实时读取）

**这份 notebook 的顺序按优先级：**

1. 真实 demo 的图像和视频；
2. 现行 pipeline（v3）：这一周怎么变过来的，三张图，逐块的详细解释；
3. 本周的实验结果（11-16 的新标签、E0、L）；
4. 进行中的任务和 LIBERO-10 的基线；
5. 时间线和决定；
6. 下一步：实验推进计划（已同意）。

**说明：**

- **数字从哪来：** 表格和图都是从服务器上已经跑完的结果里读出来的，执行 notebook 时现算。
- **ORACLE：** 指仿真里的真值；"真值计划"是上限，不是方法的结果。
- **状态标记：** 【已实现】表示代码在跑、有结果；【v3 计划】表示设计已确认，还没实现。
- **周报的规则：** 每个伦敦时间的自然周（周一到周日）只有一份周报；同一周里新内容直接替换旧内容。各实验的详细 notebook 是附录，在 `report/experiments/`。
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
from report import weekly_2026_11_16_helpers as K

set_matplotlib_formats("jpeg")
RUNS = ROOT / "runs"
PRO = RUNS / "libero_pro"
MEDIA = ROOT / "report" / "media" / "weekly_2026_11_16"   # tracked in git
MEDIA.mkdir(parents=True, exist_ok=True)
FIGURES = ROOT / "report" / "media" / "pipeline_overview"
OURS = PRO / "predicted_g1_hold_s10"   # grid plan head, plans kept, action head with the labels of 15 Nov
RELEASED = PRO / "released"            # the released checkpoint, no plan
""")

md("""
## 0 · 一页总结

**本周主线：**

- **pipeline 定为 v3，修订到第五轮**（2.1 节）。现在的设计把一次抓取拆成三件事：
  - 抓哪个物体：语义门控；
  - 抓哪里、朝什么方向：带方向的交互 affordance map，画面里所有物体都画；
  - 手怎么动：动作头，被训练成服从计划。
- **泛化定为最终目标之一**（11-20 晚，第五轮）：学会一类物体的抓法后，高矮胖瘦不同的同类物体也要抓对。为此标签改成类别级（同类不同网格共用一张 map），并加上高矮胖瘦的杯子和红杯子作为测试（2.3、2.4 节，第 6 节阶段 1）。
- **11-16：动作头的新标签（s10）设为默认**（3.1 节）。
- **E0 做完了。** LIBERO-Object 上，失败的抓取不是朝向错了，而是位置偏了：夹爪离物体还远就闭合了。按事先约定的规则，方向部分改到有把手物体的 LIBERO-10 上验证。
- **L 做完了：** 每类物体一张 3D 标准 map（LIBERO-Object 10 类、LIBERO-10 10 类），投回画面和示范对得上（3.3 节）。摩卡壶的 map 落在把手上，杯子落在杯沿：方向信息正是罐子类物体没有的。
- **E1 做完了**（3.4 节）：冻结 VLM 里"指令和格子"的相似度几乎指不到点名的物体，接近随机。按 v3 的预案，语义门控要加一个可学习的投影。另外，现在的 g1 格子头在 LIBERO-Object 首帧上只认错 3/40，主要错误仍是位置偏。
- **LIBERO-10 的基线：** 官方权重在 LIBERO-PRO 的 LIBERO-10 上几乎全失败（位置扰动 3/200，任务扰动 18/200）。
- **定下了实验推进计划**（第 6 节，项目负责人已同意）：骨架定型 → 挑下游任务 → 按对比结果有针对性地改进。

**进行中：**

- 下一步：把 3D 标准 map 投到训练帧上，做 E2 的"所有物体"标签；
- 同时做 L2c：下载 LIBERO-90 里抓杯子的示范，建杯子的类别级 map 和高矮胖瘦测试集（只用 CPU）。

**其他：**

- 仓库按研究代码的要求整理过一遍（`AGENTS.md`）；
- Mac 上可以在 MuJoCo 里打开场景、回放服务器上的测试；
- 读了 SemanticVLA 和 AROS；
- 给记忆系统写了一份"带方向的交互 affordance map"总结页。

**现在最好的数字**（论文协议的结果来自上周；本周 11-16 的新标签只在开发协议下测过，见 3.1 节）：LIBERO-PRO Object，按论文协议（每个任务 50 个起始状态，3 个种子）。

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
### 1.6 LIBERO-10：官方权重在有把手物体的任务上

方向部分改到 LIBERO-10 之后，先看官方权重在那里的样子。下面是同一个两步任务（白杯子放左盘，黄白杯子放右盘）在两种扰动下的表现：

- **左：位置扰动，物体挪了位置。** 白杯子放到了左盘；第二步伸向了没被点名的红杯子，一直没抓起来。
- **右：任务扰动，两个杯子的去处对调了。** 两个杯子都放上了盘子，但放法和训练时的原指令一样（白杯子左盘、黄白杯子右盘），没有按改过的指令对调。

最后的位置从日志里的物体坐标核对过，左右盘对应哪个物体，按任务定义文件里的目标条件确定。右边就是 LIBERO-PRO 想测的问题：模型照着训练时记住的动作做，不看改过的指令。语义门控要解决的正是这种"认错物体"。官方权重在这两套任务上几乎全失败（第 4 节的表），全部失败的逐个归类放在阶段 2 的筛选里做（第 6 节）。
""")

code("""
L10 = PRO / "libero10_released"
pair = [load_rollout_log(L10 / "libero_10_swap" / "task4_ep0.npz"),
        load_rollout_log(L10 / "libero_10_task" / "task4_ep0.npz")]
for log, name in zip(pair, ["position", "task"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released: position perturbation", "released: task perturbation"],
                             MEDIA / "libero10_released_mugs.mp4")
Video(str(clip), embed=True, width=760)
""")

pipeline_cells(md, code)

md("""
## 3 · 本周结果

### 3.1 11-16：动作头的新标签（s10）和计划噪声（s11）

11-16 是周一，按伦敦时间的自然周属于本周。

- **白天：** 检验了 15 日诊断后提出的标签修法（s10）：前半段以起点为中心旋转缩放，最后一段用被点名物体自己示范里的接近方式。
- **晚上：** 又试了给"换成另一个物体"的样本加计划噪声（s11）。

都在 LIBERO-PRO Object 上测，每个任务 20 个起始状态、种子 7：
""")

code("""
HEADS = {"13 日的标签（s9）": "s9", "15 日的标签（s10，现在的默认）": "s10", "s10 + 换物体样本的计划噪声（s11）": "s11"}
display(K.label_runs_table(PRO, HEADS))
""")

md("""
**结论：**

- **s10 设为默认（11-16 项目负责人同意）。** 任务扰动明显变好，真值计划和图像格子计划都是。代价是位置扰动下用图像格子计划时，tomato sauce 从 13 掉到 2：新动作头更严格地跟着偏了约 3 cm 的计划。
- **s11 不设为默认。** 合计没有提升。
- 逐任务的分析在附录 [`experiments/2026-11-16_label_fix_results.ipynb`](experiments/2026-11-16_label_fix_results.ipynb)。

### 3.2 E0：失败的抓取是不是朝向错了

**问题：** 失败抓取的夹爪朝向，是不是比抓住的更偏离示范？

**事先约定的通过线：** 失败比抓住多偏至少 10°。

**数据：** 论文协议的全部闭环，两种配置 × 3 个种子，每种配置 3000 个回合。

附录：[`experiments/2026-11-20_grasp_orientation.ipynb`](experiments/2026-11-20_grasp_orientation.ipynb)。
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
### 3.3 L：每类物体的 3D 标准 map

**做法**（[`interaction_labels.py`](../affordance_vla/plan/interaction_labels.py)）：

1. 从仿真模型里取物体的可视网格，在物体自己的坐标系里按面积采 3000 个表面点；
2. 示范里每次抓住物体（仿真报告抓住、并且持续 5 步）时，把夹爪位姿换到物体坐标系；
3. 表面点离闭合线越近、越正对闭合方向，权重越高；超出手指够得到的范围记为 0；
4. 同一类物体的所有抓取加起来：每个点得到交互分数、主闭合轴（转 180° 算同一条线）和方向一致度。

**抓取怎么算（改过一次）：** 一开始沿用 E0 的规则，要求在发出"闭合"后 15 步内抓住。ketchup 瓶因此只剩 21 次抓取，因为很多示范里手指早早就合上了，过了一会儿才真正抓住。现在改成"仿真报告抓住、并且持续 5 步"，每类都回到约 50 次。

**通过标准**（v3 第 5 节）：同一类物体的闭合方向在物体坐标系里聚成少数几簇；投回画面后和示范帧对得上。
""")

code("""
LABELS = RUNS / "interaction_labels"
for name, title in [("object", "LIBERO-Object"), ("libero10", "LIBERO-10")]:
    print(title)
    display(K.label_summary(LABELS / name / "summary.json"))
""")

md("""
**下面三张检查图**，每张从左到右依次是：

- map 从上面看、从侧面看：颜色是交互分数，红线是分数最高的点上的闭合轴；
- map 按物体位姿投到示范第一帧上：8×8 格子，颜色越亮分数越高；
- 这条示范真正抓住的那一帧，叉是夹爪的位置。
""")

code("""
for name, category in [("object", "alphabet_soup"), ("libero10", "moka_pot"), ("libero10", "porcelain_mug")]:
    display(Image(filename=str(LABELS / name / "check" / f"{category}.png"), width=1000))
""")

md("""
**看到的：**

- **罐子（alphabet soup）：** 两指夹在罐身两侧，闭合轴横穿罐身。
- **摩卡壶：** map 落在**把手**上，闭合轴横跨把手。
- **白杯子：** map 落在杯沿上，一指在里、一指在外。

后两类物体的方向信息，正是对称的罐子没有的。三张图里，投到第一帧的高分格子都落在示范真正抓住的位置。

**分组：**

- LIBERO-Object 的 10 类里，9 类的最大组占八成以上；tomato sauce 约 76%，有一小组换了方向。
- LIBERO-10 的杯子分成几组：抓的是杯沿上的不同位置，闭合轴跟着杯沿转。但每个表面点上的方向一致度仍有 0.93–0.97。
- LIBERO-10 里的 alphabet soup 也分成两大组（63 / 37），而在 Object 里几乎只有一组。原因还没查，可能是两个套件里它的摆放不同。

**结论：** L 通过。20 类里我逐张看了 5 类（alphabet soup、ketchup、白杯子、黄白杯子、摩卡壶），其余的检查图在服务器的 `runs/interaction_labels/*/check/`。

### 3.4 E1：语义诊断（冻结 VLM，不训练）

**问的问题：**

1. 语义门控要靠"指令和画面格子的相似度"挑物体。冻结的 VLM 里，这个相似度本身能不能指到点名的物体？
2. 现在的 g1 格子头在首帧上出错时，是认错了物体，还是认对了但位置偏？

**做法**（[`probe_semantic_prior.py`](../scripts/vla/probe_semantic_prior.py)）：

- LIBERO-PRO 的首帧共 74 帧：Object 40 帧，Spatial 34 帧，位置扰动和任务扰动大约各一半。
- 用官方权重里的 VLM，36 层每层都算：指令的 token（或只取物体名的 token）和 8×8 个格子 token 的余弦相似度，在 token 之间求平均或取最大值。
- 两种看法：
  - **峰值：** 相似度最高的格子，是不是落在点名的物体上。点名物体平均只占 3.3 格，随机水平约 5%。
  - **逐物体：** 每个物体区域里的平均相似度，最高的是不是点名的物体。随机水平是 1 / 画面里的物体数。门控乘在交互分数上，而交互分数只在物体上，所以这一看更接近门控的用法。
""")

code("""
PROBE = RUNS / "semantic_probe" / "e1"
print("峰值：相似度最高的格子落在点名物体上的比例（全部 74 帧，每种做法取最好的层）")
display(K.probe_table(PROBE / "summary.json"))
print("逐物体：点名物体的平均相似度在画面里所有物体中最高的比例（每种做法取最好的层）")
display(K.probe_object_table(PROBE / "probe.npz"))
print("g1 格子头：首帧最高分的格子落在哪里")
display(K.probe_head_table(PROBE / "probe.npz"))
""")

md("""
**看到的：**

- **峰值：** 最好的也只有 9.5%（指令、平均、第 35 层），和随机差不多。
- **逐物体，LIBERO-Object：** 最好 0.30（只用物体名、平均），随机 0.14。用整句指令时，和随机差不多。
- **逐物体，Spatial：** 数字更高，但 Spatial 的目标是两只一样的碗之一。相似度能找到"碗"，分不清是哪一只；在两只碗里随便挑一只就有一半。
- **层和合并方式：** 帧数少（40 和 34 帧，一个标准差约 6–8 个百分点），各种做法之间的差别分不出高下。
- **g1 格子头：**
  - LIBERO-Object 上 40 帧里 37 帧落在点名物体上：任务扰动 20/20 都在 2 cm 内；位置扰动 11 帧在 2 cm 内，6 帧偏了超过 2 cm。只有 3 帧认错。
  - Spatial 上（g1 没在 Spatial 上训练过）只有 3/34 落在点名的碗上。

**结论：**

1. **冻结 VLM 的余弦相似度太弱，不能直接当门控。** 按 v3 M1 已写好的预案，E2 的门控加一个小的可学习投影：指令 token 和格子 token 各投到低维，再算相似度，用"目标分数"的交叉熵只训练门控。
2. **合并方式和层：** E1 分不出高下。E2 先用最简单的：指令 token 求平均，取最后一层；投影训练好以后再比较。
3. **门控的作用在哪里看得出来：** LIBERO-Object 上 g1 已经基本不认错物体，主要错误是位置偏（和 E0 一致）。门控的作用要在 Spatial、语言扰动和 LIBERO-10 的两步任务上才看得出来。**建议 E2 的离线评测加上 Spatial**（等项目负责人确认）。
""")

md("""
## 4 · 进行中的任务和 LIBERO-10 的基线（执行时实时读取）

- **L：** LIBERO-Object 示范重新回放，补记物体朝向。只用 CPU。
- **LIBERO-10 示范回放：** 第一次全部失败。文件名带场景前缀（`KITCHEN_SCENE3_…`），被当成了指令。修好后重跑，只用 CPU。
- **LIBERO-10 基线：** 官方权重在 `libero_10_swap` 和 `libero_10_task` 上，每个任务 20 个起始状态、种子 7。任务扰动的任务 2、3 第一次没跑：它们第一个关注物体是灶台和柜子，不能抓。脚本改成取第一个可抓的物体后，在单独的运行名下补跑完了（0/40）。
- **要注意的一点：** LIBERO-10 的任务大多要先后抓两个物体。日志里记的"目标"取自任务定义的关注物体列表，对这些任务不一定是抓取的物体（任务 4 记的是盘子）。做 LIBERO-10 的标签时，每个阶段抓的是哪个物体，要从示范的接触记录里取。
""")

code("""
WORK = RUNS.resolve().parent   # runs/ links into the NAS work folder
display(K.progress_table(WORK))
""")

md("""
**官方权重在 LIBERO-PRO 的 LIBERO-10 上**（每个任务 20 个起始状态、种子 7）：
""")

code("""
for suite, name in [("libero_10_swap", "位置扰动"), ("libero_10_task", "任务扰动")]:
    table = K.task_results([PRO / run for run in K.LIBERO10_RUNS], suite)
    print(f"{name}（{suite}）：{table['成功'].sum()} / {table['回合'].sum()}，{len(table)} 个任务有结果")
    display(table)
""")

md("""
## 5 · 本周时间线和决定

| 日期 | 做了什么 | 提交 |
| --- | --- | --- |
| 11-16 | 检验 15 日的新标签（s10），项目负责人同意设为默认；晚上试了给"换成另一个物体"的样本加计划噪声（s11），合计没有提升 | 附录 `2026-11-16_label_fix_results` |
| 11-16 晚 | 第一周周报；项目负责人暂停实验，做深度调研 | `ddf3ff2`、`c75460f` |
| 11-17 | 仓库按研究代码要求整理：按领域分包，每个文件不超过 500 行，实验索引写在 `EXPERIMENTS.md`；Mac 上的 MuJoCo 本机仿真（打开场景、脚本抓取、回放服务器上的测试） | `b1b09ad` 至 `f593260` |
| 11-18 | 逐个问答理解 pipeline。发现 affordance 只给接触点、没有朝向，提出"计划里加夹爪朝向"的方案，画了第一版图 | `5d6bb35` 里的 `pose_plan_pipeline.png` |
| 11-19 | 读 SemanticVLA；**确认 v3 的 6 条**，画三张图。晚上**修订：以带方向的交互 affordance map 为核心**（5 条） | `5d6bb35`、`1245ebb` |
| 11-20 | **E0 完成**（不通过，失败的原因是位置）；日志加物体朝向（逐字节核对过）；L 回放、LIBERO-10 下载和基线、E1 开始；为记忆系统写方向性 map 总结页 | `ce7b263`、`2d447e0` |
| 11-20 晚 | **L 完成：** 每类物体的 3D 标准 map（Object、LIBERO-10）；E1 的第二个问题修好 | `9e598cd`、`ec1c792` |
| 11-20 晚 | **v3 第三轮修订**（3D 标准 map、所有物体 + 语义门控、3D 对齐、G1–G5），文档和三张图更新；修好 E1 和 LIBERO-10 的三个问题；**定下实验推进计划**（三个阶段，项目负责人同意） | `9e41d68`、`0d60d4d`、`7bf1f63` |
| 11-20 晚 | 周报改成按伦敦时间的自然周，一周一份；各实验的 notebook 移到 `report/experiments/` 作附录 | `8b61e39` |
| 11-20 晚 | **E1 完成**（3.4 节）：冻结相似度接近随机，门控按 v3 的预案加可学习投影；LIBERO-10 任务扰动的任务 2、3 补跑完 | 运行 `e797d04` |
| 11-20 晚 | **v3 第五轮：泛化定为最终目标之一。** L2 改成类别级；加高矮胖瘦测试集和红杯子（核对过：LIBERO 的杯子有 3 种网格）；G4 的测法；GPU 作业用 GPU 7 | `489ae35` |

**本周的决定**（每条改了什么见 2.1 节，原文在 [`docs/vla/README.md`](../docs/vla/README.md)）：

- **11-19，6 条：** v3 取代 v2 的方法部分；按 E0–E7 做实验；基线定为 B、D、E、F；加语言扰动；Spatial 挪到 E7；恢复实验。
- **11-19 晚，修订 5 条：** 以交互 map 为核心；标签用 L1 + L2；第一版做 2D map；朝向的主要测试加 LIBERO-10；读物体朝向放在第二阶段。另有两条细节：语义图的合并方式由 E1 选定；"转朝向"先设 ±45°。
- **11-20，第三轮 4 条：** L2 做成每类物体的 3D 标准 map；交互 map 画所有物体，语义门控挑目标；读物体朝向改用 3D 对齐；E7 加泛化测试 G3–G5。
- **11-20 晚，2 条：** 实验按三个阶段推进（第 6 节）；本项目的作业可以设队列优先级 164。
- **11-20 晚：** 周报按伦敦时间的自然周，一周一份；同一周里新内容直接替换旧内容。
- **11-20 晚，第五轮 3 条：** 泛化是最终目标之一。L2 改成类别级；加高矮胖瘦测试集和红杯子（阶段 1 建好，阶段 2 作为候选）；G4 的测法和对照。另外，本项目的 GPU 作业提交时用 `cards=7`。
""")

plan_cells(md)

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, OUT)
print("wrote", OUT)
