# 第 1 步（前半 ①）：RoboCasa365 上的交互 map 标签

日期 2026-11-26；任务 AFFO-15（Pipeline v4 执行，AFFO-8）；分支 `agent/vla-engineer/affo-15`。只用 CPU，没有 GPU 作业，没有训练。

## 1 三个问题的答案

1. **mug 能，cup 和 bowl 做不了（示范里几乎没有）。** mug：用 12 个 pretrain 实例的 103 次抓取做类别级 map，搬到 5 个 target 实例（503 次抓取），处理 ① 落点率 0.89、闭合轴中位误差 12.9°，两条线都过；逐实例看 4 个过、`mug_8` 不过（0.59 / 22.8°）。cup、bowl：pretrain 里 0 / 1 次抓取，target 里 4 / 2 次，不够检查（要求两边各 ≥ 20 次）。
2. **抽屉把手、柜门把手能；水龙头把手部分能；炉灶旋钮的“碰后动向”不能。** 同一格式（部件坐标系里的接触位置、闭合方向、碰后动向）记得下来，但一致程度不同，见 §4。
3. **第 0 步的问题在 RoboCasa 的 mug 上不存在。** 抓取方位角在物体坐标系里集中度 R = 0.97 / 1.00，在朝向机器人的坐标系里只有 0.47 / 0.33：示范者总抓把手，机器人朝向是随机的。两种处理里 ①（沿对称轴补成一圈，按机器人方位取）好于 ②（直接在朝向机器人的坐标系里记），选 ①；但普通处理只差 0.01 没过，说明在这批数据上处理本身不是关键（推断）。

## 2 做法

| 部分 | 内容 | 代码 |
| --- | --- | --- |
| 数据 | 8 个任务 pretrain 人工示范 + 5 个任务 target 示范（其余 3 个已在 NAS 上），共约 3.8 GB；MimicGen 没碰 | `scripts/data/fetch_robocasa365.sh` |
| 重放 | 从保存的状态重放，逐步记夹爪闭合/接触事件，抓放任务记被抓物体与夹爪位姿，部件任务记部件、接触点、闭合方向和接触后的运动；24 进程，CPU，约 1 小时 | `scripts/data/rc365_extract.py`，`affordance_vla/plan/rc365_events.py`，`configs/rc365_labels/extract.yaml` |
| 类别坐标系 | 沿用 `category_frame.py`（mug 的把手转到 +x） | `affordance_vla/plan/category_frame.py` |
| 对称部位 | 方位角分布；三种处理：plain（把手坐标系）、ring（① 补一圈，用时取朝向机器人的半圈和把手区）、robot（② 在朝向机器人的坐标系里记） | `affordance_vla/plan/rc365_symmetry.py` |
| 搬运检查 | 沿用第 0 步：落点率 ≥ 0.8，闭合轴误差中位数 < 20°，最近点法 | `affordance_vla/plan/transfer_check.py`，`scripts/data/rc365_label_check.py` |
| 部件 | 接触位置、闭合方向、接触后 20 步（1 s）里夹爪的运动方向，位移不到 2 cm 记为“没有定义”，各在部件坐标系 / 世界坐标系 / 机器人底座坐标系里算跨厨房一致程度 | `affordance_vla/plan/rc365_parts.py` |
| 投影 | 重放出的深度和相机参数把 3D map 或接触点投回画面 | `affordance_vla/sim/rc365_project.py`，`scripts/data/rc365_label_figures.py` |

## 3 抓放任务：mug、cup、bowl

### 3.1 示范里有多少条

| 类别 | pretrain 抓取 / 实例 | target 抓取 / 实例 | 够做检查？ |
| --- | --- | --- | --- |
| mug | 103 / 12（`CoffeeSetupMug` 102，`PickPlaceCounterToCabinet` 1） | 503 / 5（496 + 7） | 够 |
| cup | 0 / 0 | 4 / 2 | 不够 |
| bowl | 1 / 1 | 2 / 2 | 不够 |

`CoffeeSetupMug` 的 105（pretrain）和 502（target）条示范每条都抓 mug。三个抓放任务（`PickPlaceCounterToCabinet`、`…ToStove`、`…SinkToCounter`）被抓的 `obj` 是食材和小物件，容器是 pan / plate，所以 mug、cup、bowl 在这三个任务里只有 pretrain 的 mug 1 + bowl 1，target 的 mug 7 + cup 4 + bowl 2（都在 `PickPlaceCounterToCabinet`）。cup、bowl 要做就得换任务或用别的数据源，本任务没换。

### 3.2 抓取方位角（把手方向为 0°）

| | 物体坐标系 R | 朝向机器人的坐标系 R |
| --- | --- | --- |
| pretrain（103 次） | 0.97 | 0.47 |
| target（503 次） | 1.00 | 0.33 |

R = 1 表示全一致，0 表示均匀散开。抓取全落在把手一侧（直方图集中在 ±30° 内），机器人从哪边来分布很散。

### 3.3 搬到留出实例：三种处理

留出 5 个 target mug，源为 12 个 pretrain mug。逐实例的数字是（落点率 / 闭合轴中位误差）。

| 处理 | 合计（503 次） | 落点 ≥ 0.8 | 轴 < 20° | `mug_4` | `mug_5` | `mug_7` | `mug_8` | `mug_9` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 0.79 / 12.9° | ✗（差 0.01） | ✓ | 0.98 / 10.9° | 0.90 / 12.7° | 0.99 / 11.4° | 0.28 / 22.8° ✗ | 0.81 / 12.4° |
| ① ring | **0.89 / 12.9°** | ✓ | ✓ | 1.00 / 10.9° | 0.93 / 12.7° | 0.98 / 11.4° | 0.59 / 22.8° ✗ | 0.95 / 12.4° |
| ② robot | 0.33 / 14.8° | ✗ | ✓ | 0.30 / 15.4° | 0.31 / 13.7° | 0.39 / 16.6° | 0.41 / 12.2° | 0.26 / 15.2° |

上限（留出实例自己的示范对半，plain / ring / robot）：合计 1.00 / 9.8°、0.99 / 9.8°、0.81 / 13.2°。

选 ①：落点率比 plain 高 0.10 并且过线，比 ② 高 0.56。②（朝向机器人的坐标系）在自己的示范上也只有 0.81，因为把手并不总朝机器人，这个坐标系把同一个把手打散了。

### 3.4 `mug_8` 为什么不过

事实：`mug_8` 自己的示范对半是 1.00 / 10.7°，所以是搬运的问题。类别坐标系里抓取中心的半径中位数 `mug_8` 为 1.57，pretrain 的 12 个源实例都在 1.08–1.48；把手最远处 2.35，源实例 1.81–2.23。最近点搬运把源的抓取分数拿给目标表面上最近的点，`mug_8` 的抓取点落在所有源抓取点之外。`max_distance` 加到 0.4 结果不变。**推断**：`mug_8` 的把手伸得比所有源实例都远，类别归一化只拉平了身高和半径，没有拉平把手的形状。没有做进一步的实验。图 `maps_3d.png` 里 `mug_8` 的自己的抓取（下排）比其他实例偏低偏外。

## 4 部件任务：接触位置、闭合方向、碰后动向

pretrain 加 target 合并（`both`），每个部件任务一行。接触点数 = 有接触的示范数；部件 = 不同的部件实例数；一致程度 = 与其他厨房的均值方向差在容差内的比例（闭合轴 20°，碰后动向 30°），括号里是在部件坐标系里的中位误差。

| 任务（部件） | 示范 / 厨房 / 部件 | 接触位置散布（部件坐标系） | 闭合方向一致（部件 / 世界 / 底座） | 碰后动向一致（部件 / 世界 / 底座） | 碰后动向有定义的比例 |
| --- | --- | --- | --- | --- | --- |
| `OpenDrawer`（抽屉把手，滑动） | 322 / 50 / 86 | 1.8 cm | 0.85（10.8°）/ 0.78 / 0.82 | 0.73 / 0.10 / 0.72 | 0.77 |
| `OpenCabinet`（柜门把手，铰链） | 257 / 55 / 75 | 2.1 cm | 0.94（7.3°）/ 0.16 / 0.94 | 0.91 / 0.40 / 0.91 | 0.86 |
| `TurnOnSinkFaucet`（水龙头把手） | 257 / 54 / 15 | 5.3 cm | 0.75（13.5°）/ 0.37 / 0.77 | 0.49 / 0.00 / 0.47 | 0.38 |
| `TurnOffStove`（旋钮） | 259 / 57 / 63 | 32.9 cm | 0.71（14.7°）/ 0.04 / 0.12 | 0.01 / 0.04 / 0.19 | 0.26 |

- 部件坐标系里的闭合方向和碰后动向都比世界坐标系一致得多（柜门：闭合 0.94 对 0.16；抽屉的碰后动向 0.73 对 0.10）。部件坐标系和机器人底座坐标系的数字几乎一样（**推断**：机器人总是正对着操作的部件站）。
- 抽屉、柜门把手：三样都一致，一致程度 0.73–0.94。
- 水龙头把手：闭合方向一致（0.75），碰后动向弱（0.49），且 62% 的示范里接触后夹爪没有明显的平移（碰后动向没有定义）。部件实例只有 15 个。
- 旋钮：闭合方向在部件坐标系里 0.71 一致；但接触位置在部件坐标系里散布 32.9 cm，说明这个坐标系没有把旋钮对齐（**推断**：部件体的坐标原点不在旋钮上，或同一个厨房里碰的不是同一个旋钮；没有查）；碰后动向只有 26% 有定义（**推断**：旋钮原地转，夹爪几乎不平移，位移不到 2 cm），有定义的里面也不一致（0.01）。
- `OpenDrawer` 的 target 是第一次运行留下的 220 条，其余 target 部件任务取前 150 条（10 个厨房全覆盖）。

## 5 没解决的问题

- cup 和 bowl 没有检查：需要换任务或增加有 cup / bowl 的数据（例如别的原子任务），本任务没做。
- `mug_8`（把手更伸出）不过；用留一的方式做目标实例的归一化（把手长度）没试。
- 旋钮：接触位置的散布（32.9 cm）原因没查；碰后动向可能要用“绕旋钮轴的转角”而不是夹爪位移来记，没做。
- 水龙头把手的部件实例只有 15 个。
- 搬运只做了最近点；NDF 没做（这条任务不用 GPU）。
- 逐帧标签数据集的全量生成不在本任务内。

## 6 复现

```bash
# 在服务器仓库根目录（分支 agent/vla-engineer/affo-15），W=/mnt/nas/xinyizheng/affordance_vla
source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
"$RC365_PYTHON" scripts/data/rc365_extract.py --config configs/rc365_labels/extract.yaml          # ~1 h，24 进程，CPU
"$RC365_PYTHON" scripts/data/rc365_label_check.py --config configs/rc365_labels/check.yaml --stage all   # ~30 s
"$RC365_PYTHON" scripts/data/rc365_label_figures.py --config configs/rc365_labels/figures.yaml            # ~3 min
source envs/server/libero_env.sh; "$LIBERO_PYTHON" scripts/data/rc365_label_charts.py \
    --config configs/rc365_labels/check.yaml --out report/media/weekly_2026_11_23/interaction_labels
```

输出：`$W/runs/rc365_labels/extract_v1/`（逐示范事件和物体表面点），`$W/runs/rc365_labels/check_v1/`（`counts.json`、`objects.json`、`parts.json`）；日志 `$W/logs/affo15_*.log`。图在 `report/media/weekly_2026_11_23/interaction_labels/`。

不确定的地方：重放里的接触和抓取事件由夹爪与物体的接触判定，阈值写在 `rc365_events.py`；闭合轴误差按 180° 对称取较小值；没有多次随机种子（重放是确定的，检查里没有随机采样）。
