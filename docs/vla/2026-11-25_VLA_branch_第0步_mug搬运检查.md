# 第 0 步 ①：3 种 mug 轮流留一，最近点和 NDF 两种搬运的比较

日期 2026-11-25；任务 AFFO-9（Pipeline v4 执行，AFFO-8）；分支 `agent/vla-engineer/affo-9`。

## 1 要回答的问题

用两种 mug 的示范做出的 map，搬到第三种 mug 上，和第三种自己的示范对得上吗？最近点和 NDF 两种搬法哪个更好？
过关线照 Pipeline v3 第 3 节：留出示范的抓取落在搬来 map 高分区（分数前 20% 的表面）的比例不低于 0.8，闭合轴误差中位数小于 20°。

## 2 做法

| 部分 | 内容 | 代码 |
| --- | --- | --- |
| 数据 | LIBERO-90 里抓 mug 的 10 个任务（500 条示范）只取状态和动作，带物体位姿回放；与 LIBERO-10 已回放的示范合在一起，按网格分组，每个网格一份抓取集 | `scripts/data/fetch_libero_states.py`，`scripts/data/replay_libero_states.sh`，`affordance_vla/plan/category_data.py` |
| 类别坐标系 | 竖直轴朝上，原点在底面中心，高度和水平半径各归一化到 1，把手转到 +x | `affordance_vla/plan/category_frame.py` |
| 最近点搬运 | 两个源 mug 的抓取在类别坐标系叠成一张类别级 map；留出 mug 的每个表面点取最近的 map 点（超过 0.15 类别单位的点记 0 分），闭合轴按法向规则换算 | `affordance_vla/plan/category_transfer.py` |
| NDF 搬运 | NDF 公开多类别权重 `multi_category_weights.pth`，不训练；mug 网格转点云；源 mug 的抓取位姿（每个源随机取 100 个，共 200 个）在留出 mug 上用 NDF 的描述子优化（10 个起点、250 次迭代）；搬过去的抓取再用同一份接触权重代码叠成 map | `affordance_vla/plan/ndf_transfer.py` |
| 检查 | 留出 mug 的全部自己的示范抓取，对搬来的 map 算落入高分区的比例和闭合轴误差中位数（按 180° 对称取较小的） | `affordance_vla/plan/transfer_check.py`，`affordance_vla/analysis/transfer_report.py` |

另有三行对照（不计入 6 组）：**body frames**（不做类别归一化，在网格自己的坐标系里最近点）；**scene-aligned**（诊断：类别坐标系改成按场景里杯子的初始朝向对齐）；**own demos half/half**（留出 mug 自己的示范按示范分成两半，一半的 map 测另一半）。

## 3 结果

留出 mug 的示范抓取数：白 269，黄白 334，红 273。最近点用的源抓取数：607（白）、542（黄白）、603（红）。NDF 用的源抓取数每组 200。

| 留出的 mug | 搬法 | 源抓取数 | 留出示范抓取数 | 落在高分区的比例（≥ 0.8） | 闭合轴误差中位数（< 20°） | 结论 |
|---|---|---|---|---|---|---|
| 白 `porcelain_mug` | 最近点 | 607 | 269 | 0.00 ✗ | 62.7° ✗ | 不过 |
| 白 `porcelain_mug` | NDF | 200 | 269 | 0.00 ✗ | 45.1° ✗ | 不过 |
| 黄白 `white_yellow_mug` | 最近点 | 542 | 334 | 0.02 ✗ | 15.2° ✓ | 不过 |
| 黄白 `white_yellow_mug` | NDF | 200 | 334 | 0.70 ✗ | 18.6° ✓ | 不过（比例差 0.10） |
| 红 `red_coffee_mug` | 最近点 | 603 | 273 | 0.00 ✗ | 9.0° ✓ | 不过 |
| 红 `red_coffee_mug` | NDF | 200 | 273 | 1.00 ✓ | 9.6° ✓ | **过** |

对照行（留出示范抓取数同上）：

| 留出的 mug | body frames | scene-aligned（诊断） | own demos half/half |
|---|---|---|---|
| 白 | 0.12 / 4.6° | 1.00 / 20.9° | 1.00 / 3.7° |
| 黄白 | 0.77 / 14.4° | 0.77 / 16.0° | 0.97 / 11.5° |
| 红 | 0.26 / 8.1° | 1.00 / 16.9° | 1.00 / 7.1° |

（格式：落在高分区的比例 / 闭合轴误差中位数。完整表在 `$W/runs/mug_transfer/mug_mesh/results.md`、`results.json`，含平均接触权重和无接触数，两项均为 0。）

三张检查图（搬来的 map 与留出 mug 自己示范的抓取位置，投到一帧示范画面）：`report/media/weekly_2026_11_23/mug_transfer/{porcelain_mug,white_yellow_mug,red_coffee_mug}.png`。每张三栏：最近点、NDF、自己的示范 map；橙色点是留出 mug 自己的示范抓取。

## 4 结论

**事实**
- 6 组里只有 1 组过关（红杯留出、NDF）；最近点 0/3，NDF 1/3。
- NDF 在三组里比例都不低于最近点（0.00 对 0.00，0.70 对 0.02，1.00 对 0.00）；闭合轴误差中位数在白杯上更小（45.1° 对 62.7°），在黄白和红杯上与最近点相近（18.6° 对 15.2°，9.6° 对 9.0°）。
- 留出 mug 自己的示范前后半分对照，三种都在 0.97 到 1.00 之间：map 的构造本身是稳定的。
- 白杯的示范抓取在类别坐标系里离把手约 30°，黄白和红杯离把手约 170°（上一次运行的诊断）。把类别坐标系改成按场景初始朝向对齐后，比例变成 1.00、0.77、1.00。

**推断**
- 失败的主要原因在示范数据：LIBERO 里每种杯子在各任务中的初始朝向固定，示范者总是从同一侧抓，白杯的把手朝向与另外两种相反，于是“按把手对齐”的搬运（最近点；NDF 在白杯上同样）把高分放到错的一侧。这不是搬运代码的问题。
- NDF 在红杯上比最近点好，可能因为它按形状对应而不依赖把手方向对齐；这只是一个杯子上的结果，不足以说 NDF 总体更好。

## 5 没验证的事

- NDF 每组只跑了 1 个种子，源抓取是每个源随机取的 100 个（`seed: 0`），没有重复，看不到方差。
- NDF 的起点数、迭代数没有调；描述子损失中位数 0.0018 到 0.0023，没有拿它和别的值比。
- 3 种 mug 的示范朝向都是固定的，所以“搬到没见过的 mug”和“搬到不同朝向的 mug”在这批数据里混在一起，分不开。要分开需要把杯子的朝向在示范里变起来（新环境或重放时转动杯子），这不在本任务范围。
- 过关线只用于逐组判断；只有 3 个杯子，没有做显著性检验。
- NDF 的点云从仿真的视觉网格采样，是整只杯子，不是 NDF 训练时的相机局部视角，也没有深度噪声；随机起点用均匀旋转，不是 NDF 参考代码的 healpix 网格（`ndf_transfer.py` 开头说明了与参考实现的差别）。

## 6 复现

在服务器仓库根目录（`W=/mnt/nas/xinyizheng/affordance_vla`）：

```bash
bash scripts/data/run_mug_transfer_check.sh collect                      # CPU
bash scripts/server/submit_queue_job.sh 164 mug_ndf 8000 1 "cards=<有空位的卡>" -- \
    bash $PWD/scripts/data/run_mug_transfer_check.sh ndf                 # GPU，只走队列，约 45 分钟
bash scripts/data/run_mug_transfer_check.sh evaluate                     # CPU
```

前置：回放过的示范（见 `EXPERIMENTS.md`）、`$W/repos/ndf_robot`、`$W/models/ndf/multi_category_weights.pth`。设置在 `configs/mug_transfer/mug_mesh.yaml`，输出在 `$W/runs/mug_transfer/mug_mesh/`。

## 7 资源

只用了一个队列作业：`164_mug_ndf2`，GPU 1，23:45 到 00:29，约 0.73 GPU 小时，峰值显存 2.65 GB（第一次提交 `164_mug_ndf` 在 GPU 0 上因显存不足失败，没有占用计算）。上限 2 GPU 小时。
