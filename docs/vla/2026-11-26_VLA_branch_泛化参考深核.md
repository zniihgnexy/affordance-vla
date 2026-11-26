# 泛化参考深核：七篇论文对第 1 步后半缺口 a–d 的用处（AFFO-17）

日期 2026-11-26。只读调研，没有运行代码、下载数据或连接服务器。缺口 a–d 的出处：`docs/vla/2026-11-26_VLA_branch_第1步_交互map标签.md` 第 3–5 节、`docs/vla/2026-11-25_VLA_branch_Pipeline_v4.md`；NDF、Robo-ABC、RAM、VRB、AffordanceVLA 沿用 `docs/vla/2026-11-24_VLA_branch_affordance提取方法调研.md`，未重查。

**读法的限制（须知）**：论文正文是通过网页抓取工具读入的（arXiv HTML；抓取工具会把长页面压缩转述），不是我逐字通读 PDF。引用的数字没有逐条回对原文 PDF，下文标「二手」的尤其如此。要把某个数字写进论文或周报，先对照原文。GAPartNet 附录末尾约 8k 字符、MimicGen 附录 S–U 部分、RoboCasa365 论文末尾 10k 字符未读到。

## 1 总表

| # | 论文（读的版本） | 输入 | 表示 | 怎么搬到新实例 / 新类别 | 跨实例的数字 | 代码 / 权重 | 许可 | 和我们的差别 |
|---|---|---|---|---|---|---|---|---|
| 1 | DenseMatcher 2412.05268（HTML v1，二手） | 带纹理的网格，自己渲染 5 视角；重网格化到约 2000 顶点（Sec.3，A.3.2） | 顶点到顶点的稠密对应（functional map） | SD-DINO 特征 + FeatUp 投到顶点，DiffusionNet 精修，求对应；新类别不用重训（留出 3 类训练实例为 0） | Table 1 DenseCorr3D：AUC 0.845 / Err 1.74（All），0.775 / 2.82（留出类）；URSSM 0.589 / 6.08；Diff3F 0.522 / 5.96。Table 3 机器人 23/30（76.7%），Robo-ABC 30% | 代码有；权重有（Dropbox，仅 384 像素版）；评测代码和推理示例未发布；最后 push 2026-05-11 | 未声明（API license=null） | 我们：类别归一化后只看几何的最近点。它用语义特征做全局平滑的对应。把手这类局部大形变：未查到专门例子 |
| 2 | AffordDP 2412.03142（HTML v2） | RGB-D、物体点云 | Φ=(c, τ)：3D 接触点 + 接触后末端 3D 路点；无朝向无闭合方向（Sec.3，Eq.1） | CLIP 检索最近的已存 affordance → SD-DINOv2 特征 argmax 找接触点 → Point-SAM 分出部件 + ICP 求旋转 → 变换路点（Sec.3.1–3.3）；新实例和新类别走同一流程 | Tab.3 仿真统一策略，见过 / 新实例 / 新类别：PullDrawer 90.0 / 55.6 / 73.3，OpenDoor 74.4 / 52.5 / 26.7；DP3 为 41.3 / 10.0 / 3.3 和 41.1 / 5.0 / 5.6。Tab.4 真机 Pick&Place 52.5 / 65.0 / 50.0 | 代码有（`SshiJwu/AffordDP`，2025-06-24）；仿真数据在 HF；策略权重未查到 | 未声明 | 它同时做条件输入（点云 + 路点 + 接触点）和采样时梯度引导；**没有**扰动 affordance 看动作是否跟的测试。我们的 plan 含接近和闭合方向，由图像预测 |
| 3 | GAPartNet 2211.05272（HTML v2，附录末尾未读） | 点云 / RGB-D | 9 类 GAPart，NPCS 部件坐标系下的定向包围盒；类别 + 位姿 + 手写规则 | 同类几何相似、操作方式一致；操作由规则（App.C.1）而非学习 | Table 4 位姿：见过 R_e 9.9° / 交互轴 7.4°，没见过 14.8° / 11.3°；Table 2 hinge knob AP50 52.9 / 23.4；Table 5 抽屉 95.0 / 90.0，门 70.0 / 55.0（见过 / 没见过）；knob 无单独成功率 | 代码有（`PKU-EPIC/GAPartNet`，2024-10-29）；检查点在 Google Drive | 数据和工作 CC BY-NC 4.0（非商用），要填表；上游数据条款未查到 | 它给“部件类别 + 位姿 + 规则”，没有交互 map 或碰后方向标签 |
| 4 | FlowBot3D 2205.04382（HTML v6，二手） | 点云 | 逐点 3D 流向量；转动关节 f=(ω×r)/‖ω×r_max‖，滑动关节 f=v（Sec.III-B，Eq.3） | 网络（PointNet++）从点云预测流；21 类（11 类训练、10 类没见过） | 真机 Table IV 45/70（64.3%）；仿真逐类平均未能可靠读到 | 代码有（`r-pad/flowbot3d`，2023-09-18）；预训练模型和数据 README 标 TBD，未发布 | MIT | 流只由关节参数决定，不用夹爪位移，所以能表达原地转的 knob；论文没有 knob 结果 |
| 5 | MimicGen 2310.17596（HTML v1，附录部分未读） | 源示范 + 每段子任务的参考物体位姿 | 示范切成以物体为中心的子任务，变换 T_W^C′=T_W^O0·(T_W^O0′)⁻¹·T_W^C（Sec.4.2） | 同类别、几何和尺度相近、类别内坐标系对齐的刚体（FAQ 9，附录 G）；段间线性插值 | Table G.1 Mug Cleanup，原杯 D0 / 1 个没见过的杯 O1 / 12 个杯 O2：生成成功率 29.5 / 31.0 / 24.5；图像策略 80.0 / 90.7 / 75.3 | 代码有（NVlabs/mimicgen，2026-10-21，v1.0.1）；数据 CC-BY 4.0 | NVIDIA Source Code License | 它造数据，不造 affordance；RoboCasa365 的生成用它的 fork |
| 6 | RT-Affordance 2411.02704（HTML v1，二手） | 语言 + 图像 | 关键位姿序列（夹爪开合切换时刻和末帧，带朝向）；画在图上，或写成文本 | 预测器 VLA 预测 plan；人也可在 UI 上标 | Table I 抓取（未见类别，每物体 5 次）：RT-2 28%，oracle plan 76%，预测 plan 68%；Table II 新任务 3% → 70% / 70% | 未查到 | 未查到 | 训练里用真值 plan，没有逼策略依赖它的做法，没测错 plan |
| 7 | CoA-VLA 2412.20451（HTML v2，二手） | 图像 + 语言 | 四种：object（2D 框）、grasp（2D 点）、spatial（2D 区域）、movement（轨迹）；每种有文本和视觉两种形式 | 文本经 MLP、视觉经 ViT-Small，两路过 2 个 Transformer 块，FiLM 注入扩散头；按场景动态选 | 没见过的物体位姿无定量表（Sec.7.3 只有定性）；Table 1 视觉泛化 CoA-VLA 57.14%、DiffusionVLA 44.44%、OpenVLA 22.22% | 未查到官方仓库 | 未查到 | 只有删除型消融，没有改 affordance 看动作变不变 |

## 2 按缺口的建议

### 缺口 a：类别太窄（cup、bowl 示范不够）

- **a1 沿用 RoboCasa 的 MimicGen 配置给 cup、bowl 造示范。** 事实：RoboCasa 官方文档称 65 个原子任务中 64 个配了 MimicGen，配置在 `robocasa-benchmark/mimicgen` fork 的 `mimicgen/exps/templates/robocasa/`（只确认了 `PnPCounterToSink.json` 一条路径）。MimicGen 迁移到同类没见过的杯子有证据（上表 5）。**推断**：改的是任务的 `obj_groups`，加每个 cup / bowl 实例的规范坐标系一致性检查。要的数据：每任务的人类种子示范（已有 pretrain 100 条）。代价：要 MuJoCo 环境和 CPU/渲染时间，需另行批准；RoboCasa365 论文 Table 4 显示 MimicGen 数据质量参差（Human300+MG60 的 51.1% 低于纯 Human300 的 52.5%）。
- **a2 换任务。** 见第 3 节：`CoffeeServeMug`、`PickPlaceCounterToCabinet` 等有 pretrain 示范，且 `obj_groups` 可设为 `("cup","mug","bowl")`（**推断**，未验证）。不需要新示范就能增加 mug 的示范条数，但 cup / bowl 仍要靠 a1。代价：只改任务配置，要重新采集或生成。

### 缺口 b：形状差得多的实例搬不过去（`mug_8`）

- **b1 DenseMatcher 代替最近点。** 要改：`affordance_vla/plan/interaction_labels.py` 里的搬运这一步，换成“源和目标网格的稠密对应 → 把源抓取点映射过去 → 闭合轴由对应点重建”（**推断**：论文只给顶点对应，闭合轴要自己重建）。要的数据：带纹理的 `.obj`（RoboCasa 实例是否带纹理未查）。代价：许可证未声明；权重仅 384 版；评测代码和示例未发布；每对约 8–12 秒（单 A100，A.3.3），源可离线缓存；要 GPU，需批准。**推断**：论文没有把手的例子，Table 6 里椅子、动物的拓扑变化更大且仍有增益，是间接证据。
- **b2 AffordDP 的 ICP 对部件配准。** 用部件点云（把手）单独做 ICP，而不是整体缩放。要改：搬运前加一步“把手部件分割 + 配准”；要 Point-SAM，或直接用 RoboCasa 的部件标注（**推断**，RoboCasa 实例是否有把手分割未查）。代价：CPU 即可做 ICP；未许可的代码只借思路。AffordDP 没有把手形状差异的专门数字。

### 缺口 c：旋钮（碰到之后往哪边动）

- **c1 由仿真关节算标签，FlowBot3D 的公式。** 事实：旋转关节逐点流 f=(ω×r)/‖ω×r_max‖，ω 为关节轴，r 为点到轴的垂直向量（Sec.III-B）。**推断**：RoboCasa 仿真里有各部件的关节类型、轴、原点，旋钮的接触点代入即得，不依赖夹爪位移；要改 `affordance_vla/plan/rc365_parts.py`，数据只需回放时读关节信息（不新增示范）。风险：旋钮半径几厘米，幅值小，归一化后只剩方向，需自己验证；论文没有 knob 结果。代价：很低，纯 CPU。另：旋钮接触位置散布 32.9 cm 的原因仍未查，做 c1 之前应先查是不是取错了部件。
- **c2 沿用 GAPartNet 的部件类别和规则。** 事实：knob 在 GAPartNet 里是绕 z 的连续对称，操作规则是“像 round handle 一样夹住再转”（App.C.1），转轴和方向怎么算未查到。**推断**：只能借“部件类别 + 位姿框架”的想法；轴向约定从文本读不出来；数据非商用（CC BY-NC 4.0）。优先 c1。

### 缺口 d：计划怎么交给动作头

- **d1 把 plan 画到图上（RT-Affordance 主方案，对应我们的路线 D）。** 事实：把机械手轮廓投到图像，颜色编码时间顺序；需要相机内外参；也可写成文本 token。**推断**：改动小，只改输入图像，但它没有逼策略依赖 plan 的训练，仍须配我们的反事实计划训练。代价：低，RoboCasa 重放能给相机参数（`affordance_vla/sim/rc365_project.py` 已有投影）。
- **d2 CoA-VLA 的“文本 + 视觉两路，FiLM 注入头”（对应路线 B/C 之间）。** 事实：FiLM 注入扩散头；删除一路只掉 1.2–3.8 点（Table 4，LIBERO）。**推断**：这说明在那里策略对 affordance 依赖有限，支持我们的论点；新骨架的动作头是小 MLP，FiLM 比“加到动作 token 上”改动大一点，要重做接口。代价：中。

## 3 A：RoboCasa365 里会抓 cup / bowl / mug 的原子任务

出处：`robocasa/robocasa` 主分支 HEAD `7bfc60c9`（2026-11-18），只读；路径均在 `robocasa/` 下。表里的 `a_coffee.py`、`a_pick_place.py` 是简写，文件的真名是 `environments/kitchen/atomic/kitchen_coffee.py` 和 `kitchen_pick_place.py`（验收时改注，行号不变）。

| 任务 | 被抓物体类别（文件:行） | pretrain 人工 | MimicGen | target |
|---|---|---|---|---|
| CoffeeSetupMug | 固定 mug：`a_coffee.py:61` | 有（`utils/dataset_registry.py:125-`） | 有（mg / mg_5x5 / mg_5x1） | 有 human（`:133`） |
| CoffeeServeMug | 固定 mug：`a_coffee.py:77` | 有（`:118-`） | 有（mg） | 无 |
| MakeIcedCoffee | 抓冰块；`bowl` 在 `a_pick_place.py:1817`、`glass_cup` 在 `:1858` 只作容器 | 有（`:532` 附近） | 无 | 无 |
| PickPlaceCounterToCabinet | 默认 `"all"`：`a_pick_place.py:34`，采样 `:81`（会随机抽到杯碗） | 有（`:252`） | 有（三种） | 有 human，另有 `real`（`:260,263`） |
| PickPlaceCabinetToCounter | 默认 `"all"`：`:150` | 有（`:238`） | 有（mg） | 无 |
| PickPlaceCounterToSink | 默认 `"all"`：`:262` | 有（`:288`） | 有（mg） | 无 |

- 默认 `"food"` 的任务不会抓 cup / bowl / mug（它们不属于 food）：PickPlaceSinkToCounter（`:373`）、CounterToStove（`:819`）、CounterToMicrowave 等。**这与第 1 步文档第 3.1 节“三个抓放任务里 mug、cup、bowl 只有少量”一致。**
- 示范条数不在注册表里，来自 `utils/dataset_registry_utils.py:30-60` 的硬编码：human pretrain 100、target 500、real 30；MimicGen 各 10000（OpenCabinet 5000）。没有对照实际数据文件。
- 物体类别：`models/objects/kitchen_objects.py` 的 `bowl`（`:211`）、`cup`（`:563`）、`mug`（`:964`）都是 `graspable=True`；`glass_cup`（`:2648`）；`coffee_cup`（`:481`）是 drink 类。每个类别名本身就是一个组，所以 `obj_groups=("cup",)` 可用（`:2960-2972`）。
- composite 里涉及杯碗的（只列文件名，注册表情况未查）：`clearing_table/bowl_and_cup.py`、`organize_mugs_by_handle.py`、`stack_bowls_cabinet.py`、`restock_bowls.py`、`heat_mug.py`、`place_ice_in_cup.py`、`choose_measuring_cup.py` 等。

## 4 B：有没有人做过“把计划挪走，看动作跟不跟”

**答案：部分。** 没查到“把计划移到另一物体、沿新路径重写动作标签、以此训练动作头服从计划”的工作。

- 七篇之内：AffordDP（扰动测试未查到，只有去掉模块的消融）；RT-Affordance（没有扰动 plan 的实验，只有 oracle plan 与预测 plan 对比）；CoA-VLA（只有删除型消融）。
- 近邻 5 篇（全部「只读摘要」；其中 MolmoAct 和 GT-VLA 验收时读了全文，见第 6 节）：
  - MolmoAct [2508.07917](https://arxiv.org/abs/2508.07917)：用户可编辑或手绘 2D 轨迹，模型据此预测动作；训练细节未查到。
  - HAMSTER [2502.05485](https://arxiv.org/abs/2502.05485)：VLM 给 2D 末端路径，低层策略以它为引导；没有服从度指标。
  - GT-VLA [2609.31904](https://arxiv.org/abs/2609.31904)：语义目标变成渲染到观测上的 2D trace；“可控性更好”只出现在检索摘要里，摘要页未写，未核。
  - Long-Horizon Trace-Conditioned VLA [2604.21924](https://arxiv.org/abs/2604.21924)：执行器以渲染的 2D 关键点 trace 为条件。
  - ReSteer [2603.17300](https://arxiv.org/abs/2603.17300)：对“中途切换语言指令”的可控性，合成数据再微调；与我们的训练思路同构，但对象是语言。
- **推断**：最接近的是 MolmoAct（可编辑 trace）和 ReSteer（针对可控性的合成数据）；缺的是三样：计划移到别的物体并重写动作标签、把“动作跟随被改计划的程度”当独立指标、对比普通训练与反事实训练。作为新颖性主张前，须再读 MolmoAct 和 GT-VLA 全文。

## 5 没查到的

- DenseMatcher：把手类大局部形变的例子；GPU 和显存要求；数据集条数 README（24 类 599）与论文（23 类 589）不一致；仓库 `TEA-Lab` 与 README 里克隆的 `JunzheJosephZhu` 关系。
- AffordDP：路点数 N、引导的阈值 θ、γ、δ；训练好的策略权重；许可证。
- GAPartNet：各部件类别 NPCS 的原点和轴约定（图里）；knob 的转轴和方向怎么算；knob 的操作成功率；上游数据（PartNet-Mobility、AKB-48）条款。
- FlowBot3D：项目页；knob 或小部件结果；逐类平均值。
- MimicGen：RoboCasa 60 个原子任务具体名单；fork 里配置的内容；种子示范是否正是 100 条人工示范（文档没逐字写）；GPU 要求；Mug Cleanup 里各杯的尺寸和几何差异。
- RT-Affordance：代码、权重、许可、仓库；摘要和项目页的 69% / 15% 与表里 68% / 28% 对不上。
- CoA-VLA：官方代码（页面有按钮但无链接）、权重、许可；论文自身数字有出入（Sec.5.1 的 14.29% 与 Table 1 算出的 8.94 点）。
- RoboCasa：各类别实例数；composite 的注册表条目；pretrain50 是否含 CoffeeServeMug 等。

## 6 验收抽查（VLA Lead，2026-11-26）

这一节是验收时加的，前五节是 VLA Scout 的原文（只改了第 3 节的文件名注释和第 4 节的一处指引）。抽查的办法：让网页读取工具只读一张表或一节并逐字引用，再和上文对照；RoboCasa 的部分在服务器上装的源码里只读核对。

| 核对的内容 | 结果 |
|---|---|
| AffordDP Tab.3、Tab.4 的全部数字，Φ=(c, τ) 的定义 | 逐字读到，和上文一致。正文里没有扰动 affordance 的实验 |
| RT-Affordance Table I（28% / 24% / 76% / 68%）、Table II（3% / 70% / 70%） | 逐字读到，一致。引言写的是 69% 对 15%，和表对不上，上文第 5 节说的出入属实。扰动 plan 只在“以后的工作”里提到 |
| MimicGen Mug Cleanup 的图像策略 80.0 / 90.7 / 75.3，D0 的生成成功率 29.5 | 逐字读到，一致。O1、O2 的生成成功率（31.0 / 24.5）没读到，仍是二手 |
| DenseMatcher Table 1 和机器人实验 23/30 | 数字一致，但我这边也是从工具的转述里读到的（页面只读到约四成），不算逐字核实，仍是二手 |
| RoboCasa 源码（服务器上的 `7bfc60c`）：第 3 节的类名、默认 `obj_groups`、行号、注册表条目、示范条数的出处 | 全部一致；只有文件名是简写（已在第 3 节改注） |
| RoboCasa 文档：MimicGen 的 fork 和“64/65 个原子任务已配好” | 一致（`docs/use_cases/mimicgen.md`）。同一份文档写现成的 MimicGen 数据覆盖 60 个原子任务 |

没有抽查的：GAPartNet、FlowBot3D、CoA-VLA 三篇的数字。写进论文或周报之前，标「二手」的和这三篇的数字都要对照原文。

**B 的补充：MolmoAct 和 GT-VLA 读了全文（逐字）。**

- MolmoAct（2508.07917v4）：训练用的 2D 轨迹来自每条示范自己的画面（让 Molmo 逐帧指出夹爪，取当前点到结束点之间最多 5 个点，Sec.3.1），没有编辑过或重写过的轨迹，动作标签也没有改。测试时由人画 5 个点来改去向（Sec.5.6，附录 D.7）：15 次试验，成功率 75%，比用语言改高 33 个点。指标是任务进度分，没有“执行的路径离给定轨迹多近”的指标。训练里和“服从”沾边的只有一处：一半示范故意走不同的路到同一个目标（D.7）。
- GT-VLA（2609.31904v1）：轨迹是末端位置投到图像上，画在输入图里（Sec.III-B、III-C）。训练时随机去掉轨迹，或给轨迹加低频噪声（Sec.III-C）；论文没有说动作标签随之改动。**【推断】** 标签不变，这是让策略容忍轨迹的小误差，和“轨迹挪了动作就跟着挪”方向相反。测试时把目标点挪开 4–64 像素（Sec.IV-F），到 32 像素成功率仍稳，各半径的数字只在图里。“可操控”是全文的说法，但没有单独的指标；失败分类里“没跟轨迹”占 0.6%–8.3%（Sec.IV-H）。
- 所以第 4 节的答案不变，并且更具体：查过的 12 篇里，没有一篇在训练时把计划挪到别处并重写动作标签，也没有一篇把“动作跟随被改计划的程度”单独量出来。测试时改计划的有两篇（MolmoAct 由人画，GT-VLA 加噪声）。HAMSTER、Long-Horizon Trace-Conditioned VLA、ReSteer 仍只读了摘要；没查到不等于没有，写新颖性之前还要再查一轮。
