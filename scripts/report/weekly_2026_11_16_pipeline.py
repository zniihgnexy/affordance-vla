"""Section 2 of the week-2 review: how the pipeline changed, the three v3 figures, and the layer-level figures
with formulas (numbers checked against the code and the checkpoint configs on 20 Nov 2026).

Imported by build_weekly_2026_11_16.py, which owns the notebook. Only the latest version of every figure and block
is kept: when the pipeline changes, the text here is replaced, and section 2.1 records what changed and why. The
layer-level figures are rendered from report/media/pipeline_overview/detail/pipeline_v3_layers.html by
scripts/report/render_pipeline_detail.py. Markdown cells are raw strings because they hold LaTeX.
"""


def pipeline_cells(md, code) -> None:
    """Section 2: changes, then test time, training and generalisation, each with its figures."""
    md("""
## 2 · 现行 pipeline：v3（11-20 第五轮修订后），逐块解释

完整设计在 [`docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md`](../docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md)。下面只保留最新版本：三张总图是第三轮修订后的；第四轮（推进计划）和第五轮（类别级标签、同类新形状的测试）没有改图，只改了文字。2.2、2.3 节另有 5 张逐层的细节图和公式（11-20 晚，数字逐个对照代码和权重配置；交互版见[逐层图解页面](https://claude.ai/artifact/6zKN4uaLS6myH6vAh7XhQt)，私有）。以前的版本怎么变过来的，见 2.1 节。

### 2.1 pipeline 是怎么变过来的

**版本和原因：**

| 版本 | 日期 | 核心想法 | 和上一版比，变了什么 | 为什么变 |
| --- | --- | --- | --- | --- |
| 第一周在跑的系统 | 11-16 | affordance = 接触点；动作头用反事实样本训练成服从计划 | — | — |
| 加朝向的方案 | 11-18 | 计划里加夹爪朝向（单独回归 6D） | 计划从"抓哪里"扩到"夹爪怎么摆" | 逐块问答时发现：affordance 只给接触点，没有朝向 |
| v3 | 11-19 | 计划 = 抓哪个 + 抓哪里 + 夹爪朝向 | 加语义先验（借 SemanticVLA）；实验按 E0–E7；基线定为 B、D、E、F | 泛化需要语义：任务扰动下会指错物体 |
| v3 修订 | 11-19 晚 | 以带方向的交互 affordance map 为核心，计划从 map 解码 | 不再单独回归朝向，从 map 的方向场读；标签改成夹爪和物体的交互（L1、L2）；加 LIBERO-10 测朝向 | 物体不对称时，交互 map 本身就带方向 |
| E0 之后 | 11-20 | 按事先约定的规则调整 | 方向部分改到 LIBERO-10；Object 上先做位置部分 | E0：Object 上失败的原因是位置，不是朝向 |
| v3 第三轮 | 11-20 | 一次抓取 = 抓哪个（语义）× 抓哪里、朝什么方向（3D affordance）× 手怎么动（动作头） | 标签用每类物体的 3D 标准 map；map 画所有物体，语义门控挑目标；3D 对齐读物体朝向；加泛化测试 G1–G5 | 三件事各管一种变化，才能组合起来泛化；借 AROS 的 3D 交互描述 |
| v3 第五轮 | 11-20 晚 | **泛化是最终目标之一：** 学会一类物体的抓法，高矮胖瘦不同的同类物体也要抓对 | L2 改成类别级（同类不同网格对齐到共同坐标系，共用一张 map）；加高矮胖瘦测试集（缩放杯子网格，标签跟着缩放）和红杯子；G4 的测法和对照 | 项目负责人问：map 是学出来的吗，换了形状能不能抓？标签其实是从示范统计的，而 LIBERO 里一类物体通常只有一个网格，现在的 map 说明不了换形状怎么抓 |

**每个模块：第一周在跑的系统 → 现在的设计**

| 模块 | 第一周在跑的 | 现在的设计 | 状态 |
| --- | --- | --- | --- |
| VLM | Qwen3-VL-4B，冻结 | 不变；主对比时全量微调 | 已实现 |
| 交互 map 头 | 8×8 格子：交互分数（只针对点名物体）、偏移、深度；全局的事件和接近方向 | 画面里**所有物体**的交互分数，加方向场和深度；特征更细（合并前每边 16 块，不够再加 DINOv2） | 第一版已实现，扩展未实现 |
| 语义 | 没有单独的模块，由 map 头隐式学 | **语义门控**：目标分数 = 语义权重 × 交互分数 | E1 在测 |
| 解码 | map 峰值 → 3D 接触点 | 门控后的峰值 → 3D 接触点 + 夹爪朝向；第二阶段用 3D 对齐读物体朝向 | 位置已实现 |
| 计划向量 | 10 维 | 16 维（加相对朝向，6D）；要不要加由 E3 决定 | 10 维已实现 |
| 计划保持 | 保持到夹爪完成事件 | 不变 | 已实现 |
| 动作头 | s10：反事实（挪位置、换物体、换释放点） | 加"转朝向"，在 LIBERO-10 上做 | 已实现；转朝向未实现 |
| 标签 | 夹爪开合切换那一刻的末端位置 = 接触点 | L1：夹爪位姿 + 物体分割图；L2：类别级的 3D 标准 map（同类不同网格共用一张），投到每一帧的所有物体上 | 每个网格一张已实现（3.3 节）；类别级未实现 |
| 测试 | LIBERO-PRO Object 的位置扰动、任务扰动 | 加语言扰动、LIBERO-10、G3–G5；G4 用高矮胖瘦的杯子和红杯子 | LIBERO-10 基线在跑 |
| 失败归类 | 没有 | 五类，每类对应一个模块（2.4 节） | 已定 |

**没有变的：** 论点（affordance 只有在动作被训练成服从它时才有用）；模型的输入和输出仍是 2D 图像上的 map 加深度，3D 只用在标签、读物体朝向和评测上；动作头和反事实训练。

**11-20 晚按代码逐层核对后改正的说法**（pipeline 没变，是以前写得不准；2.2、2.3 节已按代码改写）：

- 现在 g1 的训练标签不是"夹爪开合那一刻的末端位置"，而是 ORACLE 接触点：仿真里物体的位置 + 示范里这个物体的平均抓取偏移。
- L2 的物体表面只取可视网格，不取碰撞几何。
- 接触权重里"正对"的系数是 clip((|n·y| − 0.3) / 0.7, 0, 1)：正对程度低于 0.3 的表面点权重为 0。
- 动作头 DiT 的 36 块全是交叉注意，没有自注意力。权重目录的配置写着有，但运行时被 `use_canonical_forward=false` 关掉（这份官方权重早于"交错自注意力"的版本）。配置里写的 DiT-B（16 层、宽 1024）也是旧的，运行时按 LLM 重设成 36 块、宽 2560。

### 2.2 测试时：从画面到动作
""")
    code("""
Image(filename=str(FIGURES / "v3_model.png"), width=1000)
""")
    md("""
**逐层的完整路径**（下图）。数字逐个对照了服务器上的代码和权重配置；紫色虚线是计划中、还没实现的部分。
""")
    code("""
DETAIL = FIGURES / "detail"   # layer-level figures, rendered from pipeline_v3_layers.html
Image(filename=str(DETAIL / "v3_detail_1_overview.png"), width=1000)
""")
    md(r"""
**B1 输入【已实现】**

- 智能体视角（左右镜像后给模型）和腕部视角，各一张 224×224 的图；指令文本；夹爪当前的位置（来自仿真，只用来算"离目标还有多远"）。
- 处理器要求每张图至少 65536 = 256² 个像素，所以先放大到 256×256。

**B2 VLM【已实现，冻结】**

- **视觉编码器：** 16 px 一块，每张图 16×16 块；ViT 24 层、宽 1024、16 头；2×2 合并后每张图 8×8 = 64 个 token，投到宽 2560。每个 token 对应 224 图上的 28×28 像素，就是 map 头的一个格子。
- **DeepStack：** ViT 第 5、11、17 层的特征各自合并后，加到 LLM 第 1、2、3 层输出里图像 token 的位置上。
- **语言模型：** Qwen3-VL-4B，36 层、宽 2560，32 个查询头 / 8 个 KV 头，FFN 9728，因果注意。
- **提示词：** "Your task is {指令}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format."（官方训练时的写法）。图像 token 在指令前面，所以图像 token 自己看不到指令。
- 输出第 1–36 层的隐状态 $h^{(1)},\dots,h^{(36)}$，map 头和动作头都读它们。诊断和消融时 VLM 冻结；主对比时全量微调。

下图：两个头怎么读 VLM。动作头第 $i$ 块只读 LLM 第 $i$ 层；map 头把 36 层按学到的权重加起来。
""")
    code("""
Image(filename=str(DETAIL / "v3_detail_2_layers.png"), width=850)
""")
    md(r"""
**B3 交互 map 头【8×8 版 g1 已实现；v3 的扩展计划中】**

g1（`plan_grid.py`）：7.89M 参数，训练 2500 步，学习率 3e-4。

$$H=\sum_{\ell=1}^{36}\mathrm{softmax}(m)_\ell\,h^{(\ell)},\qquad X=W_{2560\to512}\,\mathrm{LN}(H)$$

$$Z=\mathrm{Enc}_{2\times}\big([\,q;\,X\,]\big)\qquad(\text{512 维，8 头，FFN 2048，双向})$$

$$[\ell_c,\ \tilde o_c,\ \hat z_c]=W_{512\to4}\,Z_{\mathrm{cell}(c)},\qquad o_c=\sigma(\tilde o_c)\in[0,1]^2,\qquad c=1,\dots,64$$

$$[\text{event}\in\mathbb{R}^3,\ \tilde a]=\mathrm{MLP}_{512\to512\to6}\big(\mathrm{LN}(Z_q)\big),\qquad a=\tilde a/\lVert\tilde a\rVert$$

- $m$ 从 0 开始学，开始时就是 36 层的平均；$q$ 是放在最前面的汇总 token。双向 Transformer 让格子能读到指令。腕部视角的 token 也在序列里，但不出格子。
- 事件分三类：没有、抓、放。深度 $\hat z_c$ 的单位是 10 cm，以 1 m 为零点。
- **v3 的扩展（计划中）：** 交互分数 $I_c=\sigma(\ell_c)$ 对画面里所有物体都画；每格加方向场 $(\cos2\theta_c,\sin2\theta_c)$，读出 $\theta_c=\tfrac12\,\mathrm{atan2}(\sin2\theta_c,\cos2\theta_c)$，转 180° 算同一种抓法；更细的特征先用 ViT 合并前的 16×16 块，不够再加 DINOv2。

**B4 语义门控【计划中，E2】**

$$w_c=\frac{e^{\alpha s_c}}{\sum_{c'}e^{\alpha s_{c'}}},\qquad T_c=w_c\cdot I_c,\qquad \alpha_{\text{init}}=0$$

- α 从 0 开始，所以刚开始每个格子的权重一样，等于没有语义。目标损失只更新门控，"抓哪个"和"抓哪里"分开学。换一句指令时，交互 map 不用重学，只是挑了另一个物体。
- E1（3.4 节）显示，冻结 VLM 里的余弦相似度几乎指不到点名物体。所以按 v3 的预案加一个小的可学习投影，先用指令 token 求平均、最后一层（投影的大小待定）：

$$s_c=\cos\Big(P_t\,\mathrm{mean}_{k\in\text{指令}}\,h^{(36)}_k,\ \ P_x\,h^{(36)}_{\mathrm{cell}(c)}\Big)$$

- E2 的离线评测包括 Spatial：两只一样的碗只能靠指令里的位置词区分。

**B5 解码【位置已实现；朝向计划中】**

下图：从格子到 3D 点。
""")
    code("""
Image(filename=str(DETAIL / "v3_detail_3_decode.png"), width=850)
""")
    md(r"""
$$c^*=\arg\max_c \ell_c\ \ (\text{v3：}\arg\max_c T_c),\qquad u=\big(\mathrm{col}(c^*)+o_x\big)\cdot 28,\qquad v=\big(\mathrm{row}(c^*)+o_y\big)\cdot 28$$

$$u'=255-u\cdot\tfrac{256}{224},\qquad v'=v\cdot\tfrac{256}{224},\qquad d=\mathrm{clip}\big(1\,\mathrm{m}+0.1\,\mathrm{m}\cdot\hat z_{c^*},\ 0.2,\ 3\big)$$

$$p_{\text{world}}=T_{\text{cam}\to\text{world}}\big(d\,K^{-1}[u',v',1]^\top\big)$$

- 深度是沿光轴的距离，不是沿射线的长度，所以直接乘 $K^{-1}[u',v',1]^\top$。$K$、$T$ 是这台固定相机的标定，从仿真日志读；它不是物体的真值，测试时可以用。
- **计划中：** 夹爪朝向 = 接近方向 $a$ + 峰值格子上的闭合方向 $\theta_{c^*}$。物体朝向放在第二阶段，用 3D 对齐：把这类物体的 3D 标准 map $\mathcal M$ 放到预测的位置，绕竖直轴每 10° 转一档，投到画面上和预测的 map 比较（λ 和两项的具体定义还没定）：

$$\psi^*=\arg\max_{\psi\in\{0^\circ,10^\circ,\dots,350^\circ\}}\ \mathrm{overlap}\big(\Pi(R_z(\psi)\mathcal M),\ \hat I\big)+\lambda\,\mathrm{agree}\big(\Pi(R_z(\psi)\mathcal M),\ \hat\theta\big)$$

**B6 计划向量【10 维已实现；16 维计划中】**

$$f=\Big[\ \frac{r}{0.10\,\mathrm{m}},\ \ \tanh\Big(\frac{r}{0.02\,\mathrm{m}}\Big),\ \ a,\ \ e\ \Big]\in\mathbb{R}^{10},\qquad r=p_{\text{world}}-p_{\text{夹爪}},\qquad e=+1\ \text{抓},\ -1\ \text{放}$$

- 前 3 维是粗的距离（远处也不饱和），中间 3 维是细的距离（最后几厘米分得清），接近方向 3 维，事件 1 维；没有事件时 $f=0$。
- v3 加 6 维：目标朝向相对当前夹爪的旋转，用 6D 表示。加不加由 E3 决定。

**B7 计划保持【已实现】**

- 接触点在世界坐标里，到事件完成前不动：抓的计划要连续 5 步发出"合"，放的计划要连续 5 步发出"开"。之后再预测新计划。保持期间，每次查询只按夹爪的新位置重算 $r$。
- 动作头一次出 8 步，全部执行完再查询。

**B8 计划编码器【已实现】**

- MLP 10→512→512→2560，SiLU，最后一层零初始化，所以训练开始时等于官方策略。
- 输出加到 8 个动作 token 上：动作 token 之间不互相注意，放在旁边的额外 token 读不到。

**B9 动作头【已实现】**

$$\mathrm{tok}_j=W_3\,\mathrm{swish}\big(W_2\,[\,W_1x_{t,j};\ \sin(\tau)\,]\big)+\mathrm{pos}_j+\mathrm{MLP}_{\text{plan}}(f),\qquad j=1,\dots,8$$

$$\mathrm{tok}\leftarrow\mathrm{tok}+\mathrm{CrossAttn}\big(\mathrm{AdaLN}(\mathrm{tok},\tau),\ h^{(i)}\big),\qquad \mathrm{tok}\leftarrow\mathrm{tok}+\mathrm{FFN}\big(\mathrm{LN}(\mathrm{tok})\big),\qquad i=1,\dots,36$$

$$\hat v=\mathrm{MLP}_{2560\to1024\to7}(\mathrm{tok}),\qquad x_{k+1}=x_k+\tfrac14\,\hat v(x_k,\tau_k),\qquad \tau_k=\lfloor 1000\,k/4\rfloor,\qquad x_0\sim\mathcal N(0,I)$$

- starVLA 的 DiT，36 块、宽 2560。第 $i$ 块交叉注意 LLM 第 $i$ 层的全部 token（40 头 × 64 维）；FFN 2560→10240→2560，GELU。没有自注意力（2.1 节）。$W_1$：7→2560，$W_2$：5120→2560，$W_3$：2560→2560；$\tau$ 是时间（1000 个桶）的正弦编码。本体状态不输入。
- 训练的参数（s10，VLM 冻结）：LoRA（秩 16，α 16，36 块 × 6 个线性层）26.54M + 动作解码 MLP 2.63M + 计划编码器 1.58M = 30.75M，和训练记录一致。

**B10 执行【已实现】**

- 8 步 × 7 维的动作（Δ位置 3、Δ旋转 3、夹爪 1）按数据集统计反归一化，交给 robosuite 的 OSC 控制器算关节力矩，逆运动学由它负责。
- 20 Hz 下 8 步是 0.4 秒。执行完拿新画面，回到 B1。

### 2.3 训练
""")
    code("""
Image(filename=str(FIGURES / "v3_training.png"), width=1000)
""")
    md(r"""
**T1 交互 map 头的训练**

- **数据：** `dataset10`，即 LIBERO-Object 10 个任务的示范回放，每一帧都是一个样本；从 E2 起加 `dataset_spatial`。
- **现在 g1 的标签【已实现，ORACLE】：** 接触点 = 物体在仿真里的位置 + 示范里这个物体的平均抓取偏移（放的时候是容器位置 + 平均放置偏移）。接近方向 = 这个物体所有示范在事件前 10 步运动方向的平均。投到画面上得到格子、偏移、深度。同一帧可以换成点名另一个物体的指令，目标换成那个物体的抓取点，不需要新示范。
- **已实现的损失：**

$$\mathcal L_{\text{map}}=\mathrm{CE}(\text{event})+\mathrm{CE}(\ell,\ c^\star)+10\,\mathrm{Huber}_{0.1}(o_{c^\star})+10\,\mathrm{Huber}_{0.2}(\hat z_{c^\star})+0.5\,(1-\hat a\cdot a)$$

  事件每帧都算；接近方向在有事件的帧算；格子、偏移、深度只在有事件、并且接触点落在画面里时算。Huber 的阈值约 1 cm 和 2 cm。

**L2：每个网格一张 3D 标准 map【已实现，`ec1c792`】**

- 表面：可视网格上按面积采 3000 个点，带法向 $n$，在物体自己的坐标系里。
- 抓取：仿真报告"抓住"的上升沿，并且持续至少 5 步。用仿真里的物体位姿（ORACLE，只用于生成标签）把夹爪换到物体坐标系；闭合轴 $y$ 是一条线，不分正反：

$$g=R_o^\top(p_{ee}-p_o),\qquad y=R_o^\top R_{ee}\,e_{\text{close}}$$

下图：一次抓取给表面点的权重。离闭合线近、法向又正对它的点权重高；法向和 $y$ 垂直的点权重为 0；沿线超出 ±5 cm 够不到。
""")
    code("""
Image(filename=str(DETAIL / "v3_detail_4_contact.png"), width=600)
""")
    md(r"""
$$w_g(x)=\exp\Big(-\tfrac12\big(\tfrac{d_\perp}{1.2\,\mathrm{cm}}\big)^2\Big)\cdot\mathrm{clip}\Big(\frac{|n\cdot y|-0.3}{0.7},\ 0,\ 1\Big)\cdot\mathbf 1\big[\,|(x-g)\cdot y|\le 5\,\mathrm{cm}\,\big]$$

$$d_\perp=\big\lVert (x-g)-\big((x-g)\cdot y\big)\,y\big\rVert,\qquad S(x)=\frac{\sum_g w_g(x)}{\max_{x'}\sum_g w_g(x')}$$

$$M(x)=\sum_g w_g(x)\,y_g y_g^\top,\qquad \mathrm{axis}(x)=M(x)\ \text{的最大特征向量},\qquad A(x)=\frac{3\lambda_{\max}/\mathrm{tr}\,M-1}{2}\in[0,1]$$

- $yy^\top$ 对 $y$ 和 $-y$ 一样，所以转 180° 的两次抓取会叠加，不会抵消。$A=1$ 表示这个点上所有抓取沿同一条线，$A=0$ 表示方向均匀散开。
- 投到一帧上（每个物体都投）：$X=R_ox+p_o$，按镜像投到 256 的日志图上。只保留落在这个物体分割像素上、并且深度不超过同一 4×4 像素块里最近点 1.5 cm 的点（去掉被物体自己挡住的背面）。$\theta_x$ 是把 1 cm 长的闭合轴投到图上后的角度：

$$\mathrm{score}_c=\max_{x\in c}S(x),\qquad \mathrm{dir}_c=\frac{\sum_{x\in c}S(x)\,[\cos2\theta_x,\ \sin2\theta_x]}{\sum_{x\in c}S(x)}$$

- L2 是统计出来的，不是学的：它只复述示范者的抓法（示范捏杯沿，map 就落在杯沿）。学的是 map 头：从画面预测这张 map。

**L2c：类别级 map【计划中，第五轮】**

- 同一类的不同网格对齐到一个类别坐标系：原点在底面中心 $b$，高度 $h$ 和水平半径 $\rho$ 归一化到 1，绕竖直轴转 $\varphi$ 让把手朝 +x。所有网格的抓取在这里叠成一张 map；新网格的每个点取最近的类别 map 点的分数和闭合轴，不需要新示范：

$$\tilde x=\mathrm{diag}\big(\tfrac1\rho,\tfrac1\rho,\tfrac1h\big)\,R_z(\varphi)\,(x-b)$$

- 高矮胖瘦测试集：网格乘 $S=\mathrm{diag}(s_x,s_y,s_z)$（高 ×0.8、×1.3，粗 ×0.8、×1.2，再加两种组合），闭合轴按法向的规则变换：

$$x'=Sx,\qquad y'=\frac{S^{-1}y}{\lVert S^{-1}y\rVert}$$

- 检查：三种杯子轮流留出一种，留出的杯子自己的示范抓取八成以上落在搬来的 map 分数前 20% 的表面上，闭合轴误差中位数小于 20°。

**L1【计划中】：** 抓取那一刻的夹爪位姿加物体分割图，直接给逐像素的"接触区域 + 闭合方向"；精确版借 AROS 的交互面，用夹爪网格和物体网格算。

**v3 的损失【计划中，各项权重未定】：**

$$\mathcal L_{\text{v3}}=\underbrace{\mathrm{BCE}(I,\ \text{所有物体的标签})}_{\text{交互分数}}+\underbrace{\mathrm{CE}(T,\ \text{点名物体的标签分布})}_{\text{只更新 M1}}+\underbrace{\big\lVert\widehat{\mathrm{dir}}-\mathrm{dir}\big\rVert^2_{\text{接触区域内}}}_{\text{方向场}}+\mathrm{Huber}(\hat z)+\text{事件、接近方向}$$

- 点名物体有几种可行抓法时，标签是一个分布。目标分数那一项只更新门控，"抓哪个"和"抓哪里"不缠在一起。
- 物体位姿只用来生成标签和做评测，测试时不输入模型。

**T2 动作头的训练：反事实监督，本工作的核心【已实现，s10】**

同一帧画面，换一个计划，动作标签跟着改写。这样动作头学到的是"计划在哪就去哪"，而不是背场景。

| 种类 | 计划怎么换 | 状态 |
| --- | --- | --- |
| 示范 | 原样，计划输入加 1 cm 噪声 | 已实现 |
| 挪位置 | 接触点在水平圆盘里挪：半径 18 cm × √U，角度均匀 | 已实现 |
| 换物体 | 换成场景里另一个物体的抓取点，指令改成点名它；最后一段用那个物体自己示范里的接近 | 已实现 |
| 换释放点 | 搬运时，释放点在同样的圆盘里挪 | 已实现 |
| 转朝向 | 绕竖直轴 ±45°，平移和旋转一起改写（LIBERO-10） | 计划中（E3） |

下图：改写路径（俯视）。示范的最后一段原样搬到新接触点，前段绕起点在水平面旋转、缩放去接上它。
""")
    code("""
Image(filename=str(DETAIL / "v3_detail_5_counterfactual.png"), width=640)
""")
    md(r"""
- 最后一段的长度是 $\min(18\,\mathrm{cm},\ \tfrac12\lVert c'-p_0\rVert)$；高度按走过的路程逐渐调整；按示范的节奏走，但每步不超过约 1.28 cm：

$$a'_{xyz}=\mathrm{clip}\Big(a_{xyz}+\frac{\Delta\big(P'_{\text{走}}-P_{\text{走}}\big)}{s},\ -1,\ 1\Big),\qquad s\approx1.28\,\mathrm{cm}\ \ (\text{每单位平移动作的位移，实测})$$

- 旋转沿用示范，只在到达接触点之前监督；夹爪命令在示范闭合时离接触点的那个距离给出。离接触点不到 3 cm 的帧不做反事实。
- **损失**（只更新计划编码器、DiT 的 LoRA 和动作解码 MLP）：

$$x_t=(1-t)\,\varepsilon+t\,a,\qquad t=\frac{0.999-u}{0.999},\quad u\sim\mathrm{Beta}(1.5,\,1),\qquad \varepsilon\sim\mathcal N(0,I)$$

$$\mathcal L=\underbrace{\frac{\sum m\odot\lVert\hat v(x_t,t,f)-(a-\varepsilon)\rVert^2}{\sum m}}_{\mathcal L_{\text{flow}}}+0.2\,\mathrm{Huber}_{0.5\,\mathrm{cm}}\Big(p_0+s\sum_{k\le j}\hat a_{k,xyz},\ \ \text{目标路径}_j\Big),\qquad \hat a=x_t+(1-t)\,\hat v$$

- $m$ 是哪些动作维有标签的掩码；$\hat a$ 是一步估出的干净动作，第二项比较它要走的路径和目标路径。s10：4000 步，学习率 2e-4，454 条示范、56,852 帧。训练时喂真值计划，测试时换成 map 头预测的计划。
- 第三、五轮修订不改变动作头：计划向量的格式不变；"换物体"的样本已经教它去任何被指定的物体，所以门控挑谁它就去谁。朝向进计划（16 维）后要重训一次，这是 E3 本来就有的。

### 2.4 泛化和诊断
""")
    code("""
Image(filename=str(FIGURES / "v3_generalization.png"), width=1000)
""")
    md("""
**V1 离线首帧检查**（便宜，每次改了 map 头都先做）：在 LIBERO-PRO 每个起始状态的第一帧上看 map 的输出。

- **认对物体率：** 门控后的峰值落在指令点名的物体分割区域里的比例；
- **所有物体的 map：** 每个物体自己区域里的峰值，是否落在它的接触区域里；
- **位置误差：** 沿视线方向和垂直视线方向分开算；
- **方向误差：** 单位是度，按 180° 对称取较小的；
- **物体朝向误差（第二阶段）：** 3D 对齐读出的朝向和仿真真值之差，按物体的对称性取最小。

**V2 闭环和泛化测试：** 开发阶段每个任务 20 个起始状态、种子 7；正式报告用 50 个 × 3 个种子。

| 测试 | 改变了什么 | 主要考哪部分 | 数据 |
| --- | --- | --- | --- |
| G1 | 物体位置 | 交互 map 跟着物体走 | LIBERO-PRO `_swap`（已有） |
| G2 | 指令或目标 | 语义门控挑物体 | LIBERO-PRO `_task`、`_lan`（已有） |
| G3 | 物体绕竖直轴转 ±45° | 3D 对齐读朝向 | LIBERO-10 的杯子、摩卡壶；扰动自己加 |
| G4 | 同类但没见过的形状（高矮胖瘦） | 类别级的 3D 标准 map + map 头 | 缩放集：LIBERO-10 杯子任务里把杯子沿高度或水平方向缩放（高 ×0.8、×1.3，粗 ×0.8、×1.2，加两种组合），标签跟着缩放；红杯子：训练不用，只在 LIBERO-90 的 4 个任务里测 |
| G5 | 几种变化同时出现 | 三部分一起 | 自己组合 |

**泛化雏形成立的判据：** G5 的每次失败都能归到某一部分；把那一部分换成真值后，成功率回升。

**G4 是最终目标之一（第五轮）：**

- 训练只用白杯子、黄白杯子的原始网格；在没见过的形状上测离线和闭环两层，和见过的形状比。
- 对照：
  - 每个网格一张 map 的标签，对类别级标签；
  - 真值计划：分开是 map 的问题还是动作头的问题；
  - 官方权重。
- 红杯子任务的指令也没见过，所以同时考 G2：认错物体和抓的位置、方向错分开算。

**V3 失败归类：** 每个失败的回合归到一个要改的模块。第 6 节的推进计划就靠这张表决定"回到哪里改"。

| 失败 | 要改的模块 |
| --- | --- |
| 认错物体 | 语义门控 |
| 位置偏 | 交互分数、深度、更细的特征 |
| 方向错 | 方向场 + 3D 标准 map |
| 没跟上计划 | 动作头 + 反事实标签 |
| 抓到后失败 | 释放阶段的标签 |
""")
