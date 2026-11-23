"""Section 3 of the week-2 review: how the pipeline changed, the three v3 figures, and the layer-level figures
with formulas (numbers checked against the code and the checkpoint configs on 20 Nov 2026).

Imported by build_weekly_2026_11_16.py, which owns the notebook. Only the latest version of every figure and block
is kept: when the pipeline changes, the text here is replaced, and section 3.1 records what changed and why. The
layer-level figures are rendered from report/media/pipeline_overview/detail/pipeline_v3_layers.html by
scripts/report/render_pipeline_detail.py. Markdown cells are raw strings because they hold LaTeX.
"""


def figure(name: str, width: int, folder: str = "FIGURES") -> str:
    return f'Image(filename=str({folder} / "{name}"), width={width})'


def pipeline_cells(md, code) -> None:
    """Section 3: changes, then test time, training and generalisation, each with its figures."""
    md(r"""
## 3 · Pipeline v3 (after revision round 5)

Full design: [`docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md`](../docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md). Three overview figures, then five layer-level figures with formulas; every number was checked against the code and the checkpoint configs. Purple dashed = planned, not built yet.

### 3.1 How the pipeline changed this week

| Version | Date | Core idea | What changed | Why |
| --- | --- | --- | --- | --- |
| week-1 system | 16 Nov | affordance = contact point; action head trained to obey the plan | — | — |
| orientation proposal | 18 Nov | add gripper orientation to the plan | plan covers "how to hold", not only "where" | the contact point says nothing about orientation |
| v3 | 19 Nov | plan = which object + where + gripper orientation | semantic prior (after SemanticVLA); experiments E0–E7; baselines B, D, E, F | task perturbation needs semantics: the plan points at the wrong object |
| v3, revised | 19 Nov | directional interaction map at the core; the plan is decoded from it | orientation read from the map's direction field; labels from gripper–object interaction (L1, L2); LIBERO-10 for orientation | for asymmetric objects the map itself carries direction |
| after E0 | 20 Nov | rule agreed in advance | orientation tested on LIBERO-10; position first on Object | E0: failures on Object are position, not orientation |
| round 3 | 20 Nov | which (semantics) × where and direction (3D affordance) × how (action head) | per-category 3D map labels; map on every object + semantic gate; 3D alignment for object orientation; tests G1–G5 | each part covers one kind of change, so they compose |
| round 5 | 20 Nov | **generalise to unseen shapes of a category** | category-level L2; tall/short/fat/thin mug set; G4 protocol | L maps are demonstration statistics and LIBERO has one mesh per category |

**Corrected after checking the code (20 Nov):** g1 trains on ORACLE contacts (object position + mean demonstrated grasp offset), not on the end effector at the gripper switch; L2 uses visual meshes only; the facing factor of the contact weight is $\mathrm{clip}((|n\cdot y|-0.3)/0.7,0,1)$; the DiT has no self-attention at runtime (`use_canonical_forward=false`; the checkpoint config is stale).

**Unchanged:** the thesis; the model's input and output stay a 2D image map plus depth — 3D is used only for labels, the orientation readout and evaluation; the counterfactually trained action head.

### 3.2 Test time
""")
    code(figure("v3_model.png", 1000))
    md(r"""
**Layer by layer** (below): the plan branch (map head → gate → decoding → plan) and the action branch read the same VLM forward pass.
""")
    code('DETAIL = FIGURES / "detail"   # layer-level figures, rendered from pipeline_v3_layers.html\n'
         + figure("v3_detail_1_overview.png", 1000, "DETAIL"))
    md(r"""
**B1 Inputs [implemented]:** agent view (mirrored) and wrist view, 224×224 each, upsampled to 256×256 (the processor's minimum is 256² pixels); the instruction; the current gripper position.

**B2 VLM [implemented, frozen]:** Qwen3-VL-4B. ViT: 16-px patches, 24 layers × 1024; 2×2 merge → **8×8 = 64 tokens per view** (one token = one 28×28-px cell of the 224 image = one map cell). DeepStack adds ViT layers 5/11/17 to LLM layers 1–3. LLM: 36 layers × 2560, causal; image tokens come before the instruction, so image tokens cannot see it. Outputs $h^{(1)},\dots,h^{(36)}$. Frozen for diagnostics, fully fine-tuned for the main comparison.

How the two heads read the VLM (below): action-head block $i$ reads LLM layer $i$; the map head mixes all 36 layers.
""")
    code(figure("v3_detail_2_layers.png", 850, "DETAIL"))
    md(r"""
**B3 Interaction map head [8×8 version g1 implemented; v3 extension planned]** — 7.89M parameters:

$$H=\sum_{\ell=1}^{36}\mathrm{softmax}(m)_\ell\,h^{(\ell)},\qquad Z=\mathrm{Enc}_{2\times}\big([\,q;\ W_{2560\to512}\,\mathrm{LN}(H)\,]\big)$$

$$[\ell_c,\ \tilde o_c,\ \hat z_c]=W_{512\to4}\,Z_{\mathrm{cell}(c)},\qquad [\text{event},\ \tilde a]=\mathrm{MLP}(Z_q),\qquad c=1,\dots,64$$

A 2-layer bidirectional encoder lets the cells read the instruction; $q$ is a summary token. Per cell: score $\ell_c$, in-cell offset $o_c=\sigma(\tilde o_c)$, depth $\hat z_c$; globally: event (none / grasp / release) and approach direction $a$. **Planned:** scores $I_c=\sigma(\ell_c)$ for every object, a direction field $(\cos2\theta_c,\sin2\theta_c)$ (180°-symmetric), finer features.

**B4 Semantic gate [planned, E2]:**

$$T_c=\underbrace{\frac{e^{\alpha s_c}}{\sum_{c'}e^{\alpha s_{c'}}}}_{w_c}\cdot I_c,\qquad s_c=\cos\big(P_t\,\mathrm{mean}_{k\in\text{instr}}\,h^{(36)}_k,\ P_x\,h^{(36)}_{\mathrm{cell}(c)}\big),\qquad \alpha_{\text{init}}=0$$

$\alpha=0$ at the start means no semantics. A new instruction picks another object without relearning the map. The projections $P_t,P_x$ were added after E1 (section 4.4).

**B5 Decoding [position implemented; orientation planned]** (below):
""")
    code(figure("v3_detail_3_decode.png", 850, "DETAIL"))
    md(r"""
$$c^*=\arg\max_c T_c\ \ (\text{now: }\arg\max_c \ell_c),\quad u=(\mathrm{col}\,c^*+o_x)\cdot28,\quad v=(\mathrm{row}\,c^*+o_y)\cdot28,\quad p_{\text{world}}=T_{\text{cam}\to\text{world}}\big(d\,K^{-1}[u',v',1]^\top\big)$$

with $u'=255-u\cdot\tfrac{256}{224}$, $v'=v\cdot\tfrac{256}{224}$ (mirroring back) and $d=\mathrm{clip}(1\,\mathrm{m}+0.1\,\mathrm{m}\cdot\hat z_{c^*},0.2,3)$ along the optical axis; $K,T$ are the fixed camera's calibration, not object ground truth. **Planned:** gripper orientation from $a$ and $\theta_{c^*}$; object orientation by 3D alignment (rotate the category map about the vertical axis in 10° steps, keep the best match).

**B6 Plan vector [10 numbers implemented; 16 planned]:**

$$f=\Big[\tfrac{r}{0.10\,\mathrm{m}},\ \tanh\big(\tfrac{r}{0.02\,\mathrm{m}}\big),\ a,\ e\Big]\in\mathbb{R}^{10},\qquad r=p_{\text{world}}-p_{\text{gripper}},\quad e=\pm1\ (\text{grasp / release})$$

Coarse distance, fine distance, approach, event; planned: +6D target orientation (decided by E3).

**B7 Plan hold [implemented]:** the contact stays fixed in the world until the gripper has carried out the event (5 consecutive close / open commands); each query only recomputes $r$. The action head returns 8 steps, all executed before the next query.

**B8 Plan encoder [implemented]:** MLP 10→512→512→2560 (SiLU), last layer zero-initialised (training starts from the released policy); its output is added to each of the 8 action tokens.

**B9 Action head [implemented]** — starVLA DiT, 36 blocks × 2560, no self-attention:

$$\mathrm{tok}_j=\mathrm{enc}(x_{t,j},\tau)+\mathrm{pos}_j+\mathrm{MLP}_{\text{plan}}(f),\qquad \mathrm{tok}\leftarrow\mathrm{tok}+\mathrm{CrossAttn}\big(\mathrm{AdaLN}(\mathrm{tok},\tau),\,h^{(i)}\big)+\mathrm{FFN},\quad i=1,\dots,36$$

$$\hat v=\mathrm{MLP}_{2560\to1024\to7}(\mathrm{tok}),\qquad x_{k+1}=x_k+\tfrac14\,\hat v(x_k,\tau_k),\quad k=0,\dots,3,\quad x_0\sim\mathcal N(0,I)$$

Trained (s10, VLM frozen): LoRA 26.54M + action decoder 2.63M + plan encoder 1.58M = 30.75M. No proprioceptive state input.

**B10 Execution [implemented]:** 8 × 7 actions (Δposition, Δrotation, gripper), de-normalised, to robosuite's OSC controller at 20 Hz (it does the inverse kinematics).

### 3.3 Training
""")
    code(figure("v3_training.png", 1000))
    md(r"""
**T1 Map head.** Data: `dataset10` (LIBERO-Object demonstrations, every frame a sample); `dataset_spatial` added from E2.

- **g1 labels now [implemented, ORACLE]:** contact = object position in the simulator + the object's mean demonstrated grasp offset; approach = mean motion direction over the 10 steps before the event. The instruction can be switched to another object, whose grasp point becomes the target — no new demonstrations. Loss: $\mathrm{CE}(\text{event})+\mathrm{CE}(\ell,c^\star)+10\,\mathrm{Huber}(o)+10\,\mathrm{Huber}(\hat z)+0.5\,(1-\hat a\cdot a)$.
- **L2: 3D map per object [implemented]:** 3000 surface points with normals $n$ on the visual mesh, in the object frame. Each held grasp (≥ 5 steps) gives the gripper centre $g=R_o^\top(p_{ee}-p_o)$ and closing axis $y=R_o^\top R_{ee}e_{\text{close}}$ (simulator object pose, used for labels only). Weight of one grasp on a surface point (below):
""")
    code(figure("v3_detail_4_contact.png", 600, "DETAIL"))
    md(r"""
$$w_g(x)=\exp\!\Big(-\tfrac{d_\perp^2}{2\,(1.2\,\mathrm{cm})^2}\Big)\cdot\mathrm{clip}\Big(\tfrac{|n\cdot y|-0.3}{0.7},0,1\Big)\cdot\mathbf 1\big[|(x-g)\cdot y|\le5\,\mathrm{cm}\big]$$

$$S(x)=\frac{\sum_g w_g(x)}{\max_{x'}\sum_g w_g(x')},\qquad M(x)=\sum_g w_g(x)\,y_gy_g^\top\ \Rightarrow\ \text{axis}=\text{top eigenvector},\quad A=\frac{3\lambda_{\max}/\mathrm{tr}M-1}{2}$$

$d_\perp$ is the distance to the closing line; $yy^\top$ makes the axis 180°-symmetric; $A\in[0,1]$ is the axis agreement. Projected into any frame (visible points of each object only), it gives per cell a score $\max S$ and a direction $(\cos2\theta,\sin2\theta)$ for **every** object.

- **L2c: category-level [planned, round 5]:** align all meshes of a category, $\tilde x=\mathrm{diag}(\tfrac1\rho,\tfrac1\rho,\tfrac1h)R_z(\varphi)(x-b)$ (base centre $b$, height $h$, radius $\rho$, handle to +x), pool their grasps into one map, and move it to a new mesh by nearest point. Tall/short/fat/thin test set: $x'=Sx$, $y'=S^{-1}y/\lVert S^{-1}y\rVert$. Check: leave one of the three mugs out.
- **v3 loss [planned]:** BCE on every object's scores + CE of the gated target (updates only the gate, so "which" and "where" stay separate) + direction-field L2 inside contact regions + depth Huber + event and approach.

**T2 Action head: counterfactual supervision, the core of this work [implemented, s10].** Same image, another plan, action relabelled — the head learns "go where the plan says", not the scene.

| Kind | Plan change | Status |
| --- | --- | --- |
| demonstration | unchanged, 1 cm plan noise | implemented |
| shift | contact moved in a horizontal disc of radius 18 cm | implemented |
| other object | another object's grasp point, instruction names it; last segment from that object's own demonstrations | implemented |
| release | release point moved in the same disc | implemented |
| rotate | ±45° about the vertical, translation and rotation relabelled (LIBERO-10) | planned (E3) |

Path rewrite (below): the last segment ($\min(18\,\mathrm{cm},\tfrac12\lVert c'-p_0\rVert)$) is moved to the new contact; the first part is rotated and scaled about the start $p_0$.
""")
    code(figure("v3_detail_5_counterfactual.png", 640, "DETAIL"))
    md(r"""
$$a'_{xyz}=\mathrm{clip}\Big(a_{xyz}+\tfrac{1}{s}\,\Delta\big(P'-P\big),\,-1,\,1\Big),\qquad s\approx1.28\,\mathrm{cm}\ \text{per unit action}$$

Loss (plan encoder, LoRA and action decoder only):

$$\mathcal L=\frac{\sum m\odot\lVert\hat v(x_t,t,f)-(a-\varepsilon)\rVert^2}{\sum m}+0.2\,\mathrm{Huber}_{0.5\,\mathrm{cm}}\big(\text{path of }\hat a,\ \text{target path}\big),\qquad x_t=(1-t)\varepsilon+t\,a$$

s10: 4000 steps, lr 2e-4, 454 demonstrations, 56,852 frames. Trained with oracle plans; tested with plans predicted by the map head.

### 3.4 Generalisation and diagnosis
""")
    code(figure("v3_generalization.png", 1000))
    md(r"""
Trained on standard LIBERO only; tested zero-shot on LIBERO-PRO and LIBERO-Plus.

| Test | What changes | Mainly tests | Data |
| --- | --- | --- | --- |
| G1 | object position | the map follows the object | LIBERO-PRO `_swap` |
| G2 | instruction or target | the semantic gate | LIBERO-PRO `_task`, `_lan` |
| G3 | object rotated ±45° | orientation by 3D alignment | LIBERO-10 mugs, moka pots (own perturbation) |
| **G4 (final goal)** | **unseen shape of a known category** | category-level map + map head | scaled mugs (height ×0.8 / ×1.3, width ×0.8 / ×1.2, two combinations); the red mug (LIBERO-90 only, never trained on) |
| G5 | several at once | all three parts | own combinations |

- **G4 protocol:** train on the white and yellow-and-white mugs; test unseen shapes offline and in closed loop; controls: per-mesh labels, oracle plan, released checkpoint.
- **Offline first-frame checks** (cheap, after every map-head change): object picking rate, every object's peak in its contact region, position error, closing-direction error.
- **Prototype of generalisation holds if** every G5 failure is attributed to one part, and replacing that part with ground truth recovers the success rate.

| Failure | Module to change |
| --- | --- |
| wrong object | semantic gate |
| right object, contact off | map scores, depth, finer features |
| right place, wrong direction | direction field + 3D map |
| plan right, action did not follow | action head + counterfactual labels |
| fails after the grasp | release-phase labels |
""")
