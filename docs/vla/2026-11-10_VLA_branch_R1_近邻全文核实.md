# VLA branch R1：六篇近邻工作的全文核实

**日期：2026-11-10｜来源：R1 检索 agent 的报告，原样保存（英文）｜对应 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 第 7、8 节**

核实对象是 Pipeline v2 的三个组成部分：(a) 统一的 CAW 计划，(b) 训练时可微的计划–动作一致性，(c) episode 内的物体 affordance 状态。agent 读了六篇论文的 arXiv 全文 HTML（AffordanceVLA 两个版本都读了），看了项目页和 GitHub 仓库，并用 Semantic Scholar API 做了正向引用检索；Google Scholar 没有查。

**结论：** 六篇里没有一篇完整覆盖 (a)、(b)、(c) 中的任何一项，按 v2 的继续规则可以继续推进，但定位要调整：

- (b) 最开放；
- (a) 只能主张"第一个统一的表示"；
- (c) 只剩很窄的空间：ObjDream（2610.20223）和 Ledger（2609.34554）已经在 episode 内保留每个物体的状态，而且 ObjDream 用的正是 LIBERO-Plus 和 RoboTwin 2.0 Hard。

---

**Bottom line:** none of the six papers does (a), (b) or (c) in full. (a) and (b) look open if the claim is worded narrowly. (c) is only narrowly open: two Sep–Oct 2026 papers outside the six (ObjDream, Ledger) come close.

## 1. AffordanceVLA — https://arxiv.org/abs/2606.06155 (v1 4 Jun, v2 18 Oct 2026)

| Item | Finding |
|---|---|
| Representation | Three predictions for the current keyframe sub-goal, all in the image or object frame: **Which2Act** (Flux-VAE latent of the target-object crop, 16×32×32), **Where2Act** (2D contact heatmap), **How2Act** (16³ voxel shape latent from diffusion plus a 10-DoF layout: rotation 4, scale 3, translation 3). No approach direction, no end-effector waypoints. |
| How it enters | A Mixture-of-Transformers built on π0 with three experts: understanding, affordance generation, action. The action expert attends to the affordance tokens, which also run at inference. Stage I trains only the affordance expert. Stages II/III fine-tune everything; the vision encoder gets a lower learning rate. Not LoRA. Affordance features are perturbed during training. |
| Consistency | **None.** The paper says it "contains no explicit cross-dimensional consistency or reprojection term". There is no plan–action loss. |
| State over time | **None.** It is a "single-frame model"; the paper names temporal modeling and memory as future work. |
| Results | **LIBERO** (50 rollouts): 95.8 vs Pi0 94.4; w/o Stage II 86.2; Pi0 architecture on the same InternData-A1 data without affordance: 92.4. **CALVIN ABC→D** (1000 rollouts): 4.33 vs 3.84. **RoboCasa**: 65.0 vs Pi0 62.5 / Pi0.5 62.1. **LIBERO-Plus**: 62.9 vs Pi0 45.5 / Pi0.5 84.2 (baselines re-measured by the authors). **Real robot** (15 trials/task): 88.3 vs 70.8 basic, 82.9 vs 44.8 complex. No seeds reported. Whether LIBERO-Plus was zero-shot or fine-tuned is **not stated**. |
| Text vs tables | The headline gains (+1.5% LIBERO, +38.2% LIBERO-Plus, etc.) are relative, not absolute. The text says the data-only control "improves marginally over Pi0", but on LIBERO it scores 92.4 vs 94.4. The LIBERO-Plus camera/robot gains are attributed to InternData-A1 co-training. |
| v1 vs v2 | v1's Stage I used three datasets (AGD20K, RefSpatial, PRISM). v2 says 8.8M samples and adds RoboCasa, LIBERO-Plus, the scaling study and a 2D–3D coherence appendix. The LIBERO, CALVIN, ablation and real-robot numbers are **identical** in both versions; the paper does not say whether they were re-run. |
| Failure types | No taxonomy. There is an "Empty Picks" count (11 vs 33) and a qualitative account of Pi0's toaster failures. |
| Code | https://github.com/Skywalker-yqz/AffordanceVLA (MIT): model, training and annotation pipeline. A commented-out TODO lists weights, data/annotations and real-robot code as unreleased. |
| Overlap | (a) **partial**: target object, 2D contact region and 3D object pose, but no approach direction or waypoints. (b) **none**: conditioning only. (c) **none**. |

## 2. ACoT-VLA — https://arxiv.org/abs/2601.11404 (CVPR 2026 per the arXiv comment; proceedings not checked)

| Item | Finding |
|---|---|
| Representation | **EAR** generates a coarse reference trajectory in the policy's own action space: 15 steps, every 2nd frame (so it spans about 30 steps), vs a 10-step output chunk. That is delta end-effector on LIBERO, absolute end-effector on VLABench, absolute joints on the real robot. **IAR** adds latent queries over each VLM layer's KV cache. |
| How it enters | Built on π0.5 (SigLIP + Gemma-2B). The action head cross-attends to both EAR and IAR outputs. Two variants: full fine-tuning and a frozen LLM backbone. No LoRA. |
| Consistency | **None.** EAR and the head have separate flow-matching losses (weights 0.5/0.5). Training feeds the ground-truth reference trajectory (teacher forcing); inference is self-conditioned. |
| State over time | **None** (current observation only). |
| Results | **LIBERO** (50 trials/task): 98.5 vs π0.5 96.9. **LIBERO-Plus** (10,030 episodes, one run each): zero-shot 86.6 vs reproduced π0.5 85.7; fine-tuned on LIBERO-Plus 88.0 vs 75.7. **VLABench** (frozen variant): Intention Score 63.5 vs 60.2, Progress Score 47.4 vs 43.1. **Real robot**: 66.7 vs 61.0 / 33.8. No seeds. |
| Text vs tables | The π0.5 baseline is 96.9 in Tables 4, 6 and 12 (the officially reported numbers) but 96.6 in Tables 5 and 10. "Best on all tracks" holds only across the two variants combined. |
| Failure types | None. |
| Code | https://github.com/AgibotTech/ACoT-VLA: training and inference released; checkpoints not released. The API lists Apache-2.0 while the README badges say CC-BY/MIT. |
| Overlap | (a) **partial**: an end-effector reference path, but no contact point, approach direction or object. Its Appendix D names "contact geometry" and "3D" action reasoning as future work. (b) **none**. (c) **none**. |

## 3. A4A — https://arxiv.org/abs/2609.05892 (5 Sep 2026)

| Item | Finding |
|---|---|
| Representation | 3D trajectories of query points on the manipulated object, tool or contacted part, in a clip-level reference frame. The target is the residual over constant-velocity extrapolation for h = 1..H; **H is not given numerically**. |
| How it enters | Pretraining only, by swapping the prediction space: point inputs and outputs replace the robot-state and action interfaces, then robot fine-tuning restores them. Backbones: Octo, OpenVLA, OpenVLA-OFT (rank-32 LoRA), π0/π0.5 (LeRobot ports), and Qwen3.5-4B for RLBench. Fine-tuning mode for the others is not stated. |
| Consistency | **None.** |
| State over time | Only the base policy's history context (16 frames on RLBench); π0/π0.5 use a single frame. No memory. |
| Results | **LIBERO-Object only** (50 episodes/task): Octo 28.2→57.2, OpenVLA 66.4→76.4, OFT 98.0→98.6, π0 77.8→87.8, π0.5 94.0→96.0. **RLBench** (25 rollouts): 39.0→61.0; with the ARM4R architecture held fixed: 41.0→60.0. **Real robot** (10 rollouts/skill): mean 56.7→72.5. No seeds. |
| Concerns | The OpenVLA and π0 baselines are far below the numbers cited elsewhere (OpenVLA 88.4, π0 98.8 on LIBERO-Object). The paper claims only the initialization differs between paired runs; this could **not be verified**. Its appendix cross-references are swapped. |
| Failure types | None. |
| Code | https://github.com/ru-arcl/a4a: RLBench/ARM4R pipeline, model code, data builder and a 12-clip sample. No weights, no full corpus, and no LIBERO code found. |
| Overlap | (a) **partial**: object-point motion, but no contact point, approach direction or end-effector waypoints. (b) **none**. (c) **none**. |

## 4. Wiring Matters — https://arxiv.org/abs/2610.06318 (5 Oct 2026)

| Item | Finding |
|---|---|
| Representation | One 2D Gaussian heatmap per camera view (third-person and wrist), centred on the projection of the 3D contact point of the **next** object the gripper touches. It comes from MuJoCo contact replay, stays constant within a stage and switches at stage boundaries. |
| How it enters | A DPT-style auxiliary head reads the backbone through a stop-gradient. An intermediate head feature is added back into the action expert's visual tokens through a randomly initialized residual bridge, so the head also runs at inference. Backbone: StarVLA QwenPI (Qwen3-VL-2B). **Full fine-tuning**, single stage, loss = action loss + λ·affordance loss. |
| Consistency | **None** (feature bridge only). |
| State over time | **None.** |
| Results | LIBERO, one seed per cell, 50 trials/task: base 93.1 → 96.2 (Goal +5.6, Long +6.2). Without stop-gradient: 85.5. |
| Text vs tables | The 85.5 run is from a "preliminary configuration". §III-C says goal success collapsed "below 32%", but the table shows 62.0. The "−28" result is not in any table. The AffordanceVLA comparison mixes backbones and data, and calls AffordanceVLA's three-stage curriculum "two-stage". An author affiliation still reads "TODO". |
| Failure types | Qualitative only (right target, but precision errors at the second stage). |
| Code | None linked; GitHub search found none. |
| Overlap | (a) **partial**: next-contact location, but 2D and with no approach direction or waypoints. (b) **none**. (c) **none**. |

## 5. Pri4R — https://arxiv.org/abs/2603.01549

| Item | Finding |
|---|---|
| Representation | 1,024 scene and robot points. In simulation they come from the mesh inside a robot-centred cube; on the real robot from an off-the-shelf 3D tracker with world coordinates from a fixed camera. Target: per-step 3D displacements over the action horizon H. |
| How it enters | **Training-only** head (two MLPs) on backbone embeddings, conditioned on the current points; discarded at inference. OpenVLA-OFT is trained with LoRA (rank 32); π0/π0.5 are fully fine-tuned. |
| Consistency | **None.** |
| State over time | **None.** |
| Results | **LIBERO** (500 trials/suite): OFT 92.7→96.3 (Long 85.5→95.3), π0 87.4→90.6, π0.5 92.6→94.0. The ± values are reported but their meaning is **not stated**. **RoboCasa** (50 trials): OFT 33.1→46.3, π0.5 52.9→57.0. |
| Text vs tables | The abstract's "+40% RoboCasa" is relative (+13.2 absolute). The OFT LIBERO numbers disagree across Tables I, VI and S6. Batch size is 128 in the text vs 64 in Table S3. "Outperforms on all real tasks" ignores two cells tied at 0.0. |
| Failure types | Qualitative only: collisions, wrong grasp location, closing on empty space. |
| Code | Project page https://jiiiisoo.github.io/Pri4R/ says code and models are "Coming Soon". |
| Overlap | (a) **weak partial**: scene flow, not a plan. (b) **none**. (c) **none**. |

## 6. CoRAct — https://arxiv.org/abs/2610.39543 (29 Oct 2026)

| Item | Finding |
|---|---|
| Representation | A numeric reasoning trace (embodied chain-of-thought) per action chunk: endpoint translation, rotation and gripper command, plus **one 2D pixel waypoint** of the future gripper position. Chunk = 8 steps of 7-D Cartesian actions. A world branch predicts the next image latent conditioned on the action. |
| How it enters | The trace is a prefix to the action expert, on a Mixture-of-Transformers built from JanusFlow-1.3B, fully fine-tuned. Stage 2 freezes the backbone and trains a 512-d noise-prior policy with GRPO. |
| Consistency | **Yes, but not differentiable.** Rewards: movement error between action and trace, plus agreement between a Grounding-DINO waypoint detector run on the predicted future image and the trace's waypoint. They are applied to G=8 sampled candidates, with no backpropagation through action generation. |
| State over time | **None.** |
| Results | RoboCasa 17 tasks × 30 trials; VLABench 10 tasks × 50 trials. VLABench in-distribution 51.8 vs π0 47.0; RoboCasa in-distribution 75.5 vs π0.5 72.6. Ablation: 39.2/68.6 → 51.8/75.5. No seeds. GRPO is trained on a 5% split. |
| Text vs tables | Claims "strongest across settings", but π0 is better on VLABench texture (32.2 vs 30.0) and it only ties LaST0 on the commonsense track (34.4). |
| Failure types | None; reports alignment metrics instead (position error in cm, waypoint error in pixels). |
| Code | Project page says "Code pending". The paper says the dataset is public, but no dataset link was found on the page. |
| Overlap | (a) **partial**: endpoint motion plus a 2D waypoint, no contact point, approach direction or object. (b) **partial**: explicit but reward-based in noise space. (c) **none**. |

## Forward citations and related Sep–Nov 2026 papers

Semantic Scholar citation counts on 2026-11-10: ACoT 58, Pri4R 13, AffordanceVLA 13, A4A 3, Wiring Matters 0, CoRAct 0.

About 60 Sep–Nov citing papers were screened by abstract and 8 skimmed in full text. **None combines two of (a)/(b)/(c) in full.** Partial combinations:

| Paper | Overlap |
|---|---|
| CoRAct 2610.39543 | (a) partial + (b) partial |
| **ObjDream 2610.20223** | (c) substantial: per-episode object slots with persistent identity, geometry and visibility over a causal frame window; plus a training-only action-conditioned future object-state loss (a weak form of (b)). Uses LIBERO-Plus and RoboTwin 2.0 Hard (8 tasks). |
| HINT 2609.02653 | (a) target object only + (c) one tracked target through occlusion; training-free wrapper |
| AffordanceWAM 2609.22332 / GIFT 2609.04193 | (a) partial only; GIFT's per-object slots are current-frame training targets, not carried state |

Four related papers **outside** the citation set, found by arXiv keyword search (not exhaustive):

| Paper | Overlap |
|---|---|
| **Ledger 2609.34554** | (c) substantial: an external per-object record from a SAM3 tracker (2D positions, rest intervals, covered/occluded events) read by an LLM planner |
| **Fast Plans, Faithful Actions 2609.30833** | (a) partial: end-effector waypoints cut at gripper transitions + (b) weak: conditioning plus anti-shortcut training, no consistency loss |
| GAM 2609.23863 | (a)+(c) partial: object-centric 3D grounding plus state history; uses LIBERO-PRO and RoboTwin 2.0 |
| StateMem, SmoLSTM, CondenseVLA, AWAM | Generic memory; AWAM reports on LIBERO-Occ |

## Verdict

None of the six fully covers (a), (b) or (c). None reports wrong-object, wrong-direction or collision rates, and none evaluates on LIBERO-Occ, LIBERO-PRO or RoboTwin 2.0 Hard.

**(a)** is open only as a *unified* 3D tuple: contact point, unit approach direction, and K end-effector waypoints to the next gripper event, tied to the target object and contact region. Each piece exists somewhere already: target object and contact region (AffordanceVLA, Wiring Matters), an end-effector reference path (ACoT-VLA), gripper-segmented waypoints (Fast Plans), a 2D waypoint (CoRAct). The claim should therefore be "first joint representation", not "first to predict contacts or waypoints".

**(b)** looks most open. No differentiable loss tying the chunk's implied end-effector path to a predicted plan was found. CoRAct does reward-based alignment in noise space, Fast Plans uses conditioning, and AffordanceVLA explicitly has no consistency term. Word it as "differentiable training-time plan–action consistency", and include CoRAct-style and Fast-Plans-style baselines.

**(c)** is narrowly open. ObjDream and Ledger already carry per-episode object state with visibility or occlusion. What appears unclaimed is a lightweight learned in-policy slot holding a 3D last-known position, an affordance feature and a held/moved status, updated per chunk. That needs an ablation against ObjDream-style slots and a plain frame-history baseline.

**Not verified:** Google Scholar coverage; completeness of Semantic Scholar; A4A's baseline protocol; AffordanceVLA's LIBERO-Plus protocol; what Pri4R's ± means; ObjDream, HINT and Ledger were skimmed, not read in full.
