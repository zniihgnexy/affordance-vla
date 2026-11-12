# VLA branch R9：“动作是否服从中间表示”与反事实训练的近邻核实

**日期：2026-11-12｜来源：五个检索 agent 的报告（英文，内容未改动，只删去了重复的链接列表）｜对应 2026-11-12 确定的新论点（见 [README](README.md) 当前状态）**

2026-11-12 项目负责人同意把论点收束为“affordance 只有在动作被训练成服从它时才有用”。这次核实要回答三个问题：这个现象是不是已有人报告过；“图像不变、把接触点换到别处、解析地重写动作标签”的做法是不是已有人做过；官方 benchmark 上还有多少空间。读的都是 arXiv 全文，Google Scholar 没有查。

## 结论

1. **现象已经有人立住，不能再当贡献。**
   - Fast Plans, Faithful Actions（2609.30833）在 π0.5 加路点计划的设置里做了干预实验：把计划终点抹掉，成功率不变，动作只变 0.6%。原因也写了：动作模块同时看得到图像和语言，训练损失里没有任何东西迫使它用计划。
   - 语言被绕过也有人做过：CAST、LangForce、LIBERO-CF；LIBERO-PRO 和 LIBERO-Plus 两篇本身也报告了这一点。
2. **已有的修法分四类。**

   | 类别 | 代表 | 做法 | 没做的 |
   | --- | --- | --- | --- |
   | 噪声、丢弃、深层注入 | Fast Plans | 给计划终点加噪声、15% 换成空计划、把计划注入每一层；动作标签不变 | 只在标准 LIBERO 上测；没有“把计划挪到别处，夹爪跟不跟”的测试；抹掉计划后仍有约 88.8% 成功 |
   | 结构性瓶颈 | GAM；RT-Trajectory、Hi Robot、HAMSTER | 动作模块只看得到目标物体框内的图像，或者不给语言 | GAM 不是 VLA，用仿真深度和专门微调的 3D 检测器；接口是物体框，没有接触点 |
   | 反事实标签 | CAST、SIDO、CounterAlign | CAST 改语言，动作由另一个策略给；SIDO 改物体位置通道，解析地改写动作；CounterAlign 用换过的样本训练奖励 | 都没有把接口换成另一个物体并改指令；都不是 affordance 或接触点 |
   | 推理时干预 | Affordance Field Intervention、CAG | 不训练，测试时用 affordance 场或引导纠正动作 | 只测了 LIBERO-PRO 的位置扰动和两个任务集 |
3. **离我们做法最近的是 SIDO（2607.27890）。** 它已经有核心配方：图像不变、改一个显式的位置输入、用斜坡把动作块解析地改写。区别在于：
   - 它的通道是跟踪到的物体位置，不是 affordance 给出的接触点，也没有接近方向和抓放标志；
   - 位移很小，是为了跟上移动的物体；没有大范围挪动，也没有换成另一个物体并改指令；
   - 策略不是 VLA，没有语言，没有 LIBERO 结果，没有在静止场景上测服从性。
4. **“图像不变、把接触点换到别处或换成别的物体并改指令、解析地重写动作”这一整套，没有查到完全相同的工作。**
5. **官方 benchmark 上的空间集中在 LIBERO-PRO 的位置扰动和任务扰动。**
   - 各篇论文报告的 π0.5 基线互相对不上（见下表），所以基线必须自己按同一协议重跑。
   - 目前要比的线：只靠训练的方法里，GAM 和 CounterAlign 最强；Affordance Field Intervention 在测试时用了深度和 GPT-4o。
   - 语言改写一项已经饱和。LIBERO-Plus 上 π0.5 总分 85.7，空间小。

   LIBERO-PRO，四个数依次是 Spatial / Object / Goal / LIBERO-10：

   | 方法（数字来源） | 位置扰动 | 任务扰动 |
   | --- | --- | --- |
   | π0.5（GAM 论文的表） | .20 / .17 / .38 / .08 | .01 / .01 / 0 / .01 |
   | π0.5（CounterAlign 论文的表） | .53 / .19 / .29 / .06 | .55 / .10 / .17 / .17 |
   | π0.5（Affordance Field Intervention 论文） | 54.0 / 56.4 / 没测 / 没测 | 没测 |
   | GAM | .60 / .47 / .19 / .05 | .88 / .50 / .16 / .08 |
   | CounterAlign | .60 / .51 / .41 / .11 | .63 / .26 / .46 / .29 |
   | π0.5 + Affordance Field Intervention | 75.7 / 73.2 / 没测 / 没测 | 没测 |

## 对论点的影响（我的判断）

- **方向不变，重心要挪。** 现象本身不算贡献。贡献要落在两件事上：
  1. 反事实 affordance 监督与另外三类修法的正面比较，比的是“把接口挪到别处后动作跟不跟”和分布外的成功率；
  2. 由模型预测的 affordance 驱动时，在 LIBERO-PRO 位置和任务扰动上的结果，要明显超过 GAM 和 CounterAlign，并且在 Goal 和 LIBERO-10 两个任务集上也成立（这两个正是 GAM 的弱项）。
- **新颖性比 2026-11-11 评审时判断的薄。** 机制上有 SIDO 这个近邻，现象上有 Fast Plans。NeurIPS 的把握下降，取决于第 2 条的数字；IROS 的窄版本更现实。
- **新增必须有的对照：** Fast Plans 式（噪声加丢弃，标签不变）、SIDO 式（小位移、斜坡改写）、GAM 式瓶颈或 HINT 式目标高亮（视可实现程度）、CounterAlign 和 Affordance Field Intervention 的公开数字。
- **新增必须自己跑的基线：** 我们的 backbone 和 π0.5 在 LIBERO-PRO 四个任务集、四种扰动上的数字，同一协议。

## 没有核实的

- Google Scholar 没查；检索不保证穷尽。
- SIDO 在仿真里物体位置的来源论文没写；Affordance Field Intervention 的每任务试验次数、在 LIBERO 里深度怎么来的都没写，正文和表格的数字互相矛盾。
- CounterAlign 的“100 次试验”是每个任务还是每格没有说明。
- ECoT-Lite 的一条结论来自二手资料。

---

## Report 1 — Fast Plans, Faithful Actions (arXiv 2609.30833v1, 25 Sep 2026)

Fetched in full (the only version; 13 pages, no appendix). Title: "Fast Plans, Faithful Actions: Closing the Planning–Execution Gap in Hierarchical Vision-Language-Action Models" (Xie et al., NTU).

**1. Plan and architecture (Sec. 3, 4.1, 4.3, 4.4)**
- The plan is up to 7 waypoints (q, g, d): target robot configuration (6-D end-effector in LIBERO, 14 joints on the real robot), gripper open/closed, and duration 1–32 steps; d=0 ends the plan. It is not image-space and not a contact point.
- Segments are cut offline by AWE dynamic programming, with gripper transitions forced as keyframes and length capped at 32.
- π0.5's PaliGemma backbone (SigLIP + Gemma-2B) predicts the plan as discrete tokens, token-by-token or one waypoint per pass.
- The plan enters the 300M flow-matching expert as suffix tokens (baseline), plus "NGM": normalised displacement (q−s)/σ and a gripper embedding go through a zero-initialised MLP, and the residual is added to every layer's AdaRMS condition, gated by flow time. The expert also cross-attends to the image/language prefix.
- Training is LoRA rank 16 on backbone, vision encoder and expert (1.27–1.35% of 3.66B), from released π0.5.

**2. Anti-shortcut training (Sec. 4.3)**
- (a) No image or language masking.
- (b) Yes: "we add Gaussian noise to the endpoint (σc=0.7 in the normalized Δ^ domain) while the action supervision remains unchanged"; "with probability 0.15 we replace the goal with a learnable null embedding". A fixed phase gate also limits the deep route to coarse denoising.
- (c) No. Labels are always the demonstrated actions. The shortcut they fight is the opposite one: with naive deep injection the expert integrates the displacement "without having to look at the images".
- They name the cause in passing (the plan is ignored "especially when each training context corresponds to only one endpoint", Sec. 3) but do not add other endpoints.

**3. Evidence the plan is ignored (Sec. 6.3, Table 4, LIBERO-Long, suffix conditioning)**
- "the change in actions caused by erasing the planned endpoint amounts to only 0.6% of the change caused by re-sampling the noise".
- "erasing every spatial endpoint in the plan causes no drop in task success, i.e., U=0.0" (91.0% either way).
- The same expert responds to an image swap (281%), a language swap (158%) and duration +4 steps (19%).

**4. Test-time intervention (Sec. 3, Table 4)**
- Yes, but erasure only. "we fix o, ℓ, si, and ϵ and change only the waypoint".
- Sensitivity S: median RMS action change under the edit, divided by that under flow-noise resampling (open loop, 64 segments).
- Utility U = SR(P) − SR(P̃), "where q~i:=si" (closed loop, 500 paired episodes).

| System | S endpoint | S duration | Image response retained | Success intact | U |
|---|---|---|---|---|---|
| Suffix | 0.6% | 19% | 100% | 91.0 | 0.0 |
| Naive injection (Token-AR) | 348% | 106% | 27% | 15 | +14 |
| NGM | 42.6% | 26.4% | 82% | 96.2 | +7.4 |

- No test moves the plan elsewhere or to another object and checks the gripper goes there.
- With NGM, erased endpoints still give about 88.8% success (derived from the table).
- Gate, noise and dropout are not ablated separately.

**5. Benchmarks**
- LIBERO four suites only, in-distribution: one model on the union (1,693 episodes), 500 episodes per suite, and "each variant was trained only once".
- Averages: π0.5 as reported 96.85; Token-AR + suffix 96.70; Block-AR + suffix 95.85; Block-AR + NGM 98.45 (Long 96.2 vs 91.0).
- Real Rokae dual-arm, 3 tasks, 20 trials each: 48/60, 46/60, 51/60.
- No LIBERO-Plus, LIBERO-PRO, RoboTwin or perturbed sets.

**6. Consistency loss:** none. Only plan-token cross-entropy plus flow matching (λ=1.7).

**7. Code/weights:** none. No repo or project page in the paper, on the arXiv page, or via web search.

**8. Citations (Sec. 2.3 "Conditions ignored by the policy", Table 1)**
- Causal confusion in imitation learning (de Haan 2019) and Fighting copycat agents in behavioral cloning (Wen 2020).
- LangForce (2601.15197) and CAST (Glossop 2026); both concern language being ignored.
- Classifier-free guidance (2207.12598), Guided flows (2311.13443), Hindsight experience replay.
- Plan interfaces: HAMSTER ("failure attribution"), 3D HAMSTER (2606.31329), GAE (2510.03896, "goal-noise ablation"), NoTVLA (2510.03895), Coarse-to-Control (2606.07107).
- No prior work is cited specifically for "the executor ignores the plan".

**Verdict**
1. Already here: the diagnosis that ordinary training leaves the plan endpoint unused because images are redundant, with numbers (S=0.6%, U=0).
2. Already here: faithfulness measured by test-time plan intervention with other inputs fixed.
3. Not here: counterfactual plans with rewritten action labels. Their fix is deep injection plus noise, dropout and a gate, with unchanged labels.
4. Not here: obedience as "move the plan, the gripper goes there". They only erase endpoints or lengthen durations, and S measures action change, not target reaching.
5. Different setting: configuration waypoints with durations rather than a 3D contact with approach and grasp flag; in-distribution LIBERO, single seed, no code.

## Report 2 — ObjDream, HINT, GAM

All three papers were read in full. Two corrections to our earlier notes: HINT is not training-free and has no simulation benchmark (its policy is fine-tuned with the interface in place and evaluated on real robots only); no paper reports both LIBERO-Plus and LIBERO-PRO.

### ObjDream (2610.20223v1)

1. **Mechanism.** A frozen object encoder turns one camera view into object slots, inserted as tokens into a Qwen3-VL-4B VLA with a flow-matching action expert (§3.2). Auxiliary heads decode current and future object boxes and masks from those tokens, in training only (§3.3).
2. **Forcing.** None. A learned attention bias favours object tokens (Eq. 9) without a routing loss, and the VLM still sees full images.
3. **Intervention.** None on the policy; Table 5 perturbs only the auxiliary decoder. Removing the object interface costs 3.2 points on LIBERO-Plus (Table 2).
4. **Protocol.** LIBERO-only training, same checkpoint zero-shot on LIBERO-Plus; trials and seeds not stated. Table 1, Cam/Robot/Layout/Lang/Light/BG/Noise/Total:

| Method | LIBERO-Plus |
|---|---|
| π0 | 13.8/6.0/68.9/58.8/85.0/81.4/79.0/53.6 |
| OpenVLA-OFT | 56.4/31.9/74.2/79.5/88.7/93.3/75.8/69.6 |
| X-VLA | 23.4/89.7/71.8/75.7/88.2/96.0/62.7/71.4 |
| VLA-JEPA | 63.3/67.1/85.1/85.4/95.6/93.6/66.3/79.5 |
| OA-WAM | 80.5/89.6/82.8/85.3/96.5/95.9/75.6/83.9 |
| π0.5 | 75.4/77.5/85.7/85.6/96.9/94.6/89.7/85.7 |
| Backbone only (Table 2) | 66.4/74.7/80.0/87.1/91.7/92.4/82.8/81.2 |
| ObjDream | 66.2/75.8/81.2/90.6/98.4/98.5/88.6/84.4 |

   RoboTwin 2.0 (Fig. 4; 8 tasks, 50 clean demos, 100 rollouts each): Hard 22.75 vs starVLA 7.13, Easy 60.25 vs 47.38 (starVLA derived from stated gains). Other baselines: DP and FastWAM only.
5. **Inputs.** RGB, proprioception, language; Qwen3-VL-8B, CLIP, DINOv2 and a SAM2 backbone at test time. No depth or calibration.
6. **Failures.** None reported.
7. **Code.** https://github.com/HuailiangMa/ObjDream says "code coming soon"; no weights.

### HINT (2609.02653v2)

1. **Mechanism.** Qwen3-VL-8B picks the subtask and target whenever a small trained router detects a manipulation-phase change; Grounding DINO or Qwen localizes the target and SAM2.1 tracks its mask (§III-B). The mask reaches π0.5 or Wall-OSS-0.5 as an image overlay and an attention bias (§III-C).
2. **Steering.** The policy is not frozen. "Parameter-free" means no added parameters: each backbone is fine-tuned for 50k–80k steps on demonstrations rendered with ground-truth target overlays and patch maps (Supp. I.1). The bias is a fixed logit (strength 1.0, per repo) added to attention on target patches, in the vision encoder and from action tokens. The prompt becomes the subtask instruction. No counterfactual samples; no inference-time guidance.
3. **Intervention.** The highlight is never edited directly. Closest: in identical scenes, when the instruction changes the category-to-basket mapping, base π0.5 subtask success falls from 89.5% to 52.2% (chance) and HINT adds 38.5 points (Fig. 5). Table IV (OOD spelling, target selection): base 40.9, attention only 59.1, highlight only 88.6, both 95.5.
4. **Benchmarks.** None of the three. Three real bimanual tasks, 50–150 demos, 15–20 ID and 10 OOD trials. Full-task success, π0.5 vs π0.5+HINT: ID 10/13.3/5 vs 60/86.7/40; OOD 0/0/0 vs 30/30/30 (Tables I, II).
5. **Inputs.** Three RGB views; Qwen3-VL-8B, Grounding DINO, SAM2.1. No depth described.
6. **Failures.** Yes: first cause per failed π0.5+HINT trial (Table S6), as reasoning/pick/place/contact. No drop or collision rates.
7. **Code.** https://github.com/robot-hint/HINT: perception, router and π0.5 training code; no trained weights.

### GAM (2609.23863v2)

1. **Mechanism.** Not a VLA. A frozen promptable 3D detector (WildDet3D) turns language, points or boxes into per-object 2D boxes, metric 3D boxes and depth (§III-A). The policy sees image tokens only inside object boxes and the robot silhouette, plus per-object detection tokens (centre, extents, rotation, point cloud) (§III-B). Only a flow-matching transformer is trained.
2. **Forcing.** A structural bottleneck: everything outside the grounded objects is zeroed or cropped. The repo adds image-token dropout 0.3 and state dropout 0.2, and the released LIBERO-PRO policy takes no language (`lang_mode: none`, despite Eq. 2), so role-ordered slots [pick0, place0, pick1, place1] are its only task specification. No counterfactual relabelling.
3. **Intervention.** End-to-end only: LIBERO-PRO Task (new target) and Pos (target moved). Real robot: click-selected target among same-coloured objects, 17/20 under visual shift vs 4/20 for π0.5 (Table III).
4. **Protocol.** LIBERO-PRO: one policy on standard LIBERO, zero-shot; leaderboard baselines; no seeds. Repo: 50 episodes per task, and an instruction-to-slot lookup table produced the reported numbers. Table II, Spatial/Object/Goal/10:

| | Obj | Pos | Sem | Task | Avg |
|---|---|---|---|---|---|
| π0 | .95/.94/.94/.79 | 0/0/0/0 | .97/.90/.93/.82 | 0/0/0/0 | .45 |
| OpenVLA | .97/.98/.96/.81 | 0/0/0/0 | .97/.98/.98/.96 | 0/0/0/0 | .48 |
| X-VLA | .97/.89/.68/.62 | 0/.02/.01/0 | .96/.98/.98/.95 | 0/.08/.09/.10 | .46 |
| Flex-π | .99/.97/.76/.77 | .26/.05/.11/.05 | .97/1.0/.97/.99 | 0/0/.10/.10 | .51 |
| π0.5 | .97/.98/.97/.92 | .20/.17/.38/.08 | .97/.96/.97/.93 | .01/.01/0/.01 | .53 |
| GAM | .96/.88/.69/.50 | .60/.47/.19/.05 | .95/.99/.97/.89 | .88/.50/.16/.08 | .61 |

   No OpenVLA-OFT row. RoboTwin 2.0 (Table I; 50 tasks, per-task policies on 50 clean demos, 100 rollouts), Easy/Hard: GAM 63.0/47.6, Abot-M0 57.4/30.4, Spatial Forcing 77.2/26.7, π0.5 64.0/25.9, π0 46.4/16.3, starVLA 46.5/3.2. The detector was first fine-tuned per simulator on ground truth (boxes, depth); whether on perturbed scenes is [UNVERIFIED].
5. **Inputs.** RGB, camera calibration, robot URDF and joint state; Flan-T5 span head, WildDet3D, CoTracker3. Repo: LIBERO-PRO also feeds simulator depth to the detector.
6. **Failures.** None quantified.
7. **Code.** https://github.com/GehaoZhang6/Grounded-Action-Model: training and evaluation code; policies, tuned detectors and LIBERO cache at huggingface.co/gehaozhang.

### Cross-paper summary

- **LIBERO-Plus total:** π0.5 85.7, ObjDream 84.4. **Layout:** π0.5 85.7, VLA-JEPA 85.1.
- **LIBERO-PRO position:** GAM .60/.47 on Spatial/Object; π0.5 .38 on Goal, .08 on LIBERO-10.
- **LIBERO-PRO task:** GAM .88/.50/.16 on Spatial/Object/Goal; X-VLA and Flex-π .10 on LIBERO-10; no baseline above .10.
- **LIBERO-PRO language:** saturated (Flex-π .97–1.0). **RoboTwin Hard:** GAM 47.6.
- GAM is closest to an explicit where-to-act interface the policy cannot bypass. It lacks a contact point or grasp/release signal (boxes only; weakest when object size changes), counterfactual training or any test moving the interface independently of the scene, a VLM backbone, and RGB-only operation. HINT is the nearest 2D analogue.

## Report 3 — Does the action module obey an explicit spatial interface? (prior work)

All arXiv IDs below were checked. "Hindsight" means the labels come from the executed trajectory.

**Named earlier works**

- **RT-Trajectory (2311.01977).** (i) A 2D/2.5D end-effector sketch, concatenated with RGB. The policy gets no language: "Since the language instruction is not used, we remove the FiLM layers". (ii) Sketches are hindsight, from proprioception. Dropping language is what makes the sketch necessary. (iii) Qualitative only: "changing trajectory sketches induces RT-Trajectory to change behavior modes in a reproducible manner".
- **RT-Affordance (2411.02704).** (i) End-effector poses at key stages, drawn as hand outlines on the image. The policy also gets the image and language: "condition the policy on both affordance plans and language". (ii) Hindsight labels (gripper-close and final timesteps). No dropout or language removal. (iii) "The policy is faithful in executing the human provided affordance poses" (qualitative). Grasping success is 76% with oracle affordances and 68% with predicted ones. No edit test.
- **HAMSTER (2502.05485).** (i) A 2D path drawn on the image, plus "(optional) language". (ii) Paths come from projected proprioception (hindsight), with "a small amount of random noise". "We remove the language instruction for RVT-2". For 3D-DA, removing language hurt. (iii) "72% of failures stemmed from the low-level model failing to follow the trajectory". No edit test.
- **MolmoAct (2508.07917).** (i) Depth tokens, then trace tokens, then action tokens, all in one sequence. Actions see the image, the instruction and the trace (Eq. 4). (ii) Hindsight traces. Auxiliary data maps "the language T and the trace-overlaid image" to the action, "mainly for enabling the steerability feature". (iii) Edit test: a user-drawn trace gets "a success rate of 75%" over 15 trials, 33 points above steering by language. The paper reports no ignoring.
- **TraceVLA (2412.10345).** (i) A trace of PAST motion, overlaid on the image, plus the raw image and the instruction. It is not a plan. (ii) "with probability α, we replace the visual trace prompt image with the original image".
- **RoboPoint (2406.10721)** and **A0 (2504.12636)** have no learned action module (a motion planner or GraspNet executes). The shortcut question does not apply.
- **CoA-VLA (2412.20451).** (i) Affordances (text and visual) are FiLM-injected into a diffusion head that also gets the VLM's image and instruction features. (ii) Automatic labels; no dropout. (iii) No edit test. Removing the visual affordance only drops LIBERO success from 79.8% to 78.6% (the agent's inference: the policy relies on it weakly).
- **Hi Robot (2502.19417).** (i) The low level gets only the high-level command. (ii) Demos are segmented into skills after the fact. (iii) With commands typed by a human, "the low-level policy executes nearly flawlessly" (qualitative).
- **π0.5 (2504.16054).** (i) The same model decodes the subtask and then the actions. The text says "the action distribution does not depend on ℓ, only on ℓ̂", but Eq. 1 includes ℓ [UNVERIFIED which one is implemented]. (iii) "Implicit HL" (no subtask inference at test time) ranks second-best (Fig. 13; numbers [UNVERIFIED]).

**2024–2026 papers that study or fix the ignoring**

1. **Fast Plans, Faithful Actions (2609.30833).** Diagnosis, intervention metric and a noise/null-goal fix (see Report 1).
2. **Do What You Say (2510.16281, Oct 2025).** "even with a correct textual plan, the generated actions can still miss the intended outcomes". The fix samples actions and verifies them at test time.
3. **ECoT (2407.08693).** "the model *could* predict a particular plan and then still deviate from it". Human edits to the reasoning give +48% on the hardest tasks.
4. **ECoT-Lite (2505.08243).** With reasoning dropout, it matches full ECoT without reasoning at test time [UNVERIFIED, secondary sources].
5. **Do VLAs Mean What They Say? (2607.04681, Jul 2026).** Driving. Finds "an inconsistent coupling between reasoning quality and downstream trajectory improvement". The fix is an RL reward from a faithfulness critic.
6. **Reasoning as a Double-Edged Sword (2607.17786).** Perturbs the reasoning stage. Its plan-action consistency probe "falls to chance under adaptive attack".
7. **The same mechanism with language as the redundant input:** CAST (2508.13446): "the policy learns to ignore language, instead inferring the task and actions purely from observations". LangForce (2601.15197): "language instructions are highly predictable from visual observations alone". LIBERO-CF (2602.17659): counterfactual instructions.

Older precedents: causal confusion (de Haan 2019), copycat agents (Wen 2020). ATM (2401.00025) also drops language for its track policy.

**Verdict**
- **Largely established.** That BC ignores a redundant input, and that breaking the redundancy fixes it, is already shown for language (CAST, LangForce, LIBERO-CF) and for a spatial waypoint plan in a VLA (2609.30833).
- Earlier spatial-interface systems avoid the problem by dropping language from the low level (RT-Trajectory, ATM, Hi Robot, HAMSTER with RVT-2).
- **Still open:** affordance, contact-point and 3D-trajectory interfaces; training on counterfactual or synthetic interfaces, beyond noise and dropout; measuring obedience as success under edited interfaces, across interface types.
- **Most likely cited against a novelty claim:** 2609.30833, CAST (2508.13446) and HAMSTER (2502.05485).

## Report 4 — Counterfactual relabelling (prior work)

No paper was found that does all of the following together: keep the image, edit an explicit 3D contact or affordance input, rewrite the action by retargeting the demonstrated end-effector path analytically, and do this so the policy follows the interface. Each ingredient has been published separately. The closest single paper is SIDO (2607.27890).

**Named works**

- **CAST (2508.13446).** Rewrites the instruction and the action; observations are reused ("As no new observations are generated for these actions."). A VLM proposes a counterfactual task and an atomic command, and a learned atomic policy produces the action; nothing is retargeted analytically. The goal is obedience, explicitly: "We also need different instructions to lead to different actions, even when the observations are otherwise similar." Mainly navigation; v2 adds real-world manipulation on Bridge WidowX with distractors, roughly a 2x gain. No LIBERO results.
- **MimicGen (2310.17596).** Transforms object-centric segments to the new object pose, then executes them ("only keeps successful demonstrations"). Needs a simulator rollout, which renders new observations. No interface input.
- **SkillMimicGen (2410.18907).** Skills are re-anchored and joined by motion planning; generated demos are executed and filtered by success. No interface input.
- **DemoGen (2502.16932).** Skill segments are transformed with the object; free-space motion is replanned. No rollout is needed ("purely computational"), but it edits the observation by moving the segmented point clouds. No interface input.
- **RoboSplat (2504.13175)** and **Real2Render2Real (2505.09601).** Both re-render observations from 3D Gaussian splats. Both change the image. Neither has an interface input.
- **ECoT (2407.08693).** Intervenes on the reasoning at test time (+48% on the hardest tasks); no counterfactual training.
- **ECoT-Lite (2505.08243).** Reasoning is only switched on or off, never corrupted or edited. "learning to generate reasonings does lead to better VLA representations"; "enabling test-time reasoning does improve performance compared to disabling it".
- **Classical.** Causal confusion (de Haan et al. 2019) fixes the problem "through targeted interventions" by interacting with the environment or querying the expert. Hindsight relabelling (GCSL) changes the goal to fit the fixed action; counterfactual relabelling does the reverse. Copycat (Wen et al. 2020) is the same kind of shortcut as memorised scene layouts.

**2025–2026 hits**

- **SIDO (2607.27890).** See Report 5.
- **CounterAlign (2608.21740).** See Report 5.
- **LIBERO-CF / CAG (2602.17659).** Counterfactual Action Guidance at inference time; no action relabelling. +9.7% language grounding for π0.5 on LIBERO-CF.
- **Affordance Field Intervention (2512.07472).** See Report 5.
- **GTA-VLA (2605.13632, ECCV 2026).** The policy is conditioned on user-given points, boxes and traces. Training cues are "derived by projecting end-effector motion": hindsight cues with added noise, not counterfactual edits.
- **CofactVLA (2608.04396).** A language-masked counterfactual branch plus projection in feature and action space. No data relabelling. [UNVERIFIED details]
- **TREAD (2606.10918)** and **S2 (2606.02735)** relabel language only.
- **The benchmarks that motivate the problem.** LIBERO-Plus: "models tend to ignore language instructions completely." LIBERO-PRO: performance "collapses to 0.0%"; outputs "remain unchanged even when given corrupted instructions."

**Verdict**
- **Not published as a whole.**
- **Partly anticipated:** SIDO has the core mechanism (same image, an edited explicit input channel, an analytically morphed action) for object pose and moving objects. CAST has counterfactual labels on the same observation, used to fight ignored conditioning, but its actions come from a learned policy. DemoGen and MimicGen retarget segments without new demonstrations, but change the observation or need a simulator.
- **Gaps left open:** the interface being an affordance or contact point; renaming the instruction to another object that is visible in the same image; evaluation of obedience on LIBERO-PRO or LIBERO-Plus.
- **Most likely cited against a novelty claim:** SIDO, CAST and DemoGen; CounterAlign as a likely fourth because it reports LIBERO-PRO results.

## Report 5 — SIDO, Affordance Field Intervention, CounterAlign

All three papers were read in full (arxiv.org/html), plus AFI's CVPR 2026 version and the project pages.

### SIDO (2607.27890v2)
1. **Policy and channel:** not a VLA; no language. Diffusion Policy (DINOv3) and Equivariant Diffusion Policy (§5). Inputs are two camera images, proprioception and object position P_t ∈ R³ (§3). The edited channel is that position (no orientation) of the one object to be grasped, in the robot base frame (App. F). At test time a predictor writes the forecast one chunk ahead (default: finite-difference extrapolation from the last 3 observations). Real observations come from FoundationPose; the simulation source is not stated [UNVERIFIED: presumably simulator state].
2. **Counterfactuals (§4.2):** pre-grasp transitions only. δ = v·T_p·[cos θ, sin θ, 0], θ uniform: planar, fixed magnitude (0.016 simulation units; T_p=16, T_a=8, App. I); 20% of samples stay unchanged. "we keep the images ... as-is and write the displaced position into the pose channel"; "This mismatch between the unchanged images and the displaced pose channel mirrors deployment". Morph (Eq. 2): ã_{t+k} = a_{t+k} + ρ_k·δ, ρ_k = min(k, T_a−1)/(T_a−1), a linear ramp over the first T_a steps, then the full shift. DynaSIDO refines this with MPPI. No image edits, new demos or simulator rollouts.
3. **Ignoring the channel:** partly. "DP (cur. pose)" has the channel but static-only training: "the policy sees the object yet was never trained to act on its motion". Its tracking slope is 0.10, against 0.42 for SIDO and 0.62 for demos (H2, Fig. 4). No test edits the pose on a static scene; all obedience evidence uses physically moving objects.
4. **Target swap:** none; same object only.
5. **Evaluation:** MimicGen Stack, Mug, Square under 5 motion patterns; real Gantry and Peachtree. 20 rollouts per cell, no seeds. Baselines: DP with and without the channel; a test-time action shift. Stack x-axis 0.15 to 0.90; Gantry x dynamic 0.35 to 0.55; Peachtree 0.45 to 0.60 (Tables 1–3). No LIBERO.
6. **Code:** project page says "Code (TBD)"; no weights.

### Affordance Field Intervention (2512.07472v1, CVPR 2026)
1. **Mechanism:** nothing is trained ("without parameter updates"). GPT-4o names the current target, Grounded-SAM segments it, and depth plus calibrated intrinsics back-project the mask (§3.2). The field is a voxel cost, distance to the target centroid plus an obstacle term, not a contact-point map. Trigger (§4.1; thresholds from CVPR supp. C.1): end-effector displacement below 2.5 cm while over 25 cm from the target. The robot rolls back to its lowest-cost recent position, samples 8 VLA chunks at each of about 10 low-cost waypoints, and executes the cheapest (§4.2).
2. **LIBERO-PRO (Table 2):** position perturbation only, official π0.5-LIBERO checkpoint only. Spatial 54.0 to 75.7 (7 of 10 tasks listed); Object 56.4 to 73.2. No Goal or Long suites, no other perturbation types, no LIBERO-Plus numbers. Trials per task are not stated [UNVERIFIED]. How depth is obtained in LIBERO is not described. The §5.3 prose contradicts the table (52.4 to 78.2; 67.3 to 82.5), and the abstract's "20.2%" matches neither.
3. **Memory-trap evidence:** qualitative Fig. 1 ("the VLA drives the end-effector toward the original location"). Real π0 Place Carrot falls from 17/20 to 3/20 at +10 cm X and 0/20 at +15 cm Y (Table 3). Where the gripper actually goes is never measured.
4. **Limitations (CVPR supp. D):** the field "does not encode grasp orientation, contact force"; static scenes; 2 Hz updates. At +15 cm AFI reaches only 3/20 and 2/20. No code or weights link anywhere.

### CounterAlign (2608.21740v1)
1. **Mechanism:** it keeps the observation and swaps the instruction, the action chunk (for another dataset chunk), or both (§3.2). These tuples train a discriminator; a second one separates expert from policy actions, and their log outputs form the reward. π0.5 is then trained by offline RL: an IQL critic plus advantage-weighted flow matching on expert tuples only (§3.3). No action is rewritten, and actor-side relabeling did not help (Table 2).
2. **LIBERO-PRO (Table 1), cells are Obj/Pos/Sem/Task:**

| Method | Spatial | Object | Goal | LIBERO-10 |
|---|---|---|---|---|
| OpenVLA-OFT | .30/.14/.65/.00 | .67/.10/.89/.00 | .85/.00/.44/.03 | .06/.00/.39/.00 |
| VLA-Adapter | .98/.00/.98/.49 | .89/.00/.99/.08 | .61/.00/.75/.12 | .47/.00/.91/.10 |
| xVLA | .90/.00/.69/.39 | .92/.05/.98/.00 | .72/.10/.94/.08 | .61/.10/.71/.20 |
| π0.5 (BC) | .99/.53/.97/.55 | .89/.19/.95/.10 | .90/.29/.95/.17 | .66/.06/.91/.17 |
| CounterAlign | .98/.60/.97/.63 | .80/.51/.99/.26 | .82/.41/.94/.46 | .64/.11/.97/.29 |

   No π0 or OpenVLA rows. Protocol (§4.1): "We first train policies on the LIBERO dataset and evaluate them on LIBERO-PRO"; "we run 100 trials" (per task or per cell unspecified); no seeds. No LIBERO-Plus table.
3. **Code:** project page says "Code (Coming Soon)"; no weights.

### Verdict on SIDO
1. SIDO already covers the core recipe: unchanged images, an edited explicit position input, a closed-form ramp-rewritten action chunk, and a policy that then follows the input.
2. Its channel is a tracked object position, not an affordance-map contact point with approach direction and grasp/release.
3. Its shifts are small planar offsets modelling motion; no large relocations, no retargeting to another object with a renamed instruction.
4. Its morph is a linear ramp over the first 8 steps of a 16-step pre-grasp chunk, not a whole-path retarget preserving the last 8 cm.
5. It has no language, VLA or LIBERO results, and no static-scene obedience test.

### Strongest LIBERO-PRO numbers seen
- **Position:** π0.5 + AFI, Spatial 75.7 and Object 73.2, using test-time depth and GPT-4o. The training-only best is CounterAlign: Spatial .60, Object .51, Goal .41, LIBERO-10 .11. The two papers' π0.5 baselines disagree on Object (56.4 against 19), so the protocols are not comparable.
- **Task:** CounterAlign: .63, .26, .46, .29.
- **Language (Sem):** saturated.
