# VLA branch 检索结果：backbone、仿真 benchmark 与近邻工作

**日期：2026-11-10｜来源：三个检索 agent 的报告，原样保存（英文）**

这些报告用于 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 的选型与定位。表中数字大多由 agent 从论文 HTML 自动抽取，部分来自 allenai vla-evaluation-harness 汇总榜（标 [A]），没有人工逐项核对；引用前要回到原文确认。部分论文自身前后数字不一致（AFI、AffordVLA、TraceVLA、A4A）。

---

## Open VLA backbones for an affordance-aware VLA (agent report, 2026-11-10)

Tags: [P] primary source checked; [S] secondary; [A] agent's assessment. Compute target: 8×A6000 48 GB (x86, headless).

| Model | Params | Licence | Framework | Fits 8×A6000? | Official sim results | Architecture | Add modality / aux head [A] | Spatial intermediates | Adoption |
|---|---|---|---|---|---|---|---|---|---|
| π0.5 (π0) | 3.62B [P] | code Apache-2.0, weights Gemma terms | JAX openpi; LeRobot PyTorch port | LoRA >22.5 GB, full >70 GB single GPU; RoboTwin doc: LoRA 1×A6000, full 4×40 GB FSDP → yes | LIBERO 96.85 (openpi) / 97.5 (LeRobot); RoboTwin2 82.7/76.8 [S, LingBot eval]; SimplerEnv WidowX 57.1 [S] | PaliGemma + flow action expert | moderate; LM head kept → can emit affordance/waypoint tokens before expert | detection + subtask co-training | baseline in all 5 checked 2026 papers |
| OpenVLA-OFT | 7.54B | MIT | PyTorch Prismatic | LoRA 25.6 GB @bs1, 44.1 GB @bs8; slow | LIBERO 97.1 | Llama-2 + parallel decoding + L1 head | moderate; no AR path | none | last push 2025-09 |
| SmolVLA | 0.45B | Apache-2.0 | LeRobot | single GPU | LIBERO 87.3; Meta-World 57.3 | SmolVLM2 + flow expert | easy, low ceiling | none | 152k dl |
| GR00T N1.7 | 3.14B | code Apache-2.0; NVIDIA Open Model Licence | PyTorch | "A6000 works, longer training"; VLM frozen default | LIBERO 97.0; SimplerEnv Bridge 62.3, Fractal 72.5 | Cosmos-Reason2-2B cut at L12 + flow DiT | moderate; DiT conditioning easy, no LM head | none | aarch64 support documented (Thor, Spark) |
| X-VLA | 0.88B | Apache-2.0, in LeRobot | PyTorch | yes [A] | LIBERO 98.1; SimplerEnv Google VM 80.4 / VA 75.7, WidowX 95.8; CALVIN 4.43; RoboTwin2 70.0/39.0 | Florence-2 + soft prompts + flow | moderate; no LM decoder | none | budget option |
| VLA-Adapter | 0.5B | MIT | Prismatic | ~29 GB @bs8; 8 h LIBERO-Object on one card | LIBERO 97.3; CALVIN 4.42 | Qwen2.5-0.5B + bridge attention | moderate | none | |
| starVLA | backbone-dependent (Qwen3-VL-4B, Qwen3.5 0.8–9B, Florence-2) | MIT | PyTorch + DeepSpeed | recipes 8–48×A100 80 GB → likely with ZeRO + small batch | Qwen3-VL-4B: LIBERO 95.4–96.6; SimplerEnv WidowX 31.6–65.3, Google VM 76.0 / VA 70.2; RoboTwin2 clean 72.5–88.2; RoboCasa-GR1 39.0–48.8 | swappable VLM + FAST/OFT/π/GR00T heads | easy: one framework file; VLM co-training script | none | 3.9k stars, paper 2604.05014 |
| InternVLA-M1 | ~4B | code MIT, weights CC-BY-NC-SA | PyTorch | 8×A100 ~20 h/suite | LIBERO 95.9; SimplerEnv Google VM 80.7 / VA 76.0, WidowX 71.7 | Qwen2.5-VL-3B + DiT | easy–moderate | yes (box/point/trace QA) | |
| MolmoAct2 | 5.44B | Apache-2.0 | PyTorch | heavy (LIBERO 32 H100) | LIBERO 97.2 / 98.1 | Molmo2-ER + flow DiT | moderate–hard | yes (depth, trace) | |
| CogACT / RDT / FLOWER | — | MIT/Apache | PyTorch | — | see report | — | — | none | stale / little used |
| Qwen-VLA | ~5.15B | no weights or code | — | — | paper only | — | — | — | not usable |
| LingBot-VLA | 4.20B | Apache-2.0, LeRobot | PyTorch | unknown | RoboTwin2 88.6/86.7 vs π0.5 82.7/76.8 | Qwen2.5-VL-3B + π0-style expert | moderate | depth-distilled variant | |

Agent's top 3: (1) starVLA Qwen3-VL-4B + flow head (same-backbone baselines on LIBERO, SimplerEnv, RoboTwin2, RoboCasa in one codebase; Afford-VLA used this setup); (2) π0.5 via LeRobot port, checked against openpi; (3) GR00T N1.7 (best aarch64 path). Budget: X-VLA-0.9B. Avoid as base: Qwen-VLA, CogACT, RDT-1B, FLOWER; MolmoAct2 and InternVLA-M1 already contain spatial intermediates (use as related baselines).

Risks: novelty (2D affordance aux already published); LIBERO saturated (96–98%) → main claim on RoboTwin2 Hard, SimplerEnv, RoboCasa, LIBERO-Plus with ≥3 seeds + CIs; recipes assume 80 GB GPUs; retrain baseline in same code; add zeroed/shuffled-target control; SAPIEN needs Vulkan headless; keep sim eval on x86; pin commits.

Sources: openpi, LeRobot pi05 docs, RoboTwin Pi05 guide, RoboTwin 2.0 (2506.18088), OpenVLA-OFT, SmolVLA (2506.01844), Isaac-GR00T + HF GR00T-N1.7-3B, X-VLA (2510.10274), VLA-Adapter, starVLA (2604.05014), InternVLA-M1 (2510.13778), MolmoAct2 (2605.02881), CogACT (2411.19650), RDT/RDT2, FLOWER, Qwen-VLA (2605.30280), LingBot-VLA, Afford-VLA (2605.24203).

---

## Simulation benchmarks for an affordance-aware VLA (agent report, 2026-11-10)

Tags: [V] primary source; [A] allenai vla-evaluation-harness leaderboard (LLM-curated, snapshot 2026-10-20, not re-checked); [U] uncertain.

| Benchmark | Papers (total / 2026) | Top result | Headroom | Clutter | Labels (depth / cam / seg / pose / contact) | Data | Headless on A6000 | Licence |
|---|---|---|---|---|---|---|---|---|
| LIBERO | 496 / 331 [A] | 99.6–99.9% [A] | none | 2–8 movable objects (median 3) [V] | depth + instance seg via flags; raw HDF5 states replay; object XML affordance regions [V] | 50 demos × 130 tasks; HDF5, LeRobot, RLDS | MuJoCo EGL [V] | MIT |
| LIBERO-Plus | 75 / 72 [A]; CVPR 2026 | 79.6% own paper → 92.4% best 2026 [A] | shrinking; camera + robot-init perturbations hardest | confounding objects, target displacement; 10,030 tasks | as LIBERO | RLDS + LeRobot 10k+ eps | EGL | HF data MIT; repo no licence file |
| LIBERO-PRO | 23 / 19 [A] | >90% models drop to 0% generalized; best mean 69.8% [A] | large | object/state/instruction/env swaps | as LIBERO | | EGL | MIT |
| LIBERO-Occ | 1 paper (2606.10862) | OFT 47.95, π0 49.30, π0.5 40.55, VIM 65.05 [V] | large, no community | 2,000 occluded instances | as LIBERO | | EGL | MIT |
| SimplerEnv | 135 / 66 [A] | WidowX VM 98.0 (RL); Google VM 91.0 / VA 90.1 [A] | small at top | 1–3 objects; VA adds distractors | RGB-D + seg; no sim demos | none (train on real Bridge/RT-1) | SAPIEN 2 Vulkan, known segfaults | MIT |
| RoboTwin 2.0 | ICML 2026; board 33 entries (28 since Aug 2026) | official 50 clean demos × 50 tasks: best Clean 91.84 / Hard 81.48; π0.5 70.66 / 46.0 [V] | large on Hard | cluttered_table from 731 objects / 147 categories | depth, point cloud, end pose, mesh/actor seg; seed replay | 100k+ traj HDF5 → LeRobot | SAPIEN 3 Vulkan; stalls on A100/H100/V100 (#83); A6000 untested | MIT code + data |
| CALVIN | 128 / 52 [A] | ABC→D 4.78/5 [A] | small | 3 blocks + articulated | depth; scene_obs block poses | 517 GB ABC→D | PyBullet EGL | MIT |
| RoboCasa / RoboCasa365 | 81 / 58 [A]; 365: ICLR 2026 | RoboCasa 24 atomic 80.6% [A]; 365 best non-agentic 64.4, Composite-Unseen 45.2 [V] | large (365) | 2,500+ kitchens, 3,200+ objects; mobile, not pure tabletop | states.npz replay; depth/seg [U] | 2,200+ h LeRobot | MuJoCo [U] | MIT / CC BY 4.0 |
| ManiSkill3 | no common VLA protocol [U] | ManiSkill2 5-task 77.0 [A] | n/a | PickClutterYCB (no demos) | full RGB-D, seg, point cloud, intrinsics/extrinsics; demos replay to any obs mode [V] | HDF5 + JSON | Vulkan + ICD JSON; Docker | Apache-2.0 |
| VLABench | 14 / 10 [A] | mean progress 69.2 (π0.5 62.2) [A] | moderate | 100 categories, 2,000+ objects | MuJoCo [U] | 10 × 500 eps | EGL [U] | MIT; repo idle since 2025-11 |
| GenManip | few | 23.0 modular; 0% end-to-end unseen | large | ~50 objects retrieved, 0–2 distractors | depth, point cloud, cams; InternData-M1 boxes/traj/grasps/masks | 244K eps gated | Isaac Sim (RT cores) | NC data |
| RLBench / COLOSSEUM | 64 / 18 [A] | 93.7 / 68.8 | small / moderate | few objects; distractor axis | full | script | CoppeliaSim needs sudo X | RLBench non-commercial |

Agent recommendation: primary = LIBERO family (claim on LIBERO-Plus; LIBERO-PRO and LIBERO-Occ as diagnostics; standard LIBERO as sanity check) — most expected, perturbations match failure modes, cheapest label extraction (raw HDF5 replay → depth, instance seg, affordance regions), MuJoCo EGL. Secondary = RoboTwin 2.0 Hard (real clutter, π0.5 46.0 vs best 81.5, MIT data; dual-arm Aloha-AgileX, SAPIEN/Vulkan). Fallback = RoboCasa365. Not main: SimplerEnv (no sim demos), CALVIN (saturated, 3 blocks), RLBench/COLOSSEUM (NC licence, X server), GenManip, VLABench (idle), ManiSkill3 (no protocol, but best label tool).

Real-robot pretraining not needed: papers fine-tune public checkpoints directly on sim demos (OFT, openpi pi05_libero, VLA-Adapter).

Risks: saturation (report perturbation splits, several seeds, failure types separately); LeRobot/RLDS ports probably lack sim states → regenerate from raw HDF5; state setting bypasses dynamics so contacts need re-execution; SAPIEN Vulkan/Docker issues; cross-paper numbers not strictly comparable.

Sources: allenai/vla-evaluation-harness leaderboard data; LIBERO (2306.03310); LIBERO-Plus (2510.13626); LIBERO-PRO (2510.03827); LIBERO-Occ (2606.10862); SimplerEnv; RoboTwin 2.0 (2506.18088) + leaderboard json + install/config docs; CALVIN; RoboCasa + leaderboard; ManiSkill demos docs; VLABench; Meta-World (LeRobot); GenManip (2506.10966); RLBench, COLOSSEUM (2603.00615); OpenVLA-OFT (2502.19645); VLA-Adapter (2509.09372); openpi.

---

## Prior art: VLAs trained with affordance / trajectory intermediates (agent report, 2026-11-10)

Raw subagent report kept for the VLA branch evidence record. Numbers were extracted automatically from arXiv HTML and not hand-checked; several papers are internally inconsistent (AFI, AffordVLA, TraceVLA, A4A). Spot-check before citing.

| # | Work | Representation | How it enters the VLA | Backbone | Sim gain vs own baseline | Code | Checked |
|---|---|---|---|---|---|---|---|
| 1 | AffordanceVLA 2606.06155 | Which2Act VAE latent; Where2Act 2D heatmap; How2Act 3D shape latent + 10-DoF layout; no temporal | affordance expert tokens attended by action expert | π0 MoT 3 experts | LIBERO 94.4→95.8; CALVIN 3.84→4.33; RoboCasa 62.5→65.0; LIBERO-Plus 45.5→62.9; no-aff ablation 92.4 | code yes, weights TODO | M/E |
| 2 | AffordVLA 2605.17517 | teacher affordance feature map (Qwen3-VL + SAM3) | training-only alignment at VLM layer 12 | π0.5 | RoboTwin2.0 5 tasks 61.2/28.8 vs π0 31.6/6.6 | none | M/E |
| 3 | A4A 2609.05892 | "action-oriented 4D affordance": future 3D trajectories of object/tool/contact points (CoTracker3 lifted) | pretraining objective then fine-tune; training-only | Octo, OpenVLA, OFT, π0, π0.5 | LIBERO-Object π0 77.8→87.8, π0.5 94.0→96.0; RLBench 39→61 | project page | M/E |
| 4 | AFI 2512.07472 (CVPR 2026) | non-learned 3D voxel affordance field (GPT-4o + Grounded-SAM + depth) | test-time plug-in, rollback + sample 8 chunks | π0, π0.5 | LIBERO-Pro π0.5 Spatial 54.0→75.7, Object 56.4→73.2 | project page | M/E |
| 5 | VLAff 2608.05215 (IROS 2026) | heatmap + MANO grasp + 6-DoF trajectory tokens | not a VLA | Qwen2.5-VL + DINOv2 | IsaacGym zero-shot 83.0 | repo unverified | M/E |
| 6 | A0 2504.12636 | 2D contact point + ~5 post-contact waypoints | hierarchical DiT + GraspNet/planner | DiT-1B | real only | code + weights | M/E |
| 7 | MolmoAct 2508.07917 | depth tokens + 2D EE trace | autoregressive visual CoT | Molmo-7B | LIBERO 86.6; SimplerEnv VM 71.6 | code/weights/data | M/E |
| 8 | InternVLA-M1 2510.13778 | point/box/trace grounding QA | auxiliary co-training; latent queries → DiT | Qwen2.5-VL-3B + DiT | SimplerEnv Google VM 66.1→80.7; WidowX 54.7→71.7; LIBERO 91.6→95.9 | code/weights/data | M/E |
| 9 | HAMSTER 2502.05485 | 2D EE path + gripper drawn on image | hierarchical, overlay into 3D policy | VILA-1.5-13B; RVT-2/3D-DA | Colosseum 0.18→0.43 | code + ckpt | M/E |
| 10 | TraceVLA 2412.10345 | past point tracks | overlay input | OpenVLA-7B; Phi-3-V | SimplerEnv 40.2→47.7 | code + weights | M/E |
| 11 | CoT-VLA 2503.22020 | future subgoal image | visual CoT | VILA-U 7B | LIBERO 81.1 | none | M/E |
| 12 | DreamVLA 2507.04447 | latent dynamic-region/depth/semantic | auxiliary queries, dropped at inference | GPT-2-style + ViT-B | CALVIN 3.64→4.44 | code + weights | M/E |
| 13 | UP-VLA 2501.18867 | future image + understanding | joint auxiliary | Show-o 1.3B | CALVIN 4.08 vs 1.44 | code | M/E |
| 14 | RoboBrain 2.0/2.5 | affordance, pointing, 2D/3D traces | planner/reward brain, not VLA | 4B–32B | — | code + weights | Abs |
| 15 | 4D-VLA 2506.22242 | 3D coords in visual tokens + history | input | InternVL-4B | LIBERO 88.6 | code | M/E |
| 16 | Embodied-R1 2508.13998 | 2D affordance/placement points, 8-pt trace | VLM → motion planner | Qwen2.5-VL-3B | SimplerEnv WidowX 56.2 | code/weights/data | M/E |
| 17 | Afford-VLA 2605.24203 | affordance tokens → 16×16 masks → embedding | action-expert conditioning | Qwen3-VL-4B + DiT flow | LIBERO 95.4→97.4 | code | M/E |
| 18 | Wiring Matters 2610.06318 | Gaussian heatmap at next contact point (sim contact solver) | aux head + stop-grad + residual bridge | StarVLA QwenPI (Qwen3-VL-2B) | LIBERO 93.1→96.2; naive co-training 85.5; GT affordance as input hurts 88.3 vs 94.0 | not stated | M/E |
| 19 | ACoT-VLA 2601.11404 (CVPR 2026) | coarse reference action trajectory H=15 + latent priors | action-expert cross-attention | π0.5 | LIBERO 96.9→98.5; LIBERO-Plus FT 75.7→88.0; VLABench IS 60.2→63.5 | code | M/E |
| 20 | OASIS 2605.25829 | camera-frame SE(3) EE trajectory H=8 | aux prediction → conditions decoder | Qwen2.5-0.5B + DA3 | LIBERO-Long 89.5→95.2; CALVIN 4.57 | ? | M/E |
| 21 | Pri4R 2603.01549 | 3D point tracks over action horizon | training-only aux head | OFT, π0, π0.5 | LIBERO OFT 92.7→96.3; RoboCasa OFT 33.1→46.3, π0.5 52.9→57.0 | to be released | M/E |
| 22 | 3D HAMSTER 2606.31329 | 3D EE waypoints | hierarchical, points into policy point cloud | Qwen3-VL-8B; 3DFA | Colosseum 36.6/38.8/44.8 | planner only | M/E |
| 23 | CoA-VLA 2412.20451 | box → grasp → placement → collision-free path | visual+text CoT, FiLM | DiffusionVLA | LIBERO 79.8 | none | M/E |
| 24 | ERVLA 2606.03784 | embodied CoT incl. EE movement + image trajectory | training-only | Qwen3-VL-4B | LIBERO-Plus 70.8→86.9; VLABench 40.9→53.2 | claimed | M/E |
| 25 | CoRAct 2610.39543 | CoT movement + image waypoints | offline reward-guided action–CoT alignment (consistency) | MoT | VLABench 39.2→51.8; RoboCasa 68.6→75.5 | page + data | Abs |
| 26 | DreamTrajectory 2608.01381 | EE trajectory denoised with actions | guidance + test-time plan-matching search | ? | MS-HAB 32.3→54.8 | ? | Abs |
| 27 | TAG 2603.24584 | object-erased counterfactual image | test-time guidance + 10% counterfactual training | π0.5, π0 | LIBERO-Plus 81.4→87.2; VLABench 29.4→55.4 | ? | M/E |
| 28 | ThinkAct 2507.16815 | 8-pt 2D trajectory as RL reward | latent plan → DiT | Qwen2.5-VL-7B | SimplerEnv VM 56.0→71.5; LIBERO 76.8→84.4 | none | M/E |
| 29 | GIFT 2609.04193 | geometry/affordance/goal-region supervision | training constraints | StarVLA-OFT, Fast-WAM | LIBERO-Plus 79.6; RoboCasa 61.4 (+12.6) | ? | Abs |
| 30 | 3D-point head 2606.27663 | one grounded 3D point | MLP into action head | GR00T-N1.6, π0.5 | LIBERO-PRO task 31.2→77.5 | ? | Abs |
| 31 | AffordTrajDP 2608.01603 | contact point propagated by object pose | conditions DP (not VLA) | DP | ManiSkill3 70.0 | ? | Abs |

Also abstract-only: AffordanceWAM 2609.22332, VLA-R1 2510.01623, VLA-OS 2506.17561, RT-Affordance 2411.02704, 3DWay (ECCV 2026), H-VLA 2609.22895, LoHo-Manip 2604.21924, AnchorRefine 2604.17787, OBEYED-VLA 2512.22519 (clutter, real only), LIBERO-Safety 2606.23686 (Tabletop Spatial Avoidance, Affordance-Aware Grasping suites).

Crowded: 2D trace CoT/overlay; affordance-mask heads into π0-style action expert; training-only auxiliary spatial/4D targets; coarse trajectory guidance. "4D affordance" name taken by A4A.
Open (per agent): unified contact + approach + EE-waypoint representation inside a VLA; differentiable affordance/trajectory→action consistency loss during training (but CoRAct, DreamTrajectory align via reward / test-time search); no affordance VLA reports wrong-object / wrong-direction / collision rates (LIBERO-Plus, VLABench, RoboTwin2.0, LIBERO-Safety have clutter suites).
Likely reviewer comparisons: AffordanceVLA, A4A, ACoT-VLA; plus Wiring Matters as cheap ablation, CoRAct if consistency is claimed.
