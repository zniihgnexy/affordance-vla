# VLA branch R2：LIBERO-Plus、LIBERO-PRO、LIBERO-Occ 的协议与数据

**日期：2026-11-10｜来源：R2 检索 agent 的报告（英文，内容未改动）｜对应 [Pipeline v2](2026-11-10_VLA_branch_Pipeline_v2.md) 第 4.2、11 节**

**结论：**

- **三个都是测试集。** 2026 年的常见做法是只用标准 LIBERO-40 的示范训练，再在三个集合上 zero-shot 评测：抽查的 7 篇论文里有 6 篇是这样做的。所以训练一次，就能拿去评测全部三个诊断集。
- **LIBERO-Plus 的训练数据不能用来提取标签。** 公开的训练数据只有 RGB、本体状态和动作，没有仿真状态、XML、深度或分割，也无法对应回具体的 BDDL 变体。
- **LIBERO-Occ 的遮挡物从第一帧就在。** 所以记忆只在目标后来变得可见时才有用；评测时要写明有没有用腕部视角。

---

## R2: protocols and data for LIBERO-Plus, LIBERO-PRO and LIBERO-Occ, checked against primary sources (2026-11-10)

How it was checked: arXiv HTML for each paper, the GitHub source code, and the Hugging Face (HF) API file listings. The RLDS zip directories and their `features.json` / `dataset_info.json` were read remotely with HTTP range requests, without downloading the archives. One original LIBERO HDF5 file was opened over HTTP with h5py. **[UNVERIFIED]** marks anything that could not be confirmed. **[inference]** marks the agent's own reasoning.

### Original LIBERO (needed by all three benchmarks)

| Item | Finding |
|---|---|
| Hosting / licence | [HF yifengzhu-hf/LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets): card says apache-2.0, about 100 GB. Code: [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), MIT. `benchmark_scripts/download_libero_datasets.py --use-huggingface` (UT Box zips are the alternative). |
| Layout | `libero_{spatial,object,goal,10,90}/<task>_demo.hdf5`, one file per task, 130 files. Suite totals: spatial 6.24 GB, object 7.44 GB, goal 6.37 GB, libero_10 13.73 GB, libero_90 66.66 GB. Per-task file sizes: spatial 0.51–0.75 GB, object 0.66–0.80 GB, goal 0.45–1.02 GB, libero_10 0.94–2.06 GB. |
| Per-demo contents (checked) | `data` attrs: `bddl_file_name`, `env_args`, `problem_info` (language), `num_demos=50`. Each `demo_i` attrs: **`model_file`** (full MJCF XML, 76 kB), **`init_state`**, `num_samples`. Datasets: **`states`**, `actions` (T,7, float64), `obs/{agentview_rgb, eye_in_hand_rgb}` (128×128), `obs/{ee_pos, ee_ori, ee_states, gripper_states, joint_states}`, `robot_states`, `rewards`, `dones`. `env_args` has `camera_depths=false` and `camera_segmentations=null`, so depth and segmentation are not stored. (The state size depends on the scene: 64 in the file the agent opened, 110 in our LIBERO-Object file.) |
| Smallest file | `libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5`, 377,038,880 B. In the four eval suites: `libero_goal/turn_on_the_stove_demo.hdf5`, 447,509,922 B. |
| Action space (checked in `env_args`) | OSC_POSE, `control_freq=20`, `control_delta=true`, kp 150, damping 1. robosuite 1.4.0 `scale_action` clips input to [-1,1], then scales linearly to ±0.05 m and ±0.5 rad per step. Goal = current world-frame EE site position + delta; rotation is an axis-angle delta. Gripper ±1. **Cumulative 0.05·a is the *commanded* goal path, not the achieved path** (OSC lags, contacts). Read the actual path from `obs/ee_pos` (HDF5) or from `state[:3]` (RLDS/LeRobot). |
| Eval convention (OpenVLA-OFT `run_libero_eval.py`) | 50 trials per task over the 50 fixed init states, seed 7, 10 wait steps. Max steps: spatial 220, object 280, goal 300, libero_10 520. Success = `done = _check_success()`, i.e. all BDDL goal predicates hold. |

### LIBERO-Plus ([arXiv 2510.13626](https://arxiv.org/abs/2510.13626) v3, also CVPR 2026; [GitHub](https://github.com/sylvestf/LIBERO-plus))

| Item | Finding |
|---|---|
| Protocol | Test-only set. The paper evaluates the public LIBERO checkpoints zero-shot. Fine-tuning on its own training data appears only as one extra result (OpenVLA-OFT mix-SFT, 79.6 total). |
| Splits | 10,030 tasks, 1 trial each (README: change `num_trials_per_task` from 50 to 1). Per-dimension counts in `task_classification.json`: Camera 1599, Robot 1550, Language 1537, Light 1142, Background 1076, Noise 1601, Layout 1525 (these match the paper). Per-suite counts in code (2402 / 2518 / 2591 / 2519) differ from paper Table 7 (2293 / 2576 / 2569 / 2592). Difficulty levels L1–L5 = how many of 4 reference models solved the task. Max steps and seeds are not specified **[UNVERIFIED]**. |
| Perturbations | **Layout**: add distractors drawn from 416 objects (`_add_N` BDDL), or randomise target pose (`_levelK_sampleN`, init states in `init_files/libero_newobj`). **Camera**: distance, azimuth/elevation and orientation, encoded in the filename `_view_h_v_scale_rot_vert_initstate_0` and parsed in `ControlEnv`. **Robot init**: `initstate_k` selects one of 500 hard-coded `MountedPanda{k}` classes with perturbed qpos. **Language**: LLM rewrites (distraction / common sense / reasoning). **Light**: XML diffuse, direction, specular, shadows. **Background**: 950 scene textures plus table textures. **Noise**: 2-D post-processing applied to the agentview RGB only (motion / gaussian / zoom blur, fog, glass; 5 severities each). |
| Training data | Paper: over 20,000 trajectories, made by executing the original LIBERO demos' (state, action) pairs inside the perturbed envs and keeping only successes. Robot-pose and target-pose variants were excluded; camera angles are 5° off the test set. **Released data has 14,347 episodes / 2,238,036 frames, which does not match the paper.** Locations: [RLDS mix](https://huggingface.co/datasets/Sylvest/libero_plus_rlds) (split zip, 75.5 GB); [per-suite](https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite) (RLDS 4,320 / 4,243 / 4,520 / 2,791 episodes, about 83 GB; LeRobot zips 3.5–5.0 GB); [LeRobot v2.1](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) (7.85 GB, 20 fps, AV1 video); [assets.zip](https://huggingface.co/datasets/Sylvest/LIBERO-plus) (6.4 GB). HF cards say MIT; the GitHub repo has no LICENSE file. |
| What the training data contains | RLDS fields: `image`, `wrist_image` (256²), `state` (8: EEF 6-D pose + 2 gripper), `joint_state` (7), `action` (7), `language_instruction`, `episode_metadata/file_path`. **No sim states, no XML, no depth or segmentation.** In 4 sampled episodes, `file_path` names only the category and task, not the specific variant, so an episode cannot be mapped back to its BDDL. LeRobot `tasks.jsonl` lists only 40 instructions, so the language rewrites seem to be absent (observed, not confirmed). |
| Leaderboard | A table in the README, updated by pull request. No hosted server. |

### LIBERO-PRO ([arXiv 2510.03827](https://arxiv.org/abs/2510.03827) v2, May 2026; [GitHub](https://github.com/Zxy-MLlab/LIBERO-PRO), MIT)

| Item | Finding |
|---|---|
| Protocol | Evaluation only, no training data. The paper uses official LIBERO checkpoints (OpenVLA per suite; π0 / π0.5 single checkpoint), 50 episodes per task. |
| Splits | For each of the 4 suites and each perturbation in {object, swap (position), lan (semantic), task}: 10 BDDL files plus init files ([HF](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro), CC-BY-4.0, about 234 kB). The environment perturbation is generated on the fly. Perturbations combine via `evaluation_config.yaml`, except task. Max steps 220 / 280 / 300 / 520. Position sweep: x/y offsets 0.1–0.5. |
| Generation | `perturbation.py` rewrites BDDL from YAML files: region swaps, asset variants (colour/scale), paraphrases, new goals, table/fixture swap. `generate_init_states.py` then makes 50 states per task via `reset()` + `get_sim_state()`, with no seed set **[inference: regenerated suites are not bit-reproducible]**. The README notes that swapping the environment can make objects move randomly. |
| July 2026 extension | 280 BDDL files (40 tasks × 7 cases, including view occlusion, runtime object move, camera, glare) with `:perturbation_config` blocks. No parser for these was found on GitHub master **[UNVERIFIED / possibly unreleased]**. |
| Leaderboard | README table (total scores: π0.5 0.53, OpenVLA 0.52, π0 0.44). Position scores are around 0 except π0.5. |

### LIBERO-Occ ([arXiv 2606.10862](https://arxiv.org/abs/2606.10862) v2; [GitHub](https://github.com/litsh/Libero-Occ), MIT, 6 stars)

| Item | Finding |
|---|---|
| Protocol | All methods are fine-tuned on the same standard LIBERO demos (OpenVLA-OFT preprocessing) and tested on occluded scenes. Main setting: primary (third-person) view only. Reference setting: the ground-truth wrist view is also given. |
| Splits | 2,000 instances = 4 suites × 10 tasks × 50 eval init states. By type: manipulated object 900, receptacle 750, dual 350. Severity s = 1 − A_occ/A_ref from instance segmentation, split by per-suite quartiles (light 500 / medium 1000 / heavy 500). The paper reports 500 rollouts per suite (100 for ablations). **The release's default `NUM_TRIALS_PER_TASK=1` gives only 10 rollouts per suite; set it to 50 to match the paper.** Suite registration for upstream LIBERO is not in the release **[inference]**. |
| Generation | Parse BDDL for targets, sample occluders along the camera-to-target ray, then three checks: visibility (render with and without the occluder), no collision, and the original demo replays successfully. The generation code is deliberately not released. Only BDDL + `.pruned_init` files ship. |
| Results | VIM 65.05 average (Spatial 76.4 / Goal 74.6 / Object 84.2 / libero_10 25.0); 74.0 with the ground-truth wrist view. Best baseline is UniVLA 57.10 (BAAI's UniVLA, not the OpenDriveLab one used in LIBERO-Plus). |
| VIM mechanism | Built on the UniVLA Emu3-MoE world model with FAST action tokens. At each query it generates VQ tokens of an *imagined wrist view* from the instruction and the **current** third-person frame, then outputs a 10-step action chunk. **No memory or tracking:** the released wrapper hard-codes `video_mode=False`, the perspective path uses the current frame only, and eval runs with `--no_gripper`. The paper lists multi-frame use as future work. |

### Recommended protocol for the main claim

Train only on the standard LIBERO-40 demos (spatial / object / goal / 10), then test zero-shot on all three benchmarks, each with its own official count:

- LIBERO-Plus: all 10,030 tasks × 1 trial, reported per dimension and as a total.
- LIBERO-PRO: 5 perturbation types × 50 trials per task.
- LIBERO-Occ: 2,000 instances × 1 trial, 500 per suite.

This is the most common protocol in 2026: in a non-exhaustive sample of seven papers, six train on standard LIBERO only. Zero-shot on LIBERO-Plus: [MergeVLA](https://arxiv.org/abs/2511.18810), [VLA-JEPA](https://arxiv.org/abs/2602.10098), [LIRA](https://arxiv.org/abs/2608.07596), [SRPO](https://arxiv.org/abs/2511.15605) and a [world-action-model robustness study](https://arxiv.org/abs/2603.22078); only [ProGAL-VLA](https://arxiv.org/abs/2604.09824) fine-tunes on LIBERO-Plus. On LIBERO-PRO, [HABILIS Brain 0](https://arxiv.org/abs/2609.25558) uses a frozen policy trained only on canonical LIBERO.

LIBERO-Plus fine-tuning should be a secondary ablation only: its training camera angles are just 5° from the test set. On LIBERO-Occ, state which observation setting is used. Using the wrist camera puts a method in the "complementary view available" setting, so compare against 74.0, not 65.05. Occluders are present from t=0, so a memory model only gains where the target becomes visible later.

### Getting labels

- **Read directly from the original HDF5:** `states`, `model_file`, `init_state`, actions, `ee_pos` / `ee_ori`, gripper and joint states, 128² RGB, language. The end-effector path is `obs/ee_pos`; there is no need to integrate deltas.
- **Needs replay** (set the sim state each step, render with `camera_depths=True` and `camera_segmentations="instance"`): depth, instance masks, object poses, camera matrices and 256² RGB for the original demos. **[inference]** The same should work for LIBERO-Plus camera / light / background / noise variants, since they appear to leave the state vector unchanged; check the dimensions first. The XML mesh paths may need remapping; otherwise instantiate the env from BDDL, as OpenVLA-OFT's `regenerate_libero_dataset.py` does.
- **Needs re-execution or new generation:** LIBERO-Plus layout and robot-init variants (state size or trajectory changes); LIBERO-Occ (an extra body is added, and occluders exist only for the 50 *eval* init states, so training on them would contaminate the test set); all LIBERO-PRO suites (evaluation only).
- **The released LIBERO-Plus RLDS / LeRobot data cannot be turned into depth or segmentation labels.** If perturbed training data is needed, regenerate it by re-executing the original demos in the repo's 8,172 `libero_mix` BDDLs. Whether those BDDLs are what produced the released data is **[UNVERIFIED]**.
