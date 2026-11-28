# Experiment index

Every command runs from the repository root. GPU work goes through the shared queue of the A6000 server
(`scripts/server/submit_queue_job.sh`), never directly. `$W` is the NAS work dir
`/mnt/nas/xinyizheng/affordance_vla`, which holds data, models and every run folder (`runs/` in the server
checkout links to `$W/runs`); none of it is in git. Results and their reading are in the records under
`docs/vla/` and the notebooks under `report/`.

Settings of the GPU jobs are environment variables of their wrapper (documented at the top of each
wrapper); the exact arguments every training run used are stored in its `metrics.json` (`args`). Moving
these settings into `configs/` is still open (AGENTS.md, Part B).

## 0. Local simulator (Mac, CPU; no model runs)

```bash
bash envs/local/setup_sim_env.sh            # conda env affordance-vla-sim (MuJoCo 3.2.3, robosuite 1.4.0, ...)
bash scripts/local/fetch_sim_data.sh        # LIBERO-PRO at eafdb80 + task files + image-free evaluation logs + L maps
bash scripts/local/sim.sh scene  --suite libero_object_swap --task 2 --episode 0     # open a task's scene
bash scripts/local/sim.sh map    --suite libero_10 --task 4                          # L's 3D interaction maps on the objects
bash scripts/local/sim.sh grasp  --suite libero_object_swap --task 2 --object ketchup_1  # scripted pick-and-place
bash scripts/local/sim.sh list   runs/libero_pro/predicted_g1_hold_s10              # logged test episodes
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
```

Defaults: `configs/sim/local.yaml`; override with `--config FILE` or `--set key=value`. Everything is shown
live in the MuJoCo viewer. `replay` reruns a server test episode with the actions the policy returned there;
checked on three logs (success and failure, with and without plans): same outcome and step count, end
effector within 0.5 mm of the server's log; on 23 Nov also on LIBERO-10 (same outcomes, within 6 mm). The
terminal narration is in English. Demo commands for the week-2 meeting: section 6 of
`report/2026-11-16_weekly_review.ipynb`. Walk-through of a test, in Chinese: `docs/local_simulation.md`.

## 1. Data (server, CPU)

| Step | Command | Output |
| --- | --- | --- |
| LIBERO demonstrations (states and actions only, by byte ranges) | `source envs/server/libero_env.sh && $LIBERO_PYTHON scripts/data/fetch_libero_states.py libero_object $W/data/libero_states/libero_object` | `$W/data/libero_states/<suite>/` |
| Replay them in the simulator, one log per demonstration | `SUITE=libero_object bash scripts/data/replay_libero_states.sh` | `$W/runs/plan_following/demos/` |
| Plan dataset (tracks, images, contact offsets) | `python scripts/data/build_plan_dataset.py --logs $W/runs/plan_following/demos --out $W/runs/plan_following/dataset10` | `dataset10/` (LIBERO-Object, 10 tasks) |
| LIBERO-Spatial | `SUITE=libero_spatial RAW=$W/data/libero_states/libero_spatial OUT=$W/runs/plan_following/demos_spatial bash scripts/data/replay_libero_states.sh`, then `build_plan_dataset.py ... --container plate_1 --bddl-dir $W/repos/LIBERO-PRO/libero/libero/bddl_files/libero_spatial` | `dataset_spatial/` |
| Replay with object orientations (Pipeline v3, L) | `OUT=$W/runs/plan_following/demos_pose bash scripts/data/replay_libero_states.sh`; LIBERO-10: `SUITE=libero_10 RAW=$W/data/libero_states/libero_10 OUT=$W/runs/plan_following/demos_libero10 MAX_JOBS=30 bash scripts/data/replay_libero_states.sh` | `demos_pose/` (500), `demos_libero10/` (500) |
| 3D canonical interaction maps (Pipeline v3, L2) | `source envs/server/libero_env.sh && PYTHONPATH=.:$PYTHONPATH $LIBERO_PYTHON scripts/data/build_interaction_labels.py --suite libero_object --demos $W/runs/plan_following/demos_pose --out $W/runs/interaction_labels/object`; LIBERO-10: `--suite libero_10 --demos .../demos_libero10 --out .../interaction_labels/libero10` | `maps.npz`, `summary.json`, `check/<category>.png` |
| LIBERO-90 mug demonstrations (Pipeline v4 step 0 (1)): states and actions of the 10 tasks that hold a mug, replayed with object poses | `source envs/server/libero_env.sh && $LIBERO_PYTHON scripts/data/fetch_libero_states.py libero_90 $W/data/libero_states/libero_90 --only <the 10 task names>`, then `SUITE=libero_90 RAW=$W/data/libero_states/libero_90 OUT=$W/runs/plan_following/demos_libero90_mug MAX_JOBS=30 bash scripts/data/replay_libero_states.sh` | `libero_90/`, `demos_libero90_mug/` (500 logs) |
| Mug transfer check (Pipeline v4 step 0 (1)): leave one mug mesh out, carry the map of the other two to it by nearest point and by NDF | `bash scripts/data/run_mug_transfer_check.sh collect`; `bash scripts/server/submit_queue_job.sh 164 mug_ndf 8000 3 "cards=<free cards>" -- bash $PWD/scripts/data/run_mug_transfer_check.sh ndf` (GPU, about 1 h at most; needs `$W/repos/ndf_robot` and `$W/models/ndf/multi_category_weights.pth`); `bash scripts/data/run_mug_transfer_check.sh evaluate`. Settings: `configs/mug_transfer/mug_mesh.yaml`; record: `docs/vla/2026-11-25_VLA_branch_第0步_mug搬运检查.md` | `$W/runs/mug_transfer/mug_mesh/` (`results.md`, `results.json`, `figures/`) |
| HOT3D demo: what can a human-hand dataset give our map (AFFO-11; CPU; hot3d env) | `bash envs/server/setup_hot3d_env.sh` once, then `bash scripts/data/fetch_hot3d_demo.sh` (2 clips + 3 meshes, 0.6 GB, no sign-up), then `source envs/server/hot3d_env.sh && PYTHONPATH=. python scripts/data/hot3d_contact_demo.py --out $W/runs/hot3d_demo/mug_white_v2` (30 s; `--threshold-mm 5`, `--hand-points 4000`, `--max-drift-deg 20` are the defaults; `--hand-points 0` uses the hand mesh vertices, `--max-drift-deg 180` does not split a hold) | `overlay.png`, `contact.png`, `maps.npz` (one map in `interaction_labels.load_maps` format), `summary.json`; copies in `report/media/weekly_2026_11_23/hot3d/`; record `docs/vla/2026-11-25_VLA_branch_HOT3D评估.md` |
| Logs for the local simulator | `python scripts/data/export_sim_logs.py --root $W/runs/libero_pro --out $W/cache/sim_logs/libero_pro --runs released ...` | image-free copies |

`dataset` (6 tasks) and `dataset_salad` (1 task) are earlier subsets of LIBERO-Object built the same way.

## 2. Action head trained to follow plans (GPU)

Wrapper `scripts/vla/run_plan_following_gpu.sh` (stages `train`, `eval`) around `scripts/vla/train_plan_following.py`:

```bash
bash scripts/server/submit_queue_job.sh 160 plan_s10 26000 3 "cards=5" -- \
  env RUN=s10_scale_own10 DATA=$W/runs/plan_following/dataset10 COUNTERFACTUAL=shift+other+release \
      SUPERVISION=actions STEPS=4000 STAGES=train \
      TRAIN_ARGS="--rigid-tail 0.18 --scale-from-start --own-approach" \
      bash ~/affordance-vla/scripts/vla/run_plan_following_gpu.sh
```

Shared by all runs: released starVLA QwenPI checkpoint (`Qwen3-VL-PI-LIBERO-4in1`, steps 100000) on
`Qwen3-VL-4B-Instruct`, frozen VLM, LoRA rank 16, lr 2e-4, 8 frames x 2 noise draws per step, path weight
0.2, shift radius 0.18 m, 5 held-out demonstrations per instruction, seed 7. The table is generated from each
run's `metrics.json`; `TRAIN_ARGS` lists the options that differ from today's defaults.

| Run (`RUN`) | `DATA` | `COUNTERFACTUAL` | `SUPERVISION` | `STEPS` | `TRAIN_ARGS` |
| --- | --- | --- | --- | --- | --- |
| `s1_none` | `dataset_salad` | none | path | 1200 | `-` |
| `s1_shift` | `dataset_salad` | shift | path | 1200 | `-` |
| `s1_actions` | `dataset_salad` | shift | actions | 1200 | `-` |
| `s2_none` | `dataset` | none | path | 2400 | `-` |
| `s2_other` | `dataset` | shift+other | path | 2400 | `-` |
| `s2_actions` | `dataset` | shift+other | actions | 2400 | `-` |
| `s2b_actions` | `dataset` | shift+other | actions | 2400 | `-` |
| `s3_noise` | `dataset` | none | path | 2400 | `--contact-noise 0.05 --plan-dropout 0.15` |
| `s4_actions10` | `dataset10` | shift+other | actions | 4000 | `-` |
| `s5_deep` | `dataset` | none | path | 2400 | `--contact-noise 0.05 --plan-dropout 0.15 --deep-plan` |
| `s6_release` | `dataset` | shift+other+release | actions | 2400 | `-` |
| `s7_align` | `dataset` | shift+other | actions | 2400 | `--rigid-tail 0.18 --align-first` |
| `s8_align_release` | `dataset` | shift+other+release | actions | 2400 | `--rigid-tail 0.18 --align-first` |
| `s9_align_release10` | `dataset10` | shift+other+release | actions | 4000 | `--rigid-tail 0.18 --align-first` |
| `s10_scale_own10` | `dataset10` | shift+other+release | actions | 4000 | `--rigid-tail 0.18 --scale-from-start --own-approach` |
| `s11_scale_own_noise10` | `dataset10` | shift+other+release | actions | 4000 | `--other-noise 0.02 --rigid-tail 0.18 --scale-from-start --own-approach` |

The default labels for new action heads are those of `s10` (decision of 2026-11-16). Label rules changed in
code between runs with the same arguments (`s2_actions` and `s2b_actions` differ only in the label fix of
11-13), so runs before 2026-11-16 are reproduced exactly only with the code of their date (git history and
the records in `docs/vla/`).

## 3. Plan head (GPU)

Wrapper `scripts/vla/run_plan_predictor_gpu.sh` around `scripts/vla/train_plan_predictor.py` (data `dataset10`,
lr 3e-4, hidden 512, seed 7, tested on the first frames of the `released` LIBERO-PRO run):

| Run | Head | Steps | Command settings |
| --- | --- | --- | --- |
| `p1` | regress (contact coordinates) | 3000 | `RUN=p1` |
| `g1` | grid (scores the 8 x 8 image cells) | 2500 | `RUN=g1 HEAD=grid STEPS=2500` |

**E1 of Pipeline v3 (semantic probe, no training):** for the first frame of every LIBERO-PRO rollout that kept its
images, instruction-to-cell similarity maps of the frozen VLM on every layer, and the head `g1` scored the same way.

```bash
bash scripts/server/submit_queue_job.sh 164 semantic_probe_e1_fix2 22000 6 -- \
    env RUN=e1 bash ~/affordance-vla/scripts/vla/run_semantic_probe_gpu.sh
```

Output `$W/runs/semantic_probe/e1/{probe.npz,summary.json}`. Defaults: `RUNS="released spatial_released_seed7"`,
`PREDICTOR=g1`, offsets from `dataset10` (frames whose target has no grasp offset there are scored without the 3D
error).

## 4. LIBERO-PRO evaluation (GPU policy server + CPU simulators)

Wrapper `scripts/eval/run_libero_pro_gpu.sh` (simulator side: `scripts/eval/rollout_libero_logged.py`, whose
episode loop is `affordance_vla/sim/episode.py`). Default suites: `libero_object_swap` (position perturbation)
and `libero_object_task` (task perturbation), 10 tasks, `TRIALS=20` starts per task, `SEED=7`.

| Run folder (`NAME`) | Settings |
| --- | --- |
| `released` | (none: the released checkpoint without a plan) |
| `oracle_plan_<head>` | `ADAPTER=<head> PLAN=oracle` |
| `predicted_g1_hold_<head>` | `ADAPTER=<head> PLAN=predicted PREDICTOR=g1 HOLD=1` |
| `full_released_seed<s>`, `full_oracle_s9_seed<s>`, `full_g1_hold_s9_seed<s>` | as above with `TRIALS=50 SEED=<s>`, s = 7, 8, 9 (the papers' protocol) |
| `spatial_released_seed<s>` | `SUITES="libero_spatial_swap libero_spatial_task" TRIALS=50 SEED=<s>` |

```bash
bash scripts/server/submit_queue_job.sh 162 pro_g1_hold_s10 16000 3 "cards=5" -- \
  env NAME=predicted_g1_hold_s10 ADAPTER=s10_scale_own10 PLAN=predicted PREDICTOR=g1 HOLD=1 \
      bash ~/affordance-vla/scripts/eval/run_libero_pro_gpu.sh
```

## 5. Notebooks (server, CPU; read finished runs only)

One weekly review per London calendar week, from `report/`:
`python ../scripts/report/build_weekly_<YYYY_MM_DD>.py <YYYY-MM-DD>_weekly_review.ipynb` (the week's Monday).
Experiment notebooks (appendices of their week), from `report/experiments/`:
`python ../../scripts/report/build_<name>.py <notebook>.ipynb`. Then
`python -m jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`. Each builder's docstring
names its notebook and inputs; media go to the tracked `report/media/`.
Week 3 (`report/2026-11-23_weekly_review.ipynb`, built by `scripts/report/build_weekly_2026_11_23.py`, one module per section
`scripts/report/weekly_2026_11_23_<topic>.py`) reads only tracked files, so it also builds and executes on the Mac (env with
nbformat and jupyter, e.g. `affordance3d`); its section 7 lists the demo commands of the week.
The layer-level pipeline figures are rendered on the Mac (needs Google Chrome):
`python scripts/report/render_pipeline_detail.py report/media/pipeline_overview/detail/pipeline_v3_layers.html`.

## 6. RoboCasa365 set-up check (Pipeline v4 step 0, AFFO-10; server)

Record: `docs/vla/2026-11-25_VLA_branch_第0步_RoboCasa365搭建.md`. Needs the conda env
`/data/xinyizheng/envs/robocasa365` (simulator, CPU) and `starvla` (training, GPU); data and assets on the NAS.

```bash
bash envs/server/setup_robocasa365_env.sh                 # env, pinned code, assets (22 GB), idempotent
bash scripts/data/fetch_robocasa365.sh [--dry-run] [TASK ...]   # default CoffeeSetupMug OpenDrawer TurnOffStove, 1.4 GB
source envs/server/robocasa365_env.sh                     # $RC365_PYTHON, software rendering
"$RC365_PYTHON" scripts/data/replay_robocasa365.py --task CoffeeSetupMug OpenDrawer TurnOffStove --episodes 5 \
    --out $W/runs/robocasa365_replay/<run>                # re-render from saved states; depth, segmentation, PSNR, timings
"$RC365_PYTHON" scripts/report/make_rc365_figures.py --replay $W/runs/robocasa365_replay/<run> \
    --out report/media/weekly_2026_11_23/robocasa365      # figures of the weekly review
```

Training and evaluation of starVLA on three tasks (configs `configs/rc365/`; wrappers document their settings):

```bash
# QwenOFT (MLP head), frozen VLM; FREEZE= trains everything; CONFIG=...qwenpi_rc365_3tasks.yaml is QwenPI
bash scripts/server/submit_queue_job.sh 164 rc365_oft 30000 1 "cards=6 7" -- \
    env RUN=<run> CONFIG=$PWD/configs/rc365/qwenoft_rc365_3tasks.yaml BATCH=8 STEPS=300 \
    bash $PWD/scripts/rc365/run_train_gpu.sh              # -> $W/runs/robocasa365_starvla/<run>/{train.log,PEAK_MEMORY.txt,checkpoints/}
bash scripts/server/submit_queue_job.sh 164 rc365_eval 20000 1 "cards=6 7" -- \
    env CKPT=$W/runs/robocasa365_starvla/<run>/checkpoints/steps_300_pytorch_model.pt EPISODES=10 \
    bash $PWD/scripts/rc365/run_eval_gpu.sh               # -> $W/runs/robocasa365_eval/eval_<run>/{summary.txt,client_<Task>.log,videos/}
```

| Run | Settings | Result |
| --- | --- | --- |
| `qwenoft_rc365_frozen` | QwenOFT, frozen VLM, batch 8, 300 steps | peak 11.42 GiB, 0.445 s/step model time |
| `qwenpi_rc365_3tasks_frozen` | QwenPI, frozen VLM, batch 8 | out of memory (46.41 GiB) |
| `qwenpi_rc365_frozen_b1` | QwenPI, frozen VLM, batch 1 | out of memory (41.46 GiB) |
| `qwenpi_rc365_full_b1` | QwenPI, all weights, batch 1 | out of memory (37.30 GiB) |
| `qwenoft_rc365_full_b1` | QwenOFT, all weights, batch 1 | out of memory in the first backward (29.64 GiB allocated, 42.22 GiB reserved) |
| `eval_qwenoft_frozen_300` | evaluation of the 300-step QwenOFT, 10 episodes x 3 tasks | 0/30 successes, 4.2 / 5.1 / 5.2 min per episode, 55 min wall |

## 8. RoboCasa365 interaction labels (Pipeline v4 step 1 (1), AFFO-15; server, CPU only)

Record: `docs/vla/2026-11-26_VLA_branch_第1步_交互map标签.md`. Needs the `robocasa365` env (`envs/server/robocasa365_env.sh`);
the charts run in any env with matplotlib (`envs/server/libero_env.sh`). Data and outputs on the NAS (`$W/runs/rc365_labels/`).

```bash
bash scripts/data/fetch_robocasa365.sh --dry-run --split pretrain CoffeeSetupMug ...   # sizes first; --split target|pretrain, human demos only
source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
"$RC365_PYTHON" scripts/data/rc365_extract.py --config configs/rc365_labels/extract.yaml       # replay 8 tasks, record grasps / touches (24 procs, ~1 h)
"$RC365_PYTHON" scripts/data/rc365_label_check.py --config configs/rc365_labels/check.yaml --stage all   # counts, mug transfer + treatments, part consistency (~1 min)
"$RC365_PYTHON" scripts/data/rc365_label_figures.py --config configs/rc365_labels/figures.yaml          # labels projected on rendered frames
source envs/server/libero_env.sh; "$LIBERO_PYTHON" scripts/data/rc365_label_charts.py --config configs/rc365_labels/check.yaml \
    --out report/media/weekly_2026_11_23/interaction_labels
```

| Run | Settings | Result |
| --- | --- | --- |
| `extract_v1` | 8 atomic tasks, pretrain + target human demonstrations (part tasks: first 150 target episodes) | 1647 episodes replayed (1717 files incl. an earlier partial run); mug 103 pretrain / 503 target grasps, cup 0 / 4, bowl 1 / 2; record section 3 |
| `check_v1` | pass lines of step 0 (landing >= 0.8, closing-axis median < 20 deg); treatments plain / ring / robot | mug: plain 0.79 / 12.9 deg, ring 0.89 / 12.9 deg (pass), robot 0.33 / 14.8 deg; 4 of 5 held-out mugs pass, mug_8 does not; parts: closing line 0.71-0.94 alike across kitchens, stove knob motion not; record sections 3-4 |

## 9. RoboCasa365 label follow-ups (Pipeline v4 step 1 (3), AFFO-18; server, CPU only)

Record: `docs/vla/2026-11-26_VLA_branch_第1步_旋钮标签与把手对齐.md`. Reads `extract_v1` of section 8 (nothing is replayed except for the part figures); outputs in `$W/runs/rc365_labels/check_v2/`.

```bash
source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
"$RC365_PYTHON" scripts/data/rc365_joint_check.py --config configs/rc365_labels/joint_check.yaml     # knob frame diagnosis; joint-derived vs gripper label of 4 part tasks (seconds)
"$RC365_PYTHON" scripts/data/rc365_mug_align.py --config configs/rc365_labels/mug_align.yaml         # mug transfer check, handle aligned (seconds); --set handle_x0=1.1 --set out=... for the sensitivity
"$RC365_PYTHON" scripts/data/rc365_joint_figures.py --config configs/rc365_labels/joint_figures.yaml # labels on rendered frames (~3 min)
source envs/server/libero_env.sh; export PYTHONPATH=$PWD:$PYTHONPATH
"$LIBERO_PYTHON" scripts/data/rc365_joint_charts.py --config configs/rc365_labels/joint_check.yaml --out report/media/weekly_2026_11_23/label_followups
"$LIBERO_PYTHON" scripts/data/rc365_mug_figure.py --config configs/rc365_labels/mug_align.yaml --out report/media/weekly_2026_11_23/label_followups
```

| Run | Settings | Result |
| --- | --- | --- |
| `check_v2/joint_parts.json`, `knob_diagnosis.json` | `joint_check.yaml`: four part tasks, new label = joint axis / anchor and the sign of the joint change (1 s window or whole demonstration) | stove-knob contact scatter 32.9 -> 2.02 cm with the joint anchor as origin (body origin is 49 cm from the knob); drawer / door handle motion consistent across kitchens 0.73 -> 1.00, 0.91 -> 0.98; faucet 0.49 -> 0.91 (67 deg from the gripper direction); knob translation direction 0.15, turn axis + sign 0.85-0.93 per stove design (record sections 2-3) |
| `check_v2/mug_align.json` | `mug_align.yaml`: handle length aligned (`length`) or registered (`registered`) before the nearest-point transfer; plain / ring | `none` rows equal AFFO-15's; `length` + ring: 0.94 / 12.0 deg, `mug_8` 0.84 / 10.9 deg (pass); `registered` + ring: `mug_8` 0.87 / 20.5 deg (axis 0.5 deg over the line) (record section 4) |
| `check_v1_repro/` | `check.yaml --set out=runs/rc365_labels/check_v1_repro` | `counts.json`, `objects.json`, `parts.json` byte-identical to `check_v1/` |

## 10. RoboCasa365 base policy (Pipeline v4 step 1 (2), AFFO-16; server)

Record: `docs/vla/2026-11-26_VLA_branch_第1步_底座策略.md`. QwenOFT with LoRA on the language-model layers, trained with the
MLP action head on the PRETRAIN split of eight atomic tasks (MimicGen + human demonstrations, no target demonstrations),
evaluated on seen and unseen object instances. One card at a time, queue priority 164, cap 40 GPU hours.
Settings: `configs/rc365/qwenoft_lora_rc365_8tasks.yaml` (training), `configs/rc365/base_policy_eval.yaml` (evaluation),
`configs/rc365/data_registry_affo16/data_config.py` (mixtures). Needs the envs of section 6; run from the repo root.

```bash
# data (CPU): MimicGen pretrain, 55.0 GB for the eight tasks (--dry-run prints sizes; --max-gb stops above N GB); human: --source human
bash scripts/rc365/fetch_pretrain_demos.sh --source mg [--dry-run]
MIX=affo16_rc365_pretrain_<Task> bash scripts/rc365/prepare_train_data.sh      # CPU: step index + statistics of one task (or the whole mixture)
# checks (CPU)
bash scripts/rc365/check_train_eval_alignment.sh                              # training data path vs evaluation input, same raw step
source envs/server/robocasa365_env.sh && PYTHONPATH=$W/repos/starVLA_rc365:. "$RC365_PYTHON" scripts/rc365/check_env_conditions.py --task OpenDrawer
# LoRA trial: peak memory and step time at several batch sizes
bash scripts/server/submit_queue_job.sh 164 rc365_lora_trial 40000 1 "cards=0 1 2 3 5 6 7" -- \
    env PREFIX=affo16_lora_trial bash $PWD/scripts/rc365/run_lora_trial_gpu.sh
# training (restartable: the same RUN continues from its latest checkpoint)
bash scripts/server/submit_queue_job.sh 164 rc365_base 40000 1 "cards=0 1 2 3 5 6 7" -- \
    env RUN=affo16_base_oft_lora bash $PWD/scripts/rc365/run_train_base_gpu.sh      # -> $W/runs/robocasa365_starvla/<RUN>/
# evaluation: fold the adapters into the weights (CPU), then server + 24 simulator clients (BENCH=1: measure server capacity first)
python scripts/rc365/merge_lora_checkpoint.py --ckpt $W/runs/robocasa365_starvla/<RUN>/checkpoints/steps_N_pytorch_model.pt \
    --out $W/runs/robocasa365_starvla/<RUN>_merged_N
bash scripts/server/submit_queue_job.sh 164 rc365_eval_base 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env CKPT=$W/runs/robocasa365_starvla/<RUN>_merged_N/checkpoints/steps_N_pytorch_model.pt RUN=<name> \
    OVERRIDES="conditions=[A,B,C]" bash $PWD/scripts/rc365/run_eval_base_gpu.sh     # -> $W/runs/robocasa365_eval/<name>/
python scripts/rc365/summarise_eval.py $W/runs/robocasa365_eval/<name>              # table.md, summary.json
python scripts/rc365/video_montage.py --out sheet.png <videos>                      # look at failures
python scripts/rc365/classify_failures.py $W/runs/robocasa365_eval/<name>           # failures.json: class of every failed episode
/data/xinyizheng/envs/starvla/bin/python scripts/report/make_base_policy_figures.py --eval $W/runs/robocasa365_eval/<name> \
    --train $W/runs/robocasa365_starvla/<RUN> --out report/media/weekly_2026_11_23/base_policy   # success, failure and loss figures
```

Result (2026-11-27): run `affo16_base_oft_lora` (48000 steps, batch 16, seed 42) evaluated as `affo16_eval_base_48000`:
A 74/160, B 74/160, C 82/160; 26.5 of the 40 GPU hours. Details and the open problems: the record file above.

## 11. RoboCasa365 base-policy checks (Pipeline v4 step 1 (4a), AFFO-19; server)

Record: `docs/vla/2026-11-27_VLA_branch_第1步_底座检查.md`. Three checks of the section-10 base policy before any method is
added: success against training steps, why the arm stops at the stove knob and the faucet, and seen against unseen object
instances with more episodes. Cap: 4 GPU hours of the 13.5 left of AFFO-16's 40; one card at a time, queue priority 164.
The checkpoints are the merged ones of section 10 (`scripts/rc365/merge_lora_checkpoint.py`, output folders
`affo16_base_oft_lora_merged_{16000,32000,48000}`). The evaluation code is section 10's; the new parts are all off unless
asked for: `--probes part,objects` (`affordance_vla/sim/rc365_probe.py`), `first_episode` (config key), `CONFIG` (job
variable). Every episode seed is `crc32(base_seed/task/index)`, so the same index is the same kitchen across runs and splits.

```bash
# (1) 16000 / 32000 steps, condition A, 8 tasks x 20 episodes (configs/rc365/base_checks_steps.yaml)         -> affo19_steps<N>_A
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_steps 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=steps bash $PWD/scripts/rc365/run_base_checks_gpu.sh
# (2) TurnOffStove, TurnOnSinkFaucet with the part probe, n_act 8 and 4 at the same time (base_checks_stuck.yaml)   -> affo19_stuck_nact{8,4}
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_stuck 25000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=stuck bash $PWD/scripts/rc365/run_base_checks_gpu.sh
# (3) four pick tasks, A and B, episodes 20-79 (base_checks_instances.yaml; EXTRA="episodes=N" cuts the episodes)    -> affo19_instances_AB_20_79
bash scripts/server/submit_queue_job.sh 164 rc365_affo19_instances 20000 1 "cards=0 1 2 3 5 6 7" -- \
    env PHASE=instances bash $PWD/scripts/rc365/run_base_checks_gpu.sh
# objects of the episodes AFFO-16 already ran (CPU; same seeds, no policy): category and instance of every object
source envs/server/robocasa365_env.sh && PYTHONPATH=$W/repos/starVLA_rc365:. "$RC365_PYTHON" scripts/rc365/record_objects.py \
    --out $W/runs/robocasa365_eval/affo19_objects_0_19 --tasks CoffeeSetupMug PickPlaceCounterToCabinet \
    PickPlaceCounterToStove PickPlaceSinkToCounter --conditions A B --stop 20
# tables (CPU, any python with the repo): <D> = $W/runs/robocasa365_eval, O = $D/affo19_summary
python scripts/rc365/summarise_checks.py steps --run 16000=$D/affo19_steps16000_A --run 32000=$D/affo19_steps32000_A \
    --run 48000=$D/affo16_eval_base_48000 --out $O
python scripts/rc365/summarise_checks.py stuck --nact8 $D/affo19_stuck_nact8 --nact4 $D/affo19_stuck_nact4 \
    --reference $D/affo16_eval_base_48000 --out $O
python scripts/rc365/summarise_checks.py instances --new $D/affo19_instances_AB_20_79 --old $D/affo16_eval_base_48000 \
    --objects $D/affo19_objects_0_19/objects.jsonl --out $O
# figures for the weekly review (matplotlib, e.g. the starvla env)
/data/xinyizheng/envs/starvla/bin/python scripts/report/make_base_checks_figures.py --summary $O --nact8 $D/affo19_stuck_nact8 \
    --nact4 $D/affo19_stuck_nact4 --out report/media/weekly_2026_11_23/base_checks
```

Result (2026-11-28; all three runs finished, 3.5 of the 4 GPU hours; details in the record file): (1) condition A, 8 tasks x 20
episodes, success 27/160 (17 %) at 16 000, 59/160 (37 %) at 32 000, 74/160 (46 %) at 48 000 steps; 32 000 -> 48 000 paired
15 only-earlier against 30 only-later, sign test p = 0.036, so still rising after 32 000 (inference, medium confidence).
(2) TurnOffStove 0/20 for both n_act 8 and 4; TurnOnSinkFaucet 6/20 (n_act 8) and 8/20 (n_act 4); most failures reach the
part (stove: 17 of 20 within 8 cm; 7 touched without turning it) rather than stopping short; the n_act=8 rerun matches
AFFO-16 in 39 of 40 outcomes. (3) four pick tasks, 75 episodes per cell (AFFO-16's 20 + new indices 20-74; the job was cut
from 80 to 75 by `EXTRA="episodes=75"` to fit the GPU budget, the folder keeps the name `_20_79`): A 170/300, B 163/300,
paired only-A 47 / only-B 40, p = 0.52, A-B = +2.3 points (95 %: -3.8 to +8.4).
