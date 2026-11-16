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
bash scripts/local/fetch_sim_data.sh        # LIBERO-PRO at eafdb80 + task files + image-free evaluation logs
bash scripts/local/sim.sh scene  --suite libero_object_swap --task 2 --episode 0     # open a task's scene
bash scripts/local/sim.sh grasp  --suite libero_object_swap --task 2 --object ketchup_1  # scripted pick-and-place
bash scripts/local/sim.sh list   runs/libero_pro/predicted_g1_hold_s10              # logged test episodes
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
```

Defaults: `configs/sim/local.yaml`; override with `--config FILE` or `--set key=value`. Everything is shown
live in the MuJoCo viewer. `replay` reruns a server test episode with the actions the policy returned there;
checked on three logs (success and failure, with and without plans): same outcome and step count, end
effector within 0.5 mm of the server's log. Walk-through of a test, in Chinese: `docs/local_simulation.md`.

## 1. Data (server, CPU)

| Step | Command | Output |
| --- | --- | --- |
| LIBERO demonstrations (states and actions only, by byte ranges) | `source envs/server/libero_env.sh && $LIBERO_PYTHON scripts/data/fetch_libero_states.py libero_object $W/data/libero_states/libero_object` | `$W/data/libero_states/<suite>/` |
| Replay them in the simulator, one log per demonstration | `SUITE=libero_object bash scripts/data/replay_libero_states.sh` | `$W/runs/plan_following/demos/` |
| Plan dataset (tracks, images, contact offsets) | `python scripts/data/build_plan_dataset.py --logs $W/runs/plan_following/demos --out $W/runs/plan_following/dataset10` | `dataset10/` (LIBERO-Object, 10 tasks) |
| LIBERO-Spatial | `SUITE=libero_spatial RAW=$W/data/libero_states/libero_spatial OUT=$W/runs/plan_following/demos_spatial bash scripts/data/replay_libero_states.sh`, then `build_plan_dataset.py ... --container plate_1 --bddl-dir $W/repos/LIBERO-PRO/libero/libero/bddl_files/libero_spatial` | `dataset_spatial/` |
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

From `report/`: `python ../scripts/report/build_<name>.py <notebook>.ipynb`, then
`python -m jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`. Each builder's docstring
names its notebook and inputs; media go to the tracked `report/media/`.
