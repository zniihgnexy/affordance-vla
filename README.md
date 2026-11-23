# Affordance-aware VLA training

The VLA direction of the affordance project. Thesis (since 12 November 2026): an affordance helps a
vision-language-action policy only when the action is trained to obey it. The method is counterfactual
affordance supervision: the same image is paired with a plan moved elsewhere, and the action labels are
rewritten along the moved path, so the action head must follow the plan; a small plan head on the VLM
predicts the plan at test time. So far everything runs at diagnostic scale (frozen VLM, two small heads,
LIBERO and LIBERO-PRO); the main comparison with full VLA training is still to come.

This repository was split from the `vla` branch of [affordance3D](https://github.com/zniihgnexy/affordance3D)
on 10 November 2026 with its full history; the environment-memory direction stays there.
Agents: read [AGENTS.md](AGENTS.md) first (the project lead's requirements for research code).

## Layout

```
affordance_vla/      shared code, by domain
  plan/              the plan the action head reads; labels made from demonstrations; keeping a plan
  heads/             the plan-conditioned action head (LoRA on starVLA) and the plan head (grid / regress)
  training/          counterfactual samples, plan datasets, targets and losses
  sim/               LIBERO / LIBERO-PRO: test-episode loop, rollout logs, cameras, instructions, viewer, scripted grasp
  analysis/          reading closed-loop rollouts
  legacy/            pre-split SceneFun3D / memory code, kept for the old notebooks
  config.py          YAML configs with a base and key=value overrides
configs/             configuration files (configs/sim/local.yaml: the local simulator)
scripts/             entrypoints by purpose: data/ vla/ (train, serve) eval/ report/ server/ local/ legacy/
envs/                environment specs and setup: server/ (starVLA, LIBERO), local/ (Mac simulator), legacy/
report/              executed notebooks, their helpers, tracked media
docs/                records (docs/vla/ for this direction); docs/local_simulation.md
tests/               pytest, mirroring affordance_vla/
EXPERIMENTS.md       every experiment's command and settings
```

## Where things live

- **Code:** GitHub; the working copy where commits are made is `~/affordance-vla` on the A6000 server, and a
  Mac clone pulls from GitHub.
- **Data, models, run outputs:** only on the server's NAS (`/mnt/nas/xinyizheng/affordance_vla`: `data/`,
  `models/`, `runs/`, `repos/`), never in git.
- **On a Mac:** the full code plus what the local simulator needs (git-ignored): the LIBERO-PRO checkout in
  `third_party/` and image-free evaluation logs and contact offsets in `runs/`, fetched by
  `scripts/local/fetch_sim_data.sh`.

## Running things

- **Local simulator (Mac):** `bash scripts/local/sim.sh scene | grasp | replay | list` — open a LIBERO-PRO
  scene, run a scripted grasp, or replay a logged test episode live in the MuJoCo viewer
  ([docs/local_simulation.md](docs/local_simulation.md), in Chinese).
- **Experiments (server):** data preparation, training, evaluation and notebooks, with exact commands and
  settings: [EXPERIMENTS.md](EXPERIMENTS.md). GPU work goes through the shared queue only.
- **Checks:** `python -m pytest` and `python -m ruff check .` (server: starVLA env; Mac: `affordance-vla-sim`).

## Start here

1. [VLA documentation index](docs/vla/README.md) — all design, search and setup records.
2. [Pipeline v3](docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md) — the current design (in Chinese): a directional interaction affordance map on every object, a semantic gate that picks the named one, and an action head trained to follow the plan; experiments run in three stages. It replaces the method part of [Pipeline v2](docs/vla/2026-11-10_VLA_branch_Pipeline_v2.md).
3. [V0 server setup record](docs/vla/2026-11-10_VLA_branch_V0_服务器搭建记录.md) — what is installed on the A6000 server and how the GPU queue works.

### Weekly reviews

One notebook per London calendar week (Monday to Sunday), named after its Monday; within a week newer content replaces older content. In Chinese.

- [Weekly review, 16-22 Nov](report/2026-11-16_weekly_review.ipynb) — real demos first, the VLA model and data we build on and how the affordance map reaches the actions, how the pipeline changed this week, the Pipeline v3 figures with a block-by-block explanation, the new labels of 16 Nov, E0, the 3D canonical interaction maps (L), the LIBERO-10 baseline and the three-stage plan (executed notebook).
- [Weekly review, 9-15 Nov](report/2026-11-09_weekly_review.ipynb) — in Chinese: the week's timeline, the main results with tables and a figure read from the finished runs, the negative results, and the gap to the target research scope with a plan to the January checkpoint (executed notebook).

### Experiment notebooks (appendices of their week, `report/experiments/`)

- [E0: are failed grasps a wrong gripper orientation?](report/experiments/2026-11-20_grasp_orientation.ipynb) — first step of Pipeline v3, from existing logs only: failed first grasps lie only 1.4 degrees further from the nearest demonstration than held ones (pass line 10); the difference is position, 4.7 against 2.0 cm from the object centre at closing (executed notebook).
- [New labels for moved plans, tested](report/experiments/2026-11-16_label_fix_results.ipynb) — two ten-task heads that differ only in their labels: with the labels of 15 Nov the task perturbation rises from 152 to 166 of 200 (oracle plan) and from 136 to 151 (map plans), the orange juice from 8 to 18; with map plans the tomato sauce of the position perturbation drops from 13 to 2 (executed notebook).
- [Why three tasks of the task perturbation fail](report/experiments/2026-11-15_task_perturbation_failures.ipynb) — a diagnosis from existing logs: the training labels for "another object named" make a detour when the object asked for is close to the start, and copy the original target's final approach (executed notebook; record in [`docs/vla`](docs/vla/2026-11-15_VLA_branch_任务扰动失败诊断.md)).
- [LIBERO-PRO under the papers' protocol](report/experiments/2026-11-14_libero_pro_full_protocol.ipynb) — 50 start states per task and three evaluation seeds: without any oracle plan 0.660 ± 0.012 (position) and 0.681 ± 0.012 (task); with an oracle plan 0.920 and 0.753; the released policy 0.000 and 0.090 (executed notebook).
- [Predicted plans on LIBERO-PRO](report/experiments/2026-11-13_plan_predictor.ipynb) — the plan is predicted from the camera images and the instruction by a small head on the frozen VLM: a regressed contact does no better than the released policy, a score map over the image cells reaches 133/200 and 136/200 with the ten-task action head trained with the labels of 13 Nov, against 185/200 and 152/200 with an oracle plan (executed notebook; record in [`docs/vla`](docs/vla/2026-11-13_VLA_branch_预测计划驱动闭环.md)).
- [LIBERO-PRO, Object suite](report/experiments/2026-11-12_libero_pro_object.ipynb) — the released policy against an action head steered by an oracle plan on the official position and task perturbations: 0/200 and 20/200 against 164/200 and 136/200 (executed notebook; the plan is ground truth, so this is an upper bound).
- [Plan-following experiment](report/experiments/2026-11-11_plan_following_experiment.ipynb) — the released action head is given an oracle plan and a small add-on is trained: it follows the plan only once training includes moved plans (executed notebook; record in [`docs/vla`](docs/vla/2026-11-11_VLA_branch_计划跟随实验.md)).
- [Pipeline walk-through with an affordance map](report/experiments/2026-11-11_affordance_pipeline_walkthrough.ipynb) — one moment of a real rollout followed through the nine steps of the pipeline, with the input and output of each step drawn; the affordance map and the plan are computed from the simulator (executed notebook, for discussion).
- [Pipeline demo](report/experiments/2026-11-10_vla_pipeline_demo.ipynb) — one LIBERO task step by step with the released starVLA checkpoint: training sample, model internals, open loop, 21 closed-loop rollouts, and the planned labels (executed notebook).
- [Datasets and simulation report](report/experiments/2026-11-10_vla_datasets_sim_report.ipynb) — benchmark overview and one LIBERO task in detail (executed notebook).

## Background carried over from affordance3D

The earlier documents and prototype are the brainstorming base shared with the memory direction. They are kept for reference and are not maintained here.

- [Research handoff and first experiments](docs/research/实验前研究交接_2026-11-06.md), [pipeline history](docs/pipeline.md), [literature](docs/literature.md), [datasets and simulation](docs/data-and-simulation.md), [project scope](docs/项目总控.md), and the [documentation index](docs/README.md).
- Prototype code for RGB-D geometry, DINOv2 features, voxel memory and SceneFun3D (`affordance_vla/legacy/`, `report/legacy/`, `report/experiments/2026-11-09_datasets_report.ipynb`). `affordance_vla/geometry.py` is reused by the VLA code.

Dataset and platform descriptions are not proof of access, installation or compatibility; individual resources and their licences should be checked before use.
