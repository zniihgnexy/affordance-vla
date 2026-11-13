# Affordance-aware VLA training

This repository holds the VLA direction of the affordance project: a training method for vision-language-action models in cluttered tabletop manipulation. The VLA predicts a contact–approach–waypoint plan, its actions are kept consistent with that plan during training, and a lightweight per-episode affordance state tracks objects that become occluded or are moved. The claim to test is higher closed-loop success in simulation than the same backbone trained the usual way, with fewer wrong-object, wrong-direction and collision failures.

Everything here is a research plan with data and label checks, one walk-through of the pipeline with a released open-source checkpoint, and a first diagnostic training run on top of that checkpoint (11 November 2026: does the action head follow a plan it is given?). The full method has not been trained yet.

On 10 November 2026 this repository was split from the `vla` branch of [affordance3D](https://github.com/zniihgnexy/affordance3D), with its full history. The environment-memory direction (`mem`) stays in affordance3D.

## Start here

1. [VLA documentation index](docs/vla/README.md) — all design, search and setup records.
2. [Pipeline v2](docs/vla/2026-11-10_VLA_branch_Pipeline_v2.md) — the current design: method, backbone and benchmark choices, experiment configurations, stages V0–V5.
3. [Pipeline demo](report/2026-11-10_vla_pipeline_demo.ipynb) — one LIBERO task step by step with the released starVLA checkpoint: training sample, model internals, open loop, 21 closed-loop rollouts, and the planned labels (executed notebook).
4. [Plan-following experiment](report/2026-11-11_plan_following_experiment.ipynb) — the released action head is given an oracle plan and a small add-on is trained: it follows the plan only once training includes moved plans (executed notebook; record in [`docs/vla`](docs/vla/2026-11-11_VLA_branch_计划跟随实验.md)).
5. [LIBERO-PRO, Object suite](report/2026-11-12_libero_pro_object.ipynb) — the released policy against an action head steered by an oracle plan on the official position and task perturbations: 0/200 and 20/200 against 164/200 and 136/200 (executed notebook; the plan is ground truth, so this is an upper bound).
6. [Predicted plans on LIBERO-PRO](report/2026-11-13_plan_predictor.ipynb) — the plan is predicted from the camera images and the instruction by a small head on the frozen VLM: a regressed contact does no better than the released policy, a score map over the image cells reaches 133/200 and 136/200 with the ten-task action head trained with the labels of 13 Nov, against 185/200 and 152/200 with an oracle plan (executed notebook; record in [`docs/vla`](docs/vla/2026-11-13_VLA_branch_预测计划驱动闭环.md)).
7. [Pipeline walk-through with an affordance map](report/2026-11-11_affordance_pipeline_walkthrough.ipynb) — one moment of a real rollout followed through the nine steps of the pipeline, with the input and output of each step drawn; the affordance map and the plan are computed from the simulator (executed notebook, for discussion).
8. [Datasets and simulation report](report/2026-11-10_vla_datasets_sim_report.ipynb) — benchmark overview and one LIBERO task in detail (executed notebook).
9. [V0 server setup record](docs/vla/2026-11-10_VLA_branch_V0_服务器搭建记录.md) — what is installed on the A6000 server and how the GPU queue works.

## Code

- `affordance3d/libero_io.py` reads LIBERO demonstration files and BDDL scene descriptions.
- `affordance3d/caw_labels.py` derives contact–approach–waypoint labels from end-effector paths and gripper commands, and checks how well summed actions track the path.
- `affordance3d/plan_consistency.py` compares the path an action chunk asks for with the plan still ahead; `affordance3d/episode_state.py` keeps one rule-based slot per object through an episode; `affordance3d/rollout_outcome.py` labels a rollout from simulator grasp flags.
- `affordance3d/plan_input.py` turns a contact, an approach direction and the kind of gripper event into the plan input of the action head; `affordance3d/counterfactual.py` and `affordance3d/plan_samples.py` build training samples whose plan is moved elsewhere, from the demonstration tracks stored by `affordance3d/plan_dataset.py`; `affordance3d/path_loss.py` is the plan–action consistency loss; `affordance3d/lora.py` and `affordance3d/plan_policy.py` put the plan input and low-rank updates on the released action head; `affordance3d/plan_stages.py` splits a rollout into reached, held and delivered; `affordance3d/plan_targets.py` and `affordance3d/plan_predictor.py` train a small readout of the VLM that predicts the plan; `affordance3d/benchmark_text.py` gives the instruction of a perturbed LIBERO-PRO task. `affordance3d/plan_grid.py` is the plan head that scores the image cells of the agent view; `affordance3d/plan_hold.py` keeps a predicted plan until its gripper event; `affordance3d/plan_audit.py` tells which object each logged plan pointed at.
- `affordance3d/rollout_log.py` reads the per-step episode logs and `affordance3d/libero_camera.py` projects world points into the image the policy sees and lifts a depth image back to world points; `affordance3d/affordance_map.py` builds an affordance map from simulator ground truth and gives the 3D point a map stands for.
- `report/vla_report_helpers.py` and `scripts/build_vla_report.py` build the datasets report; `report/vla_pipeline_demo_helpers.py`, `report/vla_pipeline_demo_labels.py` and `scripts/build_vla_pipeline_demo.py` build the pipeline demo; `report/plan_following_helpers.py`, `scripts/build_plan_following_report.py` and `scripts/conclusions_plan_following.md` build the plan-following notebook; `report/affordance_pipeline_helpers.py`, `scripts/build_affordance_pipeline_walkthrough.py` and `scripts/conclusions_affordance_pipeline.md` build the walk-through. `report/libero_pro_helpers.py` and `report/plan_predictor_helpers.py` with `scripts/build_libero_pro_report.py` and `scripts/build_plan_predictor_report.py` build the two LIBERO-PRO notebooks.
- `scripts/remote/` sets up the starVLA training stack and the LIBERO simulator on the A6000 server, and holds the probe, rollout, training and evaluation scripts that go through the GPU queue.
- **Working copy:** since 10 November 2026 the working copy is `~/affordance-vla` on the A6000 server, with data, models and outputs on the NAS; the layout is in the V0 server setup record.
- **Environment:** on the server, the starVLA env built by `scripts/remote/setup_starvla_env.sh`. [`requirements.txt`](requirements.txt) is the earlier Mac environment. Run the tests with `python -m pytest`.

## Background carried over from affordance3D

The earlier documents and prototype are the brainstorming base shared with the memory direction. They are kept for reference and are not maintained here.

- [Research handoff and first experiments](docs/research/实验前研究交接_2026-11-06.md), [pipeline history](docs/pipeline.md), [literature](docs/literature.md), [datasets and simulation](docs/data-and-simulation.md), [project scope](docs/项目总控.md), and the [documentation index](docs/README.md).
- Prototype code for RGB-D geometry, DINOv2 features, voxel memory and SceneFun3D (`affordance3d/`, `notebooks/`, `report/2026-11-09_datasets_report.ipynb`). `affordance3d/geometry.py` and `affordance3d/memory.py` are reused by the VLA plan.

Dataset and platform descriptions are not proof of access, installation or compatibility; individual resources and their licences should be checked before use.
