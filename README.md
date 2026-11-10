# Affordance-aware VLA training

This repository holds the VLA direction of the affordance project: a training method for vision-language-action models in cluttered tabletop manipulation. The VLA predicts a contact–approach–waypoint plan, its actions are kept consistent with that plan during training, and a lightweight per-episode affordance state tracks objects that become occluded or are moved. The claim to test is higher closed-loop success in simulation than the same backbone trained the usual way, with fewer wrong-object, wrong-direction and collision failures.

Everything here is a research plan with data and label checks. No model has been trained yet.

On 10 November 2026 this repository was split from the `vla` branch of [affordance3D](https://github.com/zniihgnexy/affordance3D), with its full history. The environment-memory direction (`mem`) stays in affordance3D.

## Start here

1. [VLA documentation index](docs/vla/README.md) — all design, search and setup records.
2. [Pipeline v2](docs/vla/2026-11-10_VLA_branch_Pipeline_v2.md) — the current design: method, backbone and benchmark choices, experiment configurations, stages V0–V5.
3. [Datasets and simulation report](report/2026-11-10_vla_datasets_sim_report.ipynb) — benchmark overview and one LIBERO task in detail (executed notebook).
4. [V0 server setup record](docs/vla/2026-11-10_VLA_branch_V0_服务器搭建记录.md) — what is installed on the A6000 server and how the GPU queue works.

## Code

- `affordance3d/libero_io.py` reads LIBERO demonstration files and BDDL scene descriptions.
- `affordance3d/caw_labels.py` derives contact–approach–waypoint labels from end-effector paths and gripper commands, and checks how well summed actions track the path.
- `report/vla_report_helpers.py` and `scripts/build_vla_report.py` build the report notebook.
- `scripts/remote/` sets up and probes the starVLA training stack on the A6000 server.
- **Environment:** [`requirements.txt`](requirements.txt), installed with conda + pip (commands at the top of the file). Run the tests with `python -m pytest`.

## Background carried over from affordance3D

The earlier documents and prototype are the brainstorming base shared with the memory direction. They are kept for reference and are not maintained here.

- [Research handoff and first experiments](docs/research/实验前研究交接_2026-11-06.md), [pipeline history](docs/pipeline.md), [literature](docs/literature.md), [datasets and simulation](docs/data-and-simulation.md), [project scope](docs/项目总控.md), and the [documentation index](docs/README.md).
- Prototype code for RGB-D geometry, DINOv2 features, voxel memory and SceneFun3D (`affordance3d/`, `notebooks/`, `report/2026-11-09_datasets_report.ipynb`). `affordance3d/geometry.py` and `affordance3d/memory.py` are reused by the VLA plan.

Dataset and platform descriptions are not proof of access, installation or compatibility; individual resources and their licences should be checked before use.
