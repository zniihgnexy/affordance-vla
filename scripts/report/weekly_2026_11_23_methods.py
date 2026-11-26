"""Section 6 of the week-3 review (2026-11-23): how to extract the grasp style of a category (survey of methods).

Imported by build_weekly_2026_11_23.py, which owns the notebook. Every entry is copied from
docs/vla/2026-11-24_VLA_branch_affordance提取方法调研.md (AFFO-5): abstracts, project pages and repository front
pages, plus a full-text round for NDF, Robo-ABC, RAM and VRB. Nothing was downloaded or run. "not found" = the
survey did not find it.
"""


def methods_cells(md, code) -> None:
    """One table, then the recommended route."""
    md(r"""
## 6 · How to extract an affordance for a whole category (survey)

**Question (AFFO-5, 24 Nov):** the way one object category is grasped must carry over to new instances of it. From which data, in which representation, by which transfer? Which one or two routes fit our pipeline? Abstracts and project pages for 10 methods (round 1), then full texts for four of them plus RAM (round 2); **nothing was downloaded or run**. Our own approach (demonstration statistics → a 3D map per category) is not counted among them.

| Method | Data it needs | Output | Transfer to a new instance | Direction? | Cross-instance evidence (as reported) | How it connects | Limit for us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **NDF** 2021 (2112.05124), full text | point cloud; 5–10 demonstrations | 6-DoF gripper pose | descriptor matching + optimisation | **yes, 3D** | unseen ShapeNet instances: mug upright 0.88 total success, any pose 0.58; 1 demonstration: 0.46; real robot qualitative only | external planner; MIT code | shapes only from ShapeNet; single-view point clouds not tested; no precedent as an offline label maker |
| Robo-ABC 2024 (2401.07487), full text | RGB; human-video memory of 51 categories | 2D contact point | retrieval + diffusion-feature correspondence | no (grasp direction from AnyGrasp) | contact success 60.7% (second best 29.1%); mug not listed alone | external | 2D point, no closing axis |
| RAM 2024 (2407.04689), full text | RGB-D; retrieval memory | 3D contact point + direction after contact | retrieval + feature correspondence | after contact, not gripper pose | simulation 52.6% on 70+ objects, 10 classes | external; also generates demonstrations for distillation | mug only in real-robot videos, no number |
| VRB 2023 (2304.08488), full text | RGB; egocentric human video | 2D contact heat-map + wrist path | end-to-end | 2D path only | cabinet / door / drawer kNN success 0.6 / 1.0 / 1.0; knob not tested; no mug or bowl grasp number | **trains** imitation or RL policies (not a VLA) | 2D, no grasp direction |
| Actron3D 2025 (2510.12971) | a few human RGB demonstrations | object-centric geometry, contact, motion functions | retrieve and align to the target scene | not stated | not found | external; needs RGB-D plus grasp and control modules | extra modules needed |
| Dense Object Nets 2018 | multi-view RGB-D, self-supervised | dense descriptors | nearest descriptor | no | said to generalise to unseen objects; numbers not found | external | points only |
| RoboPoint 2024 | synthetic data | image key-points from language | VLM | no | +21.8% spatial-affordance accuracy over GPT-4o and PIVOT (abstract) | external | no direction |
| Where2Act 2021 | simulated interaction | per-pixel actionability + trajectory | end-to-end | trajectory | cross-category, number not found | external | articulated parts, not grasping a mug |
| kPAM 2019, GG-CNN 2018 | key-point labels / grasp data | 3D key-points / per-pixel grasp | category key-points / end-to-end | no | not found / real 83–88% | external | key-points need hand definition; GG-CNN is only a model-head reference |
| AffordanceVLA 2026 (2606.06155), full text | 8.8M samples | object latent, 2D contact heat-map, 3D shape | end-to-end | no approach direction | LIBERO 95.8 vs 94.4 | **trains** a VLA; weights and data not released | the nearest neighbour; also the one to differ from |

- **Only AffordanceVLA trains a VLA; VRB trains imitation or RL policies; the other eight are external labels or planners.** So the contribution "a training method" has to come from our own obedience training; these methods only produce labels (inference).
- **Recommended route 1: our own category-level 3D map + nearest-point transfer.** It inherits the labels we have, needs no new data, the transfer is geometric and testable. Risk: LIBERO has meshes for only 3 mugs.
- **Recommended route 2: replace the correspondence with a descriptor, NDF first, Robo-ABC or RAM second.** NDF is the only one that carries a 6-DoF relative pose to a new shape. Risk: all are external-planner methods; they supply labels or a transfer step at best.
- **Minimal test the survey proposed (run in step 0 ①, section 3):** leave one mug out, nearest point vs NDF. Result: 1 of 6 pairs passes.
- **Not worth doing now:** VRB (2D, no grasp direction); features-only methods (points without axes) need a second layer to get the closing axis.
- **Not settled:** NDF on single-view point clouds; whether our noisy teleoperated grasps are cleaner or worse than the clean poses NDF expects (inference).
""")
