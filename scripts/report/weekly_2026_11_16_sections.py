"""Sections 5 (plan and decisions) and 6 (live demo commands for the local MuJoCo simulator) of the week-2 review.

Imported by build_weekly_2026_11_16.py, which owns the notebook; sections 2 and 3 are in weekly_2026_11_16_vla.py
and weekly_2026_11_16_pipeline.py.
Only the latest version of every block is kept. The commands of section 6 were run on the Mac on 23 Nov 2026
(scripts/local/sim.sh; the replays reproduced the server's outcomes).
"""


def plan_cells(md) -> None:
    """Section 5: skeleton and replaceable parts, the three stages, next steps, decisions of the week."""
    md("""
## 5 · Plan and decisions

**Approach (project lead, agreed 20 Nov):** fix the pipeline skeleton first; then pick downstream tasks; then use the comparisons to go back to specific modules or add new ideas. Each failure is attributed to one module (section 3.4), which says where to go back to.

| Skeleton (fixed at decision point 1) | Replaceable parts (changed in stage 3, by results) |
| --- | --- |
| VLM reads image and instruction | frozen vs full fine-tuning |
| interaction map head: scores for every object, depth, direction; event and approach | features: 8×8 → pre-merge 16×16 → DINOv2 → wrist view |
| semantic gate picks the target | size of the learnable projection; spatial language |
| decode to a plan vector; plan hold | object orientation: none → 3D alignment |
| action head trained to obey through counterfactuals | kinds of counterfactual, label rules |
| labels from simulated interaction (L1, category-level L2) | L1 → interaction surface; map transfer: nearest point → parts → semantic correspondence |
| evaluation: offline first-frame checks, closed loop, failure attribution | downstream task set (stage 2) |

| Stage | When | What | Ends with |
| --- | --- | --- | --- |
| 1 · skeleton | to ~10 Dec | E1 ✓, L ✓; E2 semantic gate (with Spatial); E3 orientation upper bound on LIBERO-10; minimal E4 (8×8 head + direction field); L2c category-level mug map + tall/short/fat/thin test set | decision point 1: skeleton fixed |
| 2 · pick tasks | ~11–17 Dec | inference only on candidate sets (LIBERO-PRO Object / 10 / Spatial, LIBERO-Plus, G3 and G4 sets, drawers and knobs): released vs skeleton vs skeleton + oracle plan, failures in 5 classes | decision point 2: task set and protocol |
| 3 · improve | ~18 Dec – end of Jan | rounds: find the failure class we lose most, change its module, keep a change only above 2× noise (~12/200); baselines B, D, E, F; full fine-tuning mid-Jan; paper-protocol numbers end of Jan | NeurIPS 2027 or IROS 2027 |

**Next steps (stage 1)**

1. **E2:** project the 3D maps onto the training frames (labels for every object); gate with a learnable projection; map head trained with Spatial; offline check including Spatial, then closed loop on Object. Pass: better object picking than g1 on the same data by > 2× noise; closed loop above 129 (position) and at least 145 (task).
2. **L2c** (CPU, alongside E2): LIBERO-90 mug demonstrations; category-level mug map with a leave-one-mug-out check; the scaled mug test set.
3. **E3:** LIBERO-10 plan dataset with the object of each phase from the contact records; oracle orientation in the plan (16 numbers) vs not.

**Decisions this week** (full text in [`docs/vla/README.md`](../docs/vla/README.md))

- **19 Nov:** Pipeline v3 replaces v2's method; baselines B, D, E, F; add the `_lan` suite. Evening: the directional interaction map is the core; labels L1 + L2; first version is a 2D map plus depth.
- **20 Nov, round 3:** L2 = per-category 3D map; a map for every object plus a multiplicative semantic gate; object orientation by 3D alignment; generalisation tests G1–G5.
- **20 Nov:** the three-stage plan; this project's GPU jobs use queue priority 164 and GPU 7.
- **20 Nov, round 5:** generalisation to unseen shapes of a category is a final goal → category-level L2, the scaled mug set, the G4 protocol.
- **20 Nov, after E1:** learnable projection in the gate; Spatial in E2.
- **Reporting:** one review per London calendar week; experiment notebooks are appendices in `report/experiments/`.
""")


def demo_cells(md) -> None:
    """Section 6: copy-paste commands that show the week's results live in MuJoCo on the Mac."""
    md("""
## 6 · Live demo in MuJoCo (Mac)

Run from the repository root on the Mac. Each command opens a MuJoCo window (mouse: rotate and zoom; space: pause; double-click an object, then Ctrl + right-drag to move it; close the window to end) and narrates in the terminal. **Replays** re-run the server's logged actions in the same scene, start state and seed: no model runs locally, and at the end the outcome and gripper path are compared with the server's record.

**Once:**

```bash
bash envs/local/setup_sim_env.sh       # conda env affordance-vla-sim: MuJoCo 3.2.3, robosuite 1.4.0 (as on the server)
bash scripts/local/fetch_sim_data.sh   # LIBERO-PRO, perturbed task files, evaluation logs, the L maps
```

**1. The test scene, with each object's demonstrated grasp point (orange):**

```bash
bash scripts/local/sim.sh scene --suite libero_object_swap --task 2 --episode 0
```

**2. Position perturbation, the salad dressing moved: released checkpoint (fails), then ours (succeeds; orange dot = the plan's contact):**

```bash
bash scripts/local/sim.sh replay runs/libero_pro/released/libero_object_swap/task2_ep0.npz
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0.npz
```

**3. Task perturbation, the tomato sauce named: released (fails), ours (succeeds):**

```bash
bash scripts/local/sim.sh replay runs/libero_pro/released/libero_object_task/task2_ep0.npz
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_task/task2_ep0.npz
```

**4. A failure of ours: the predicted contact is ~3 cm off and the gripper closes beside the tomato sauce (E0):**

```bash
bash scripts/local/sim.sh replay runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task5_ep0.npz
```

**5. L: the 3D interaction maps on the objects** (yellow → red = grasp contact score, blue line = closing axis; the maps follow an object when you drag it):

```bash
bash scripts/local/sim.sh map --suite libero_10 --task 4            # mugs: rim
bash scripts/local/sim.sh map --suite libero_10 --task 8            # moka pots: handle
bash scripts/local/sim.sh map --suite libero_object_swap --task 2   # LIBERO-Object: both sides
```

**6. LIBERO-10, released checkpoint on the mug task:** position perturbation (reaches for the red mug), then task perturbation (places both mugs as in training, ignoring the swapped instruction):

```bash
bash scripts/local/sim.sh replay runs/libero_pro/libero10_released/libero_10_swap/task4_ep0.npz
bash scripts/local/sim.sh replay runs/libero_pro/libero10_released/libero_10_task/task4_ep0.npz
```

**Options:** `--set viewer.realtime=0.5` plays at half speed; `bash scripts/local/sim.sh list runs/libero_pro/predicted_g1_hold_s10` lists every episode of a run with its outcome. Details: [`docs/local_simulation.md`](../docs/local_simulation.md).
""")
