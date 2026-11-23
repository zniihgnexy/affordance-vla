"""Build report/2026-11-16_weekly_review.ipynb, the review of the London week 16-22 Nov 2026 (cells only; execute it).

Usage on the server (starvla env, from report/):
    python ../scripts/report/build_weekly_2026_11_16.py 2026-11-16_weekly_review.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-16_weekly_review.ipynb
Reads finished runs only: runs/libero_pro/{released,predicted_g1_hold_s10,oracle_plan_s*,predicted_g1_hold_s*,
full_*_s9_seed*,libero10_released*}, runs/interaction_labels, runs/semantic_probe/e1 and the replayed demonstrations.
Writes its videos to report/media/weekly_2026_11_16/ (tracked). Nothing runs on a GPU. In English, for the group
meeting of 23 Nov. Section 2 comes from weekly_2026_11_16_vla.py, section 3 from
weekly_2026_11_16_pipeline.py, sections 5 and 6 from weekly_2026_11_16_sections.py; only the latest version
of every figure and block is kept.
"""
import sys
from pathlib import Path

import nbformat as nbf
from weekly_2026_11_16_pipeline import pipeline_cells
from weekly_2026_11_16_sections import demo_cells, plan_cells
from weekly_2026_11_16_vla import vla_cells

OUT = Path(sys.argv[1])
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Week 2 review · 16–22 Nov 2026 (London time)

**Repository:** `affordance-vla` · **for the group meeting of 23 Nov 2026**

**Order:** 0 summary · 1 real demos · 2 the VLA, the data and the affordance map · 3 Pipeline v3 · 4 results · 5 plan and decisions · 6 live demo in MuJoCo

- Every table and figure is computed from finished runs on the server when the notebook executes.
- **ORACLE** = simulator ground truth; an "oracle plan" is an upper bound, not a result of the method.
- **[implemented]** = code runs and has results; **[planned]** = design agreed, not built yet.
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.sim.rollout_log import load_rollout_log
from report import grasp_orientation_helpers as G
from report import vla_pipeline_demo_helpers as D
from report import weekly_2026_11_16_helpers as K

set_matplotlib_formats("jpeg")
RUNS = ROOT / "runs"
PRO = RUNS / "libero_pro"
MEDIA = ROOT / "report" / "media" / "weekly_2026_11_16"   # tracked in git
MEDIA.mkdir(parents=True, exist_ok=True)
FIGURES = ROOT / "report" / "media" / "pipeline_overview"
OURS = PRO / "predicted_g1_hold_s10"   # grid plan head, plans kept, action head with the labels of 15 Nov
RELEASED = PRO / "released"            # the released checkpoint, no plan
""")

md("""
## 0 · Summary

**Thesis (unchanged):** affordance helps a VLA only if the action is *trained to obey it*. We train the action head with counterfactual plans (same image, plan moved, action relabelled).

**This week**

- **What we build on** (section 2): the released starVLA QwenPI LIBERO checkpoint (Qwen3-VL-4B + flow-matching action head); trained on LIBERO, tested zero-shot on LIBERO-PRO; the affordance map reaches the actions as one decoded contact point.
- **Pipeline v3, five revision rounds** (section 3). A grasp = *which object* (semantic gate) × *where and in which direction* (directional interaction map, drawn on every object) × *how to move* (action head that obeys the plan).
- **Generalisation is a final goal** (round 5): after learning how to grasp one category, grasp unseen shapes of it (tall / short / fat / thin mugs). Labels become category-level; new test set G4.
- **Results** (section 4):
  - new action-head labels **s10** are the default (task perturbation 152 → 166 with oracle plans, 136 → 151 with predicted plans);
  - **E0 fails:** failed grasps are *mispositioned* (gripper 4.7 cm from the object centre vs 2.0 cm when successful), not misoriented;
  - **L done:** a 3D interaction map per object category, from demonstration grasps (moka pot → handle, mug → rim);
  - **E1:** frozen-VLM instruction–image similarity is near chance → the semantic gate needs a learnable projection;
  - **LIBERO-10 baseline:** the released checkpoint gets 3/200 (position) and 18/200 (task perturbation).
- **Plan** (section 5): stage 1 skeleton (to ~10 Dec) → stage 2 pick downstream tasks → stage 3 comparison-driven rounds (to end of Jan).

**Best numbers so far** (LIBERO-PRO Object, paper protocol: 50 starts per task × 3 seeds; from week 1):

| | position perturbation | task perturbation |
| --- | --- | --- |
| released checkpoint | 0.000 | 0.090 |
| ours: predicted plan | 0.660 | 0.681 |
| oracle plan (upper bound) | 0.920 | 0.753 |
""")

md("""
## 1 · Real demos

All frames come from real closed-loop logs on the server. "Ours" = grid plan head `g1` + plan hold + action head `s10`; 20 starts per task, seed 7. Trained on standard LIBERO-Object only; tested zero-shot on LIBERO-PRO (**position perturbation:** objects moved; **task perturbation:** same scene, another object named). Orange outline = the named object.
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
for axis, (name, title) in zip(axes, [("scene_train", "training: LIBERO-Object (salad dressing named)"),
                                       ("scene_swap", "test: position perturbation (it moved)"),
                                       ("scene_task", "test: task perturbation (tomato sauce named)")]):
    axis.imshow(plt.imread(FIGURES / f"{name}.png"))
    axis.set_title(title, fontsize=10)
    axis.set_axis_off()
fig.tight_layout()
""")

md("""
### 1.1 Position perturbation: released checkpoint vs ours

The salad dressing moved. The released checkpoint reaches where it usually was in training; our plan head finds it in the image and the action head follows the plan.
""")

code("""
TASK = 2   # the salad dressing task
pair = [load_rollout_log(RELEASED / "libero_object_swap" / f"task{TASK}_ep0.npz"),
        load_rollout_log(OURS / "libero_object_swap" / f"task{TASK}_ep0.npz")]
for log, name in zip(pair, ["released", "ours"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released checkpoint", "ours: map plan kept + s10"], MEDIA / "swap_released_vs_ours.mp4")
Video(str(clip), embed=True, width=760)
""")

md("""
### 1.2 Task perturbation: same scene, the tomato sauce is named
""")

code("""
pair = [load_rollout_log(RELEASED / "libero_object_task" / f"task{TASK}_ep0.npz"),
        load_rollout_log(OURS / "libero_object_task" / f"task{TASK}_ep0.npz")]
for log, name in zip(pair, ["released", "ours"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released checkpoint", "ours: map plan kept + s10"], MEDIA / "task_released_vs_ours.mp4")
Video(str(clip), embed=True, width=760)
""")

md("""
### 1.3 Inside the method: the plan drawn on the image

Left: agent view with the VLM's 8 × 8 image cells; the orange cell and cross are the plan sent to the action head (the predicted contact, kept until the gripper has grasped). Right: wrist view. Below: first query, gripper starts closing, 0.6 s later, last step.
""")

code("""
log = load_rollout_log(OURS / "libero_object_swap" / f"task{TASK}_ep0.npz")
clip = K.plan_video(log, MEDIA / "ours_plan_overlay.mp4")
display(Video(str(clip), embed=True, width=760))
K.plan_strip(log);
""")

md("""
### 1.4 A failure: the contact is off (what E0 found)

The gripper closes next to the named object instead of around it (see the wrist view). E0 measures the same thing over all episodes (section 4.2).
""")

code("""
failed = K.first_failure(OURS, ["libero_object_swap", "libero_object_task"])
if failed is None:
    print("no failed rollout with images and a missed first grasp in this run")
else:
    print(failed.path.parent.name, failed.path.name, "-", failed.meta["instruction"])
    clip = K.plan_video(failed, MEDIA / "ours_failure.mp4")
    display(Video(str(clip), embed=True, width=760))
    K.plan_strip(failed);
""")

md("""
### 1.5 LIBERO-10: the released checkpoint on mugs

Two-step task: white mug → left plate, yellow-and-white mug → right plate.

- **Left, position perturbation:** the white mug is placed; the second reach goes to the red mug, which was not named.
- **Right, task perturbation (the two destinations swapped):** both mugs end up placed *as in training*, ignoring the changed instruction (checked from the logged object positions). This is the "wrong object" failure the semantic gate targets.
""")

code("""
L10 = PRO / "libero10_released"
pair = [load_rollout_log(L10 / "libero_10_swap" / "task4_ep0.npz"),
        load_rollout_log(L10 / "libero_10_task" / "task4_ep0.npz")]
for log, name in zip(pair, ["position", "task"]):
    print(f"{name}: {log.meta['instruction']} -> {'success' if log.meta['success'] else 'failure'}")
clip = D.render_rollout_grid(pair, ["released: position perturbation", "released: task perturbation"],
                             MEDIA / "libero10_released_mugs.mp4")
Video(str(clip), embed=True, width=760)
""")

vla_cells(md, code)
pipeline_cells(md, code)

md("""
## 4 · Results

### 4.1 New action-head labels (s10) and plan noise (s11)

s10 (from the 15 Nov diagnosis): for an "other object" counterfactual, the path is rotated and scaled about the start, and the last segment is the named object's own demonstrated approach. s11 adds plan noise to those samples. LIBERO-PRO Object, 20 starts per task, seed 7:
""")

code("""
HEADS = {"s9 (labels of 13 Nov)": "s9", "s10 (labels of 15 Nov, default)": "s10", "s11 = s10 + plan noise": "s11"}
display(K.label_runs_table(PRO, HEADS))
""")

md("""
- **s10 is the default** (agreed 16 Nov): task perturbation improves with both oracle and predicted plans. Cost: with predicted plans, tomato sauce under position perturbation drops 13 → 2, because the new head follows a plan that is ~3 cm off more strictly.
- **s11 is not adopted:** no overall gain. Per task: appendix [`experiments/2026-11-16_label_fix_results.ipynb`](experiments/2026-11-16_label_fix_results.ipynb).

### 4.2 E0: are failed grasps misoriented?

Pre-registered pass line: failed grasps deviate from the demonstrations by at least 10° more than successful ones. Data: all paper-protocol rollouts, 2 configurations × 3 seeds (3000 episodes each). Appendix: [`experiments/2026-11-20_grasp_orientation.ipynb`](experiments/2026-11-20_grasp_orientation.ipynb).
""")

code("""
demos = G.demo_grasps(sorted((RUNS / "plan_following" / "demos").glob("*.npz")))
reference = G.demo_rotations(demos)
configs = {"predicted plan (g1 + s9)": "full_g1_hold_s9_seed{}", "oracle plan (s9)": "full_oracle_s9_seed{}"}
for name, pattern in configs.items():
    attempts = pd.concat([G.rollout_attempts(PRO / pattern.format(seed), ["libero_object_swap", "libero_object_task"],
                                             reference) for seed in (7, 8, 9)], ignore_index=True)
    gap, passed = G.verdict(attempts)
    print(f"{name}: failed grasps deviate {gap:.1f} deg more -> {'pass' if passed else 'fail'} (pass line 10 deg)")
    display(G.group_table(attempts)[["episodes", "angle_median", "yaw_median", "tilt_median"]])
    display(G.distance_table(attempts))
""")

md("""
- **E0 fails:** failed grasps are oriented like the demonstrations (only 1–2° more off), and the demonstrations use essentially one grasp (from the top).
- **The difference is position:** distance from gripper to object centre at closing is ~2.0 cm (grasped and succeeded) vs ~4.7 cm (missed).
- **As agreed in advance:** the orientation branch moves to LIBERO-10 (mugs, moka pots); on LIBERO-Object we work on the position part of the map (scores, depth, finer features).

### 4.3 L: 3D interaction map per object category

For every grasp in the demonstrations (simulator reports a grasp, held ≥ 5 steps), the gripper pose is moved into the object's frame, and the surface points near the closing line that face it get weight. Summed over a category: per surface point an interaction score, a main closing axis (180°-symmetric) and an agreement score. Formulas in section 3.3; code [`interaction_labels.py`](../affordance_vla/plan/interaction_labels.py). Pass criterion: closing axes of a category form few clusters, and the map projected into a frame matches where the demonstration grasps.
""")

code("""
LABELS = RUNS / "interaction_labels"
for name, title in [("object", "LIBERO-Object"), ("libero10", "LIBERO-10")]:
    print(title)
    display(K.label_summary(LABELS / name / "summary.json"))
""")

md("""
Check figures, left to right: the map from two sides (colour = score, red = closing axis at the top points); the map projected onto the first frame (8 × 8 cells); the frame where the demonstration grasps (× = gripper). The two left panels use the object's own axes, so for the soup can "from above" shows its side. Section 6 shows the same maps live on the objects in MuJoCo.
""")

code("""
for name, category in [("object", "alphabet_soup"), ("libero10", "moka_pot"), ("libero10", "porcelain_mug")]:
    display(Image(filename=str(LABELS / name / "check" / f"{category}.png"), width=1000))
""")

md("""
- Soup can: fingers on both sides of the body. Moka pot: on the **handle**. Mug: on the **rim**, one finger inside. The last two carry direction information that symmetric cans do not.
- 9 of 10 LIBERO-Object categories have > 80% of grasps in one axis cluster (tomato sauce 76%). LIBERO-10 mugs split into several clusters (grasps around the rim), but the axis agreement per surface point stays 0.93–0.97. Open: alphabet soup in LIBERO-10 splits 63 / 37.
- **L passes.** The maps are **statistics of the demonstrations, not learned**: they repeat how the demonstrators grasped. What is learned is the map head that predicts them from the image.

### 4.4 E1: does the frozen VLM already point at the named object?

74 LIBERO-PRO first frames (Object 40, Spatial 34). For each of the 36 layers: cosine similarity between instruction tokens (or only the object-name tokens) and the 64 cell tokens, merged by mean or max. Two readings: does the best cell lie on the named object (chance ≈ 5%); does the named object have the highest mean similarity among the objects in view (chance = 1 / objects) — the second is what a multiplicative gate sees.
""")

code("""
print("best cell on the named object (all 74 frames, best layer per variant)")
display(K.probe_table(PROBE / "summary.json"))
print("named object has the highest mean similarity among the objects in view")
display(K.probe_object_table(PROBE / "probe.npz"))
print("g1 grid head: where its best first-frame cell lands")
display(K.probe_head_table(PROBE / "probe.npz"))
""")

md("""
- **Frozen similarity is close to chance** (best peak 9.5%; object level 0.30 vs 0.14 chance on Object). On Spatial it finds "a bowl" but not which of the two identical bowls. Layer and merge differences are within noise.
- **g1 rarely picks the wrong object on Object** (3/40); its errors are position (6/20 over 2 cm under position perturbation). On Spatial, which it never saw, 3/34.
- **Decisions:** the gate gets a small learnable projection (instruction tokens, mean, last layer first), as v3 planned for this case; E2's offline checks include **Spatial**, where words must separate identical objects, and the map head trains on Spatial too (agreed 20 Nov).

### 4.5 LIBERO-10 baseline: the released checkpoint on LIBERO-PRO

20 starts per task, seed 7. Tasks 2 and 3 of the task perturbation were rerun separately (their first object of interest was a fixture).
""")

code("""
for suite, name in [("libero_10_swap", "position perturbation"), ("libero_10_task", "task perturbation")]:
    table = K.task_results([PRO / run for run in K.LIBERO10_RUNS], suite)
    print(f"{name} ({suite}): {table['successes'].sum()} / {table['episodes'].sum()}, {len(table)} tasks")
    display(table)
""")

plan_cells(md)
demo_cells(md)

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, OUT)
print("wrote", OUT)
