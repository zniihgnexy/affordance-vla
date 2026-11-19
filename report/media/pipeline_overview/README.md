# Pipeline figures

Every figure in this folder is drawn by the cvpr-figure skill from the YAML spec beside its PNG. The pictures inside
the figures are real crops, and they live in this folder too.

| Figure | What it shows | Status |
| --- | --- | --- |
| `pipeline_overview.png` | the pipeline as it runs now; it is the figure in the first section of `report/2026-11-16_weekly_review.ipynb` | current system |
| `v3_model.png` | Pipeline v3 at test time. The frozen VLM feeds the interaction affordance map head, which outputs an interaction score, depth and a direction field (gripper closing direction). A semantic prior picks the named object. The plan (contact and gripper orientation) is decoded from the map into 16 numbers, and the action head follows it | **guide since 19 Nov 2026** |
| `v3_training.png` | How v3 trains its two heads. The map head learns from gripper-object interaction labels: L1 comes from demonstration grasps, L2 from per-category canonical maps projected by object pose (labels only). The action head learns from counterfactual samples | **guide since 19 Nov 2026** |
| `v3_generalization.png` | v3 generalisation. Training is on LIBERO-Object and testing is zero-shot on LIBERO-PRO, with LIBERO-10 (mugs, moka pots) added later for orientation. The interaction map has to generalise; the action head only has to obey. Each failure is traced back to one module | **guide since 19 Nov 2026** |
| `pose_plan_pipeline.png` | first proposal (18 Nov): adds only the gripper orientation; superseded by the v3 figures | record |

The v3 figures belong to `docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md`. The project lead confirmed it on
19 Nov 2026, together with the revision that evening that puts the interaction affordance map at its centre. Purple dashed boxes mark what v3 adds and has not implemented yet. Every other part of the figures, and
all the crops, show the current system.

| File | What it is |
| --- | --- |
| `<name>.yaml` | the spec of figure `<name>`: boxes, arrows, colour roles and caption |
| `<name>.png` | the rendered figure, at 3x its size in points |
| `input.png`, `plan.png`, `outcome.png` | frames of a real LIBERO-PRO rollout (`runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0`, success): position perturbation, salad dressing, the grid plan head with plan hold, and the s10 action head |
| `counterfactual.png` | first frame of a real demonstration (salad dressing, demo 0). Dark is the demonstration's own way to the grasp; orange is the same way rewritten to the ketchup by the 15 Nov rule |
| `scene_train.png`, `scene_swap.png`, `scene_task.png` | first frames with the object the instruction names outlined in orange. `scene_train`: the demonstration above (LIBERO-Object). `scene_swap`: the rollout above (position perturbation). `scene_task`: the task perturbation in the same salad-dressing scene, where the instruction names the tomato sauce (`predicted_g1_hold_s10/libero_object_task/task2_ep0`, success) |

Remake the crops on the server (starvla env, no GPU). This rewrites the first four crops byte for byte:

```bash
W=/mnt/nas/xinyizheng/affordance_vla; R=$W/runs/libero_pro/predicted_g1_hold_s10
D=$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz
python scripts/report/pipeline_figure_crops.py --rollout $R/libero_object_swap/task2_ep0.npz --demo $D \
    --data $W/runs/plan_following/dataset10 --other ketchup_1 --out report/media/pipeline_overview \
    --scene scene_train=$D --scene scene_swap=$R/libero_object_swap/task2_ep0.npz \
    --scene scene_task=$R/libero_object_task/task2_ep0.npz
```

To render a figure you need a machine with the cvpr-figure skill, cairosvg and a CJK font; this was done on the
Mac. `NAME` is the stem of a spec. The skill writes Helvetica, which has no Chinese glyphs, so the font is swapped
in the SVG before rasterising:

```bash
python <skill>/scripts/render.py $NAME.yaml -o out/$NAME -f svg
sed -E "s/font-family='[^']*'/font-family='Hiragino Sans GB'/g" out/$NAME.svg > out/zh.svg
python -c "import cairosvg; cairosvg.svg2png(url='out/zh.svg', write_to='$NAME.png', scale=3.0)"
python <skill>/scripts/validate.py $NAME.yaml --svg out/$NAME.svg
```

Audit results:

- All figures pass with 0 errors. `pipeline_overview` and `pose_plan_pipeline` also have no warnings.
- The v3 figures keep two kinds of warning:
  - They are taller than a paper page's height budget. They are made for the docs, not for a paper.
  - Some labels are long, because Chinese lines are counted as words.

Hiragino has no U+2212 minus sign, so write labels without it.

Add `-f svg,pptx` to get a PowerPoint file to edit by hand. Once that file has been edited, it becomes the source
of the figure.
