# Pipeline figures

Every figure in this folder is drawn by the cvpr-figure skill from the YAML spec beside its PNG, rendered with
`scripts/report/render_figure_spec.py` (text in Liberation Sans, the Helvetica metrics the engine sizes boxes for). Since
23 Nov 2026 the v3 figures and the layer-level figures are in English (week-2 group meeting). The pictures inside
the figures are real crops, and they live in this folder too.

| Figure | What it shows | Status |
| --- | --- | --- |
| `pipeline_overview.png` | the pipeline as it runs now; it is the figure in the first section of `report/2026-11-09_weekly_review.ipynb` | current system |
| `v3_model.png` | Pipeline v3 at test time, redrawn on 23 Nov in the structure of `detail/v3_detail_1_overview.png`: the inputs; the open-source starVLA QwenPI model in gold (vision encoder, prompt, 36-layer LLM; the DiT action head); our map head g1, the planned semantic gate and per-object scores, decoding to a 3D contact, the 10-number plan vector, the plan hold, the plan encoder; actions out. The links ①–⑥ between starVLA and our parts are numbered as in section 2.2 of `report/2026-11-16_weekly_review.ipynb` | **current, 23 Nov 2026** |
| `v3_training.png` | How the two heads train, redrawn on 23 Nov at the same detail: (1) the map head on the frozen VLM, its four losses and their ORACLE labels (object position + mean demo grasp offset, projected into the agent view), with the planned per-object and direction losses and the planned L2 / L2c labels; (2) the action head: the kinds of counterfactual plan, how the path is rewritten, the flow-matching + path loss, and the 30.75M trained parameters of s10 | **current, 23 Nov 2026** |
| `v3_generalization.png` | Generalisation, redrawn on 23 Nov at the same detail: (1) trained on LIBERO-Object, tested zero-shot on LIBERO-PRO, with the best numbers so far; (2) tests G1–G5, G4 (unseen shapes of a category) the final goal; (3) the diagnosis loop from the offline first-frame check through closed loop to the five failure classes and their modules | **current, 23 Nov 2026** |
| `detail/v3_detail_<n>_<name>.png` | five layer-level figures of v3 after round 5 (whole test-time path; how the two heads read the VLM layers; decoding a cell to a 3D point; contact weights of one grasp; rewriting a counterfactual path). Numbers checked against the code and checkpoint configs on 20 Nov 2026; shown with their formulas in sections 3.2 and 3.3 of `report/2026-11-16_weekly_review.ipynb`. Figure 1 also marks the open-source starVLA model (gold zone) and the links ①–⑥ to our parts (23 Nov; sections 2.1 and 2.2 of that review) | **current, 20 Nov 2026** |
| `pose_plan_pipeline.png` | first proposal (18 Nov): adds only the gripper orientation; superseded by the v3 figures | record |

The v3 figures belong to `docs/vla/2026-11-19_VLA_branch_Pipeline_v3.md`. The project lead confirmed it on
19 Nov 2026, together with the revision that evening that puts the interaction affordance map at its centre, and the third revision of 20 Nov (3D canonical maps as labels, maps for every object with a semantic gate, 3D alignment for object orientation, generalisation tests). Purple dashed boxes mark what v3 adds and has not implemented yet. Every other part of the figures, and
all the crops, show the current system.

| File | What it is |
| --- | --- |
| `detail/pipeline_v3_layers.html` | source of the layer-level figures: a page with inline SVG and KaTeX formulas. `python scripts/report/render_pipeline_detail.py report/media/pipeline_overview/detail/pipeline_v3_layers.html` (needs Google Chrome, so on the Mac) writes the five PNGs at 2x, light theme |
| `<name>.yaml` | the spec of figure `<name>`: boxes, arrows, colour roles and caption |
| `paper.yaml` | colours shared by the three v3 figures: gold = the open-source starVLA QwenPI model (released weights, frozen); orange = ours, trained; grey = data and our steps without weights; red = losses; purple dashed = planned |
| `<name>.png` | the rendered figure, at 3x its size in points |
| `input.png`, `wrist.png`, `plan.png`, `outcome.png` | frames of a real LIBERO-PRO rollout (agent view, and the wrist view at the same step) (`runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0`, success): position perturbation, salad dressing, the grid plan head with plan hold, and the s10 action head |
| `counterfactual.png` | first frame of a real demonstration (salad dressing, demo 0). Dark is the demonstration's own way to the grasp; orange is the same way rewritten to the ketchup by the 15 Nov rule |
| `scene_train.png`, `scene_swap.png`, `scene_task.png` | first frames with the object the instruction names outlined in orange. `scene_train`: the demonstration above (LIBERO-Object). `scene_swap`: the rollout above (position perturbation). `scene_task`: the task perturbation in the same salad-dressing scene, where the instruction names the tomato sauce (`predicted_g1_hold_s10/libero_object_task/task2_ep0`, success) |

Remake the crops on the server (starvla env, no GPU). This rewrites the first five crops byte for byte:

```bash
W=/mnt/nas/xinyizheng/affordance_vla; R=$W/runs/libero_pro/predicted_g1_hold_s10
D=$W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz
python scripts/report/pipeline_figure_crops.py --rollout $R/libero_object_swap/task2_ep0.npz --demo $D \
    --data $W/runs/plan_following/dataset10 --other ketchup_1 --out report/media/pipeline_overview \
    --scene scene_train=$D --scene scene_swap=$R/libero_object_swap/task2_ep0.npz \
    --scene scene_task=$R/libero_object_task/task2_ep0.npz
```

Render a figure on the server, from the repository root, with a copy of the cvpr-figure skill (it lives in `~/.claude/skills/cvpr-figure` on the Mac; copy its `scripts/` folder over). `render_figure_spec.py` writes the SVG with the skill, sets the text in Liberation Sans (Helvetica's metrics, which the engine sizes boxes for) and makes the PDF with LibreOffice and the PNG from it:

```bash
python scripts/report/render_figure_spec.py <skill dir> report/media/pipeline_overview/v3_model.yaml --dpi 288
python <skill dir>/scripts/validate.py report/media/pipeline_overview/v3_model.yaml --svg report/media/pipeline_overview/v3_model.svg
```

Only the spec and the PNG are kept in git. Audit of the v3 figures (23 Nov): 0 errors. The warnings left are deliberate: the boxes carry several lines of numbers (the detail asked for), the figures are taller than a paper page allows (they are for the notebook and the docs), and the simulator crops are 256 px.

Add `-f svg,pptx` to get a PowerPoint file to edit by hand. Once that file has been edited, it becomes the source
of the figure.
