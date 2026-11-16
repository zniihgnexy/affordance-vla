# Pipeline figure of the weekly review

`pipeline_overview.png` is the figure in the first section of `report/2026-11-16_weekly_review.ipynb`.

| File | What it is |
| --- | --- |
| `pipeline_overview.yaml` | the figure spec (cvpr-figure skill: boxes, arrows, colour roles) |
| `input.png`, `plan.png`, `outcome.png` | frames of a real LIBERO-PRO rollout: position perturbation, salad dressing, the grid plan head with plan hold and the s10 action head (`runs/libero_pro/predicted_g1_hold_s10/libero_object_swap/task2_ep0`, success) |
| `counterfactual.png` | first frame of a real demonstration (salad dressing, demo 0): its own way to the grasp (dark) and the same way rewritten to the ketchup by the 15 Nov rule (orange) |
| `pipeline_overview.png` | the rendered figure, 3x the size in points |

Remake the crops on the server (starvla env, no GPU):

```bash
W=/mnt/nas/xinyizheng/affordance_vla; R=$W/runs/libero_pro/predicted_g1_hold_s10/libero_object_swap
python scripts/remote/pipeline_figure_crops.py --rollout $R/task2_ep0.npz \
    --demo $W/runs/plan_following/demos/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz \
    --data $W/runs/plan_following/dataset10 --other ketchup_1 --out report/media/pipeline_overview
```

Render the figure (any machine with the cvpr-figure skill, cairosvg and a CJK font; done on the Mac). The skill
writes Helvetica, which has no Chinese glyphs, so the font is swapped in the SVG before rasterising:

```bash
python <skill>/scripts/render.py pipeline_overview.yaml -o out/pipeline_overview -f svg
sed -E "s/font-family='[^']*'/font-family='Hiragino Sans GB'/g" out/pipeline_overview.svg > out/zh.svg
python -c "import cairosvg; cairosvg.svg2png(url='out/zh.svg', write_to='pipeline_overview.png', scale=3.0)"
python <skill>/scripts/validate.py pipeline_overview.yaml --svg out/pipeline_overview.svg   # 0 errors, 0 warnings
```
