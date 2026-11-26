# Figures of `RESEARCH_PROPOSAL.md`

| File | What it shows |
| --- | --- |
| `map_extraction.png` / `.yaml` | figure 2: from demonstrations to a training label, in five steps, with real pictures |
| `map_in_vla.png` / `.yaml` | figure 3: how the interaction map becomes a plan at test time, and four ways for the plan to reach the action head (A built on LIBERO, see `affordance_vla/heads/plan_policy.py`; B to D are ideas, not built) |
| other `*.png` | the tiles inside the two figures, cut from tracked figures of earlier tasks by `scripts/report/proposal_figure_crops.py` (AFFO-9 mug transfer, AFFO-10 RoboCasa365, AFFO-11 HOT3D, week-2 LIBERO-PRO frames) |

Figure 1 of the proposal is `../pipeline_overview/v4_model.png`. Colours follow `../pipeline_overview/paper.yaml`.

Remake (Mac; the figure engine is the cvpr-figure skill, rendered through its SVG step and Google Chrome as described
in `../pipeline_overview/README.md`, because the Mac has no LibreOffice):

```bash
python scripts/report/proposal_figure_crops.py
python <skill dir>/scripts/render.py report/media/research_proposal/map_extraction.yaml -o map_extraction -f svg
python <skill dir>/scripts/render.py report/media/research_proposal/map_in_vla.yaml -o map_in_vla -f svg
# then a headless-Chrome screenshot of each SVG at 3x, fonts set to Arial
```
