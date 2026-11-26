"""Section 1 of the week-3 review (2026-11-23): the Pipeline v4 figure and what changed from v3.

Imported by build_weekly_2026_11_23.py, which owns the notebook and provides `ROOT`, `Image` and `display`.
Facts: docs/vla/2026-11-25_VLA_branch_Pipeline_v4.md, sections 1-3 (approved 25 Nov 2026). The figure is
report/media/pipeline_overview/v4_model.yaml (spec) and v4_model.png (render); how to redraw it is in
report/media/pipeline_overview/README.md.
"""


def pipeline_cells(md, code) -> None:
    """Figure first, then the v3 to v4 table."""
    md(r"""
## 1 · Pipeline v4

Approved by the project lead on 25 Nov. One sentence: **understanding names the next object to touch, the interaction map says where to touch it and in which direction, and the action is trained to obey that plan.** The thesis is unchanged (affordance helps only if the action is trained to obey it); v4 adds who decides which object the plan points at. Source: `docs/vla/2026-11-25_VLA_branch_Pipeline_v4.md`.
""")
    code('Image(filename=str(ROOT / "report" / "media" / "pipeline_overview" / "v4_model.png"), width=1100)')
    md(r"""
*Gold = open-source starVLA; orange = our parts built and trained for LIBERO in v3; grey = ours, no weights; purple dashed = new in v4, **designed, not built**. The frame is a recorded RoboCasa365 demonstration. Nothing in this figure runs on RoboCasa365 yet: the orange parts exist for LIBERO, and the backbone (QwenPI or QwenOFT) is an open decision (section 0).*

**What changed from v3 to v4** (Pipeline v4, section 2; items marked *design* have no experiment behind them):

| | v3 | v4 |
| --- | --- | --- |
| Environment | LIBERO (tabletop), zero-shot on LIBERO-PRO | RoboCasa365 kitchens; LIBERO and LIBERO-PRO kept to connect to earlier results |
| Tasks | pick and place | atomic tasks (pick and place, open drawer, open door, knob), then composite tasks of several steps |
| Interaction map | grasp contact and closing direction | also drawer and door handles and knobs, plus a direction to move after touching *(design)* |
| Understanding | none; the semantic gate picks an object from one frame and the instruction | a new part: reads the recent frames and the instruction, answers which step we are in and what the next interaction target is |
| Who picks the target | the instruction | the understanding part *(design)* |
| Unseen instances of a category | 3 mugs, plus scaled mugs | instances RoboCasa holds out: 27 mugs, 22 bowls, 22 cups |
| Carrying a map to a new instance | nearest point | nearest point and NDF, compared (step 0 ①, section 3) |
| Training scale | frozen-VLM diagnostics, 1 training seed | main results with full fine-tuning, at least 3 training seeds *(plan)*; light training only for quick comparisons |
| Compute | planned for one 8×A6000 server | not a constraint for now (project lead, 25 Nov); full fine-tuning on one 48 GB card is still unsolved (section 4) |

Unchanged: starVLA as backbone family, the model works on 2D images (map plus depth), simulator ground truth is used for labels and evaluation only, never as a model input at test time.
""")
