"""Section of the week-3 review (23-29 Nov 2026): can the map of two mugs be carried to a third one?

Pipeline v4, step 0 (1): leave one mug mesh out, carry the map of the other two to it by nearest point in a
category frame and by NDF, and check it against the held-out mug's own demonstrated grasps. Numbers: the run
`runs/mug_transfer/mug_mesh` (results.json, results.md), recorded in
docs/vla/2026-11-25_VLA_branch_第0步_mug搬运检查.md. Figures: report/media/weekly_2026_11_23/mug_transfer/.

`mug_transfer_cells(md, code)` adds the cells to the notebook that owns it; the notebook defines `ROOT` (the
repository root) and imports `Image` and `display` from IPython.display, as in report/2026-11-16_weekly_review.ipynb.
"""

MUGS = ("porcelain_mug", "white_yellow_mug", "red_coffee_mug")


def mug_transfer_cells(md, code) -> None:
    """Figures first, then the conclusions (at most ten lines), then the commands."""
    md(r"""
## 3 · Step 0 ①: carrying the mug map to a mug nobody demonstrated (AFFO-9)

The check: build the map from two mugs' demonstrations, carry it to the third mug, and compare it with where the demonstrators grasped that third mug. Each figure: *left* nearest-point transfer in the category frame, *middle* NDF transfer, *right* the mug's own demonstrations; orange = the held-out mug's own grasps, projected onto one demonstration frame.
""")
    code('MUG_FIGURES = ROOT / "report" / "media" / "weekly_2026_11_23" / "mug_transfer"\n'
         f'for name in {MUGS!r}:\n'
         '    display(Image(filename=str(MUG_FIGURES / f"{name}.png"), width=1000))')
    md(r"""
| Held-out mug (own grasps) | Method (source grasps) | Landing rate (pass ≥ 0.8) | Median axis error (pass < 20°) |
| --- | --- | --- | --- |
| white `porcelain_mug` (269) | nearest point (607) | 0.00 ✗ | 62.7° ✗ |
| | NDF (200) | 0.00 ✗ | 45.1° ✗ |
| yellow-white `white_yellow_mug` (334) | nearest point (542) | 0.02 ✗ | 15.2° ✓ |
| | NDF (200) | 0.70 ✗ | 18.6° ✓ |
| red `red_coffee_mug` (273) | nearest point (603) | 0.00 ✗ | 9.0° ✓ |
| | NDF (200) | 1.00 ✓ | 9.6° ✓ |

- **1 of 6 pairs passes** (red mug held out, NDF); nearest point passes 0 of 3, NDF 1 of 3. NDF's landing rate is never below nearest point's.
- **The map itself is stable:** a mug's own demonstrations split in half agree at 0.97–1.00.
- **The data, not the transfer code, is the main limit (inference):** LIBERO fixes each mug's start orientation, so demonstrators always grasp from one side. The white mug's handle points the other way from the other two; its grasps lie ~30° from the handle in the category frame, the others' ~170°. Turning the category frame to the scene instead gives 1.00 / 0.77 / 1.00.
- **NDF was one seed**, 100 sampled grasps per source mug, no tuning: a single red-mug win is not a general result.
- **Consequence for the plan:** with fixed orientations, "unseen shape" and "unseen orientation" cannot be separated on these mugs.
""")
    md(r"""
```bash
# repository root of the A6000 server; W=/mnt/nas/xinyizheng/affordance_vla
bash scripts/data/run_mug_transfer_check.sh collect                       # CPU
bash scripts/server/submit_queue_job.sh 164 mug_ndf 8000 1 "cards=<free cards>" -- \
    bash $PWD/scripts/data/run_mug_transfer_check.sh ndf                  # GPU through the queue, ~45 min
bash scripts/data/run_mug_transfer_check.sh evaluate                      # CPU: table and figures
cat $W/runs/mug_transfer/mug_mesh/results.md
```
""")
