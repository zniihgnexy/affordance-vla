"""Section 4b of the week-3 review (2026-11-23): step 1 (1), interaction labels from RoboCasa365 demonstrations (AFFO-15).

Imported by build_weekly_2026_11_23.py, which owns the notebook and provides `ROOT`, `Image` and `display`.
Every number is copied from docs/vla/2026-11-26_VLA_branch_第1步_交互map标签.md, which cites the run folders on the
NAS (runs/rc365_labels/check_v1). Figures: report/media/weekly_2026_11_23/interaction_labels/, made by
scripts/data/rc365_label_charts.py and scripts/data/rc365_label_figures.py.
"""

MEDIA = "report/media/weekly_2026_11_23/interaction_labels"
FRAMES = ("CoffeeSetupMug", "OpenDrawer", "OpenCabinet", "TurnOnSinkFaucet", "TurnOffStove", "PickPlaceSinkToCounter")


def interaction_label_cells(md, code) -> None:
    """Figures first, then the conclusions (at most ten lines), then the commands."""
    md(r"""
## 4b · Step 1 (1): interaction labels from RoboCasa365 demonstrations (AFFO-15)

The label of one grasp or touch: where the gripper closes, along which line, and (for parts) where it moves next. Built from replayed human demonstrations, CPU only. *Frames:* the map (mug) or the contact point (red), the closing line (white) and the next motion (green arrow), projected with the replayed depth.
""")
    code(
        f'for name in {FRAMES!r}:\n    display(Image(filename=str(ROOT / "{MEDIA}" / f"{{name}}_ep0.png"), width=1000))'
    )
    md(r"""
**Mug: carrying the map to held-out instances.** Top: the carried map ((1) ring fill) on the five target mugs; bottom: where the demonstrators grasped. Orange sits at the handle of every mug; `mug_8`'s handle reaches further out than any source mug's.
""")
    code(f'display(Image(filename=str(ROOT / "{MEDIA}/maps_3d.png"), width=1000))')
    code(
        f'display(Image(filename=str(ROOT / "{MEDIA}/azimuth.png"), width=700))\n'
        f'display(Image(filename=str(ROOT / "{MEDIA}/transfer.png"), width=1000))'
    )
    md(r"""
**Parts: how alike the label is across kitchens** (share of touches within 20° / 30° of the other kitchens' mean).
""")
    code(f'display(Image(filename=str(ROOT / "{MEDIA}/parts_consistency.png"), width=900))')
    md(r"""
| Mug, 12 pretrain → 5 target instances (503 grasps) | Landing (pass ≥ 0.8) | Closing-axis median (pass < 20°) |
| --- | --- | --- |
| plain (handle frame) | 0.79 ✗ | 12.9° ✓ |
| **(1) ring fill, robot side** | **0.89 ✓** | **12.9° ✓** |
| (2) robot-facing frame | 0.33 ✗ | 14.8° ✓ |
| own demonstrations, halves | 1.00 | 9.8° |

- **Mug: yes.** Per instance 4 of 5 pass with (1); `mug_8` does not (0.59 / 22.8°). Inference: its handle sticks out further than every source handle (grasp radius 1.57 against 1.08–1.48), so nearest-point transfer misses it.
- **Cup and bowl: not testable.** Only 0 / 1 pretrain grasps (cup / bowl) and 4 / 2 in target; the three pick-and-place tasks mostly move food, pans and plates.
- **The step-0 problem is gone here.** Grasp azimuth concentration is 0.97 / 1.00 in the object frame but 0.47 / 0.33 in the robot-facing frame: demonstrators always take the handle, the robot's side varies. (1) is chosen (it passes and beats (2) by 0.56), but plain misses by 0.01 only, so the treatment matters little on this data (inference).
- **Handles: same format works.** Closing line across kitchens: drawer 0.85, cabinet 0.94, faucet 0.75 within 20°; motion after the touch: 0.73, 0.91, 0.49. In the part frame these beat the world frame by a wide margin (cabinet closing 0.94 vs 0.16).
- **Stove knob: not yet.** Contact scatter in the part frame is 33 cm and the next motion is defined for only 26% of touches (the gripper hardly translates); cause not checked.
""")
    md(r"""
```bash
# repository root of the A6000 server, branch agent/vla-engineer/affo-15; W=/mnt/nas/xinyizheng/affordance_vla
source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
"$RC365_PYTHON" scripts/data/rc365_extract.py --config configs/rc365_labels/extract.yaml             # replay, CPU, ~1 h
"$RC365_PYTHON" scripts/data/rc365_label_check.py --config configs/rc365_labels/check.yaml --stage all  # ~30 s
"$RC365_PYTHON" scripts/data/rc365_label_figures.py --config configs/rc365_labels/figures.yaml          # frames
source envs/server/libero_env.sh; "$LIBERO_PYTHON" scripts/data/rc365_label_charts.py \
    --config configs/rc365_labels/check.yaml --out report/media/weekly_2026_11_23/interaction_labels
```
""")
