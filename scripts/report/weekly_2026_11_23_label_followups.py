"""Section 4c of the week-3 review (2026-11-23): step 1 (3), the two open items of the interaction labels (AFFO-18).

Imported by build_weekly_2026_11_23.py, which owns the notebook and provides `ROOT`, `Image` and `display`.
Every number is copied from docs/vla/2026-11-26_VLA_branch_第1步_旋钮标签与把手对齐.md, which cites the run folders on
the NAS (runs/rc365_labels/check_v2). Figures: report/media/weekly_2026_11_23/label_followups/, made by
scripts/data/rc365_joint_figures.py, rc365_joint_charts.py and rc365_mug_figure.py.
"""

MEDIA = "report/media/weekly_2026_11_23/label_followups"
FRAMES = ("OpenDrawer_ep0", "OpenCabinet_ep0", "TurnOnSinkFaucet_ep0", "TurnOffStove_ep8", "TurnOffStove_ep40")


def label_followup_cells(md, code) -> None:
    """Figures first, then the conclusions (at most ten lines), then the commands."""
    md(r"""
## 4c · Step 1 (3): the knob's next motion from the joint, and `mug_8`'s handle (AFFO-18)

Two items left open by section 4b, CPU only. *Frames:* the contact point (red), where the part moves at the contact by the joint (green), where the gripper moved by the old label (orange), the joint axis (cyan). The stove has two designs: knobs on the front panel (`stove`, horizontal axis) and on the top (`stovetop`, vertical axis).
""")
    code(f'for name in {FRAMES!r}:\n    display(Image(filename=str(ROOT / "{MEDIA}" / f"{{name}}.png"), width=1000))')
    md(r"""
**Why the knob's contact position spread 33 cm:** the body's origin is 49 cm from the knob (the joint anchor is 2 cm from the contact), and the instruction picks the burner, so the origin differs per demonstration. With the anchor as the origin the spread is 2.0 cm.
""")
    code(f'display(Image(filename=str(ROOT / "{MEDIA}/knob_frame_before_after.png"), width=1000))')
    md(r"""
**Old label (gripper displacement) against the new one (joint), same demonstrations:**
""")
    code(f'display(Image(filename=str(ROOT / "{MEDIA}/part_label_comparison.png"), width=1000))')
    md(r"""
**`mug_8`: handle length aligned before the transfer.** Red: its own demonstrated grasps; colour: the best fifth of the carried map.
""")
    code(f'display(Image(filename=str(ROOT / "{MEDIA}/mug_8_align_before_after.png"), width=1000))')
    md(r"""
| Mug, 12 pretrain → 5 target instances (503 grasps) | Landing (pass ≥ 0.8) | Closing-axis median (pass < 20°) | `mug_8` |
| --- | --- | --- | --- |
| (1) ring fill (section 4b) | 0.89 ✓ | 12.9° ✓ | 0.59 / 22.8° ✗ |
| **+ handle length aligned** | **0.94 ✓** | **12.0° ✓** | **0.84 / 10.9° ✓** |
| + handle registered (ICP) | 0.94 ✓ | 13.3° ✓ | 0.87 / 20.5° ✗ |

- **Knob: the translation direction is unusable, turn axis plus sign is usable.** The contact is 1.9 cm from the axis, so the direction across kitchens agrees only 0.15 (old label 0.01). Per stove design the turn axis and sign agree 0.85–0.93 over the first second and 1.00 over the whole demonstration (259 of 259 turn the joint down).
- **Drawer and door handles:** the joint label agrees across kitchens 1.00 and 0.98 (old 0.73, 0.91) and points the same way as the gripper (median 11–12° apart).
- **Faucet handle: usable** (0.91 against 0.49), but 67° from where the gripper moves (it also slides along the axis). Which one the action head should follow is open.
- **`mug_8` passes with the handle length aligned and the ring fill** (0.84 / 10.9°); the other four held-out mugs stay above the lines. The alignment reads the held-out mug's surface only. Caveat: chosen after seeing `mug_8` fail, tested on the same five mugs.
""")
    md(r"""
```bash
# repository root of the A6000 server, branch agent/vla-engineer/affo-18; W=/mnt/nas/xinyizheng/affordance_vla
source envs/server/robocasa365_env.sh; export PYTHONPATH=$PWD
"$RC365_PYTHON" scripts/data/rc365_joint_check.py --config configs/rc365_labels/joint_check.yaml        # seconds
"$RC365_PYTHON" scripts/data/rc365_mug_align.py --config configs/rc365_labels/mug_align.yaml            # seconds
"$RC365_PYTHON" scripts/data/rc365_joint_figures.py --config configs/rc365_labels/joint_figures.yaml    # frames, ~3 min
source envs/server/libero_env.sh; export PYTHONPATH=$PWD:$PYTHONPATH
"$LIBERO_PYTHON" scripts/data/rc365_joint_charts.py --config configs/rc365_labels/joint_check.yaml --out report/media/weekly_2026_11_23/label_followups
"$LIBERO_PYTHON" scripts/data/rc365_mug_figure.py --config configs/rc365_labels/mug_align.yaml --out report/media/weekly_2026_11_23/label_followups
```
""")
