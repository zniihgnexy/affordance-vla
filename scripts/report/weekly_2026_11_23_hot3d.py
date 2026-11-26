"""Section 5 of the week-3 review: can a human-hand dataset (HOT3D) give our interaction map, and what should the
data-collection team record. Facts checked on 25 Nov 2026 against the HOT3D paper (arXiv 2411.19167), project page,
GitHub toolkit, the Hugging Face dataset and the licence page; numbers are copied from
runs/hot3d_demo/mug_white_v2/summary.json (the record is docs/vla/2026-11-25_VLA_branch_HOT3D评估.md, AFFO-11).

Imported by build_weekly_2026_11_23.py, which owns the notebook. The figures are made by
scripts/data/hot3d_contact_demo.py and live in report/media/weekly_2026_11_23/hot3d/ (tracked).
"""


def hot3d_cells(md, code) -> None:
    """Section 5: figures first, then at most ten lines of conclusions, then the commands."""
    md(r"""
## 5 · HOT3D: can human-hand data give our interaction map?

**Real data, ground-truth annotations** (motion capture), no model involved: [HOT3D](https://facebookresearch.github.io/hot3d/) (Meta, CVPR 2025) = egocentric recordings of 19 people handling 33 rigid objects, with per-frame 6-DoF object poses, hand meshes and scanned object meshes. Below: two 5-second clips of one person holding the white mug (`P0015_dbe00981`).
""")
    code('from pathlib import Path\n'
         'from IPython.display import Image\n'
         'ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())\n'
         'HOT3D = ROOT / "report" / "media" / "weekly_2026_11_23" / "hot3d"\n'
         'Image(filename=str(HOT3D / "overlay.png"), width=620)')
    md(r"""
*Top of the figure above: the annotated hand meshes (thumb in orange) and the mug mesh drawn on the camera frame (Aria RGB, fisheye). Below: the same 10 seconds turned into labels.*
""")
    code('Image(filename=str(HOT3D / "contact.png"), width=1000)')
    md(r"""
*Top row: how often each point of the mug is within 5 mm of a hand point (300 frames; grey = never). Cyan lines: the thumb-to-fingers closing axes of the 5 held grasps found. Bottom row: the same grasps turned into a map of `interaction_labels.py`, the format of our demonstration maps.*

- **Yes, it fills our map format.** Surface points (scanned mesh), grasps in the object frame (hand mesh + object pose) and the closing axis (thumb pad → finger pads) all come out; the result loads with `load_maps`.
- **Three things are missing:** (1) a label for "this is a held grasp" (we guess it from thumb–finger opposition, ≥ 5 frames); (2) the step from a human hand to a parallel gripper (the 9 cm mug body is at the edge of our `MAX_REACH_M`); (3) enough shapes per category: 2 mugs, 1 bowl, 0 cups, rigid objects only (no drawers, doors, knobs).
- **Numbers:** 145 of 300 frames have a hand within 5 mm of the mug (143 / 147 at 3 / 10 mm); 34% of its surface is touched at 5 mm (20% / 65%); 5 held grasps, 75–93 mm between the pads; the closing axis of one continuous hold drifts by 48° and 71° (the hand re-grips), so a hold is split where it turns 20°.
- **Not in HOT3D:** contact, action, task or language labels; depth; any robot. Hand annotations are non-commercial (CC BY-NC-SA); the UmeTrack hand mesh needs no MANO licence.
- **For the data-collection team (top 3 of 10 in the record):** per-frame object pose + mesh, per-frame hand mesh in the same frame, and quality masks; then hold start/end, ≥ 8–10 shapes per category (mug, bowl, cup, bottle, can), and articulated handles and knobs.
""")
    md(r"""
**Run it** (server, CPU, ~30 s; needs the `hot3d` env from `envs/server/setup_hot3d_env.sh`; not tried on the Mac):

```bash
bash scripts/data/fetch_hot3d_demo.sh          # 2 clips + 3 meshes, 0.6 GB from Hugging Face, no sign-up
source envs/server/hot3d_env.sh
PYTHONPATH=. python scripts/data/hot3d_contact_demo.py --out $W/runs/hot3d_demo/mug_white_v2   # add --threshold-mm 10 to see the threshold matter
```
""")
