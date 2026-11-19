"""Build report/2026-11-20_grasp_orientation.ipynb, E0 of Pipeline v3 (cells only; execution happens separately).

Usage on the server (starvla env, from report/):
    python ../scripts/report/build_grasp_orientation_report.py 2026-11-20_grasp_orientation.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-20_grasp_orientation.ipynb
Needs runs/plan_following/demos (the replayed demonstrations) and runs/libero_pro/full_{g1_hold,oracle}_s9_seed{7,8,9}
(the papers' protocol, see report/2026-11-14_libero_pro_full_protocol.ipynb). Nothing is run on a GPU.
The closing text is in conclusions_grasp_orientation.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_grasp_orientation.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# E0: do failed grasps hold the gripper at the wrong angle?
**VLA direction** · Pipeline v3, step E0 · LIBERO-PRO, Object suite · 20 Nov 2026 · from existing logs, nothing new was run

- **The question:** Pipeline v3 adds the gripper orientation to the plan, read from a directional interaction
  affordance map. That is only worth it on this suite if failed grasps close the gripper at a different angle
  from the held ones and from the demonstrations of the same object.
- **Pass criterion (agreed 19 Nov):** the median angle between a failed grasp and the nearest demonstrated grasp of
  the same object is at least 10 degrees larger than for held grasps. If not, the orientation branch pauses on
  LIBERO-Object and is tested on a suite with handled objects (LIBERO-10) instead.
- **Real:** the replayed demonstrations of the ten LIBERO-Object tasks and the rollouts of the papers' protocol
  (50 starts per task, seeds 7, 8, 9) with the ten-task action head of 13 Nov (`s9_align_release10`).
- **ORACLE:** the plans of the `oracle` runs. Angles are in world coordinates, because the logs do not record
  object orientations yet; the objects stand upright.

```text
(1) how the demonstrations grasp each object
        ↓
(2) every rollout's first close command near the named object: held or not, and its angle to the demonstrations
        ↓
(3) what differs instead: how far from the object the gripper closes
        ↓
(4) the images at a few failed closes
```
""")

code("""
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from report import grasp_orientation_helpers as G

set_matplotlib_formats("jpeg")
RUNS = ROOT / "runs"
LIBERO_PRO = RUNS / "libero_pro"
SUITES = ["libero_object_swap", "libero_object_task"]
CONFIGS = {"map plan kept (g1 + s9)": "full_g1_hold_s9_seed{}", "oracle plan (s9)": "full_oracle_s9_seed{}"}
SEEDS = (7, 8, 9)
""")

md("""
## 1. How the demonstrations grasp

The first held grasp of its own target in every replayed demonstration. *Tilt* is how far the fingers lean from
straight down. The *closing yaw spread* is how much the direction in which the fingers close varies about the
vertical between demonstrations of the same object.
""")

code("""
demos = G.demo_grasps(sorted((RUNS / "plan_following" / "demos").glob("*.npz")))
reference = G.demo_rotations(demos)
print(f"{len(demos)} demonstrations with a held grasp of their target")
display(G.demo_spread(demos))
""")

md("""
## 2. Failed and held grasps against the demonstrations

One row per rollout: its first close command near the named object. It is *held* when the grasp registers within
0.75 s. The angle is to the nearest demonstrated grasp of the same object, counting a turn of 180 degrees about the
approach axis as the same grasp. All three seeds are pooled per configuration.
""")

code("""
attempts = {}
for name, pattern in CONFIGS.items():
    attempts[name] = pd.concat([G.rollout_attempts(LIBERO_PRO / pattern.format(seed), SUITES, reference)
                                for seed in SEEDS], ignore_index=True)
    gap, passed = G.verdict(attempts[name])
    print(f"{name}: failed minus held, median angle {gap:.1f} deg -> {'PASS' if passed else 'does not pass'}")
    display(G.group_table(attempts[name]))
""")

code("""
G.angle_figure(attempts["map plan kept (g1 + s9)"]);
""")

md("""
Per object (median angle in degrees, number of rollouts in brackets), map plans kept:
""")

code("""
display(G.per_object_table(attempts["map plan kept (g1 + s9)"]))
""")

md("""
## 3. What differs instead: where the gripper closes

The distance from the gripper to the named object's centre at the close command. It includes the object's own size,
so it is compared between outcomes of the same objects, not read as an error.
""")

code("""
for name, table in attempts.items():
    print(name)
    display(G.distance_table(table))
G.distance_figure(attempts["map plan kept (g1 + s9)"]);
""")

md("""
## 4. The images at some failed closes

Agent view and wrist view at the close command: among the failed attempts whose logs kept their images, the three
that closed furthest from the object and the three with the largest angle to the demonstrations (map plans kept).
""")

code("""
table = attempts["map plan kept (g1 + s9)"]
failed = G.with_images(table[table["group"] == "not held"], LIBERO_PRO)
print(f"{len(failed)} failed first attempts kept their images")
picked = pd.concat([failed.nlargest(3, "distance_m"), failed.nlargest(3, "angle_to_demo_deg")])
G.grasp_frames(picked, LIBERO_PRO);
""")

md(CLOSING)

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(notebook, OUT)
print("wrote", OUT)
