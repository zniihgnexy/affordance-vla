"""Build report/experiments/2026-11-15_task_perturbation_failures.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/experiments/):
    python ../../scripts/report/build_task_perturbation_report.py 2026-11-15_task_perturbation_failures.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-15_task_perturbation_failures.ipynb
Needs runs/libero_pro/full_oracle_s9_seed{7,8,9} (the papers' protocol, see
report/experiments/2026-11-14_libero_pro_full_protocol.ipynb) and runs/plan_following/dataset10. Nothing is run on a GPU.
The closing text is in conclusions_task_perturbation.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_task_perturbation.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Why three tasks of the task perturbation fail even with the oracle plan
**VLA direction** · LIBERO-PRO, Object suite · action head trained on ten tasks with the labels of 13 Nov (`s9_align_release10`) · 15 Nov 2026 · a diagnosis from existing rollouts, nothing new was run

- **The question:** under the papers' protocol the action head solves 0.92 of the position perturbation with the oracle plan but only 0.75 of the task perturbation, and three tasks of the latter almost always fail: ketchup (31 of 150), milk (24) and orange juice (54). The plan is right, so the action head does something wrong. What?
- **Real:** the 1500 rollouts per perturbation of the papers' protocol (seeds 7, 8, 9) and their logs; the demonstrations of the ten LIBERO-Object tasks.
- **ORACLE:** the plan given in all these rollouts. The training labels examined in sections 3 and 4 are computed here with the same code that made them for training.

```text
(1) per task: in whose scene it is set, how long the gripper takes, how often it runs out of time
        ↓
(2) one failing rollout next to the same object at the same place in the other perturbation
        ↓
(3) the training labels for "another object named" in these scenes: their path
        ↓
(4) where the last part of the labelled approach comes from
```
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.training.counterfactual import Retarget
from affordance_vla.training.plan_dataset import load_tracks
from affordance_vla.plan.plan_input import offsets_from_json
from affordance_vla.sim.rollout_log import load_rollout_log
from report import libero_pro_helpers as L
from report import plan_predictor_helpers as P
from report import task_perturbation_helpers as T

set_matplotlib_formats("jpeg")
plt.rcParams.update({"figure.dpi": 90})
pd.set_option("display.max_colwidth", 60)
PRO = ROOT / "runs" / "libero_pro"
DATA = ROOT / "runs" / "plan_following" / "dataset10"
offsets = offsets_from_json((DATA / "offsets.json").read_text())
SEEDS = (7, 8, 9)
TASK, SWAP = "libero_object_task", "libero_object_swap"


def paths(suite):
    return [path for seed in SEEDS for path in L.episode_paths(PRO / f"full_oracle_s9_seed{seed}", suite)]


# the object that LIBERO-Object's own task k asks for = the target of task k in the position perturbation
scene_targets = {L.task_of(log): log.target for log in L.load_suite(PRO / "full_oracle_s9_seed7", SWAP)}
print(scene_targets)
""")

md("""
## 1 · Per task: whose scene, how long, how often out of time
**This step:** LIBERO-PRO builds each task of the task perturbation on the scene of LIBERO-Object's task with the same index and asks for another object of that scene. For every task: the object asked for, whose scene it is, how far the gripper starts from the grasp point of the object asked for, how far that object stands from the scene's original target, the solved rollouts, when the gripper first comes within 2 cm of the grasp point, how many rollouts never get there or run to the step limit (280), and how many of the first 150 steps the gripper spends almost still (< 2 mm per step) while still more than 5 cm away. The same for the position perturbation below, for reference.

**Status:** real rollouts with the ORACLE plan, three seeds (150 per task).
""")

code("""
task_rows = T.task_rows(paths(TASK), scene_targets, offsets)
display(T.task_table(task_rows))
print("position perturbation, for reference:")
display(T.task_table(T.task_rows(paths(SWAP), scene_targets, offsets)))
""")

md("""
## 2 · One failing rollout next to a solved one with the same object at the same place
**This step:** the orange juice stands at the same place in task 6 of the task perturbation and in task 9 of the position perturbation, and the plan given is the same. Its distance to the grasp point and its height above it, step by step, for one rollout of each (seed 7, start 0), with a ketchup and a milk rollout of the task perturbation; then camera frames of the same rollouts.

**Status:** real rollouts with the ORACLE plan.
""")

code("""
CASES = {"task pert., orange juice (scene of butter)": (TASK, "task6_ep0"),
         "position pert., orange juice, same place": (SWAP, "task9_ep0"),
         "task pert., ketchup (scene of bbq sauce)": (TASK, "task3_ep0"),
         "task pert., milk (scene of ketchup)": (TASK, "task4_ep0")}
logs = {name: load_rollout_log(PRO / "full_oracle_s9_seed7" / suite / f"{stem}.npz")
        for name, (suite, stem) in CASES.items()}
T.course_figure(logs, offsets)
plt.show()
T.frames_figure(logs)
plt.show()
""")

md("""
## 3 · The training labels for "another object named" in these scenes
**This step:** for a frame of a demonstration, a training sample that names another object of the scene gets an action label made by moving the demonstrated path (from the gripper to the original target's grasp point) to the other object's grasp point. Here that is done at the first frame of every demonstration of each scene, for the object the task perturbation asks for there, under the two rules used so far: the rule of 11 Nov (the shift blended in until the last 8 cm) and the rule of 13 Nov used by this action head (a shift longer than the path before the last 18 cm made first). For each: the labelled path's length over the straight distance (`detour`), how far it first moves away from the object asked for (`away`), and its lowest height above that grasp point while it is still more than 4 cm aside (`low_aside`). Medians over the demonstrations of the scene.

**Status:** the demonstrations are real; the moved paths are what the training labels asked for.
""")

code("""
tracks = load_tracks(DATA / "tracks.npz")
pairs = {row.task: (row.asked, row.scene_of) for row in task_rows}
RULES = {"rule of 11 Nov": Retarget(), "rule of 13 Nov": Retarget(rigid_tail=0.18, align_first=True)}
display(T.label_table(tracks, pairs, offsets, RULES))
""")

md("""
**The orange juice in the scene of the butter, drawn.** One demonstration of the butter task from its first frame, and its path moved to the orange juice under the two rules; dots are the next 8 steps the labels ask for.
""")

code("""
track = next(t for t in tracks if t.target == "butter_1" and "orange_juice_1" in t.names)
path = T.approach_path(track)
juice = track.object_pos[0, track.names.index("orange_juice_1")] + offsets.grasp["orange_juice_1"]
P.retarget_figure(path, juice, RULES)
plt.show()
""")

md("""
## 4 · Where the last part of the labelled approach comes from
**This step:** the last part of a moved path keeps the shape of the demonstrated approach to the original target. For each object, the shape of the last 18 cm of its own demonstrated approaches: how high above and how far aside of the grasp point that part starts, and how far it travels sideways while lower than 4 cm.

**Status:** real demonstrations.
""")

code("""
display(T.tail_table(tracks))
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
