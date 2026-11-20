"""Build report/experiments/2026-11-11_plan_following_experiment.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/experiments/):
    python ../../scripts/report/build_plan_following_report.py 2026-11-11_plan_following_experiment.ipynb
    jupyter nbconvert --to notebook --execute --inplace 2026-11-11_plan_following_experiment.ipynb
Needs runs/plan_following/{dataset,<runs>} (scripts/vla/run_plan_following_gpu.sh and the CPU steps in
its header) and the rollouts of the released checkpoint in runs/vla_pipeline_demo/rollouts.
The text of the conclusions is in conclusions_plan_following.md next to this script, so that the numbers
quoted there can be checked against the executed tables in one place.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CONCLUSIONS = Path(__file__).with_name("conclusions_plan_following.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Plan-following experiment: does the released action head follow a plan it is given?
**VLA direction** · LIBERO-Object · released starVLA checkpoint `Qwen3-VL-PI-LIBERO-4in1` with a small trained add-on · 11 Nov 2026 · a diagnostic before any training of the full method

- **The question:** if the action head is told where the next grasp or release should happen, does it go there, also when that place is not where the scene usually has it? This is the half of the method that turns a plan into actions. If it fails with a perfect plan, predicting plans is pointless.
- **Real:** the demonstrations, the released VLA, the training of the add-on, the open-loop numbers and the closed-loop rollouts in the simulator.
- **ORACLE:** the plan itself. In training it comes from the demonstration; in the rollouts it comes from simulator ground truth (object position plus the average grasp point of the demonstrations). No model predicts it.
- **Not the paper's configuration:** the VLM is frozen and is the 4B model, and only a plan encoder, low-rank updates of the action head and its output layer are trained (about 31 M of 7.8 B parameters). The method itself trains the whole VLA.
- **Result in short:** given only as an extra input, the plan is ignored. Once training also shows the same frame with the plan moved elsewhere, the gripper goes to the planned grasp point in 18 of 18 rollouts with a moved target and in 27–30 of 30 that name another object, and the named object ends in the basket in 19–20 of 30 instead of 6. Relabelled actions do this as well as the consistency loss. The conclusions are in the last section.

```text
(1) demonstrations → recorded simulator states → per-step logs                   real (state replay, no dynamics)
        ↓
(2) plan input: where the next gripper event happens, relative to the gripper    ORACLE (demonstration / simulator)
        ↓
(3) training samples: the demonstration, and the same frame with a moved plan    real (built from demonstrations)
        ↓
(4) training on the frozen released checkpoint: plan encoder + low-rank updates  real (GPU queue, one shared card)
        ↓
(5) open loop on held-out demonstrations: does the chunk go where the plan says  real
        ↓
(6) closed loop: the 21 rollouts of the 10 Nov pipeline demo, now with a plan    real (plan: ORACLE)
```

<sub>To re-run on the A6000 server: `scripts/data/fetch_libero_states.py`, `replay_libero_states.sh`, `build_plan_dataset.py` (CPU), then `run_plan_following_gpu.sh` through the GPU queue for each run (see its header), then execute this notebook.</sub>
""")

code("""
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.training.plan_dataset import load_tracks
from affordance_vla.plan.plan_input import offsets_from_json
from report import plan_following_helpers as P
from report import vla_pipeline_demo_helpers as H

set_matplotlib_formats("jpeg")
plt.rcParams.update({"figure.dpi": 90})
pd.set_option("display.max_colwidth", 140)

RUNS = ROOT / "runs" / "plan_following"
MEDIA = ROOT / "report" / "media"  # videos made by this notebook; tracked in git so a pull brings them
DATASET = RUNS / "dataset"                                  # six tasks of LIBERO-Object
BASELINE = ROOT / "runs" / "vla_pipeline_demo" / "rollouts"  # the released policy, 10 Nov 2026
MAIN = "+ moved plans (path loss)"
RELABELLED = "+ moved plans (relabelled actions)"
MORE = "rollouts_oracle_more"                               # the perturbed sets again, from start states 2-5
FOLDERS = {"demonstrations only": RUNS / "s2_none", MAIN: RUNS / "s2_other",
           RELABELLED: RUNS / "s2_actions"}                        # six tasks, every run has the plan input
FIRST = {"one task: demonstrations only": RUNS / "s1_none",
         "one task: + shifted contacts (path loss)": RUNS / "s1_shift",
         "one task: + shifted contacts (relabelled actions)": RUNS / "s1_actions"}  # the first check, salad dressing only
stats = json.loads((DATASET / "stats.json").read_text())
tracks = load_tracks(DATASET / "tracks.npz")
contact = offsets_from_json((DATASET / "offsets.json").read_text())
print(f"{stats['kept']} of {stats['logs']} demonstrations kept, {stats['rows']} frames; "
      f"{1000 * stats['motion_scale_m_per_unit']:.1f} mm of gripper motion per unit of translation action")
""")

md("""
## 1 · Demonstrations from their recorded simulator states
**This step:** every raw LIBERO demonstration stores the simulator state of each step. Loading the states one by one (no physics) and rendering gives the images, the gripper and object positions and the finger contacts of exactly the recorded demonstration. On 10 Nov the labels came from re-executing the actions, which reproduced only 5 of 6 demonstrations.

**Status:** real. The check on one demonstration gave the same grasp and release steps as the action replay (51 and 112), and the gripper position matched the recorded one within 0.5 mm. For five of the six tasks only the states and actions were fetched from the raw files (byte ranges, under 1 % of each file), because whole files would have taken hours on this server's connection.

The table shows, per target object, how much the grasp point varies between demonstrations once it is expressed relative to the object. That spread is the error of the oracle plan in the rollouts, which uses the average.
""")

code("""
display(P.offsets_table(DATASET))
dropped = pd.Series([H.pretty(name.rsplit("_demo", 1)[0]) for name in stats["dropped"]]).value_counts()
print(f"{len(stats['dropped'])} demonstrations dropped (not exactly one grasp followed by one release in the "
      f"contact flags):", ", ".join(f"{count} of '{task}'" for task, count in dropped.items()))
""")

md("""
## 2 · How the plan enters the action head
**This step:** the plan input is ten numbers: the vector from the gripper to the contact point of the next gripper event (in units of 10 cm, and again squashed so that the last centimetres are resolved), the direction the gripper should arrive from, and +1 for a grasp or −1 for a release. A small network turns them into one vector that is added to each of the eight action tokens.

**Status:** real code; the numbers are ORACLE.

**Why by addition:** in this checkpoint the action head has no self-attention. Each action token only reads the VLM's hidden states, so a plan placed as extra tokens beside the head's 32 learned tokens would never be read. For the same reason those 32 tokens do not affect the actions and are left out here. The plan network starts at zero, so before training the policy is the released one: the largest difference in predicted velocity is shown below (bf16 rounding).
""")

code("""
display(P.facts_table(FOLDERS | FIRST))
""")

md("""
## 3 · Training samples: the demonstration, and the same frame with a moved plan
**This step:** each sampled frame gives a demonstration sample: the demonstrated 8-step chunk with the true plan (plus 1 cm of noise on the contact). A frame on the way to a grasp also gives samples whose plan points somewhere else: a contact shifted by up to 18 cm, or the grasp point of another object of the scene together with an instruction that names it. The image stays the same.

**Status:** real. No demonstrated action exists for a moved plan. The target is the demonstrated path with its end moved: the shift is taken up early and the last 8 cm keep their shape. The translation of such a sample is supervised only by the **plan–action consistency loss**: the path asked for by the one-step estimate of the chunk against the moved path. Rotation and gripper follow the demonstration (the gripper closes at the demonstrated distance from the contact).
""")

code("""
track = next(t for t in tracks if t.target == "salad_dressing_1")
step = 12
grasp = track.labels[0].contact
other = "ketchup_1"
contacts = {"contact shifted 9 cm": grasp + np.array([-0.08, 0.05, 0.0]),
            f"grasp of the {H.pretty(other)}": track.object_pos[step, track.names.index(other)] + contact.grasp[other]}
fig = P.moved_paths_figure(track, step, contacts, max_step=stats["motion_scale_m_per_unit"])
plt.show()
""")

md("""
## 4 · Training
**This step:** the VLM is frozen. Trained are the plan network, low-rank updates (rank 16) of the attention and feed-forward layers of all 36 blocks of the action head, and its output layer. Each run goes through the GPU queue on one shared card.

**Status:** real. Every run has the plan input. The three runs on six tasks differ in the moved plans: none; moved plans with the path loss; moved plans supervised by relabelled actions instead (the flow-matching loss on the demonstrated command plus the correction, no path loss). Three earlier runs on one task (shifted contacts only, 1,200 steps) are kept as a first check; they used an earlier form of the target path (the travelled path instead of the commanded one). The right panel is the path loss during training (a Huber distance in cm on noisy one-step estimates); runs without it stay at zero.
""")

code("""
fig = P.training_figure(FOLDERS | FIRST)
plt.show()
""")

md("""
## 5 · Open loop on held-out demonstrations
**This step:** on frames of demonstrations that were not used for training, sample a chunk (4 flow steps) and compare. `demo_action_error`: mean error of the translation actions against the demonstration (action units). `*_path_cm`: mean distance of the chunk's path to the moved path. `*_if_ignored`: the same distance for the demonstrated chunk, i.e. what a head that ignores the plan would score.

**Status:** real; no simulator. Before training the head scores like one that ignores the plan, as it must.
""")

code("""
display(P.open_loop_table(FOLDERS | FIRST))
""")

md("""
## 6 · Closed loop: the 21 rollouts of the pipeline demo, now with the oracle plan
**This step:** the same starts and perturbations as on 10 Nov: 5 standard starts; 10 rollouts whose instruction names another object of the scene; 6 rollouts with the salad dressing moved by 4, 9 or 16 cm. At every query (every 8 steps) the simulator side sends the oracle plan: the grasp point of the named object while it is not held, the release point over the basket while it is, nothing once it lies in the basket.

**Status:** real rollouts; the plan is ORACLE. A rollout counts when the object named by the instruction is the first one grasped and ends in the basket. The first column is the released policy without any plan (10 Nov).
""")

code("""
runs = {"released policy, no plan": P.load_run(BASELINE)}
runs |= {name: P.load_run(folder / "rollouts_oracle") for name, folder in FOLDERS.items()}
with_empty_plan = runs | {f"{MAIN}, empty plan in the rollout": P.load_run(FOLDERS[MAIN] / "rollouts_zero")}
display(P.success_table(with_empty_plan))
""")

md("""
**More start states.** Six and ten rollouts per set are few, so the two perturbed sets were repeated from four further start states (2–5) for the three trained policies: 12 more rollouts with the target moved and 20 more with another object named. The released policy without a plan was not run from these starts; "demonstrations only" stands in for it, since it scores like the released policy on the first 21. First table: the further starts alone. Second table: all starts together. Third table: how easily chance alone gives a difference of that size between two policies (Fisher's exact test). The test takes the rollouts as independent although the policies share start states, and the six comparisons are not corrected, so the p-values are a rough guide.
""")

code("""
more = {name: P.load_run(folder / MORE) for name, folder in FOLDERS.items()}
both = {name: P.merge_runs(runs[name], more[name]) for name in FOLDERS}
display(P.success_table(more).drop(index="standard"))
display(P.success_table(both))
display(P.chance_table(both, [("demonstrations only", MAIN), ("demonstrations only", RELABELLED), (MAIN, RELABELLED)]))
""")

md("""
**Where the rollouts stop.** Following the plan and finishing the task are different things, so each rollout of the three trained policies (all starts) is split into stages: did the gripper come within 2 cm of the planned grasp point (checked at every step against the contact of the latest plan), was the object knocked over before it was held (it sank more than 3 cm below its start height), was it held, was it in the basket at the end.
""")

code("""
GROUPS = {name: (folder / "rollouts_oracle", folder / MORE) for name, folder in FOLDERS.items()}
for name in ("target moved 4–16 cm", "another object named"):
    print(name)
    display(P.stage_table(GROUPS, name))
""")

md("""
**How jerky the commands are.** Mean change of the translation command from one step to the next within a chunk, over the chunks requested before the gripper first touches an object (action units; the commands themselves average about 0.3). The path loss compares where the chunk ends up step by step, which leaves a jump forth and back between neighbouring steps almost unpunished.
""")

code("""
display(P.roughness_table({"released policy, no plan": (BASELINE,)} | GROUPS))
""")

md("""
Every rollout of the first 21, side by side.
""")

code("""
for name in ("target moved 4–16 cm", "another object named"):
    print(name)
    display(P.outcome_details(runs, name))
""")

md("""
The first check, on one task with shifted contacts only (the "another object" set was not run). The first two adapters were evaluated twice: once with an oracle plan that switched on the raw finger-contact flag, which made the plan jump back to the grasp 6 times in 22 rollouts, and once with the plan held through contact losses shorter than 5 steps (used everywhere else in this notebook).
""")

code("""
first_runs = {"released policy, no plan": runs["released policy, no plan"]}
first_runs |= {name: P.load_run(folder / "rollouts_oracle") for name, folder in FIRST.items()}
first_runs |= {f"{name}, first evaluation": P.load_run(folder / "rollouts_oracle_first")
               for name, folder in list(FIRST.items())[:2]}
display(P.success_table(first_runs).drop(index="another object named"))
""")

md("""
**Control added on 12 Nov 2026: noise and dropout on the plan, labels unchanged.** "Fast Plans, Faithful Actions" (arXiv 2609.30833) reports the same problem for waypoint plans. Its remedy adds noise to the plan, drops the plan at random and feeds it into every layer of the action expert, while the action labels stay the demonstrated ones. The run below takes the first two ingredients (5 cm of noise on the contact, the plan dropped in 15 % of the samples) with our way of feeding the plan, and has no moved plans. Same starts as above: 6 per perturbed condition. The last table is its open-loop check.
""")

code("""
CONTROL = "noise and dropout, labels unchanged"
control_groups = {"demonstrations only": GROUPS["demonstrations only"],
                  CONTROL: (RUNS / "s3_noise" / "rollouts_oracle",), RELABELLED: GROUPS[RELABELLED]}
control_runs = {name: P.merge_runs(*[P.load_run(folder) for folder in group]) for name, group in control_groups.items()}
display(P.success_table(control_runs))
display(P.chance_table(control_runs, [("demonstrations only", CONTROL), (CONTROL, RELABELLED)]))
for name in P.PERTURBED_SETS:
    print(name)
    display(P.stage_table(control_groups, name))
display(P.open_loop_table({CONTROL: RUNS / "s3_noise"}))
""")

md("""
Gripper paths from above for the two perturbed sets, first 21 rollouts (green: the named object was delivered; ★: where it starts).
""")

code("""
layout = runs["released policy, no plan"]["standard"][0]
fig = P.rollout_paths_figure(runs, P.SETS["target moved 4–16 cm"], layout)
fig.suptitle("target moved 4–16 cm", y=1.02)
plt.show()
fig = P.rollout_paths_figure(runs, P.SETS["another object named"], layout)
fig.suptitle("another object named", y=1.02)
plt.show()
""")

md("""
Distance from the gripper to the planned contact at every query, perturbed rollouts of the first 21. It falls to zero at the grasp, jumps when the plan switches to the release point, and falls again. A line that stays flat above zero is a rollout that does not follow its plan.
""")

code("""
fig = P.reach_figure({name: folder / "rollouts_oracle" for name, folder in FOLDERS.items()},
                     P.SETS["target moved 4–16 cm"] + P.SETS["another object named"])
plt.show()
""")

code("""
clip_logs = {"released policy, no plan": (BASELINE, "shift_16cm", 0)}
clip_logs |= {name: (folder / "rollouts_oracle", "shift_16cm", 0) for name, folder in FOLDERS.items()}
clip_logs |= {"moved plans (path loss): asked for the milk": (FOLDERS[MAIN] / "rollouts_oracle", "ask_milk", 0)}
logs = [H.load_rollout_log(folder / f"{tag}_ep{start}.npz") for folder, tag, start in clip_logs.values()]
CLIP = MEDIA / "plan_following_rollouts.mp4"
H.render_rollout_grid(logs, list(clip_logs), CLIP)
Video(str(CLIP), embed=True, width=1100)
""")

md(CONCLUSIONS)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
