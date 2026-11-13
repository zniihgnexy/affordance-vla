"""Build report/2026-11-13_plan_predictor.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/):
    python ../scripts/build_plan_predictor_report.py 2026-11-13_plan_predictor.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-13_plan_predictor.ipynb
Needs runs/plan_predictor/{p1,g1} from scripts/remote/run_plan_predictor_gpu.sh (HEAD=regress and HEAD=grid)
and the closed-loop folders under runs/libero_pro from scripts/remote/run_libero_pro_gpu.sh: released,
oracle_plan_s2, and predicted_<run>[_hold] for the predictors (all through the GPU queue; see the headers of
the two job scripts). Closed-loop folders that are missing or incomplete are left out of the tables.
The closing text is in conclusions_plan_predictor.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_plan_predictor.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# Predicting the plan from the frozen VLM: a regressed contact against a map over image cells
**VLA direction** · LIBERO-PRO (arXiv 2510.03827), Object suite · released starVLA checkpoint `Qwen3-VL-PI-LIBERO-4in1`, VLM frozen · 13 Nov 2026 · the first closed loop without an oracle plan

- **The question:** the notebook of 12 Nov showed that an action head trained to obey a plan solves most perturbed tasks when the plan is right. Here the plan is predicted from the camera images and the instruction. Does a predicted plan point at the right place when the target has moved or another object is asked for, and how much of the oracle result is left?
- **Real:** both plan predictors and everything they output; the action head (trained with counterfactual plans on six of the ten tasks, 11 Nov); all rollouts; the benchmark's scenes, start states and goal checks.
- **ORACLE:** the training targets of the predictors (the contact of the next gripper event in the demonstrations, from simulator state), the contacts drawn as rings in the pictures, and the last column of the closed-loop table, which is given the oracle plan as an upper bound. The camera calibration used by the map head is taken from the simulator.
- **Protocol:** the predictors saw only standard LIBERO-Object demonstrations (454 demonstrations of ten tasks, of which 5 per task were kept out for testing). Closed loop: 10 tasks × 20 start states (0–19) per perturbation; the benchmark's papers use 50. One run, one seed.

```text
(1) the two predictors                                       real (trained here on ORACLE targets)
        ↓
(2) contact error on frames they never trained on            real predictions against ORACLE contacts
        ↓
(3) the same in pictures; the map over image cells           real
        ↓
(4) closed loop on LIBERO-PRO: predicted plan → action head  real rollouts; last column ORACLE
        ↓
(5) where the plans pointed during the rollouts              real
        ↓
(6) what happened to the named object                        real
```

<sub>To re-run on the A6000 server, through the GPU queue: `scripts/remote/run_plan_predictor_gpu.sh` with `RUN=p1 HEAD=regress` and `RUN=g1 HEAD=grid`; `scripts/remote/run_libero_pro_gpu.sh` with `NAME=predicted_g1 ADAPTER=s2_actions PLAN=predicted PREDICTOR=g1` (add `HOLD=1` and `NAME=predicted_g1_hold` for the kept plan; the same for `p1`); then execute this notebook.</sub>
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance3d").is_dir())
sys.path.insert(0, str(ROOT))
from affordance3d.plan_grid import camera_of_log
from affordance3d.plan_input import offsets_from_json
from affordance3d.rollout_log import load_rollout_log
from report import libero_pro_helpers as L
from report import plan_predictor_helpers as P
from report import vla_pipeline_demo_helpers as H

set_matplotlib_formats("jpeg")
plt.rcParams.update({"figure.dpi": 90})
pd.set_option("display.max_colwidth", 120)

REGRESS, GRID = "regressed contact", "map over image cells"
PREDICTORS = {REGRESS: ROOT / "runs" / "plan_predictor" / "p1", GRID: ROOT / "runs" / "plan_predictor" / "g1"}
metrics = {name: P.load_metrics(run) for name, run in PREDICTORS.items()}
found = {name: P.load_predictions(run) for name, run in PREDICTORS.items()}
P.check_same_items(found)
DEMO = ROOT / "runs" / "plan_following" / "demos" / "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz"
camera = camera_of_log(load_rollout_log(DEMO))
offsets = offsets_from_json((ROOT / "runs" / "plan_following" / "dataset10" / "offsets.json").read_text())
print({name: sorted(facts["held_out"]) + sorted(facts["benchmark"]) for name, facts in metrics.items()})
""")

md("""
## 1 · The two predictors
**This step:** both read the hidden states of the frozen VLM (36 layers, about 180 tokens: 64 per camera image, then the instruction) and output the next gripper event (grasp, release, none), where the gripper should be at that event, and the direction it should arrive from. They differ in how the place is given.

- **Regressed contact:** four learned queries read the whole token sequence and output three world coordinates.
- **Map over image cells:** each of the 8 × 8 image tokens of the agent view gets a score, "the contact is seen in this cell", plus the position inside the cell and the depth along the viewing ray. The best cell, its offset and its depth give the 3D contact through the camera calibration. Because the VLM reads the images before the instruction, a small two-layer encoder lets the cells read the instruction first. In training the agent view is shifted at random by up to two cells and the target cell moves with it.

**Status:** real. Both are trained on the same frames. A frame before the grasp is also given with an instruction that names another object of the scene, whose grasp point is then the target; this needs no extra demonstrations. The table gives sizes and training length.
""")

code("""
display(P.facts_table(metrics))
""")

md("""
## 2 · Contact error on frames the predictors never trained on
**This step:** the predicted contact against the oracle contact, in centimetres, on five sets of frames: held-out demonstrations under their own instruction (next grasp, next release), the same frames with another object named, and the first frame of LIBERO-PRO rollouts under each perturbation (two start states per task).

**Status:** real predictions; the contacts they are compared with are ORACLE. "Another object nearer" counts the items whose predicted contact lies closer to another object's grasp point than to the right one: the plan points at the wrong object.
""")

code("""
display(P.offline_table(metrics))
""")

md("""
## 3 · The same in pictures, and the map
**This step:** the first frame of one rollout per task, under each perturbation. The ring is the oracle grasp point of the object asked for; the markers are the two predictions, projected into the image. Below them, the scores the map head gives to the 64 cells for the position perturbation.

**Status:** real predictions on images the predictors never saw; ring = ORACLE.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    P.prediction_figure(found, camera, P.items_of(found[REGRESS], f"pro {suite}", every=2))
    plt.show()
""")

code("""
for label, suite in L.SUITES.items():
    print("map over image cells ·", label)
    P.map_figure(found[GRID], camera, P.items_of(found[GRID], f"pro {suite}", every=2))
    plt.show()
""")

md("""
## 4 · Closed loop on LIBERO-PRO
**This step:** the plan is predicted from the images at run time and given to the action head; nothing comes from the simulator. Every task of the Object suite from 20 start states under each perturbation.

**Status:** real rollouts. The action head is the same in every column with a plan. "At every query" asks the predictor for a new plan every 8 steps. "Kept" predicts a plan once, keeps its contact in world coordinates until the gripper has carried out the event (it closed, or opened), and only then asks again. The last column is given the ORACLE plan.
""")

code("""
PRO = ROOT / "runs" / "libero_pro"
RELEASED, ORACLE = "released policy, no plan", "ORACLE plan"
CANDIDATES = {RELEASED: "released",
              f"{REGRESS}, at every query": "predicted_p1", f"{REGRESS}, kept": "predicted_p1_hold",
              f"{GRID}, at every query": "predicted_g1", f"{GRID}, kept": "predicted_g1_hold",
              ORACLE: "oracle_plan_s2"}
FOLDERS = {name: PRO / folder for name, folder in CANDIDATES.items()
           if all(len(list((PRO / folder / suite).glob("task*_ep*.npz"))) >= 200 for suite in L.SUITES.values())}
runs = {name: {suite: L.load_suite(folder, suite) for suite in L.SUITES.values()} for name, folder in FOLDERS.items()}
display(L.score_table(runs).T)
""")

md("""
**For comparison, what other papers report on LIBERO-PRO** (position and task perturbation; the Object suite is the second number of each cell). Quoted, not re-run; they use 50 trials per task and their baselines disagree with each other, so the comparison is loose.
""")

code("""
display(L.published_table())
""")

md("""
## 5 · Where the plans pointed during the rollouts
**This step:** every logged plan is turned back into the contact it pointed at and compared with the oracle grasp point of the object asked for, at every query before anything is held. First the plan of the first query of each rollout, then the error along the approach, then per task.

**Status:** real plans; the grasp points they are compared with are ORACLE. A grasp plan "points at" the object whose grasp point is nearest to it, if that is within 10 cm. In the figure, query n only counts the rollouts that had held nothing by then, so its later points rest on fewer and slower rollouts.
""")

code("""
PREDICTED = [name for name in FOLDERS if name not in (RELEASED,)]
audits = {name: {suite: [P.grasp_plans(log, offsets) for log in runs[name][suite]] for suite in L.SUITES.values()}
          for name in PREDICTED}
for label, suite in L.SUITES.items():
    print(label)
    display(P.pointing_table({name: run[suite] for name, run in audits.items()}).T)
""")

code("""
P.approach_figure({name: run for name, run in audits.items() if name != ORACLE})
plt.show()
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(P.per_task_pointing({name: run[suite] for name, run in audits.items() if name != ORACLE}))
""")

md("""
## 6 · What happened to the object asked for
**This step:** the outcome of every rollout judged by what the gripper held: the named object was delivered (`success`), another object was grasped first (`wrong object`), nothing was held (`no grasp`), or the named object was held but did not end in the basket (`not delivered`). Then, for the policies with a plan, how far the rollouts got relative to the plan they were given.

**Status:** real. In the second table "the planned grasp point" is the one the policy was given, right or wrong: the row shows whether the action head went where the plan said.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(L.outcome_counts(runs, suite))
display(pd.DataFrame({f"{name}: {label.split(':')[0]}": L.stage_counts(folder, suite)
                      for name, folder in FOLDERS.items() if name != RELEASED for label, suite in L.SUITES.items()}))
""")

code("""
TASK = 2   # the salad dressing task in both suites
SHOWN = [name for name in FOLDERS if name not in (RELEASED, ORACLE)][-2:]
clips = {f"{name} - {label.split(':')[0]}": load_rollout_log(FOLDERS[name] / suite / f"task{TASK}_ep0.npz")
         for label, suite in L.SUITES.items() for name in SHOWN}
CLIP = ROOT / "report" / "media" / "plan_predictor_closed_loop.mp4"   # tracked in git
H.render_rollout_grid(list(clips.values()), list(clips), CLIP)
Video(str(CLIP), embed=True, width=1100)
""")

md("""
## 7 · A second action head: moved plans that come down from above, and moved releases
**Why:** with a right plan, most remaining failures of the action head are tall objects knocked over before they are held, and objects let go outside the basket (section 6, last column; the notebook of 12 Nov). The action labels for a moved plan are the demonstrated path moved to the new contact. With the rule used so far the shift is blended in until the last 8 cm, and with less than 8 cm left it is put into the next step while the descent goes on. So the labels ask the gripper to come down while it is still to the side. Moved plans also only covered the way to the grasp, not the carry to the basket.

**This step:** two changes to the labels, and an action head trained with both (`s8_align_release`), on the same six tasks and for the same 2400 steps as the head of 11 Nov:
- **Align first:** a shift that does not fit into the path before the last 18 cm is made first, in a straight line, and the demonstrated approach follows, moved (`affordance3d/counterfactual.py`, `--align-first --rigid-tail 0.18`).
- **Moved releases:** frames in which the object is carried are also paired with a release point moved within the same 18 cm disc (`--counterfactual shift+other+release`).
- In the same commit a label error was fixed: when a moved path needs more steps than the demonstration had left, those steps used to get the demonstration's commands from after its grasp (lifting) added to them. The heads of 11 Nov had that error; a head with the fix alone is still training (`s2b_actions`).

**Status:** the figure and the first table are real demonstrations and real rollouts; the moved paths are what the labels ask for. The rollouts of section 7 are real; their plans are ORACLE unless a column says "map over image cells".
""")

code("""
from affordance3d.counterfactual import Retarget
from affordance3d.plan_dataset import load_tracks

tracks = load_tracks(ROOT / "runs" / "plan_following" / "dataset" / "tracks.npz")
track = next(t for t in tracks if t.target == "salad_dressing_1")
grasp = track.labels[0].event.step
path = track.ee_pos[grasp - 16:grasp + 1]   # 16 steps before the grasp
RULES = {"rule of 11 Nov (last 8 cm moved as a whole)": Retarget(),
         "align first (last 18 cm moved as a whole)": Retarget(rigid_tail=0.18, align_first=True)}
P.retarget_figure(path, path[-1] + [0.0, 0.06, 0.0], RULES)
plt.show()
""")

md("""
**How high the gripper is as it closes in on a tall object.** For every rollout whose named object is tall (bbq sauce, ketchup, milk, orange juice, salad dressing): the height above the grasp point when the gripper first comes within 12, 9, 6, 4 and 2 cm of it horizontally, before it first touches the object. Median over rollouts, in cm.
""")

code("""
TALL_WORDS = [name[:-2] for name in P.TALL_OBJECTS]
DEMOS = ROOT / "runs" / "plan_following" / "demos"
demos = [load_rollout_log(path, images=False) for path in sorted(DEMOS.glob("*_demo*.npz"))
         if any(word in path.name for word in TALL_WORDS)]
HEADS = {"head of 11 Nov + ORACLE plan": PRO / "oracle_plan_s2",
         "head of 13 Nov + ORACLE plan": PRO / "oracle_plan_s8"}
HEADS = {name: folder for name, folder in HEADS.items()
         if all(len(list((folder / suite).glob("task*_ep*.npz"))) >= 200 for suite in L.SUITES.values())}
tall = {"demonstrations": demos}
for name, folder in HEADS.items():
    for label, suite in L.SUITES.items():
        logs = L.load_suite(folder, suite)
        tall[f"{name}, {label.split(':')[0]}"] = [log for log in logs if log.target in P.TALL_OBJECTS]
display(P.approach_height_table(tall, offsets))
""")

md("""
**On the six training tasks** (the experiment of 11 Nov: the same 5 standard starts, the target moved 4–16 cm from 6 starts, another object named from 6 starts; ORACLE plan). One row per head; a success is the named object grasped first and left in the basket.
""")

code("""
from report import plan_following_helpers as F

SIX = ROOT / "runs" / "plan_following"
SIX_HEADS = {"11 Nov: moved grasps": "s2_actions", "moved grasps, label fix": "s2b_actions",
             "moved grasps and releases (labels before the fix)": "s6_release",
             "align first, label fix": "s7_align", "13 Nov: align first + moved releases, label fix": "s8_align_release"}
def rollout_folders(run):
    return tuple(folder for folder in (SIX / run / "rollouts_oracle", SIX / run / "rollouts_oracle_more")
                 if folder.is_dir())


SIX_HEADS = {name: run for name, run in SIX_HEADS.items() if rollout_folders(run)}
six = {name: F.merge_runs(*[F.load_run(folder) for folder in rollout_folders(run)]) for name, run in SIX_HEADS.items()}
display(F.success_table(six))
for set_name in F.PERTURBED_SETS:
    print(set_name)
    display(F.stage_table({name: rollout_folders(run) for name, run in SIX_HEADS.items()}, set_name))
""")

md("""
**On LIBERO-PRO Object** (10 tasks × 20 starts per perturbation): the head of 13 Nov with the ORACLE plan and with the plans of the map head, kept per phase, next to the head of 11 Nov.
""")

code("""
SECOND = {"head of 11 Nov + map, kept": "predicted_g1_hold", "head of 13 Nov + map, kept": "predicted_g1_hold_s8",
          "head of 11 Nov + ORACLE plan": "oracle_plan_s2", "head of 13 Nov + ORACLE plan": "oracle_plan_s8",
          "11 Nov, moved releases + ORACLE plan": "oracle_plan_s6"}
SECOND = {name: PRO / run for name, run in SECOND.items()
          if all(len(list((PRO / run / suite).glob("task*_ep*.npz"))) >= 200 for suite in L.SUITES.values())}
second = {name: {suite: L.load_suite(folder, suite) for suite in L.SUITES.values()} for name, folder in SECOND.items()}
display(L.score_table(second).T)
for label, suite in L.SUITES.items():
    print(label)
    display(L.outcome_counts(second, suite))
display(pd.DataFrame({f"{name}: {label.split(':')[0]}": L.stage_counts(folder, suite)
                      for name, folder in SECOND.items() for label, suite in L.SUITES.items()}))
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
