"""Build report/experiments/2026-11-12_libero_pro_object.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/experiments/):
    python ../../scripts/report/build_libero_pro_report.py 2026-11-12_libero_pro_object.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-12_libero_pro_object.ipynb
Needs runs/libero_pro/{released,oracle_plan_s2} from scripts/eval/run_libero_pro_gpu.sh (through the GPU
queue; see its header; runs/libero_pro/oracle_plan is added as a third column once it is complete) and one replayed demonstration in runs/plan_following/demos for the unperturbed
scene. The closing text is in conclusions_libero_pro.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_libero_pro.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# LIBERO-PRO, Object suite: the released policy, and the same policy steered by an oracle plan
**VLA direction** · LIBERO-PRO (arXiv 2510.03827) · released starVLA checkpoint `Qwen3-VL-PI-LIBERO-4in1` · 12 Nov 2026 · the first numbers on an official perturbed benchmark

- **The question:** on the benchmark other papers report, how far does the released policy fall when the target is moved or another object is asked for, and how much of that would an action head that obeys a plan recover if the plan were right?
- **Real:** the benchmark's scenes, start states and goal checks; the released policy; the action head trained with counterfactual plans on 11 Nov; all rollouts. That head saw six of the ten LIBERO-Object tasks: the bbq sauce, butter, chocolate pudding and orange juice were never targets in its training. A head trained on all ten tasks is added as a third column when its run is complete.
- **ORACLE:** the plan given to the second policy (object position from the simulator plus the average grasp point of the demonstrations). Its column is an upper bound for a plan head, not a result of a method.
- **Protocol:** both policies saw only standard LIBERO demonstrations. 10 tasks × 20 start states (0–19) per perturbation; the benchmark's papers use 50. A rollout counts when the benchmark's goal check holds. One run, one seed.

```text
(1) the two perturbations of one task                      real (benchmark files)
        ↓
(2) solved rollouts: released policy / oracle-plan policy  real rollouts; the plan is ORACLE
        ↓
(3) per task, and what happened to the named object        real
        ↓
(4) numbers other papers report on the same benchmark      quoted, not re-run
```

<sub>To re-run on the A6000 server: `envs/server/setup_libero_pro.sh` (CPU), then `scripts/eval/run_libero_pro_gpu.sh` through the GPU queue with `NAME=released` and with `NAME=oracle_plan_s2 ADAPTER=s2_actions PLAN=oracle` (ten-task head: `NAME=oracle_plan ADAPTER=s4_actions10 PLAN=oracle`), then execute this notebook.</sub>
""")

code("""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Video, display
from matplotlib_inline.backend_inline import set_matplotlib_formats

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from affordance_vla.sim.rollout_log import load_rollout_log
from report import libero_pro_helpers as L
from report import vla_pipeline_demo_helpers as H

set_matplotlib_formats("jpeg")
plt.rcParams.update({"figure.dpi": 90})
pd.set_option("display.max_colwidth", 120)

PRO = ROOT / "runs" / "libero_pro"
RELEASED, ORACLE = "released policy, no plan", "action head from six tasks + ORACLE plan"
TEN = "action head from ten tasks + ORACLE plan"
FOLDERS = {RELEASED: PRO / "released", ORACLE: PRO / "oracle_plan_s2"}
if all(len(list((PRO / "oracle_plan" / suite).glob("task*_ep*.npz"))) >= 200 for suite in L.SUITES.values()):
    FOLDERS[TEN] = PRO / "oracle_plan"
runs = {name: {suite: L.load_suite(folder, suite) for suite in L.SUITES.values()} for name, folder in FOLDERS.items()}
print({name: {suite: len(logs) for suite, logs in run.items()} for name, run in runs.items()})
""")

md("""
## 1 · The two perturbations
**This step:** the same task, "pick up the salad dressing and place it in the basket", as LIBERO has it and as LIBERO-PRO changes it. The position perturbation lets the target trade places with another object; the instruction stays. The task perturbation leaves the scene and asks for another object of it.

**Status:** real; the frames are the first camera image of a rollout. LIBERO takes a task's instruction from its file name, which the task perturbation does not change: the new object is only named inside the task file. The instruction used here is the trained wording with that object put in (`affordance_vla/sim/benchmark_text.py`).
""")

code("""
TASK = 2   # the salad dressing task in both suites
standard = load_rollout_log(ROOT / "runs" / "plan_following" / "demos"
                            / "pick_up_the_salad_dressing_and_place_it_in_the_basket_demo0.npz")
shown = {label.split(":")[0]: load_rollout_log(PRO / "released" / suite / f"task{TASK}_ep0.npz")
         for label, suite in L.SUITES.items()}
fig = L.starts_figure(standard, shown)
plt.show()
""")

md("""
## 2 · Solved rollouts
**This step:** every task of the suite from 20 start states, under each perturbation, for the two policies.

**Status:** real rollouts. The second column is given the ORACLE plan at every query, so it shows what the action head can do when the plan is right.
""")

code("""
display(L.score_table(runs))
""")

md("""
## 3 · Per task, and what happened to the named object
**This step:** the same counts for each task, then the outcome of every rollout judged by what the gripper held: the named object was delivered (`success`), another object was grasped first (`wrong object`), nothing was held (`no grasp`), or the named object was held but did not end in the basket (`not delivered`).

**Status:** real. For the policy with a plan, the last table splits its rollouts by how far they got.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(L.per_task_table(runs, suite))
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(L.outcome_counts(runs, suite))
display(pd.DataFrame({f"{name.split(' +')[0]}: {label.split(':')[0]}": L.stage_counts(folder, suite)
                      for name, folder in FOLDERS.items() if name != RELEASED for label, suite in L.SUITES.items()}))
""")

code("""
SHORT = {RELEASED: "released", ORACLE: "ORACLE plan"}
clips = {f"{SHORT[name]} - {label.split(':')[0]}": load_rollout_log(FOLDERS[name] / suite / f"task{TASK}_ep0.npz")
         for label, suite in L.SUITES.items() for name in SHORT}
CLIP = ROOT / "report" / "media" / "libero_pro_object.mp4"   # tracked in git
H.render_rollout_grid(list(clips.values()), list(clips), CLIP)
Video(str(CLIP), embed=True, width=1100)
""")

md("""
## 4 · What other papers report on LIBERO-PRO
**This step:** published numbers for the same two perturbations, on all four suites; the Object suite is the second number of each cell.

**Status:** quoted from the papers, not re-run here. They are not directly comparable with section 2: the papers use 50 trials per task, their baselines disagree with each other (three different values for π0.5 on the Object suite's position perturbation), and it is not known which instruction each of them fed under the task perturbation.
""")

code("""
display(L.published_table())
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
