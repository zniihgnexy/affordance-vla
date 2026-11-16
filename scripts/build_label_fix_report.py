"""Build report/2026-11-16_label_fix_results.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/):
    python ../scripts/build_label_fix_report.py 2026-11-16_label_fix_results.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-16_label_fix_results.ipynb
Needs runs/libero_pro/{oracle_plan,predicted_g1_hold}_{s9,s10} from scripts/remote/run_libero_pro_gpu.sh (20 start
states, seed 7) and runs/plan_following/dataset10. The closing text is in conclusions_label_fix.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_label_fix.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# New labels for moved plans: the test on LIBERO-PRO Object
**VLA direction** · LIBERO-PRO, Object suite · released starVLA checkpoint, VLM frozen · 16 Nov 2026 · two ten-task action heads that differ only in their labels

- **The question:** the diagnosis of 15 Nov traced three failing tasks of the task perturbation (ketchup, milk, orange juice) to the training labels for "another object named": they made a detour when the object asked for was close to the gripper's start, and copied the final approach of the scene's original target. With labels that scale the path about the start and take the final approach from the object asked for, do these tasks recover, and do the other tasks hold?
- **Real:** both action heads, trained the same way on all ten LIBERO-Object tasks except for the labels: `s9_align_release10` (labels of 13 Nov: align first) and `s10_scale_own10` (labels of 15 Nov: `--scale-from-start --own-approach`); the plan head that scores image cells (`g1`) and its plans, kept per phase; all rollouts.
- **ORACLE:** the plan of the columns marked "oracle"; the grasp points the plans are compared with.
- **Protocol:** both perturbations, 10 tasks × 20 start states, seed 7 (the protocol of the 20-start runs, not the papers' 50 states and three seeds).
""")

code("""
from pathlib import Path
import sys

import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance3d").is_dir())
sys.path.insert(0, str(ROOT))
from affordance3d.plan_input import offsets_from_json
from report import libero_pro_helpers as L
from report import plan_predictor_helpers as P
from report import task_perturbation_helpers as T

pd.set_option("display.max_colwidth", 60)
PRO = ROOT / "runs" / "libero_pro"
offsets = offsets_from_json((ROOT / "runs" / "plan_following" / "dataset10" / "offsets.json").read_text())
RUNS = {"labels of 13 Nov + oracle plan": "oracle_plan_s9",
        "labels of 15 Nov + oracle plan": "oracle_plan_s10",
        "labels of 13 Nov + map plans, kept": "predicted_g1_hold_s9",
        "labels of 15 Nov + map plans, kept": "predicted_g1_hold_s10"}
runs = {name: {suite: L.load_suite(PRO / run, suite) for suite in L.SUITES.values()} for name, run in RUNS.items()}
""")

md("""
## 1 · Solved rollouts
**This step:** rollouts in which the benchmark's goal check held, per perturbation, for the two heads with the oracle plan and with kept map plans.

**Status:** real rollouts; two columns get the ORACLE plan.
""")

code("""
display(L.score_table(runs).T)
""")

md("""
## 2 · Per task
**This step:** the same counts per task. The three tasks of the diagnosis are 3 (ketchup), 4 (milk) and 6 (orange juice) of the task perturbation.

**Status:** real.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(L.per_task_table(runs, suite))
""")

md("""
## 3 · How the rollouts went
**This step:** the measures of the diagnosis for each task of the task perturbation and each head: when the gripper first reaches the grasp point, how often it never does or runs to the step limit, and how long it stalls far from the object; then how far the rollouts got (held, knocked over, let go outside the basket).

**Status:** real; the reach is measured against the ORACLE grasp point.
""")

code("""
scene_targets = {L.task_of(log): log.target for log in runs["labels of 13 Nov + oracle plan"]["libero_object_swap"]}
for name in ("labels of 13 Nov + oracle plan", "labels of 15 Nov + oracle plan"):
    print(name)
    rows = T.task_rows(L.episode_paths(PRO / RUNS[name], "libero_object_task"), scene_targets, offsets)
    display(T.task_table(rows).drop(columns=["gripper start to grasp point (cm)",
                                             "original target to the object asked for (cm)"]))
display(pd.DataFrame({f"{name}: {label.split(':')[0]}": L.stage_counts(PRO / run, suite)
                      for name, run in RUNS.items() for label, suite in L.SUITES.items()}))
""")

md("""
## 4 · Where the map plans pointed
**This step:** for the two heads with kept map plans, the first plan of each rollout against the ORACLE grasp point of the object asked for. With kept plans the first plan is the same for both heads (same start images, same plan head), so a difference in what follows comes from the action head.

**Status:** real plans; grasp points ORACLE.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    audits = {name: [P.grasp_plans(log, offsets) for log in runs[name][suite]]
              for name in RUNS if "map" in name}
    display(P.per_task_pointing(audits))
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
