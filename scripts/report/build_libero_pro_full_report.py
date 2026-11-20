"""Build report/experiments/2026-11-14_libero_pro_full_protocol.ipynb (cells only; execution happens separately).

Usage on the server (starvla env, from report/experiments/):
    python ../../scripts/report/build_libero_pro_full_report.py 2026-11-14_libero_pro_full_protocol.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace 2026-11-14_libero_pro_full_protocol.ipynb
Needs runs/libero_pro/full_{released,g1_hold_s9,oracle_s9}_seed{7,8,9} from scripts/eval/run_libero_pro_gpu.sh
with TRIALS=50 and SEED=7, 8, 9 (through the GPU queue; see docs/vla/2026-11-13_VLA_branch_预测计划驱动闭环.md).
Seeds that are missing or incomplete are shown as such and left out of the means. The closing text is in
conclusions_libero_pro_full.md next to this script.
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1])
CLOSING = Path(__file__).with_name("conclusions_libero_pro_full.md").read_text()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))  # noqa: E731

md("""
# LIBERO-PRO, Object suite, under the papers' protocol
**VLA direction** · LIBERO-PRO (arXiv 2510.03827) · released starVLA checkpoint `Qwen3-VL-PI-LIBERO-4in1`, VLM frozen · 14 Nov 2026 · 50 start states per task, three evaluation seeds

- **The question:** the best configuration of 13 Nov solved 133 and 136 of 200 rollouts without any oracle plan, on 20 start states per task and one seed. Does it hold on all 50 start states the papers use, over several seeds, and where does it stand next to the published numbers?
- **Real:** the benchmark's scenes, start states and goal checks; the released policy; the action head trained with the labels of 13 Nov on all ten LIBERO-Object tasks (`s9_align_release10`); the plan head that scores the image cells of the agent view (`g1`) and its plans, each kept until the gripper has closed or opened; all rollouts.
- **ORACLE:** the plan of the third configuration (object position from the simulator plus the average grasp point of the demonstrations), an upper bound for the plan head. The plan head turns a pixel and a depth into a 3D point with the simulator's camera calibration.
- **Protocol:** 10 tasks × 50 start states × 2 perturbations = 1000 rollouts per configuration and seed; seeds 7, 8 and 9 (the seed of the policy's sampling noise and of the simulator). All policies saw only standard LIBERO demonstrations. One trained model per configuration: the seeds vary the evaluation, not the training.

```text
(1) success per seed, mean and spread                      real rollouts
        ↓
(2) next to the numbers other papers report                 quoted, not re-run
        ↓
(3) per task, pooled over the seeds                         real
        ↓
(4) what happened to the named object                       real
```

<sub>To re-run on the A6000 server, through the GPU queue: `scripts/eval/run_libero_pro_gpu.sh` with `TRIALS=50 SEED=<seed>` and `NAME=full_released_seed<seed>`; `NAME=full_g1_hold_s9_seed<seed> ADAPTER=s9_align_release10 PLAN=predicted PREDICTOR=g1 HOLD=1`; `NAME=full_oracle_s9_seed<seed> ADAPTER=s9_align_release10 PLAN=oracle`. Then execute this notebook.</sub>
""")

code("""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "affordance_vla").is_dir())
sys.path.insert(0, str(ROOT))
from report import libero_pro_helpers as L

pd.set_option("display.max_colwidth", 120)
PRO = ROOT / "runs" / "libero_pro"
SEEDS = (7, 8, 9)
EXPECTED = 500   # rollouts per perturbation and seed: 10 tasks x 50 start states
RELEASED = "released policy, no plan"
MAP = "ten-task head of 13 Nov + map plans, kept"
ORACLE = "ten-task head of 13 Nov + ORACLE plan"
CONFIGS = {RELEASED: "full_released_seed{}", MAP: "full_g1_hold_s9_seed{}", ORACLE: "full_oracle_s9_seed{}"}


def load(folder):
    has_logs = lambda suite: (folder / suite).is_dir() and any((folder / suite).glob("task*_ep*.npz"))  # noqa: E731
    return {suite: L.load_suite(folder, suite) if has_logs(suite) else [] for suite in L.SUITES.values()}


runs = {name: {seed: load(PRO / pattern.format(seed)) for seed in SEEDS} for name, pattern in CONFIGS.items()}
counts = {name: {seed: [len(logs) for logs in run.values()] for seed, run in seeds.items()}
          for name, seeds in runs.items()}
print("rollouts per configuration, seed and perturbation:", counts)
""")

md("""
## 1 · Success per seed
**This step:** rollouts in which the benchmark's goal check held, for each seed, and the mean and standard deviation of the success rate over the complete seeds (500 rollouts each).

**Status:** real rollouts. The third configuration is given the ORACLE plan.
""")

code("""
table = L.seed_table(runs, expected=EXPECTED)
display(table.style.format({"mean": "{:.3f}", "std over seeds": "{:.3f}"}, na_rep="–"))
""")

md("""
## 2 · Next to the published numbers
**This step:** the means of section 1 next to the numbers other papers report for the same two perturbations of the Object suite.

**Status:** the published numbers are quoted, not re-run. The papers use 50 start states per task too; how many seeds they ran, and which instruction they fed under the task perturbation, is not always stated. GAM uses the simulator's depth; the affordance-field method uses depth and GPT-4o at test time. Three papers report three different values for π0.5 under the position perturbation.
""")

code("""
rows = {name: {"position": values[1], "task": tasks[1], "what it is": what}
        for name, what, values, tasks in L.PUBLISHED}
for name in CONFIGS:
    rows[f"ours: {name}"] = {"what it is": "this notebook, mean over complete seeds",
                             "position": table.loc[(name, "position"), "mean"],
                             "task": table.loc[(name, "task"), "mean"]}
display(pd.DataFrame.from_dict(rows, orient="index")[["what it is", "position", "task"]]
        .style.format({"position": "{:.2f}", "task": "{:.2f}"}, na_rep="–"))
""")

md("""
## 3 · Per task, pooled over the seeds
**This step:** for each task, the solved rollouts of each configuration over all seeds (150 per task when all seeds are complete).

**Status:** real.
""")

code("""
pooled = {name: {suite: [log for run in seeds.values() for log in run[suite]] for suite in L.SUITES.values()}
          for name, seeds in runs.items()}
for label, suite in L.SUITES.items():
    print(label)
    display(L.per_task_table(pooled, suite))
""")

md("""
## 4 · What happened to the named object
**This step:** the outcome of every rollout judged by what the gripper held: the named object was delivered (`success`), another object was grasped first (`wrong object`), nothing was held (`no grasp`), or the named object was held but did not end in the basket (`not delivered`); then, for the two configurations with a plan, how far the rollouts got. Pooled over the seeds.

**Status:** real.
""")

code("""
for label, suite in L.SUITES.items():
    print(label)
    display(L.outcome_counts(pooled, suite))
stages = {}
for name in (MAP, ORACLE):
    for label, suite in L.SUITES.items():
        parts = [L.stage_counts(PRO / CONFIGS[name].format(seed), suite) for seed in SEEDS
                 if len(runs[name][seed][suite])]
        stages[f"{name}: {label.split(':')[0]}"] = sum(parts[1:], parts[0]) if parts else pd.Series(dtype=int)
display(pd.DataFrame(stages))
""")

md(CLOSING)

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, OUT)
print("wrote", OUT)
