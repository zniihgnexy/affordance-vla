import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rc365/eval_policy.py"
spec = importlib.util.spec_from_file_location("eval_policy", SCRIPT)
eval_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_policy)

CFG = {"tasks": ["OpenDrawer", "OpenCabinet"], "conditions": ["A", "B"], "episodes": 10, "chunk": 5}


def test_wilson_interval_brackets_the_rate_and_widens_for_few_episodes():
    low, high = eval_policy.wilson(5, 20)
    assert low < 0.25 < high
    assert eval_policy.wilson(0, 20)[0] == 0.0
    assert (high - low) < (eval_policy.wilson(1, 4)[1] - eval_policy.wilson(1, 4)[0])


def test_summary_counts_per_cell_and_condition():
    rows = [{"task": "OpenDrawer", "condition": c, "success": s, "steps": 100} for c, s in
            [("A", True), ("A", False), ("B", False), ("B", False)]]
    summary = eval_policy.summarise(rows)
    assert summary["cells"]["OpenDrawer/A"]["rate"] == 0.5
    assert summary["by_condition"]["B"]["wins"] == 0 and summary["by_condition"]["A"]["n"] == 2


def test_jobs_skip_finished_slices_and_keep_partial_ones():
    assert len(eval_policy.jobs(CFG, set())) == 8
    done = {("OpenDrawer", "A", i) for i in range(5)} | {("OpenDrawer", "B", i) for i in range(4)}
    remaining = eval_policy.jobs(CFG, done)
    assert ("OpenDrawer", "A", 0, 5) not in remaining
    assert ("OpenDrawer", "B", 0, 5) in remaining
    assert len(remaining) == 7
