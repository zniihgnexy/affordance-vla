import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rc365/eval_policy.py"
spec = importlib.util.spec_from_file_location("eval_policy", SCRIPT)
eval_policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_policy)

CFG = {"tasks": ["OpenDrawer", "OpenCabinet"], "conditions": ["A", "B"], "episodes": 10, "chunk": 5}


def test_jobs_skip_finished_slices_and_keep_partial_ones():
    assert len(eval_policy.jobs(CFG, set())) == 8
    done = {("OpenDrawer", "A", i) for i in range(5)} | {("OpenDrawer", "B", i) for i in range(4)}
    remaining = eval_policy.jobs(CFG, done)
    assert ("OpenDrawer", "A", 0, 5) not in remaining
    assert ("OpenDrawer", "B", 0, 5) in remaining
    assert len(remaining) == 7


def test_client_command_carries_the_cell_and_the_shared_settings():
    cfg = {**CFG, "base_seed": 7, "n_act": 8, "videos_per_cell": 3}
    command = eval_policy.client_command(cfg, ("OpenDrawer", "B", 5, 10), 5601, Path("/out"))
    assert command[command.index("--task") + 1] == "OpenDrawer"
    assert command[command.index("--start") + 1:command.index("--start") + 3] == ["5", "--stop"]
    assert command[command.index("--base-seed") + 1] == "7" and command[command.index("--port") + 1] == "5601"
