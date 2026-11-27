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


def test_first_episode_starts_the_slices_later_and_defaults_to_zero():
    cfg = {**CFG, "first_episode": 5, "episodes": 20, "chunk": 5}
    slices = [(j[2], j[3]) for j in eval_policy.jobs(cfg, set()) if j[:2] == ("OpenDrawer", "A")]
    assert slices == [(5, 10), (10, 15), (15, 20)]
    assert eval_policy.jobs({**CFG}, set()) == eval_policy.jobs({**CFG, "first_episode": 0}, set())


def test_probes_reach_the_client_only_when_configured():
    cfg = {**CFG, "base_seed": 7, "n_act": 4, "videos_per_cell": 0}
    plain = eval_policy.client_command(cfg, ("OpenDrawer", "A", 0, 5), 5601, Path("/out"))
    both = {**cfg, "probes": ["part", "objects"]}
    probed = eval_policy.client_command(both, ("OpenDrawer", "A", 0, 5), 5601, Path("/out"))
    assert "--probes" not in plain
    assert probed[probed.index("--probes") + 1] == "part,objects" and probed[probed.index("--n-act") + 1] == "4"


def test_the_checks_configs_extend_the_base_and_name_what_differs():
    from affordance_vla.config import load_config

    root = Path(__file__).resolve().parents[2] / "configs/rc365"
    stuck = load_config(root / "base_checks_stuck.yaml", ["n_act=4"])
    assert stuck["tasks"] == ["TurnOffStove", "TurnOnSinkFaucet"] and stuck["probes"] == ["part"]
    assert stuck["n_act"] == 4
    assert stuck["base_seed"] == 1234 and stuck["clients"] == 14
    instances = load_config(root / "base_checks_instances.yaml")
    assert (instances["first_episode"], instances["episodes"], instances["conditions"]) == (20, 80, ["A", "B"])
    assert len(eval_policy.jobs(instances, set())) == 4 * 2 * 12  # 96 slices of 5 = four rounds of 24 clients
    steps = load_config(root / "base_checks_steps.yaml")
    assert steps["conditions"] == ["A"] and steps["episodes"] == 20 and len(eval_policy.jobs(steps, set())) == 40
