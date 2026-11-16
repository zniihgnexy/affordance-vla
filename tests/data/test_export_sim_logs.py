"""Image-free copies of rollout logs for replaying elsewhere (scripts/data/export_sim_logs.py)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "data" / "export_sim_logs.py"
spec = importlib.util.spec_from_file_location("export_sim_logs", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_a_copy_keeps_everything_but_the_images_and_is_not_rewritten(tmp_path):
    source = tmp_path / "runs" / "r1" / "libero_object_swap" / "task0_ep0.npz"
    source.parent.mkdir(parents=True)
    np.savez(source, meta=json.dumps({"suite": "libero_object_swap"}), actions=np.zeros((3, 7)),
             agentview=np.zeros((4, 8, 8, 3), np.uint8), wrist=np.zeros((4, 8, 8, 3), np.uint8),
             seg=np.zeros((4, 8, 8), np.uint8), ee_pos=np.ones((4, 3)))
    (source.parent / "task0_summary.json").write_text("[]")
    assert export.export_run(tmp_path / "runs", "r1", tmp_path / "out") == (1, 0)
    with np.load(tmp_path / "out" / "r1" / "libero_object_swap" / "task0_ep0.npz") as copy:
        assert sorted(copy.files) == ["actions", "ee_pos", "meta"]
    assert (tmp_path / "out" / "r1" / "libero_object_swap" / "task0_summary.json").is_file()
    assert export.export_run(tmp_path / "runs", "r1", tmp_path / "out") == (0, 1)
