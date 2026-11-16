"""Reading what a logged test episode recorded about the policy (affordance_vla/sim/logged_episode.py)."""
import numpy as np

from affordance_vla.sim.logged_episode import LoggedPolicy, read_queries, task_index
from affordance_vla.sim.narration import describe_plan, plan_source


def write_log(path, plans=True):
    arrays = {"chunk_steps": np.array([0, 8]), "chunks": np.zeros((2, 8, 7), dtype=np.float32)}
    if plans:
        arrays["plans"] = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0], [0.5, 0, 0, 0, 0, 0, 0, 0, 1, -1.0]],
                                   dtype=np.float32)
        arrays["plan_predicted"] = np.array([True, False])
    np.savez(path, **arrays)
    return path


def test_each_query_has_its_step_chunk_and_plan(tmp_path):
    queries = read_queries(write_log(tmp_path / "task2_ep0.npz"))
    assert [q.step for q in queries] == [0, 8]
    assert queries[0].chunk.shape == (8, 7) and queries[0].predicted and not queries[1].predicted
    assert queries[1].plan[-1] == -1.0


def test_a_run_without_plans_gives_queries_without_plans(tmp_path):
    assert all(q.plan is None for q in read_queries(write_log(tmp_path / "task2_ep0.npz", plans=False)))


def test_the_task_index_comes_from_the_evaluation_file_name(tmp_path):
    assert task_index(tmp_path / "task7_ep13.npz", {}) == 7


def test_the_logged_policy_repeats_the_executed_actions():
    actions = np.arange(14, dtype=float).reshape(2, 7)
    policy = LoggedPolicy(actions)
    np.testing.assert_array_equal(policy.act(None, None, 1), actions[1])


def test_a_plan_is_described_by_its_event_and_contact(tmp_path):
    query = read_queries(write_log(tmp_path / "task2_ep0.npz"))[0]
    text, contact = describe_plan(query, np.zeros(3), "predicted")
    np.testing.assert_allclose(contact, [0.1, 0.0, 0.0])  # reach of 1 = 10 cm
    assert "抓取" in text and "10.0 cm" in text and "新预测" in text
    assert plan_source(query, "oracle").endswith("（ORACLE）")
