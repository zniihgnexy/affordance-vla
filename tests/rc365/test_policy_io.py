import numpy as np
import pytest
from PIL import Image

from affordance_vla.sim import rc365_policy_io as io


def _obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs = {key: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for key in io.VIDEO_KEYS}
    obs[io.LANGUAGE_KEY] = "pick the mug"
    return obs


def test_example_holds_the_three_cameras_in_training_order_at_224():
    obs = _obs()
    example = io.build_example(obs)
    assert [im.shape for im in example["image"]] == [(224, 224, 3)] * 3
    for key, image in zip(io.VIDEO_KEYS, example["image"]):
        assert np.array_equal(image, np.asarray(Image.fromarray(obs[key]).resize((224, 224))))
    assert example["lang"] == "pick the mug" and "state" not in example


def test_example_needs_every_camera_and_the_instruction():
    obs = _obs()
    del obs[io.VIDEO_KEYS[2]]
    with pytest.raises(KeyError):
        io.build_example(obs)


def test_example_refuses_a_state_it_cannot_feed_like_training():
    with pytest.raises(NotImplementedError):
        io.build_example(_obs(), include_state=True)


def test_action_steps_split_the_chunk_into_the_environment_keys():
    chunk = np.arange(16 * 12, dtype=float).reshape(16, 12)
    steps = io.action_steps(chunk, 8)
    assert len(steps) == 8
    assert np.array_equal(steps[1]["action.end_effector_position"], chunk[1, 0:3])
    assert np.array_equal(steps[1]["action.base_motion"], chunk[1, 7:11])
    assert steps[7]["action.control_mode"].shape == (1,)


@pytest.mark.parametrize("chunk,n", [(np.zeros((16, 7)), 8), (np.zeros((4, 12)), 8), (np.zeros((16, 12)), 0)])
def test_action_steps_reject_a_wrong_chunk(chunk, n):
    with pytest.raises(ValueError):
        io.action_steps(chunk, n)


def test_episode_seed_is_stable_and_separates_cells():
    assert io.episode_seed(1234, "OpenDrawer", "A", 3) == io.episode_seed(1234, "OpenDrawer", "A", 3)
    seeds = {io.episode_seed(1234, t, c, i) for t in ("OpenDrawer", "OpenCabinet") for c in "AB" for i in range(20)}
    assert len(seeds) == 80
    assert io.episode_seed(1234, "OpenDrawer", "A", 3) != io.episode_seed(1235, "OpenDrawer", "A", 3)


def test_numpy_scalars_become_plain_numbers_for_json():
    import json

    from affordance_vla.sim.rc365_eval import _plain

    assert json.dumps([_plain(np.int64(3)), _plain(None), _plain("x")]) == '[3, null, "x"]'
