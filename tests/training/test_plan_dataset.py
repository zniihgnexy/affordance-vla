"""Tests for storing the tracks of many replayed demonstrations in one file."""
import numpy as np
import pytest

from affordance_vla.training.plan_dataset import frame_offsets, load_tracks, save_tracks
from affordance_vla.training.plan_samples import track_from_arrays


def make_track(steps: int, names: tuple[str, ...], grasp: int, release: int):
    rng = np.random.default_rng(steps)
    actions = rng.uniform(-1, 1, size=(steps, 7))
    actions[:, 6] = -1.0
    actions[grasp - 3:release - 2, 6] = 1.0
    grasped = np.zeros(steps + 1, dtype=bool)
    grasped[grasp:release] = True
    return track_from_arrays(ee_pos=np.cumsum(rng.normal(size=(steps + 1, 3)), axis=0) * 0.01, actions=actions,
                             grasped=grasped, object_pos=rng.normal(size=(steps + 1, len(names), 3)), names=names,
                             target=names[0], instruction=f"pick up the {names[0]}")


def test_tracks_survive_a_round_trip_through_one_file(tmp_path):
    # Arrange: two demonstrations of different length and with a different number of objects
    tracks = [make_track(40, ("bottle_1", "basket_1"), grasp=15, release=30),
              make_track(55, ("can_1", "basket_1", "box_1"), grasp=20, release=45)]

    # Act
    save_tracks(tmp_path / "tracks.npz", tracks)
    restored = load_tracks(tmp_path / "tracks.npz")

    # Assert
    assert len(restored) == 2
    for original, loaded in zip(tracks, restored):
        np.testing.assert_allclose(loaded.ee_pos, original.ee_pos)
        np.testing.assert_allclose(loaded.actions, original.actions)
        np.testing.assert_allclose(loaded.object_pos, original.object_pos)
        assert loaded.names == original.names and loaded.target == original.target
        assert loaded.instruction == original.instruction
        assert [label.event for label in loaded.labels] == [label.event for label in original.labels]
        assert loaded.command_flips == original.command_flips


def test_frame_offsets_give_the_first_state_row_of_each_demonstration():
    # Arrange
    tracks = [make_track(40, ("bottle_1", "basket_1"), 15, 30), make_track(55, ("can_1", "basket_1"), 20, 45)]

    # Act
    offsets = frame_offsets(tracks)

    # Assert: 41 and 56 state rows
    assert offsets.tolist() == [0, 41, 97]


def test_save_tracks_rejects_an_empty_list(tmp_path):
    # Act / Assert
    with pytest.raises(ValueError, match="no tracks"):
        save_tracks(tmp_path / "tracks.npz", [])


def test_load_tracks_rejects_a_file_whose_offsets_do_not_match_its_demonstrations(tmp_path):
    # Arrange: a file that lists one demonstration but two blocks of rows
    save_tracks(tmp_path / "tracks.npz", [make_track(40, ("bottle_1", "basket_1"), 15, 30)])
    with np.load(tmp_path / "tracks.npz") as data:
        arrays = {key: data[key] for key in data.files}
    arrays["demo_start"] = np.array([0, 20, 41])
    np.savez_compressed(tmp_path / "broken.npz", **arrays)

    # Act / Assert
    with pytest.raises(ValueError, match="1 demonstrations but 3 offsets"):
        load_tracks(tmp_path / "broken.npz")
