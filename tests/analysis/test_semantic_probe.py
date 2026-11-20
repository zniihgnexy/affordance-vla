"""Scoring 8 x 8 maps against the segmentation (E1 of Pipeline v3)."""
import numpy as np
import pytest

from affordance_vla.analysis.semantic_probe import (
    cell_coverage,
    merge_similarity,
    object_at_peak,
    object_phrase,
    text_token_indices,
)


def test_object_phrase_turns_a_scene_name_into_words():
    assert object_phrase("tomato_sauce_1") == "tomato sauce"
    assert object_phrase("akita_black_bowl_2") == "akita black bowl"


def test_cell_coverage_splits_the_segmentation_into_cells_row_by_row():
    seg = np.zeros((16, 16), np.uint8)
    seg[0:2, 2:4] = 1  # object 0 fills cell (row 0, column 1) of a 8 x 8 grid
    seg[14:16, 0:1] = 2  # object 1 fills half of cell (row 7, column 0)

    coverage = cell_coverage(seg, objects=2)

    assert coverage.shape == (2, 64)
    assert coverage[0, 1] == 1.0 and coverage[0].sum() == 1.0
    assert coverage[1, 56] == 0.5 and coverage[1].sum() == 0.5


def test_cell_coverage_rejects_a_size_that_does_not_split():
    with pytest.raises(ValueError, match="does not split"):
        cell_coverage(np.zeros((10, 16), np.uint8), objects=1)


def test_merge_similarity_by_mean_and_by_max():
    cells = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    text = np.array([[2.0, 0.0], [0.0, 3.0]])

    mean = merge_similarity(text, cells, "mean")
    top = merge_similarity(text, cells, "max")

    np.testing.assert_allclose(mean, [0.5, 0.5, np.sqrt(0.5)])
    np.testing.assert_allclose(top, [1.0, 1.0, np.sqrt(0.5)])


def test_merge_similarity_rejects_an_unknown_merge_and_empty_text():
    with pytest.raises(ValueError, match="merge"):
        merge_similarity(np.ones((1, 2)), np.ones((3, 2)), "sum")
    with pytest.raises(ValueError, match="no text"):
        merge_similarity(np.zeros((0, 2)), np.ones((3, 2)), "mean")


def test_object_at_peak_names_the_object_on_the_best_cell_or_none():
    coverage = np.zeros((2, 4))
    coverage[0, 1] = 0.8
    coverage[1, 2] = 0.02  # too little of the cell to count
    assert object_at_peak(np.array([0.0, 1.0, 0.5, 0.0]), coverage) == 0
    assert object_at_peak(np.array([0.0, 0.0, 1.0, 0.0]), coverage) is None


def test_text_token_indices_finds_the_instruction_inside_the_prompt():
    pieces = ["Your", " task", " is", " pick", " up", " the", " tomato", " sauce", ".", " To", " identify"]
    assert text_token_indices(pieces, "pick up the tomato sauce").tolist() == [3, 4, 5, 6, 7]
    assert text_token_indices(pieces, "Tomato Sauce").tolist() == [6, 7]
    assert text_token_indices(pieces, "ketchup").tolist() == []
    assert text_token_indices(pieces, "").tolist() == []


def test_text_token_indices_takes_the_first_occurrence():
    pieces = [" the", " white", " mug", " and", " the", " yellow", " and", " white", " mug"]
    assert text_token_indices(pieces, "white mug").tolist() == [1, 2]
