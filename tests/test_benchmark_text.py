"""The instruction of a perturbed benchmark task, in the wording the policy was trained on."""
import pytest

from affordance3d.benchmark_text import bddl_language, perturbed_instruction

TRAINED = "pick up the salad dressing and place it in the basket"
ORIGINAL = "Pick the salad dressing and place it in the basket"


def test_an_unchanged_task_keeps_the_trained_instruction():
    # the position perturbation moves objects but leaves the task file's language as it was
    assert perturbed_instruction(TRAINED, ORIGINAL, ORIGINAL) == TRAINED


def test_a_renamed_object_is_renamed_in_the_trained_wording():
    # Arrange: the task perturbation asks for another object of the scene
    perturbed = "Pick the tomato sauce and place it in the basket"

    # Act
    instruction = perturbed_instruction(TRAINED, ORIGINAL, perturbed)

    # Assert: the trained phrase "pick up the" is kept, only the object changes
    assert instruction == "pick up the tomato sauce and place it in the basket"


def test_several_changed_spans_are_all_applied():
    trained = "put the black bowl on the plate"
    original = "Put the black bowl on the plate"
    perturbed = "Put the white mug on the stove"

    assert perturbed_instruction(trained, original, perturbed) == "put the white mug on the stove"


def test_a_change_that_cannot_be_located_falls_back_to_the_task_files_language():
    # Arrange: the file calls the object "akita black bowl", the trained instruction only "black bowl"
    trained = "pick up the black bowl and place it on the plate"
    original = "Pick the akita black bowl and place it on the plate"
    perturbed = "Pick the cookie box and place it on the plate"

    # Act
    instruction = perturbed_instruction(trained, original, perturbed)

    # Assert: the perturbed language itself, lower-cased
    assert instruction == "pick the cookie box and place it on the plate"


def test_the_language_line_is_read_from_a_task_file():
    text = "(define (problem LIBERO_Floor_Manipulation)\n  (:domain robosuite)\n" \
           "  (:language Pick the tomato sauce and place it in the basket)\n    (:regions\n"

    assert bddl_language(text) == "Pick the tomato sauce and place it in the basket"


def test_a_task_file_without_a_language_line_is_rejected():
    with pytest.raises(ValueError, match="language"):
        bddl_language("(define (problem X)\n  (:domain robosuite)\n")


def test_two_objects_that_trade_places_are_both_renamed():
    # each change is located where the original words stood, not by searching the edited text again
    trained = "put the bowl on the plate"

    instruction = perturbed_instruction(trained, "Put the bowl on the plate", "Put the plate on the bowl")

    assert instruction == "put the plate on the bowl"


def test_a_changed_word_is_not_matched_inside_another_word():
    # "up" also occurs inside "cup"
    instruction = perturbed_instruction("move the cup up", "Move the cup up", "Move the cup down")

    assert instruction == "move the cup down"


def test_an_added_word_is_inserted_next_to_its_neighbour():
    instruction = perturbed_instruction("pick up the bowl", "Pick the bowl", "Pick the red bowl")

    assert instruction == "pick up the red bowl"


def test_a_removed_word_leaves_no_gap():
    instruction = perturbed_instruction("pick up the red bowl", "Pick the red bowl", "Pick the bowl")

    assert instruction == "pick up the bowl"
