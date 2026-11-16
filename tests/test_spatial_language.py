"""Where the objects of a LIBERO-Spatial scene start, in the words of its instructions."""
from affordance3d.spatial_language import initial_regions, instruction_naming, location_phrases, named_instruction

BDDL = """
  (:init
    (In akita_black_bowl_1 wooden_cabinet_1_top_region)
    (On akita_black_bowl_2 wooden_cabinet_1_top_side)
    (Open wooden_cabinet_1_top_region)
    (On plate_1 main_table_plate_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
"""
INSTRUCTION = "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"


def test_the_initial_region_of_every_placed_object_is_read_from_the_task_file():
    assert initial_regions(BDDL) == {"akita_black_bowl_1": "wooden_cabinet_1_top_region",
                                     "akita_black_bowl_2": "wooden_cabinet_1_top_side",
                                     "plate_1": "main_table_plate_region"}


def test_objects_whose_region_has_words_in_the_instructions_get_their_phrase():
    assert location_phrases(BDDL) == {"akita_black_bowl_1": "in the top drawer of the wooden cabinet",
                                      "akita_black_bowl_2": "on the wooden cabinet"}


def test_the_instruction_for_the_other_bowl_says_where_that_bowl_is():
    # Act
    other = instruction_naming(INSTRUCTION, location_phrases(BDDL), "akita_black_bowl_1", "akita_black_bowl_2")

    # Assert
    assert other == "pick up the black bowl on the wooden cabinet and place it on the plate"


def test_no_instruction_is_made_when_a_phrase_is_missing_or_not_in_the_instruction():
    phrases = location_phrases(BDDL)

    assert instruction_naming(INSTRUCTION, phrases, "akita_black_bowl_1", "plate_1") is None
    assert instruction_naming("pick up the black bowl and place it on the plate", phrases, "akita_black_bowl_1",
                              "akita_black_bowl_2") is None


def test_no_instruction_is_made_when_both_bowls_stand_at_a_place_with_the_same_words():
    phrases = {"akita_black_bowl_1": "on the stove", "akita_black_bowl_2": "on the stove"}

    assert instruction_naming("pick up the black bowl on the stove and place it on the plate", phrases,
                              "akita_black_bowl_1", "akita_black_bowl_2") is None


def test_an_object_suite_instruction_names_the_other_object_by_its_name():
    other = named_instruction("pick up the alphabet soup and place it in the basket", "alphabet_soup_1",
                              "cream_cheese_1")

    assert other == "pick up the cream cheese and place it in the basket"


def test_a_spatial_instruction_names_the_other_bowl_by_its_place():
    other = named_instruction(INSTRUCTION, "akita_black_bowl_1", "akita_black_bowl_2", location_phrases(BDDL))

    assert other == "pick up the black bowl on the wooden cabinet and place it on the plate"


def test_no_instruction_is_made_when_the_other_object_cannot_be_told_apart_in_words():
    # Arrange: two identical bowls and no phrases: renaming would give the same instruction
    instruction = "pick up the akita black bowl and place it on the plate"

    # Act / Assert
    assert named_instruction(instruction, "akita_black_bowl_1", "akita_black_bowl_2") is None
