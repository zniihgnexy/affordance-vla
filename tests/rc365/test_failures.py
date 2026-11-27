from affordance_vla.analysis import rc365_failures as rf


def trace(openings, xyz=(0.0, 0.0, 0.5)):
    return [[8 * k, *xyz, g] for k, g in enumerate(openings)]


def test_grasp_task_classes_follow_the_gripper():
    assert rf.classify_failure("CoffeeSetupMug", trace([0.04] * 10)) == "open_end"
    assert rf.classify_failure("CoffeeSetupMug", trace([0.04, 0.0, 0.0, 0.001])) == "closed_empty"
    assert rf.classify_failure("PickPlaceSinkToCounter", trace([0.04, 0.016, 0.016, 0.04])) == "open_end"
    assert rf.classify_failure("PickPlaceSinkToCounter", trace([0.04, 0.016, 0.016, 0.017])) == "holding"


def test_articulated_task_is_stalled_only_when_the_arm_stops_moving_late():
    moving = [[8 * k, 0.1 * k, 0.0, 0.5, 0.04] for k in range(20)]
    still_late = moving[:8] + [[8 * k, 0.8, 0.0, 0.5, 0.04] for k in range(8, 20)]
    assert rf.classify_failure("TurnOffStove", moving) == "kept_moving"
    assert rf.classify_failure("TurnOffStove", still_late) == "stalled"


def test_count_failures_ignores_successes_and_groups_by_cell():
    rows = [
        {"task": "OpenDrawer", "condition": "A", "success": True, "trace": trace([0.04] * 5)},
        {"task": "OpenDrawer", "condition": "A", "success": False, "trace": trace([0.04] * 5)},
        {"task": "CoffeeSetupMug", "condition": "B", "success": False, "trace": trace([0.04] * 5)},
    ]
    assert rf.count_failures(rows) == {"CoffeeSetupMug/B": {"open_end": 1}, "OpenDrawer/A": {"stalled": 1}}
