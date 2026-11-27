from affordance_vla.analysis.rc365_stalls import classify, episode_features, stall_report


def episode(dist, touch, q, success=False, task="TurnOffStove", still=True):
    path = [[i * 8, 0.0, 0.0, 0.0, 0.04] for i in range(10)]
    if not still:
        path = [[i * 8, 0.1 * i, 0.0, 0.0, 0.04] for i in range(10)]
    return {
        "task": task,
        "success": success,
        "trace": path,
        "part": {"dist_cm": dist, "touch": touch, "q": q, "q0": q[0]},
    }


def test_each_failure_takes_the_first_class_that_holds():
    far = episode([30.0, 20.0, 12.0], [0, 0, 0], [1.0, 1.0, 1.0])
    near = episode([30.0, 5.0, 6.0], [0, 0, 0], [1.0, 1.0, 1.0])
    pressed = episode([30.0, 2.0, 3.0], [0, 1, 1], [1.0, 1.0, 1.01])
    turned = episode([30.0, 2.0, 3.0], [0, 1, 1], [1.0, 0.6, 0.5])
    assert [classify(r) for r in (far, near, pressed, turned)] == [
        "not_near",
        "near_no_contact",
        "contact_no_motion",
        "moved_not_enough",
    ]


def test_a_success_is_not_a_failure_class():
    assert classify(episode([1.0], [1], [0.0], success=True)) == "success"


def test_features_find_the_closest_approach_and_the_motion():
    f = episode_features(episode([30.0, 5.0, 9.0], [0, 1, 1], [1.0, 0.7, 1.2]))
    assert f["min_dist_cm"] == 5.0 and f["step_of_min_dist"] == 1 and f["first_touch_step"] == 1
    assert f["touch_steps"] == 2 and abs(f["joint_motion_rad"] - 0.3) < 1e-9 and f["final_dist_cm"] == 9.0


def test_report_counts_stalled_failures_separately():
    rows = [
        episode([30.0, 20.0], [0, 0], [1.0, 1.0]),  # stalled, not near
        episode([30.0, 2.0], [0, 1], [1.0, 1.0], still=False),  # kept moving, contact, no motion
        episode([2.0], [1], [0.0], success=True),
    ]
    r = stall_report(rows)["TurnOffStove"]
    assert (r["episodes"], r["successes"], r["failures"], r["stalled"]) == (3, 1, 2, 1)
    assert r["failure_classes"]["not_near"] == 1 and r["failure_classes"]["contact_no_motion"] == 1
    assert r["stalled_classes"]["not_near"] == 1 and r["stalled_classes"]["contact_no_motion"] == 0
