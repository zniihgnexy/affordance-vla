from affordance_vla.analysis.rc365_paired import (
    agreement,
    outcomes,
    paired_by_task,
    paired_counts,
    paired_difference,
    sign_test_p,
    smallest_detectable_difference,
)


def rows(task, condition, flags):
    return [
        {"task": task, "condition": condition, "index": i, "success": f, "steps": 10, "trace": [[0, 0.0]]}
        for i, f in enumerate(flags)
    ]


def test_paired_counts_split_the_four_outcomes():
    a = outcomes(rows("T", "A", [1, 1, 0, 0, 1]))
    b = outcomes(rows("T", "B", [1, 0, 1, 0, 0]))
    assert paired_counts(a, b) == {"both": 1, "only_first": 2, "only_second": 1, "neither": 1}
    assert paired_by_task(a, b)["T"]["both"] == 1


def test_paired_counts_ignore_keys_one_side_lacks():
    assert paired_counts({("T", 0): True, ("T", 1): True}, {("T", 0): False}) == {
        "both": 0,
        "only_first": 1,
        "only_second": 0,
        "neither": 0,
    }


def test_sign_test_is_exact_and_two_sided():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0
    assert abs(sign_test_p(0, 5) - 2 / 32) < 1e-12
    assert abs(sign_test_p(3, 0) - 0.25) < 1e-12  # the AFFO-16 SinkToCounter flips: 3 against 0


def test_paired_difference_interval_covers_the_difference():
    d = paired_difference(12, 10, 80)
    assert abs(d["difference"] - 0.025) < 1e-12 and d["low"] < 0.025 < d["high"]
    assert paired_difference(0, 0, 80) == {"difference": 0.0, "low": 0.0, "high": 0.0}


def test_smallest_detectable_difference_needs_a_lopsided_split():
    # 20 discordant pairs: 15 against 5 gives p = 0.041, 14 against 6 gives p = 0.115
    assert abs(smallest_detectable_difference(80, 20) - (2 * 15 - 20) / 80) < 1e-12
    assert smallest_detectable_difference(80, 3) is None


def test_agreement_counts_same_outcomes_and_lists_flips():
    old = rows("T", "A", [1, 0, 1])
    new = rows("T", "A", [1, 1, 1])
    new[2]["trace"] = [[1, 1.0]]
    result = agreement(new, old)
    assert result["matched"] == 3 and result["same_success"] == 2 and result["same_trace"] == 2
    assert result["success_flips"] == [("T", 1, False, True)]
