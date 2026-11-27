from affordance_vla.analysis.rc365_checks import by_task, category_table, comparison, instances_report, steps_report


def rows(task, condition, flags, objects=None):
    return [
        {
            "task": task,
            "condition": condition,
            "index": i,
            "success": bool(f),
            **({"objects": objects[i]} if objects else {}),
        }
        for i, f in enumerate(flags)
    ]


def test_by_task_pools_all_rows():
    result = by_task(rows("T", "A", [1, 0, 1, 1]) + rows("U", "A", [0, 0]))
    assert result["T"]["wins"] == 3 and result["all"]["n"] == 6 and result["all"]["wins"] == 3


def test_comparison_gives_counts_pairs_and_p():
    result = comparison(rows("T", "A", [1, 1, 1, 0]), rows("T", "B", [0, 0, 0, 0]))
    assert result["counts"]["only_first"] == 3 and result["pairs"] == 4
    assert abs(result["p_sign_test"] - 0.25) < 1e-12
    assert result["paired_difference"]["difference"] == 0.75


def test_steps_report_compares_neighbouring_checkpoints():
    report = steps_report({16000: rows("T", "A", [0, 0]), 32000: rows("T", "A", [1, 0]), 48000: rows("T", "A", [1, 1])})
    assert list(report["adjacent_pairs"]) == ["16000->32000", "32000->48000"]
    assert report["success"]["48000"]["all"]["wins"] == 2
    assert report["adjacent_pairs"]["16000->32000"]["all"]["counts"]["only_second"] == 1


def test_category_table_counts_by_moved_object_category():
    obj = lambda c: {"obj": {"category": c, "instance": c + "_1"}}  # noqa: E731
    seen = rows("T", "A", [1, 0], [obj("mug"), obj("cup")])
    unseen = rows("T", "B", [0, 0], [obj("mug"), obj("cup")])
    assert category_table(seen, unseen) == {
        "cup": {"seen": [0, 1], "unseen": [0, 1]},
        "mug": {"seen": [1, 1], "unseen": [0, 1]},
    }
    assert instances_report(seen, unseen)["by_category"]["mug"]["seen"] == [1, 1]
