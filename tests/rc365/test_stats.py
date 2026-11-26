from affordance_vla.analysis import rc365_stats


def test_wilson_interval_brackets_the_rate_and_widens_for_few_episodes():
    low, high = rc365_stats.wilson(5, 20)
    assert low < 0.25 < high
    assert rc365_stats.wilson(0, 20)[0] == 0.0
    assert (high - low) < (rc365_stats.wilson(1, 4)[1] - rc365_stats.wilson(1, 4)[0])


def test_summary_counts_per_cell_and_condition_and_prints_a_table():
    rows = [{"task": "OpenDrawer", "condition": c, "success": s} for c, s in
            [("A", True), ("A", False), ("B", False), ("B", False)]]
    summary = rc365_stats.summarise(rows)
    assert summary["cells"]["OpenDrawer/A"]["rate"] == 0.5
    assert summary["by_condition"]["B"]["wins"] == 0 and summary["by_condition"]["A"]["n"] == 2
    table = rc365_stats.markdown_table(summary, ["OpenDrawer", "OpenCabinet"], ["A", "B"])
    assert "| OpenDrawer | 1/2 (50%;" in table and "| OpenCabinet | - | - |" in table
