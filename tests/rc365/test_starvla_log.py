from affordance_vla.analysis.starvla_log import read_series, smoothed

LOG = """
INFO | {'action_dit_loss':
       0.25,
       'timing/model':
       1.5,
INFO | {'action_dit_loss':
       1.2e-01,
       'timing/model':
       1.75,
"""


def test_series_are_read_in_order_across_wrapped_lines(tmp_path):
    path = tmp_path / "train.log"
    path.write_text(LOG)
    assert read_series(path, "action_dit_loss") == [0.25, 0.12]
    assert read_series(path, "timing/model") == [1.5, 1.75]


def test_smoothed_is_a_trailing_mean():
    assert smoothed([1.0, 3.0, 5.0], 2) == [1.0, 2.0, 4.0]
