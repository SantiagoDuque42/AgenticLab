import numpy as np

from rul.data import load, rul_labels
from rul.evaluate import split_units, truncate_randomly
from rul.pipeline import fit


def test_labels_count_down_to_zero():
    frame = load()
    rul = rul_labels(frame)
    unit1 = rul[frame["unit"] == 1]
    assert unit1[-1] == 0
    assert np.all(np.diff(unit1) == -1)


def test_fit_and_predict_last():
    frame = load()
    train, val = split_units(frame)
    model = fit(train)
    cut, true = truncate_randomly(val)
    pred = model.predict_last(cut)
    assert pred.shape == true.shape
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)


def test_split_holds_out_whole_engines():
    train, val = split_units(load())
    assert not set(np.unique(train["unit"])) & set(np.unique(val["unit"]))
