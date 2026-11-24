import numpy as np

from rul.data import load, rul_labels
from rul.evaluate import split_units, truncate_randomly


def test_labels_count_down_to_zero():
    frame = load()
    rul = rul_labels(frame)
    unit1 = rul[frame["unit"] == 1]
    assert unit1[-1] == 0
    assert np.all(np.diff(unit1) == -1)


def test_predict_last_shape():
    frame = load()
    _, val = split_units(frame)
    cut, true = truncate_randomly(val)
    assert len(np.unique(cut["unit"])) == len(true)



def test_split_holds_out_whole_engines():
    train, val = split_units(load())
    assert not set(np.unique(train["unit"])) & set(np.unique(val["unit"]))
