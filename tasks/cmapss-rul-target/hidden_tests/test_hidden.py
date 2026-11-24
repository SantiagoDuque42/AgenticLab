"""Hidden grading test for cmapss-rul-target: the official FD001 test set.

The agent never sees test_FD001 or RUL_FD001; the model is fitted here, from the
pristine training file, so editing data/ can't help either.
"""

import os
from pathlib import Path

import numpy as np

from miniagent.harness.metrics import record_metric
from rul.data import load
from rul.pipeline import fit, rmse

SHARED = Path(os.environ["MINIAGENT_SHARED_DIR"]) / "cmapss"
TARGET_RMSE = 19.0


def test_rmse_on_official_test_set():
    model = fit(load(SHARED / "train_FD001.txt.gz"))
    test = load(SHARED / "test_FD001.txt.gz")
    pred = model.predict_last(test)
    true = np.loadtxt(SHARED / "RUL_FD001.txt.gz")
    assert pred.shape == true.shape
    assert np.all(np.isfinite(pred))
    score = rmse(pred, true)
    record_metric("test_rmse", score)
    assert score <= TARGET_RMSE, f"test RMSE {score:.2f} > target {TARGET_RMSE}"
