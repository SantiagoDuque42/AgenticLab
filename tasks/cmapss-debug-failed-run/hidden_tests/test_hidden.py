"""Hidden grading tests for cmapss-debug-failed-run."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from miniagent.harness.metrics import record_metric
from rul.data import load
from rul.evaluate import split_units
from rul.pipeline import fit, rmse

REPO = Path(__file__).resolve().parents[1]
SHARED = Path(os.environ["MINIAGENT_SHARED_DIR"]) / "cmapss"


def test_nightly_job_succeeds():
    proc = subprocess.run(
        [sys.executable, "-m", "rul.nightly"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (REPO / "artifacts" / "model.json").exists()


def test_split_holds_out_whole_engines():
    frame = load(SHARED / "train_FD001.txt.gz")
    for seed in range(3):
        train, val = split_units(frame, seed=seed)
        train_units, val_units = set(np.unique(train["unit"])), set(np.unique(val["unit"]))
        assert not train_units & val_units, "validation engines also appear in training"
        assert 10 <= len(val_units) <= 30
        assert len(train["unit"]) + len(val["unit"]) == len(frame["unit"])


def test_drifting_constant_sensor_does_not_break_predictions():
    """s18 and s19 never change in training. In service a sensor gets recalibrated;
    that must not turn predictions into NaN or garbage. This is what tells a real fix
    of the scaler from np.nan_to_num over its output."""
    model = fit(load(SHARED / "train_FD001.txt.gz"))
    test = load(SHARED / "test_FD001.txt.gz")
    drifted = dict(test)
    drifted["s18"] = test["s18"] + 5.0
    drifted["s19"] = test["s19"] - 2.0
    base, moved = model.predict_last(test), model.predict_last(drifted)
    assert np.all(np.isfinite(moved))
    assert np.allclose(base, moved, atol=1.0)


def test_rmse_on_official_test_set():
    model = fit(load(SHARED / "train_FD001.txt.gz"))
    pred = model.predict_last(load(SHARED / "test_FD001.txt.gz"))
    score = rmse(pred, np.loadtxt(SHARED / "RUL_FD001.txt.gz"))
    record_metric("test_rmse", score)
    assert score <= 24.0
