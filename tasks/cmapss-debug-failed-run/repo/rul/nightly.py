"""Nightly training job: train on all engines, report validation RMSE, save the model.

Run it: python -m rul.nightly
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

from rul.data import load
from rul.evaluate import validation_rmse
from rul.pipeline import fit

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.json"
log = logging.getLogger("nightly")


def main() -> int:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level="INFO")
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    frame = load()
    log.info("loaded %d rows, %d units", len(frame["unit"]), len(np.unique(frame["unit"])))
    try:
        score = validation_rmse(frame)
        log.info("validation RMSE %.2f", score)
        if not np.isfinite(score):
            raise RuntimeError(f"validation RMSE is {score}")
        model = fit(frame)
    except Exception:
        log.exception("training failed")
        return 1
    MODEL_PATH.parent.mkdir(exist_ok=True)
    MODEL_PATH.write_text(
        json.dumps(
            {
                "validation_rmse": score,
                "weights": model.regressor.weights.tolist(),
                "bias": model.regressor.bias,
                "mean": model.scaler.mean.tolist(),
                "std": model.scaler.std.tolist(),
            }
        )
    )
    log.info("saved %s", MODEL_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
