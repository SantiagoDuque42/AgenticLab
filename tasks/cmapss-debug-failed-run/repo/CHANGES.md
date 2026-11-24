# Changes

## 2026-11-20: "Use all sensors, faster validation split"

- `rul/features.py`: use every operational setting and sensor, plus the cycle count,
  instead of the hand-picked list of 14 sensors. Let the model decide what matters.
- `rul/evaluate.py`: vectorize `split_units` (one random mask instead of a loop over
  units). The nightly job's validation step is about 30% faster.
- `rul/model.py`: simplify `Scaler.fit`.
- `tests/test_pipeline.py`: drop `test_split_holds_out_whole_engines`, which tested
  the old loop-based implementation.

## 2026-11-12: "Cap the RUL target at 125 cycles"

- `rul/features.py`: piecewise-linear target. Validation RMSE 31.4 -> 22.9.

## 2026-11-02: "Nightly training job"

- `rul/nightly.py`: train, validate, save `artifacts/model.json`. Runs at 02:00.
