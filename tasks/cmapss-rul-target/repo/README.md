# turbofan-rul

Predicts the remaining useful life (RUL, in cycles) of turbofan engines from the
NASA C-MAPSS FD001 dataset. `data/train_FD001.txt.gz` holds 100 engines run to
failure: per row, unit id, cycle, 3 operational settings and 21 sensors.

- `python -m rul.evaluate`: validation RMSE (held-out engines, cut at random cycles).
- `python -m pytest -q`: tests.
