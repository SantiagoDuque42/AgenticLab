# turbofan-rul

Predicts the remaining useful life (RUL, in cycles) of turbofan engines from the
NASA C-MAPSS FD001 dataset. `data/train_FD001.txt.gz` holds 100 engines run to
failure.

- `python -m rul.nightly`: the nightly job (validate, train, save the model). Logs of
  past runs are in `logs/`.
- `python -m rul.evaluate`: validation RMSE only.
- `python -m pytest -q`: tests.
