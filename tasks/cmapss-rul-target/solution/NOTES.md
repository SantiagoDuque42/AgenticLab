# What was tried (reference solution)

| change                                              | test RMSE |
|-----------------------------------------------------|-----------|
| starting model: all columns, ridge, uncapped target |      32.0 |
| cap target at 125 cycles                            |      21.9 |
| + only the 14 informative sensors, + cycle          |      22.5 |
| + quadratic feature crosses (still ridge)           |     ~20.1 |
| 14 sensors + 30-cycle trailing means, MLP(32)       |      17.5 |
