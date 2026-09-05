# Targeted benchmark

Entries are mean (Monte Carlo SE). Error metrics and coefficients use successful runs; every failure remains in the raw data and counts below. Diagnostic fractions use all runs where available. NaN denotes an undefined quantity or unavailable Monte Carlo SE.

| Design | Method | RMSE log-ratio | MAE lift | Spearman | Failures/reps | Warnings |
|---|---|---:|---:|---:|---:|---:|
| Log-linear, alocação padrão | DML-linear | 0.1337 (0.0687) | 0.1196 (0.0667) | 0.9509 (0.0023) | 0/2 | 0 |
| Log-linear, alocação padrão | DML-quadratic | 0.1636 (0.1017) | 0.1217 (0.0736) | 0.9509 (0.0023) | 0/2 | 0 |
| Log-linear, alocação padrão | Log-mean-linear | 0.1078 (0.0105) | 0.0769 (0.0069) | 0.8264 (0.0036) | 0/2 | 0 |
| Log-linear, alocação padrão | Log-mean-quadratic | 0.1896 (0.0896) | 0.1440 (0.0714) | 0.8342 (0.0018) | 0/2 | 0 |
| Log-linear, alocação padrão | S-learner | 0.6753 (0.2249) | 0.4836 (0.0526) | 0.5321 (0.0520) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | DML-linear | 0.3849 (0.0048) | 0.4147 (0.0005) | 0.9797 (0.0006) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | 0.1364 (0.0513) | 0.1538 (0.0502) | 0.9797 (0.0006) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | 0.4022 (0.0013) | 0.3874 (0.0123) | 0.8736 (0.0103) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | 0.1664 (0.0592) | 0.1928 (0.0366) | 0.8579 (0.0037) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | S-learner | 0.4038 (0.0298) | 0.5404 (0.1079) | -0.0920 (0.6055) | 0/2 | 0 |
| Curvo, alocação padrão | DML-linear | 0.4031 (0.0088) | 0.4218 (0.0222) | 0.9797 (0.0006) | 0/2 | 0 |
| Curvo, alocação padrão | DML-quadratic | 0.2246 (0.0407) | 0.3083 (0.0096) | 0.9797 (0.0006) | 0/2 | 0 |
| Curvo, alocação padrão | Log-mean-linear | 0.4148 (0.0262) | 0.4144 (0.0234) | -0.0027 (0.8554) | 0/2 | 0 |
| Curvo, alocação padrão | Log-mean-quadratic | 0.1706 (0.0402) | 0.2269 (0.0385) | 0.8598 (0.0002) | 0/2 | 0 |
| Curvo, alocação padrão | S-learner | 0.5334 (0.0984) | 0.5735 (0.0160) | -0.2706 (0.5412) | 0/2 | 0 |
| Curvo, pouco overlap | DML-linear | 0.6160 (0.0645) | 0.5691 (0.0664) | -0.9797 (0.0006) | 0/2 | 0 |
| Curvo, pouco overlap | DML-quadratic | 0.9653 (0.7351) | 0.6126 (0.3790) | 0.0006 (0.9797) | 0/2 | 0 |
| Curvo, pouco overlap | Log-mean-linear | 0.6387 (0.0131) | 0.6130 (0.0100) | -0.8641 (0.0026) | 0/2 | 0 |
| Curvo, pouco overlap | Log-mean-quadratic | 0.4722 (0.2305) | 0.4516 (0.1905) | -0.8558 (0.0036) | 0/2 | 0 |
| Curvo, pouco overlap | S-learner | 0.6763 (0.1466) | 0.6086 (0.0621) | 0.3905 (0.0911) | 0/2 | 0 |

## Linear heterogeneity under curved responses

These are fitted coefficients of X1 in the linear dose term. For a model omitting curvature they are allocation-dependent summaries, not automatically the structural heterogeneity. No structural-parameter coverage calculation is made.

| Design | Method | theta1 (MCSE) | Mean marginal SE | Successful runs |
|---|---|---:|---:|---:|
| Curvo, dose aleatorizada uniforme | DML-linear | -0.2338 (0.1161) | 0.1448 | 2/2 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | -0.2519 (0.1426) | 0.1480 | 2/2 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | -0.3046 (0.1187) | nan | 2/2 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | -0.2636 (0.1218) | nan | 2/2 |
| Curvo, alocação padrão | DML-linear | -0.0394 (0.0212) | 0.1651 | 2/2 |
| Curvo, alocação padrão | DML-quadratic | -0.2329 (0.0785) | 0.1853 | 2/2 |
| Curvo, alocação padrão | Log-mean-linear | -0.0178 (0.0292) | nan | 2/2 |
| Curvo, alocação padrão | Log-mean-quadratic | -0.1420 (0.0114) | nan | 2/2 |
| Curvo, pouco overlap | DML-linear | 0.5073 (0.1124) | 0.3399 | 2/2 |
| Curvo, pouco overlap | DML-quadratic | 0.9445 (0.9719) | 0.7710 | 2/2 |
| Curvo, pouco overlap | Log-mean-linear | 0.5304 (0.0414) | nan | 2/2 |
| Curvo, pouco overlap | Log-mean-quadratic | 0.3154 (0.2735) | nan | 2/2 |

## Log-mean conversion predictions

Poisson loss is a mean-fitting criterion for binary outcomes. The log link does not constrain predictions to [0, 1]; no clipping is applied. Grid = independent test X crossed with doses {0, -0.8, -0.4, 0.4, 0.8}.

| Design | Method | Invalid train fraction (MCSE) | Invalid grid fraction (MCSE) |
|---|---|---:|---:|
| Log-linear, alocação padrão | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Log-linear, alocação padrão | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, alocação padrão | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, alocação padrão | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, pouco overlap | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, pouco overlap | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |

The structured log-mean models restrict the baseline as well as the relative curve, so this comparison does not perfectly isolate orthogonalization. S-learner clipping at 0.001 follows the historical benchmark and is recorded. Paired differences (candidate minus DML-linear) and their actual pair counts are in targeted_paired.csv.
