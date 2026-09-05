# Targeted benchmark

Entries are mean (Monte Carlo SE). Error metrics and coefficients use successful runs; every failure remains in the raw data and counts below. Diagnostic fractions use all runs where available. NaN denotes an undefined quantity or unavailable Monte Carlo SE.

| Design | Method | RMSE log-ratio | MAE lift | Spearman | Failures/reps | Warnings |
|---|---|---:|---:|---:|---:|---:|
| Log-linear, alocação padrão | DML-linear | 0.1337 (0.0687) | 0.1196 (0.0667) | 1.0000 (0.0000) | 0/2 | 0 |
| Log-linear, alocação padrão | DML-quadratic | 0.1636 (0.1017) | 0.1217 (0.0736) | 1.0000 (0.0000) | 0/2 | 0 |
| Log-linear, alocação padrão | Log-mean-linear | 0.1078 (0.0105) | 0.0769 (0.0069) | 1.0000 (0.0000) | 0/2 | 0 |
| Log-linear, alocação padrão | Log-mean-quadratic | 0.1896 (0.0896) | 0.1440 (0.0714) | 1.0000 (0.0000) | 0/2 | 0 |
| Log-linear, alocação padrão | S-learner | 0.6753 (0.2249) | 0.4836 (0.0526) | 0.5599 (0.0526) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | DML-linear | 0.3849 (0.0048) | 0.4147 (0.0005) | 1.0000 (0.0000) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | 0.1364 (0.0513) | 0.1538 (0.0502) | 1.0000 (0.0000) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | 0.4022 (0.0013) | 0.3874 (0.0123) | 1.0000 (0.0000) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | 0.1664 (0.0592) | 0.1928 (0.0366) | 1.0000 (0.0000) | 0/2 | 0 |
| Curvo, dose aleatorizada uniforme | S-learner | 0.4038 (0.0298) | 0.5404 (0.1079) | -0.0996 (0.6182) | 0/2 | 0 |

## Linear heterogeneity under curved responses

These are fitted coefficients of X1 in the linear dose term. For a model omitting curvature they are allocation-dependent summaries, not automatically the structural heterogeneity. No structural-parameter coverage calculation is made.

| Design | Method | theta1 (MCSE) | Mean marginal SE | Successful runs |
|---|---|---:|---:|---:|
| Curvo, dose aleatorizada uniforme | DML-linear | -0.2338 (0.1161) | 0.1448 | 2/2 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | -0.2519 (0.1426) | 0.1480 | 2/2 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | -0.3046 (0.1187) | nan | 2/2 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | -0.2636 (0.1218) | nan | 2/2 |

## Log-mean conversion predictions

Poisson loss is a mean-fitting criterion for binary outcomes. The log link does not constrain predictions to [0, 1]; no clipping is applied. Grid = independent test X crossed with doses {0, -0.8, -0.4, 0.4, 0.8}.

| Design | Method | Invalid train fraction (MCSE) | Invalid grid fraction (MCSE) |
|---|---|---:|---:|
| Log-linear, alocação padrão | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Log-linear, alocação padrão | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | 0.0000 (0.0000) | 0.0000 (0.0000) |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | 0.0000 (0.0000) | 0.0000 (0.0000) |

The structured log-mean models restrict the baseline as well as the relative curve, so this comparison does not perfectly isolate orthogonalization. S-learner clipping at 0.001 follows the historical benchmark and is recorded. Paired differences (candidate minus DML-linear) and their actual pair counts are in targeted_paired.csv.
