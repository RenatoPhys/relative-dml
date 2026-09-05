# Targeted benchmark

Entries are mean (Monte Carlo SE). Error metrics and coefficients use successful runs; every failure remains in the raw data and counts below. Diagnostic fractions use all runs where available. NaN denotes an undefined quantity or unavailable Monte Carlo SE.

| Design | Method | RMSE log-ratio | MAE lift | Spearman | Failures/reps | Warnings |
|---|---|---:|---:|---:|---:|---:|
| Log-linear, alocação padrão | DML-linear | 0.0808 (0.0089) | 0.0687 (0.0070) | 1.0000 (0.0000) | 0/30 | 0 |
| Log-linear, alocação padrão | DML-quadratic | 0.1067 (0.0090) | 0.0909 (0.0076) | 1.0000 (0.0000) | 0/30 | 0 |
| Log-linear, alocação padrão | Log-mean-linear | 0.0829 (0.0092) | 0.0712 (0.0074) | 1.0000 (0.0000) | 0/30 | 0 |
| Log-linear, alocação padrão | Log-mean-quadratic | 0.1041 (0.0090) | 0.0879 (0.0078) | 0.9333 (0.0667) | 0/30 | 0 |
| Log-linear, alocação padrão | S-learner | 0.3358 (0.0153) | 0.2944 (0.0239) | 0.4483 (0.0646) | 0/30 | 0 |
| Curvo, dose aleatorizada uniforme | DML-linear | 0.3782 (0.0007) | 0.4185 (0.0012) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | 0.0794 (0.0041) | 0.0965 (0.0059) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | 0.3862 (0.0015) | 0.3996 (0.0018) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | 0.0736 (0.0046) | 0.0885 (0.0058) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, dose aleatorizada uniforme | S-learner | 0.3165 (0.0142) | 0.3903 (0.0239) | 0.3983 (0.0764) | 0/30 | 0 |
| Curvo, alocação padrão | DML-linear | 0.3918 (0.0023) | 0.4291 (0.0014) | 0.4667 (0.1642) | 0/30 | 0 |
| Curvo, alocação padrão | DML-quadratic | 0.0754 (0.0057) | 0.0968 (0.0079) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, alocação padrão | Log-mean-linear | 0.3953 (0.0038) | 0.4178 (0.0017) | 0.4667 (0.1642) | 0/30 | 0 |
| Curvo, alocação padrão | Log-mean-quadratic | 0.0705 (0.0055) | 0.0877 (0.0067) | 1.0000 (0.0000) | 0/30 | 0 |
| Curvo, alocação padrão | S-learner | 0.3243 (0.0179) | 0.4139 (0.0389) | 0.4061 (0.0758) | 0/30 | 0 |
| Curvo, pouco overlap | DML-linear | 0.5037 (0.0125) | 0.4730 (0.0097) | -1.0000 (0.0000) | 0/30 | 0 |
| Curvo, pouco overlap | DML-quadratic | 0.1478 (0.0214) | 0.1906 (0.0246) | 0.6000 (0.1486) | 0/30 | 0 |
| Curvo, pouco overlap | Log-mean-linear | 0.5363 (0.0148) | 0.4944 (0.0135) | -1.0000 (0.0000) | 0/30 | 0 |
| Curvo, pouco overlap | Log-mean-quadratic | 0.1363 (0.0166) | 0.1702 (0.0192) | 0.6667 (0.1384) | 0/30 | 0 |
| Curvo, pouco overlap | S-learner | 0.3519 (0.0154) | 0.4055 (0.0172) | 0.1054 (0.0993) | 0/30 | 0 |

## Linear heterogeneity under curved responses

These are fitted coefficients of X1 in the linear dose term. For a model omitting curvature they are allocation-dependent summaries, not automatically the structural heterogeneity. No structural-parameter coverage calculation is made.

| Design | Method | theta1 (MCSE) | Mean marginal SE | Successful runs |
|---|---|---:|---:|---:|
| Curvo, dose aleatorizada uniforme | DML-linear | -0.2029 (0.0135) | 0.0666 | 30/30 |
| Curvo, dose aleatorizada uniforme | DML-quadratic | -0.2007 (0.0137) | 0.0680 | 30/30 |
| Curvo, dose aleatorizada uniforme | Log-mean-linear | -0.2523 (0.0168) | nan | 30/30 |
| Curvo, dose aleatorizada uniforme | Log-mean-quadratic | -0.2027 (0.0138) | nan | 30/30 |
| Curvo, alocação padrão | DML-linear | -0.0415 (0.0136) | 0.0725 | 30/30 |
| Curvo, alocação padrão | DML-quadratic | -0.2082 (0.0134) | 0.0771 | 30/30 |
| Curvo, alocação padrão | Log-mean-linear | -0.0526 (0.0176) | nan | 30/30 |
| Curvo, alocação padrão | Log-mean-quadratic | -0.2066 (0.0138) | nan | 30/30 |
| Curvo, pouco overlap | DML-linear | 0.3088 (0.0279) | 0.1306 | 30/30 |
| Curvo, pouco overlap | DML-quadratic | -0.1960 (0.0407) | 0.1866 | 30/30 |
| Curvo, pouco overlap | Log-mean-linear | 0.3758 (0.0305) | nan | 30/30 |
| Curvo, pouco overlap | Log-mean-quadratic | -0.1912 (0.0349) | nan | 30/30 |

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
