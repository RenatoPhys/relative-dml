"""Deterministic checks of the additional experiment's reporting protocol."""
from dataclasses import replace
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.exceptions import ConvergenceWarning

pd = pytest.importorskip('pandas')  # Experiment dependencies remain optional.
from continuous_q_paper import targeted_benchmarks as benchmark


def test_log_mean_basis_and_unclipped_invalid_predictions():
    X = np.array([[-1., -.5], [1., .5]])
    assert_allclose(benchmark.log_mean_design(X, [-.4, .8], 2),
                    [[-1., -.5, -.4, .4, .16], [1., .5, .8, .8, .64]])

    class FixedLogMean:
        coef_ = np.array([0., 0., 1., 0.])

        def predict(self, design):
            return 2*np.exp(design[:, 2])

    ratios, diagnostics = benchmark.predict_ratios(FixedLogMean(), X, 'Log-mean-linear')
    assert_allclose(ratios, np.exp(benchmark.TARGETS)[:, None]*np.ones((1, len(X))))
    assert diagnostics['invalid_grid_fraction'] == .8
    assert benchmark.invalid_fraction([0., 1., -1., 2., np.nan]) == .6


def test_constant_true_effect_has_no_spearman():
    scenario = replace(benchmark.SCENARIOS[0], heterogeneity=0.)
    X = np.array([[-1., -.5], [1., .5]])
    truths = np.array([np.full(len(X), np.exp(t*scenario.effect))
                       for t in benchmark.TARGETS])
    result = benchmark.score_ratios(scenario, truths, X)
    assert result['log_ratio_rmse'] == 0.
    assert result['lift_mae'] == 0.
    assert np.isnan(result['rank_spearman'])


@pytest.mark.parametrize('heterogeneity_sign, expected_rank', [(1., 1.), (-1., -1.)])
def test_curved_log_mean_preserves_exact_profile_ties(heterogeneity_sign, expected_rank):
    scenario = benchmark.SCENARIOS[2]
    X = np.array([[-1., -.77], [1., -.5], [-1., .01],
                  [1., .37], [-1., .93], [1., .999]])

    class CurvedLogMean:
        coef_ = np.array([.65, .4, scenario.effect,
                          heterogeneity_sign*scenario.heterogeneity, scenario.curvature])

        def predict(self, design):
            return scenario.baseline*np.exp(design @ self.coef_)

    ratios, _ = benchmark.predict_ratios(CurvedLogMean(), X, 'Log-mean-quadratic')
    for profile in (-1., 1.):
        values = ratios[:, X[:, 0] == profile]
        assert np.array_equal(values, np.repeat(values[:, :1], values.shape[1], axis=1))
    result = benchmark.score_ratios(scenario, ratios, X)
    assert result['rank_spearman'] == expected_rank
    if heterogeneity_sign == 1.:
        assert result['log_ratio_rmse'] < 1e-15


def test_every_failed_method_is_retained(monkeypatch):
    class FailedFit:
        def fit(self, *args, **kwargs):
            raise RuntimeError('deliberate fit failure')

    monkeypatch.setattr(benchmark, 'make_models',
                        lambda seed: [(method, FailedFit()) for method in benchmark.METHODS])
    rows = benchmark.run_replication(benchmark.SCENARIOS[0], 0, 100, 100, 731905)
    assert len(rows) == 5
    assert all(not row['success'] and not row['fit_success'] for row in rows)
    assert all(row['error'] == 'RuntimeError: deliberate fit failure' for row in rows)
    assert len({row['converters'] for row in rows}) == 1
    summary = benchmark.summarize(pd.DataFrame(rows))
    assert summary.failures.sum() == 5
    assert summary.successes.sum() == 0
    assert summary.log_ratio_rmse.isna().all()


def test_convergence_warning_counts_as_failure_without_dropping_diagnostics(monkeypatch):
    class UnconvergedLogMean:
        def fit(self, design, y):
            self.coef_ = np.zeros(design.shape[1])
            self.intercept_ = np.log(.1)
            self.n_iter_ = 2000
            warnings.warn('iteration budget exhausted', ConvergenceWarning)
            return self

        def predict(self, design):
            return np.full(len(design), .1)

    monkeypatch.setattr(benchmark, 'make_models',
                        lambda seed: [('Log-mean-linear', UnconvergedLogMean())])
    row = benchmark.run_replication(benchmark.SCENARIOS[0], 0, 100, 100, 731905)[0]
    assert row['fit_success'] and not row['success']
    assert row['convergence_warning']
    assert row['poisson_iterations'] == 2000
    assert row['invalid_train_fraction'] == 0.
    assert row['invalid_grid_fraction'] == 0.
    assert 'ConvergenceWarning' in row['error']
    assert 'iteration budget exhausted' in row['warnings']


def test_paired_mcse_counts_failures_and_undefined_metrics():
    rows = []
    for rep in range(3):
        for method in benchmark.METHODS:
            value = float(rep) if method == 'DML-linear' else float(rep+1+2*rep)
            rows.append(dict(scenario='design', label='Design', method=method, rep=rep,
                             success=not (rep == 1 and method == 'DML-quadratic'),
                             warnings='', convergence_warning=False,
                             log_ratio_rmse=value, lift_mae=value, lift_bias=value,
                             rank_spearman=np.nan))
    df = pd.DataFrame(rows)
    pairs = benchmark.paired_summary(df)
    row = pairs[(pairs.method == 'DML-quadratic') & (pairs.metric == 'log_ratio_rmse')].iloc[0]
    assert row.pairs == 2 and row.excluded_pairs == 1 and row.total_reps == 3
    assert row.difference == 3.
    assert row.difference_mcse == pytest.approx(2.)
    assert pairs[pairs.metric == 'rank_spearman'].pairs.eq(0).all()
    summary = benchmark.summarize(df)
    result = summary[summary.method == 'DML-quadratic'].iloc[0]
    assert result.failures == 1 and result.log_ratio_rmse_n == 2


def test_output_cannot_overwrite_a_previous_run(tmp_path):
    out = tmp_path/'results'
    benchmark.prepare_output(out)
    marker = out/'old.csv'
    marker.write_text('historical data', encoding='utf-8')
    with pytest.raises(ValueError, match='not overwritten'):
        benchmark.prepare_output(out)
    assert marker.read_text(encoding='utf-8') == 'historical data'


def test_seed_streams_and_curved_causal_surface_are_fixed():
    train_seed, test_seed = benchmark.replication_seeds(731905, 0)
    assert train_seed != test_seed
    assert (train_seed, test_seed) == benchmark.replication_seeds(731905, 0)
    assert train_seed != benchmark.replication_seeds(731905, 1)[0]
    curved = benchmark.SCENARIOS[1:]
    X, t, _ = benchmark.generate(curved[0], 100, np.random.default_rng(train_seed))
    for scenario in curved[1:]:
        assert_allclose(benchmark.response(scenario, X, t), benchmark.response(curved[0], X, t))
        X_other, _, _ = benchmark.generate(scenario, 100, np.random.default_rng(train_seed))
        assert_allclose(X_other, X)
