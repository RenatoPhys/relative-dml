"""Exact moment checks and contracts for the common quadratic curvature."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from relative_dml import ContinuousDML, fit_multiplicative_dml
from relative_dml._moment import _fit_multiplicative_basis


def quadratic_cells():
    """Integrate rational treatment/outcome probabilities by repeated cells.

    Each of two profiles has 3600 observations. The allocation probabilities
    are (1/4, 1/2, 1/4) and (1/6, 1/3, 1/2), respectively. The relative curve
    has coefficients (log(2), log(3/2), log(2)); all six risks are rational.
    """
    groups = []
    for profile, counts, successes in [
        (0, [900, 1800, 900], [75, 150, 300]),
        (1, [600, 1200, 1800], [20, 60, 540]),
    ]:
        for centered_dose, count, converters in zip([-1., 0., 1.], counts, successes):
            groups.append(np.column_stack([
                np.full(count, profile), np.full(count, centered_dose),
                np.r_[np.ones(converters), np.zeros(count - converters)],
            ]))
    cells = np.vstack(groups)
    profile, d, y = cells.T
    H = np.column_stack([d, d * profile, d ** 2])
    mean = np.where(profile == 0, 0., 1 / 3)
    second_moment = np.where(profile == 0, 1 / 2, 2 / 3)
    m_H = np.column_stack([mean, mean * profile, second_moment])
    b = np.where(profile == 0, 1 / 12, 1 / 20)
    beta = np.log([2., 1.5, 2.])
    return profile, d, y, H, m_H, b, beta


@pytest.mark.parametrize('correct', ['baseline', 'moments', 'both'])
def test_quadratic_moment_exact_double_robustness(correct):
    _, _, y, H, true_m_H, true_b, beta = quadratic_cells()
    assert np.linalg.matrix_rank(H - true_m_H) == 3
    b_hat = true_b if correct in ['baseline', 'both'] else np.full(len(y), .2)
    m_H = true_m_H if correct in ['moments', 'both'] else np.zeros_like(H)

    # This verifies an exact population identity before exercising the solver.
    psi = (H - m_H) * (y * np.exp(-H @ beta) - b_hat)[:, None]
    assert_allclose(psi.mean(axis=0), 0, atol=1e-14)
    estimate = _fit_multiplicative_basis(H, m_H, y, b_hat)
    assert_allclose(estimate.theta, beta, atol=1e-9)
    assert estimate.moment_norm < 1e-10
    assert np.isfinite(estimate.se).all()
    assert (np.asarray(estimate.se) > 0).all()
    covariance = np.asarray(estimate.cov)
    assert covariance.shape == (3, 3)
    assert_allclose(covariance, covariance.T, atol=1e-12)
    assert_allclose(np.diag(covariance), np.asarray(estimate.se) ** 2)


def test_linear_moment_wrapper_preserves_generalized_moment_result():
    profile, d, y, H, m_H, b, _ = quadratic_cells()
    V = np.column_stack([np.ones(len(y)), profile])
    old_api = fit_multiplicative_dml(d, y, V, b, m_H[:, 0])
    basis_api = _fit_multiplicative_basis(H[:, :2], m_H[:, :2], y, b)
    assert_allclose(old_api.theta, basis_api.theta, atol=1e-12)
    assert_allclose(old_api.cov, basis_api.cov, atol=1e-12)


class AuditedMeanRegressor(RegressorMixin, BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.ids_ = set(X[:, 0])
        self.mean_ = np.mean(y)
        type(self).fits.append((X[:, 0].astype(int).copy(), np.asarray(y).copy()))
        return self

    def predict(self, X):
        assert self.ids_.isdisjoint(X[:, 0])
        return np.full(len(X), self.mean_)


class AuditedReferenceClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, reference_dose=2.):
        self.reference_dose = reference_dose

    def fit(self, X, y):
        self.ids_ = set(X[:, 0])
        self.classes_ = np.array([0., 1.])
        self.mean_ = np.mean(y)
        return self

    def predict_proba(self, X):
        assert self.ids_.isdisjoint(X[:, 0])
        assert_allclose(X[:, -1], self.reference_dose)
        return np.tile([1 - self.mean_, self.mean_], (len(X), 1))


def test_quadratic_nuisances_are_cross_fitted_at_training_reference():
    profile, d, y, _, _, _, _ = quadratic_cells()
    X = np.column_stack([np.arange(len(y)), profile])
    t = d + 2
    modifiers = profile[:, None]
    AuditedMeanRegressor.fits = []
    treatment = AuditedMeanRegressor()
    outcome = AuditedReferenceClassifier(reference_dose=2)
    model = ContinuousDML(outcome, treatment, reference_dose=2,
                          dose_degree=2).fit(X, t, y, modifiers)

    assert not hasattr(treatment, 'mean_')
    assert not hasattr(outcome, 'mean_')
    assert len(AuditedMeanRegressor.fits) == 2 * model.n_splits
    assert set(model.fold_ids_) == set(range(model.n_splits))
    for fold in range(model.n_splits):
        test = model.fold_ids_ == fold
        train = ~test
        assert_allclose(model.treatment_mean_oof_[test], t[train].mean())
        assert_allclose(model.treatment_second_moment_oof_[test], (d[train] ** 2).mean())
        assert_allclose(model.baseline_oof_[test], y[train].mean())
        # The second nuisance includes conditional variance; squaring the
        # first nuisance is wrong even with a nonzero training reference.
        assert np.all(model.treatment_second_moment_oof_[test]
                      > (model.treatment_mean_oof_[test] - 2) ** 2 + .1)
        fits = [(ids, target) for ids, target in AuditedMeanRegressor.fits
                if set(ids) == set(np.flatnonzero(train))]
        assert len(fits) == 2
        assert any(np.array_equal(target, t[ids]) for ids, target in fits)
        assert any(np.array_equal(target, d[ids] ** 2) for ids, target in fits)

    V = np.column_stack([np.ones(len(y)), profile])
    expected_mean = np.column_stack([
        (model.treatment_mean_oof_ - 2)[:, None] * V,
        model.treatment_second_moment_oof_,
    ])
    assert_allclose(model.basis_mean_oof_, expected_mean)
    assert model.coef_.shape == model.se_.shape == (3,)
    assert model.cov_.shape == (3, 3)
    assert_allclose(np.diag(model.cov_), model.se_ ** 2)
    assert np.isfinite(model.predict_ratio(X[:4], 3, 2, modifiers[:4])).all()


def test_default_degree_matches_explicit_linear_fit_and_old_slope_call():
    profile, d, y, _, _, _, _ = quadratic_cells()
    X = profile[:, None]
    options = dict(outcome_model=DummyClassifier(), treatment_model=DummyRegressor(),
                   reference_dose=2)
    default = ContinuousDML(**options).fit(X, d + 2, y, X)
    linear = ContinuousDML(**options, dose_degree=1).fit(X, d + 2, y, X)
    assert default.get_params()['dose_degree'] == 1
    assert clone(linear).dose_degree == 1
    assert_allclose(default.coef_, linear.coef_, atol=1e-12)
    assert_allclose(default.cov_, linear.cov_, atol=1e-12)
    assert_allclose(default.predict_slope(X[:4], X[:4]),
                    linear.predict_slope(X[:4], effect_features=X[:4], dose=2.5))
    assert_allclose(default.predict_ratio(X[:4], 3, 1, X[:4]),
                    np.exp(2 * default.predict_slope(X[:4], X[:4])))


def test_common_quadratic_curve_needs_no_effect_features():
    profile, d, y, _, _, _, _ = quadratic_cells()
    selected = profile == 0
    X = np.ones((selected.sum(), 1))
    model = ContinuousDML(DummyClassifier(), DummyRegressor(),
                          reference_dose=2, dose_degree=2).fit(X, d[selected] + 2, y[selected])
    assert model.coef_.shape == model.se_.shape == (2,)
    assert model.cov_.shape == (2, 2)
    assert model.basis_mean_oof_.shape == (len(X), 2)
    assert_allclose(model.predict_slope(X[:4]), np.full(4, model.coef_[0]))
    assert_allclose(model.predict_slope(X[:4], dose=2.5), model.coef_.sum())


@pytest.fixture
def quadratic_prediction_model():
    # Known coefficients isolate prediction algebra from nuisance estimation.
    model = ContinuousDML(reference_dose=2, dose_degree=2)
    model.coef_ = np.array([-.3, .4, .2])
    model.n_features_in_ = 2
    model.n_effect_features_ = 1
    model.dose_range_ = (1., 3.)
    model.reference_dose_ = 2.
    model.dose_degree_ = 2
    return model


def test_quadratic_ratios_share_one_curve_and_obey_composition(quadratic_prediction_model):
    model = quadratic_prediction_model
    X = np.zeros((4, 2))
    modifiers = np.array([[-1.], [0.], [.5], [1.]])
    treatment = np.array([1.1, 1.6, 2.4, 2.9])
    reference = np.array([2.2, 1.2, 2.7, 1.8])
    intermediate = 2.3
    linear_slope = -.3 + .4 * modifiers[:, 0]
    expected_log = ((treatment - reference) * linear_slope
                    + .2 * ((treatment - 2) ** 2 - (reference - 2) ** 2))
    ratio = model.predict_ratio(X, treatment, reference, modifiers)
    assert_allclose(np.log(ratio), expected_log, atol=1e-14)
    assert_allclose(model.predict_lift(X, treatment, reference, modifiers), ratio - 1)
    assert_allclose(model.predict_ratio(X, treatment, treatment, modifiers), 1)
    assert_allclose(ratio * model.predict_ratio(X, reference, treatment, modifiers), 1)
    assert_allclose(ratio, model.predict_ratio(X, treatment, intermediate, modifiers)
                    * model.predict_ratio(X, intermediate, reference, modifiers))
    assert_allclose(model.predict_ratio(X, 2.5, 1.5, modifiers), np.exp(linear_slope))


def test_quadratic_slope_is_dose_specific_log_risk_derivative(quadratic_prediction_model):
    model = quadratic_prediction_model
    X = np.zeros((4, 2))
    modifiers = np.array([[-1.], [0.], [.5], [1.]])
    treatment = np.array([1.1, 1.6, 2.4, 2.9])
    at_reference = -.3 + .4 * modifiers[:, 0]
    assert_allclose(model.predict_slope(X, modifiers), at_reference)
    assert_allclose(model.predict_slope(X, effect_features=modifiers, dose=2), at_reference)
    assert_allclose(model.predict_slope(X, modifiers, dose=2.5), at_reference + .2)
    eps = 1e-6
    numerical = (np.log(model.predict_ratio(X, treatment + eps, 2, modifiers))
                 - np.log(model.predict_ratio(X, treatment - eps, 2, modifiers))) / (2 * eps)
    assert_allclose(model.predict_slope(X, modifiers, dose=treatment), numerical, atol=1e-9)


@pytest.mark.parametrize('bad_degree', [0, 3, -1, 1.5, '2', None])
def test_unsupported_dose_degrees_fail_clearly(bad_degree):
    _, d, y, _, _, _, _ = quadratic_cells()
    with pytest.raises(ValueError, match='dose_degree'):
        ContinuousDML(DummyClassifier(), DummyRegressor(), dose_degree=bad_degree).fit(
            np.ones((len(y), 1)), d, y)


@pytest.mark.parametrize('values', [[0., 1.], [-1., 1.]])
def test_two_doses_cannot_identify_quadratic_curve(values):
    t = np.repeat(values, 600)
    y = np.r_[np.ones(60), np.zeros(540), np.ones(120), np.zeros(480)]
    model = ContinuousDML(DummyClassifier(), DummyRegressor(), dose_degree=2)
    with pytest.raises(RuntimeError, match='identif|condicionado|singular|rank'):
        model.fit(np.ones((len(y), 1)), t, y)


@pytest.mark.parametrize('bad_dose', [np.nan, np.inf, [1., 2.], [[2.], [2.], [2.], [2.]],
                                       [2., 2., 2., np.nan], .9, 3.1])
def test_quadratic_predictions_validate_dose_shape_finiteness_and_range(
        quadratic_prediction_model, bad_dose):
    model = quadratic_prediction_model
    X = np.zeros((4, 2))
    modifiers = np.zeros((4, 1))
    with pytest.raises(ValueError):
        model.predict_slope(X, modifiers, dose=bad_dose)
    with pytest.raises(ValueError):
        model.predict_ratio(X, bad_dose, 2, modifiers)
    with pytest.raises(ValueError):
        model.predict_ratio(X, 2, bad_dose, modifiers)


def test_quadratic_predictions_require_matching_features(quadratic_prediction_model):
    model = quadratic_prediction_model
    X = np.zeros((4, 2))
    with pytest.raises(ValueError, match='effect_features'):
        model.predict_slope(X)
    for modifiers in [np.zeros((4, 2)), np.zeros((3, 1)), np.full((4, 1), np.nan)]:
        with pytest.raises(ValueError):
            model.predict_ratio(X, 2.5, 2, modifiers)
    with pytest.raises(ValueError):
        model.predict_slope(np.zeros((4, 3)), np.zeros((4, 1)))


@pytest.mark.parametrize('invalid', [
    'H_vector', 'm_H_vector', 'missing_row', 'missing_column', 'nan_basis',
    'nan_mean', 'nan_baseline', 'nonbinary_y', 'bad_baseline', 'bad_start',
    'nan_start', 'infinite_start',
])
def test_generalized_moment_rejects_invalid_inputs(invalid):
    _, _, y, H, m_H, b, _ = quadratic_cells()
    start = None
    if invalid == 'H_vector':
        H = H[:, 0]
    elif invalid == 'm_H_vector':
        m_H = m_H[:, 0]
    elif invalid == 'missing_row':
        m_H = m_H[:-1]
    elif invalid == 'missing_column':
        m_H = m_H[:, :-1]
    elif invalid == 'nan_basis':
        H[0, 0] = np.nan
    elif invalid == 'nan_mean':
        m_H[0, 0] = np.nan
    elif invalid == 'nan_baseline':
        b[0] = np.nan
    elif invalid == 'nonbinary_y':
        y[0] = 2
    elif invalid == 'bad_baseline':
        b[0] = 1.1
    elif invalid == 'bad_start':
        start = np.zeros(2)
    elif invalid == 'nan_start':
        start = np.array([np.nan, 0., 0.])
    elif invalid == 'infinite_start':
        start = np.array([0., np.inf, 0.])
    with pytest.raises(ValueError):
        _fit_multiplicative_basis(H, m_H, y, b, start=start)
