import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier
from relative_dml import (DiscreteQLearner, DiscreteDML, ContinuousQLearner,
                          ContinuousDML, fit_multiplicative_dml)
from relative_dml.discrete import _aipw


def arm_table():
    # Saturated finite population: exactly 10%, 20%, 30% conversion.
    a = np.repeat(['control', 'high', 'low'], 600)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(600-k)] for k in [60, 120, 180]])
    return np.ones((len(a), 1)), a, y


@pytest.mark.parametrize('estimator', [
    DiscreteQLearner(DummyClassifier(), DummyClassifier()),
    DiscreteDML(DummyClassifier(), DummyClassifier(), DummyRegressor()),
])
def test_multiclass_empirical_ratios_and_labels(estimator):
    X, a, y = arm_table()
    model = clone(estimator).fit(X, a, y)
    assert_allclose(model.predict_ratio(X[:3], 'high', 'control'), 2, atol=1e-12)
    assert_allclose(model.predict_lift(X[:3], 'low', 'control'), 2, atol=1e-12)
    assert_allclose(model.predict_ratio(X[:3], 'control', 'control'), 1)
    assert_allclose(model.predict_ratio(X[:3], 'control', 'high'), .5)
    with pytest.raises(ValueError):
        model.predict_ratio(X[:3], 'missing', 'control')


@pytest.mark.parametrize('correct', ['propensity', 'outcome', 'both', 'neither'])
def test_aipw_exact_double_robustness(correct):
    e, mu = .3, .2
    eh = e if correct in ['propensity', 'both'] else .6
    mh = mu if correct in ['outcome', 'both'] else .4
    # Exact expectation over (A,Y); Y outside this arm does not enter the signal.
    signal = _aipw(np.array([1., 0., 0.]), np.array([1, 1, 0]), mh, eh)
    expected = np.dot([e*mu, e*(1-mu), 1-e], signal)
    assert_allclose(expected - mu, (1-e/eh)*(mh-mu), atol=1e-14)
    if correct != 'neither':
        assert_allclose(expected, mu, atol=1e-14)
    else:
        assert abs(expected - mu) > .05


def test_first_order_ratio_correction_is_not_exactly_dr():
    # Correct propensity cancels in expectation, yet a wrong baseline
    # leaves a nonzero remainder for the one-step ratio pseudo-outcome.
    mu0, mu1, m0, m1 = .1, .2, .2, .3
    ratio = mu1/mu0
    initial = m1/m0
    corrected = initial + (mu1-m1)/m0 - initial*(mu0-m0)/m0
    assert_allclose(corrected, 1.75)
    assert_allclose(corrected-ratio, (1-mu0/m0)*(initial-ratio))


@pytest.mark.parametrize('correct', ['mean', 'baseline', 'both'])
def test_multiplicative_moment_exact_dr(correct):
    # True mu(0)=.1, mu(1)=.2, so slope=log(2). Repeated cells
    # integrate the binary outcome exactly, without Monte Carlo tolerance.
    a = np.repeat([0., 1.], 1000)
    y = np.r_[np.ones(100), np.zeros(900), np.ones(200), np.zeros(800)]
    b = np.full(2000, .1 if correct in ['baseline', 'both'] else .3)
    ma = np.full(2000, .5 if correct in ['mean', 'both'] else .2)
    est = fit_multiplicative_dml(a, y, np.ones((2000, 1)), b, ma)
    assert_allclose(est.theta, [np.log(2)], atol=1e-10)
    assert est.se[0] > 0 and est.moment_norm < 1e-10


def test_continuous_q_joint_density_identity():
    # A two-dose saturated example verifies the implementation's exact
    # joint-density formula, including the duplicated positive records.
    X, labels, y = arm_table()
    a = np.repeat([-1., 0., 1.], 600)
    model = ContinuousQLearner(DecisionTreeClassifier(max_depth=2)).fit(X, a, y)
    assert_allclose(model.predict_ratio(X[:5], 1, -1), 3, atol=1e-12)
    assert_allclose(model.predict_lift(X[:5], 0, -1), 1, atol=1e-12)
    assert_allclose(model.predict_ratio(X[:5], [-1, 0, 1, 0, -1], -1), [1, 2, 3, 2, 1])
    with pytest.raises(ValueError, match='outside'):
        model.predict_ratio(X[:5], 2, 0)


class HonestClassifier(ClassifierMixin, BaseEstimator):
    """Reject any nuisance prediction on a training observation's id."""
    def fit(self, X, y):
        self.ids_ = set(X[:, 0])
        self.classes_ = np.unique(y)
        self.probs_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def predict_proba(self, X):
        assert self.ids_.isdisjoint(X[:, 0])
        return np.tile(self.probs_, (len(X), 1))


class HonestRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.ids_ = set(X[:, 0])
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        assert self.ids_.isdisjoint(X[:, 0])
        return np.full(len(X), self.mean_)


def test_discrete_cross_fitting_has_no_nuisance_leakage():
    X, a, y = arm_table()
    X[:, 0] = np.arange(len(X))
    model = DiscreteDML(HonestClassifier(), HonestClassifier(), DummyRegressor()).fit(X, a, y)
    assert set(model.fold_ids_) == {0, 1, 2}
    assert np.isfinite(model.pseudo_outcomes_).all()
    assert np.any(model.pseudo_outcomes_ < 0)  # Signals are not silently clipped.


def test_continuous_cross_fitting_and_nonzero_reference():
    rng = np.random.default_rng(913)
    n = 12000
    X = np.arange(n, dtype=float)[:, None]
    a = rng.uniform(1, 3, n)
    y = rng.binomial(1, .2*np.exp(-.4*(a-2)))
    model = ContinuousDML(HonestClassifier(), HonestRegressor(), reference_dose=2).fit(X, a, y)
    assert abs(model.coef_[0] + .4) < .12
    assert set(model.fold_ids_) == {0, 1, 2}
    assert_allclose(model.predict_ratio(X[:4], 2.5, 1.5), np.exp(model.coef_[0]))
    assert_allclose(model.predict_ratio(X[:4], 2, 2), 1)
    assert_allclose(model.predict_lift(X[:4], 2.5, 1.5), np.exp(model.coef_[0])-1)


def test_continuous_heterogeneity_recovery_and_feature_contract():
    rng = np.random.default_rng(39)
    n = 24000
    X = rng.choice([-1., 1.], (n, 1))
    a = rng.uniform(-1, 1, n)
    y = rng.binomial(1, .2*np.exp(a*(-.3-.25*X[:, 0])))
    model = ContinuousDML(treatment_model=DummyRegressor()).fit(X, a, y, X)
    assert_allclose(model.coef_, [-.3, -.25], atol=.12)
    with pytest.raises(ValueError, match='effect_features'):
        model.predict_slope(X[:4])
    with pytest.raises(ValueError):
        model.predict_slope(X[:4], np.ones((4, 2)))
    assert_allclose(model.predict_ratio(X[:4], .5, -.5, X[:4]),
                    np.exp(model.predict_slope(X[:4], X[:4])))


@pytest.mark.parametrize('estimator', [DiscreteQLearner(), DiscreteDML(),
                                       ContinuousQLearner(), ContinuousDML()])
def test_input_validation(estimator):
    X, a, y = arm_table()
    a = np.repeat([-1., 0., 1.], 600)
    for bad_y in [np.full(len(y), 2), np.zeros(len(y)), y[:, None], y[:-1]]:
        with pytest.raises(ValueError):
            clone(estimator).fit(X, a, bad_y)
    with pytest.raises(ValueError):
        clone(estimator).fit(X, np.full(len(a), np.nan), y)
    with pytest.raises(ValueError):
        clone(estimator).fit(X, np.ones(len(a)), y)


def test_sparse_arms_and_unidentified_design_fail_clearly():
    X, a, y = arm_table()
    y[a == 'high'] = 0
    for model in [DiscreteQLearner(), DiscreteDML()]:
        with pytest.raises(ValueError):
            model.fit(X, a, y)
    with pytest.raises(RuntimeError, match='condicionado'):
        fit_multiplicative_dml(np.tile([0, 1], 20), np.tile([0, 1], 20),
                               np.ones((40, 2)), np.full(40, .2), np.full(40, .5))


def test_response_clipping_is_visible():
    X, a, y = arm_table()
    model = DiscreteDML(DummyClassifier(), DummyClassifier(),
                        DummyRegressor(strategy='constant', constant=-.1)).fit(X, a, y)
    with pytest.warns(RuntimeWarning, match='clipped'):
        assert_allclose(model.predict_response(X[:2], 'control'), model.clip)
