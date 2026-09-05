"""A classification Q baseline and the manuscript's multiplicative DML."""
import warnings
import numpy as np
from scipy.special import logit
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from ._utils import features, training_data, learner, probability, check_clip, dose
from ._moment import _fit_multiplicative_basis


class ContinuousQLearner(BaseEstimator):
    """Joint density-ratio classification of converters versus all records.

    The balanced source-class odds equal mu(t,x)/P(Y=1); their ratio at two
    doses cancels the marginal prevalence. This is a plug-in Q construction,
    closely related to an S-learner; it has no DR or efficiency guarantee.
    The classifier must support sample_weight. Positive records appear in
    both sources, so they are not additional independent observations.
    """
    def __init__(self, classifier=None, clip=1e-3, random_state=0):
        self.classifier = classifier
        self.clip = clip
        self.random_state = random_state

    def fit(self, X, t, y):
        X, t, y = training_data(X, t, y)
        t = np.asarray(t, float)
        check_clip(self.clip)
        self.dose_range_ = (t.min(), t.max())
        if not np.isfinite(t).all() or self.dose_range_[0] == self.dose_range_[1]:
            raise ValueError('Continuous treatment must be finite and vary.')
        self.n_features_in_ = X.shape[1]
        design = np.column_stack([X, t])
        converters = design[y == 1]
        n, nc = len(X), len(converters)
        # Equal total source weights; mean sample weight one preserves the
        # default regularization scale. Split original records before tuning.
        weights = np.r_[np.full(nc, (n + nc) / (2 * nc)),
                        np.full(n, (n + nc) / (2 * n))]
        self.classifier_ = learner(self.classifier, True, self.random_state)
        self.classifier_.fit(np.vstack([converters, design]),
                             np.r_[np.ones(nc), np.zeros(n)], sample_weight=weights)
        return self

    def predict_ratio(self, X, treatment, reference):
        check_is_fitted(self, 'classifier_')
        X = features(X, self.n_features_in_)
        def log_odds(value):
            t = dose(value, len(X), self.dose_range_)
            p = probability(self.classifier_, np.column_stack([X, t]))
            if np.any((p < self.clip) | (p > 1 - self.clip)):
                warnings.warn('Source-class probability clipped; inspect ratio stability.',
                              RuntimeWarning, stacklevel=2)
            return logit(np.clip(p, self.clip, 1 - self.clip))
        return np.exp(log_odds(treatment) - log_odds(reference))

    def predict_lift(self, X, treatment, reference):
        return self.predict_ratio(X, treatment, reference) - 1


class ContinuousDML(BaseEstimator):
    """Cross-fitted multiplicative effects with optional common curvature.

    X contains confounders; effect_features contains prespecified effect
    modifiers. Omit effect_features for a common relative slope. The slope
    is per unit of dose (elasticity only if dose is log-price).
    With d=t-reference_dose, g=d*[1,V(x)]@theta (+ kappa*d**2 when
    dose_degree=2). coef_ orders theta first, then the common kappa.
    Baseline, dose mean and (for degree 2) centered second moment are
    learned outside each fold. cov_ and se_ use the iid sandwich,
    not inference robust to arbitrary persistent nuisance misspecification.
    """
    def __init__(self, outcome_model=None, treatment_model=None, n_splits=3,
                 reference_dose=0.0, random_state=0, dose_degree=1):
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.n_splits = n_splits
        self.reference_dose = reference_dose
        self.random_state = random_state
        self.dose_degree = dose_degree

    def _effect_design(self, X, effect_features):
        if effect_features is None:
            if self.n_effect_features_:
                raise ValueError('Supply the same effect_features columns used during fit.')
            return np.ones((len(X), 1))
        V = features(effect_features, self.n_effect_features_)
        if len(V) != len(X):
            raise ValueError('effect_features must have one row per observation.')
        return np.column_stack([np.ones(len(X)), V])

    def fit(self, X, t, y, effect_features=None):
        X, t, y = training_data(X, t, y)
        t = np.asarray(t, float)
        if not np.isscalar(self.dose_degree) or self.dose_degree not in (1, 2):
            raise ValueError('dose_degree must be 1 or 2.')
        if np.asarray(self.reference_dose).ndim != 0:
            raise ValueError('reference_dose must be a finite scalar.')
        self.dose_range_ = (t.min(), t.max())
        dose(self.reference_dose, len(X), self.dose_range_)
        if not np.isfinite(t).all() or self.dose_range_[0] == self.dose_range_[1]:
            raise ValueError('Treatment must be finite and vary.')
        if self.dose_degree == 2 and len(np.unique(t)) < 3:
            raise RuntimeError('Quadratic effect is not identified with fewer than three doses (identificação).')
        if self.n_splits < 2 or np.bincount(y.astype(int)).min() < self.n_splits:
            raise ValueError('Need n_splits >= 2 and at least n_splits events and non-events.')
        self.n_features_in_ = X.shape[1]
        self.n_effect_features_ = 0 if effect_features is None else features(effect_features).shape[1]
        V = self._effect_design(X, effect_features)
        self.dose_degree_ = self.dose_degree
        self.reference_dose_ = float(self.reference_dose)
        with np.errstate(over='raise', invalid='raise'):
            centered = t-self.reference_dose_
            H = centered[:, None]*V
            if self.dose_degree_ == 2:
                squared = centered**2
                H = np.column_stack([H, squared])
                self.treatment_second_moment_oof_ = np.empty(len(X))
            elif hasattr(self, 'treatment_second_moment_oof_'):
                del self.treatment_second_moment_oof_
        self.baseline_oof_ = np.empty(len(X))
        self.treatment_mean_oof_ = np.empty(len(X))
        self.fold_ids_ = np.empty(len(X), int)
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (tr, te) in enumerate(folds.split(X, y)):
            self.fold_ids_[te] = fold
            treatment = learner(self.treatment_model, False, self.random_state)
            treatment.fit(X[tr], t[tr])
            self.treatment_mean_oof_[te] = treatment.predict(X[te])
            if self.dose_degree_ == 2:
                second_moment = learner(self.treatment_model, False, self.random_state)
                second_moment.fit(X[tr], squared[tr])
                self.treatment_second_moment_oof_[te] = second_moment.predict(X[te])
            outcome = learner(self.outcome_model, True, self.random_state)
            outcome.fit(np.column_stack([X[tr], t[tr]]), y[tr])
            at_reference = np.column_stack([X[te], np.full(len(te), self.reference_dose)])
            self.baseline_oof_[te] = probability(outcome, at_reference)
        self.basis_mean_oof_ = (self.treatment_mean_oof_-self.reference_dose_)[:, None]*V
        if self.dose_degree_ == 2:
            self.basis_mean_oof_ = np.column_stack(
                [self.basis_mean_oof_, self.treatment_second_moment_oof_])
        self.estimate_ = _fit_multiplicative_basis(
            H, self.basis_mean_oof_, y, self.baseline_oof_)
        self.coef_ = np.asarray(self.estimate_.theta)
        self.se_ = np.asarray(self.estimate_.se)
        self.cov_ = np.asarray(self.estimate_.cov)
        return self

    def predict_slope(self, X, effect_features=None, *, dose=None):
        """Derivative of log-risk at dose; defaults to the training reference."""
        check_is_fitted(self, 'coef_')
        X = features(X, self.n_features_in_)
        V = self._effect_design(X, effect_features)
        # Keep the existing effect_features positional argument unchanged.
        from ._utils import dose as checked_dose
        t = checked_dose(self.reference_dose_ if dose is None else dose,
                         len(X), self.dose_range_)
        with np.errstate(over='raise', invalid='raise'):
            slope = V @ self.coef_[:V.shape[1]]
            if self.dose_degree_ == 2:
                slope = slope + 2*self.coef_[-1]*(t-self.reference_dose_)
            return slope

    def _log_effect(self, V, t):
        centered = t-self.reference_dose_
        g = centered*(V @ self.coef_[:V.shape[1]])
        if self.dose_degree_ == 2:
            g = g + self.coef_[-1]*centered**2
        return g

    def predict_ratio(self, X, treatment, reference, effect_features=None):
        check_is_fitted(self, 'coef_')
        X = features(X, self.n_features_in_)
        V = self._effect_design(X, effect_features)
        t = dose(treatment, len(X), self.dose_range_)
        t0 = dose(reference, len(X), self.dose_range_)
        with np.errstate(over='raise', invalid='raise'):
            return np.exp(self._log_effect(V, t) - self._log_effect(V, t0))

    def predict_lift(self, X, treatment, reference, effect_features=None):
        return self.predict_ratio(X, treatment, reference, effect_features) - 1
