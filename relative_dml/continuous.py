"""A classification Q baseline and the manuscript's multiplicative DML."""
import warnings
import numpy as np
from scipy.special import logit
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from ._utils import features, training_data, learner, probability, check_clip, dose
from ._moment import fit_multiplicative_dml


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
    """Cross-fitted mu(t,x)=b(x)*exp((t-reference_dose)*[1,V(x)]@coef).

    X contains confounders; effect_features contains prespecified effect
    modifiers. Omit effect_features for a common relative slope. The slope
    is per unit of dose (elasticity only if dose is log-price). Baseline
    probability and conditional dose mean are learned outside each fold.
    se_ is the manuscript's iid sandwich at the nuisance intersection,
    not inference robust to arbitrary persistent nuisance misspecification.
    """
    def __init__(self, outcome_model=None, treatment_model=None, n_splits=3,
                 reference_dose=0.0, random_state=0):
        self.outcome_model = outcome_model
        self.treatment_model = treatment_model
        self.n_splits = n_splits
        self.reference_dose = reference_dose
        self.random_state = random_state

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
        self.dose_range_ = (t.min(), t.max())
        dose(self.reference_dose, len(X), self.dose_range_)
        if not np.isfinite(t).all() or self.dose_range_[0] == self.dose_range_[1]:
            raise ValueError('Treatment must be finite and vary.')
        if self.n_splits < 2 or np.bincount(y.astype(int)).min() < self.n_splits:
            raise ValueError('Need n_splits >= 2 and at least n_splits events and non-events.')
        self.n_features_in_ = X.shape[1]
        self.n_effect_features_ = 0 if effect_features is None else features(effect_features).shape[1]
        V = self._effect_design(X, effect_features)
        self.baseline_oof_ = np.empty(len(X))
        self.treatment_mean_oof_ = np.empty(len(X))
        self.fold_ids_ = np.empty(len(X), int)
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (tr, te) in enumerate(folds.split(X, y)):
            self.fold_ids_[te] = fold
            treatment = learner(self.treatment_model, False, self.random_state)
            treatment.fit(X[tr], t[tr])
            self.treatment_mean_oof_[te] = treatment.predict(X[te])
            outcome = learner(self.outcome_model, True, self.random_state)
            outcome.fit(np.column_stack([X[tr], t[tr]]), y[tr])
            at_reference = np.column_stack([X[te], np.full(len(te), self.reference_dose)])
            self.baseline_oof_[te] = probability(outcome, at_reference)
        self.estimate_ = fit_multiplicative_dml(
            t - self.reference_dose, y, V, self.baseline_oof_,
            self.treatment_mean_oof_ - self.reference_dose)
        self.coef_ = np.asarray(self.estimate_.theta)
        self.se_ = np.asarray(self.estimate_.se)
        self.cov_ = np.asarray(self.estimate_.cov)
        return self

    def predict_slope(self, X, effect_features=None):
        check_is_fitted(self, 'coef_')
        X = features(X, self.n_features_in_)
        return self._effect_design(X, effect_features) @ self.coef_

    def predict_ratio(self, X, treatment, reference, effect_features=None):
        slope = self.predict_slope(X, effect_features)
        t = dose(treatment, len(slope), self.dose_range_)
        t0 = dose(reference, len(slope), self.dose_range_)
        with np.errstate(over='raise', invalid='raise'):
            return np.exp((t - t0) * slope)

    def predict_lift(self, X, treatment, reference, effect_features=None):
        return self.predict_ratio(X, treatment, reference, effect_features) - 1
