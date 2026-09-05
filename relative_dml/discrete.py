"""Binary/multiclass Q identity and cross-fitted AIPW response means."""
import warnings
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from ._utils import features, training_data, learner, probability, check_clip


class DiscreteQLearner(BaseEstimator):
    """Q identity: (q_t / e_t) / (q_ref / e_ref), where q uses converters.

    Models must be cloneable probabilistic classifiers. Treatments may be
    numeric or string labels. This plug-in estimator is not doubly robust.
    """
    def __init__(self, propensity_model=None, converter_model=None, clip=1e-3,
                 random_state=0):
        self.propensity_model = propensity_model
        self.converter_model = converter_model
        self.clip = clip
        self.random_state = random_state

    def fit(self, X, t, y):
        X, t, y = training_data(X, t, y)
        check_clip(self.clip)
        self.classes_ = np.unique(t)
        if len(self.classes_) < 2 or not np.array_equal(np.unique(t[y == 1]), self.classes_):
            raise ValueError('Need at least two treatment arms and converters in every arm.')
        self.n_features_in_ = X.shape[1]
        self.propensity_model_ = learner(self.propensity_model, True, self.random_state)
        self.converter_model_ = learner(self.converter_model, True, self.random_state)
        self.propensity_model_.fit(X, t)
        self.converter_model_.fit(X[y == 1], t[y == 1])
        return self

    def predict_ratio(self, X, treatment, reference):
        check_is_fitted(self, 'converter_model_')
        X = features(X, self.n_features_in_)
        def adjusted(arm):
            q = probability(self.converter_model_, X, arm)
            e = probability(self.propensity_model_, X, arm)
            if np.any((q < self.clip) | (e < self.clip)):
                warnings.warn('Small arm probability clipped; check converters and overlap.',
                              RuntimeWarning, stacklevel=2)
            return np.maximum(q, self.clip) / np.maximum(e, self.clip)
        return adjusted(treatment) / adjusted(reference)

    def predict_lift(self, X, treatment, reference):
        return self.predict_ratio(X, treatment, reference) - 1


def _aipw(y, observed, mu, propensity):
    return mu + observed * (y - mu) / propensity


class DiscreteDML(BaseEstimator):
    """Cross-fit AIPW signals, regress each arm's mean, then take their ratio.

    Consistency requires correct propensity OR all relevant outcome models,
    consistent final mean regressions, overlap and a positive reference mean.
    Pseudo-outcomes are never clipped or divided. Final response estimates
    are clipped to [clip, 1] with a warning. No pointwise CIs are supplied.
    """
    def __init__(self, propensity_model=None, outcome_model=None, final_model=None,
                 n_splits=3, clip=1e-3, random_state=0):
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.final_model = final_model
        self.n_splits = n_splits
        self.clip = clip
        self.random_state = random_state

    def fit(self, X, t, y):
        X, t, y = training_data(X, t, y)
        check_clip(self.clip)
        self.classes_, encoded = np.unique(t, return_inverse=True)
        strata = 2 * encoded + y.astype(int)
        counts = np.bincount(strata, minlength=2 * len(self.classes_))
        if len(self.classes_) < 2 or self.n_splits < 2 or counts.min() < self.n_splits:
            raise ValueError('Need n_splits >= 2 and at least n_splits events and non-events per arm.')
        self.n_features_in_ = X.shape[1]
        self.pseudo_outcomes_ = np.empty((len(X), len(self.classes_)))
        self.fold_ids_ = np.empty(len(X), int)
        clipped = 0
        folds = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for fold, (tr, te) in enumerate(folds.split(X, strata)):
            self.fold_ids_[te] = fold
            prop = learner(self.propensity_model, True, self.random_state)
            prop.fit(X[tr], t[tr])
            for j, arm in enumerate(self.classes_):
                rows = tr[t[tr] == arm]
                outcome = learner(self.outcome_model, True, self.random_state)
                outcome.fit(X[rows], y[rows])
                mu = probability(outcome, X[te])
                e = probability(prop, X[te], arm)
                clipped += np.count_nonzero(e < self.clip)
                self.pseudo_outcomes_[te, j] = _aipw(
                    y[te], t[te] == arm, mu, np.maximum(e, self.clip))
        self.propensity_clip_fraction_ = clipped / self.pseudo_outcomes_.size
        if clipped:
            warnings.warn('Out-of-fold propensities clipped; inspect propensity_clip_fraction_.',
                          RuntimeWarning, stacklevel=2)
        self.final_models_ = []
        for j in range(len(self.classes_)):
            # AIPW signals are noisier than outcomes, especially with rare
            # events: use a shallower, more regularized final regression.
            default_final = HistGradientBoostingRegressor(
                max_iter=30, learning_rate=.05, max_leaf_nodes=4,
                min_samples_leaf=200, l2_regularization=10,
                early_stopping=False, random_state=self.random_state)
            model = learner(self.final_model if self.final_model is not None else default_final,
                            False, self.random_state)
            model.fit(X, self.pseudo_outcomes_[:, j])
            self.final_models_.append(model)
        return self

    def predict_response(self, X, treatment):
        check_is_fitted(self, 'final_models_')
        X = features(X, self.n_features_in_)
        index = np.flatnonzero(self.classes_ == treatment)
        if len(index) != 1:
            raise ValueError(f'Unknown treatment arm: {treatment!r}.')
        mu = self.final_models_[index[0]].predict(X)
        if np.any((mu < self.clip) | (mu > 1)):
            warnings.warn('Estimated response clipped to [clip, 1]; ratios may be unstable.',
                          RuntimeWarning, stacklevel=2)
        return np.clip(mu, self.clip, 1)

    def predict_ratio(self, X, treatment, reference):
        return self.predict_response(X, treatment) / self.predict_response(X, reference)

    def predict_lift(self, X, treatment, reference):
        return self.predict_ratio(X, treatment, reference) - 1
