"""Small shared input checks and default learners."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.validation import check_array


def features(X, n_features=None):
    X = check_array(X, dtype=float)
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError('X has a different number of features than during fit.')
    return X


def training_data(X, t, y):
    X = features(X)
    t, y = np.asarray(t), np.asarray(y, float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(X) or len(y) != len(X):
        raise ValueError('t and y must be vectors with one entry per row of X.')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('y must be binary and contain both 0 and 1.')
    if t.dtype.kind in 'biuf' and not np.isfinite(t).all():
        raise ValueError('Treatment must be finite.')
    if any(value is None or value != value for value in t):
        raise ValueError('Treatment must not contain missing values.')
    return X, t, y


def learner(model, classifier, random_state):
    if model is not None:
        return clone(model)
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    return cls(max_iter=100, max_leaf_nodes=8, min_samples_leaf=50,
               l2_regularization=2, early_stopping=False, random_state=random_state)


def probability(model, X, label=1):
    index = np.flatnonzero(model.classes_ == label)
    if len(index) != 1:
        raise ValueError(f'Training fold has no observations of class {label!r}.')
    return model.predict_proba(X)[:, index[0]]


def check_clip(value):
    if not 0 < value < 0.5:
        raise ValueError('clip must be strictly between 0 and 0.5.')


def dose(value, n, bounds):
    value = np.asarray(value, float)
    if value.ndim == 0:
        value = np.full(n, value)
    if value.shape != (n,) or not np.isfinite(value).all():
        raise ValueError('Dose must be a finite scalar or vector with one entry per row.')
    if np.any((value < bounds[0]) | (value > bounds[1])):
        raise ValueError('Dose is outside the observed training range.')
    return value
