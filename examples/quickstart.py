"""Run after: python -m pip install -e ."""
import numpy as np
from relative_dml import DiscreteQLearner, DiscreteDML, ContinuousQLearner, ContinuousDML

rng = np.random.default_rng(20260905)
X = rng.uniform(-1, 1, (12000, 2))
b = .12*np.exp(.5*X[:, 1])
slope = -.3-.2*X[:, 0]

t = rng.uniform(-1, 1, len(X))
y = rng.binomial(1, b*np.exp(t*slope))
for model in [ContinuousQLearner(), ContinuousDML()]:
    if isinstance(model, ContinuousDML):
        model.fit(X, t, y, effect_features=X[:, :1])
        lift = model.predict_lift(X[:5], .5, 0, effect_features=X[:5, :1])
    else:
        model.fit(X, t, y)
        lift = model.predict_lift(X[:5], .5, 0)
    print(type(model).__name__, 'lift:', np.round(lift, 3))

t = rng.integers(0, 3, len(X))
y = rng.binomial(1, b*np.exp(t*slope))
for model in [DiscreteQLearner(), DiscreteDML()]:
    model.fit(X, t, y)
    print(type(model).__name__, 'lift arm 1 / arm 0:',
          np.round(model.predict_lift(X[:5], treatment=1, reference=0), 3))
