"""Additional synthetic benchmarks of the installable package, with held-out truth.

Run from the project root after installing the package:
python continuous_q_paper/scenario_experiments.py --reps 30 --n 12000
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy.special import softmax
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from relative_dml import DiscreteQLearner, DiscreteDML, ContinuousQLearner, ContinuousDML

SEED = 20270906


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    kind: str = 'continuous'
    baseline: float = .04
    effect: float = -.35
    heterogeneity: float = -.20
    confounding: float = 1.
    curvature: float = 0.
    arms: int = 2


SCENARIOS = [
    Scenario('continuous_null', 'Contínuo: efeito nulo', effect=0, heterogeneity=0),
    Scenario('continuous_common', 'Contínuo: efeito comum', heterogeneity=0),
    Scenario('continuous_heterogeneous', 'Contínuo: heterogêneo'),
    Scenario('continuous_rare', 'Contínuo: conversão rara', baseline=.008),
    Scenario('continuous_weak', 'Contínuo: pouco overlap', confounding=4.),
    Scenario('continuous_curved', 'Contínuo: curva omitida', curvature=.8),
    Scenario('binary_rct', 'Binário: aleatorizado', kind='discrete', confounding=0),
    Scenario('binary_confounded', 'Binário: confundido', kind='discrete'),
    Scenario('three_arm', 'Discreto: três braços', kind='discrete', arms=3),
    Scenario('binary_rare', 'Binário: conversão rara', kind='discrete', baseline=.008),
]


def response(s, X, t):
    b = s.baseline*np.exp(.65*X[:, 0] + .4*X[:, 1])
    g = t*(s.effect + s.heterogeneity*X[:, 0]) + s.curvature*t*t
    mu = b*np.exp(g)
    if np.any((mu <= 0) | (mu >= 1)):
        raise ValueError('Synthetic probabilities outside (0, 1).')
    return mu


def generate(s, n, rng):
    X = np.column_stack([rng.choice([-1., 1.], n), rng.uniform(-1, 1, n)])
    alpha = s.confounding*(.9*X[:, 0] + .6*X[:, 1])
    if s.kind == 'continuous':
        u = rng.uniform(size=n)
        t = 2*u-1
        keep = np.abs(alpha) > 1e-8
        t[keep] = -1 + np.log1p(u[keep]*np.expm1(2*alpha[keep]))/alpha[keep]
    else:
        e = softmax(alpha[:, None]*np.arange(s.arms), axis=1)
        t = (rng.uniform(size=n)[:, None] > np.cumsum(e, axis=1)).sum(axis=1)
    y = rng.binomial(1, response(s, X, t))
    return X, t, y


def classifier(seed):
    return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=8,
        min_samples_leaf=50, l2_regularization=2, early_stopping=False, random_state=seed)


def evaluate(s, model, X, targets, method):
    predictions, truths = [], []
    for t in targets:
        if method == 'S-learner':
            mu = model.predict_proba(np.column_stack([X, np.full(len(X), t)]))[:, 1]
            b = model.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]
            if np.any((mu < .001) | (b < .001)):
                warnings.warn('S-learner response clipped.', RuntimeWarning)
            ratio = np.maximum(mu, .001)/np.maximum(b, .001)
        elif isinstance(model, ContinuousDML):
            ratio = model.predict_ratio(X, t, 0, effect_features=X[:, :1])
        else:
            ratio = model.predict_ratio(X, t, 0)
        truth = response(s, X, t)/response(s, X, 0)
        if not np.isfinite(ratio).all() or np.any(ratio <= 0):
            raise FloatingPointError('Invalid ratio prediction.')
        predictions.append(ratio)
        truths.append(truth)
    predictions, truths = np.array(predictions), np.array(truths)
    # Ranking at the largest evaluated dose; undefined for a constant effect.
    ranking = np.nan
    if np.ptp(truths[-1]) > 1e-10 and np.ptp(predictions[-1]) > 1e-10:
        ranking = float(spearmanr(predictions[-1], truths[-1]).statistic)
    return {'log_ratio_rmse': float(np.sqrt(np.mean((np.log(predictions)-np.log(truths))**2))),
            'lift_mae': float(np.mean(np.abs(predictions-truths))),
            'lift_bias': float(np.mean(predictions-truths)), 'rank_spearman': ranking}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--test-n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--pilot-final', action='store_true',
                        help='Reproduce the initial, less regularized discrete final regression.')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    if args.reps < 2 or min(args.n, args.test_n) < 100:
        parser.error('reps >= 2 and sample sizes >= 100 required.')
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, s in enumerate(SCENARIOS):
        for rep in range(args.reps):
            seed = args.seed + 1000*case + rep
            X, t, y = generate(s, args.n, np.random.default_rng(seed))
            Xt, _, _ = generate(s, args.test_n, np.random.default_rng(seed + 1000000))
            targets = [-.8, -.4, .4, .8] if s.kind == 'continuous' else list(range(1, s.arms))
            q = ContinuousQLearner(random_state=seed) if s.kind == 'continuous' else DiscreteQLearner(random_state=seed)
            dml = ContinuousDML(random_state=seed) if s.kind == 'continuous' else DiscreteDML(random_state=seed)
            if args.pilot_final and s.kind == 'discrete':
                dml.set_params(final_model=HistGradientBoostingRegressor(
                    max_iter=100, max_leaf_nodes=8, min_samples_leaf=50,
                    l2_regularization=2, early_stopping=False, random_state=seed))
            for method, model in [('Q', q), ('DML', dml), ('S-learner', classifier(seed))]:
                row = {'scenario': s.name, 'label': s.label, 'method': method,
                       'rep': rep, 'seed': seed, 'n': args.n, 'test_n': args.test_n,
                       'conversion': float(y.mean()), 'converters': int(y.sum()),
                       'success': False, 'error': '', 'warnings': ''}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        if method == 'S-learner':
                            model.fit(np.column_stack([X, t]), y)
                        elif isinstance(model, ContinuousDML):
                            model.fit(X, t, y, effect_features=X[:, :1])
                        else:
                            model.fit(X, t, y)
                        row.update(evaluate(s, model, Xt, targets, method))
                        row['success'] = True
                        if isinstance(model, ContinuousDML):
                            row.update(theta0=model.coef_[0], theta1=model.coef_[1],
                                       moment_norm=model.estimate_.moment_norm)
                        if isinstance(model, DiscreteDML):
                            row['propensity_clip_fraction'] = model.propensity_clip_fraction_
                    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                    row['warnings'] = ' | '.join(sorted({str(w.message) for w in caught}))
                rows.append(row)
        print(f'{s.name}: {args.reps} replicações concluídas', flush=True)
        pd.DataFrame(rows).to_csv(args.out/'scenario_raw.csv', index=False)
    df = pd.DataFrame(rows)
    summaries = []
    for (scenario, method), sub in df.groupby(['scenario', 'method'], sort=False):
        ok = sub[sub.success]
        row = {'scenario': scenario, 'label': sub.label.iloc[0], 'method': method,
               'reps': len(sub), 'successes': len(ok), 'failures': len(sub)-len(ok),
               'warning_runs': int((sub.warnings != '').sum()),
               'conversion': sub.conversion.mean()}
        for metric in ['log_ratio_rmse', 'lift_mae', 'lift_bias', 'rank_spearman']:
            values = ok[metric].dropna() if metric in ok else pd.Series(dtype=float)
            row[metric] = values.mean()
            row[f'{metric}_mcse'] = values.std(ddof=1)/np.sqrt(len(values)) if len(values) > 1 else np.nan
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out/'scenario_summary.csv', index=False)
    metadata = {'seed': args.seed, 'reps': args.reps, 'n': args.n, 'test_n': args.test_n,
                'folds': 3, 'scenarios': [asdict(s) for s in SCENARIOS],
                'discrete_final': 'pilot: 100 iterations, 8 leaves, leaf size 50' if args.pilot_final else 'regularized: 30 iterations, learning rate .05, 4 leaves, leaf size 200, l2=10',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                             'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'pandas': pd.__version__},
                'note': 'Independent test sets. Metrics conditional on successful fits; failures and warnings retained.'}
    (args.out/'scenario_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary[['scenario', 'method', 'log_ratio_rmse', 'lift_mae', 'failures', 'warning_runs']].to_string(index=False))


if __name__ == '__main__':
    main()
