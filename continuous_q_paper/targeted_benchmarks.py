"""Fixed, additional benchmarks for relative-curve specification.

The Poisson loss fits a binary outcome's conditional mean; it does not assert
Poisson outcomes. Its log link can predict above one. These predictions are
counted and never clipped. The log-mean regressions also restrict the baseline,
so comparing them with DML does not isolate orthogonalization.

Historical experiments and their output files are not modified. Example:
python continuous_q_paper/targeted_benchmarks.py --reps 2 --n 3000 \
    --test-n 1000 --seed 731905 --out continuous_q_paper/results/targeted_smoke
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from relative_dml import ContinuousDML

if __package__:
    from .scenario_experiments import Scenario, classifier, generate, response
else:
    from scenario_experiments import Scenario, classifier, generate, response


SCENARIOS = (
    Scenario('linear_standard', 'Log-linear, alocação padrão'),
    Scenario('curved_randomized', 'Curvo, dose aleatorizada uniforme',
             curvature=.8, confounding=0),
    Scenario('curved_standard', 'Curvo, alocação padrão', curvature=.8),
    Scenario('curved_weak_overlap', 'Curvo, pouco overlap',
             curvature=.8, confounding=4),
)
METHODS = ('DML-linear', 'DML-quadratic', 'Log-mean-linear',
           'Log-mean-quadratic', 'S-learner')
TARGETS = (-.8, -.4, .4, .8)
GRID = (0., *TARGETS)
POISSON_PARAMS = dict(alpha=0., fit_intercept=True, solver='lbfgs',
                      max_iter=2000, tol=1e-9)
BOOSTING_PARAMS = dict(max_iter=100, max_leaf_nodes=8, min_samples_leaf=50,
                       l2_regularization=2, early_stopping=False)
METRICS = ('log_ratio_rmse', 'lift_mae', 'lift_bias', 'rank_spearman')
DIAGNOSTICS = ('invalid_train_fraction', 'invalid_grid_fraction',
               's_clipped_grid_fraction', 'converters', 'conversion',
               'moment_norm', 'jac_condition', 'jac_min_singular',
               'poisson_iterations', 'poisson_gradient_max', 'fit_seconds')


def log_mean_design(X, treatment, degree):
    """Intercept is supplied by PoissonRegressor, not duplicated in X."""
    t = np.broadcast_to(np.asarray(treatment, float), (len(X),))
    columns = [X[:, 0], X[:, 1], t, t*X[:, 0]]
    if degree == 2:
        columns.append(t*t)
    return np.column_stack(columns)


def make_models(seed):
    models = []
    for degree in (1, 2):
        models.append((METHODS[degree-1], ContinuousDML(
            outcome_model=classifier(seed),
            treatment_model=HistGradientBoostingRegressor(
                **BOOSTING_PARAMS, random_state=seed),
            n_splits=3, reference_dose=0., random_state=seed,
            dose_degree=degree)))
    models.extend((METHODS[degree+1], PoissonRegressor(**POISSON_PARAMS))
                  for degree in (1, 2))
    models.append(('S-learner', classifier(seed)))
    return models


def invalid_fraction(predictions):
    """Invalid Bernoulli means: nonfinite or outside the closed [0, 1]."""
    p = np.asarray(predictions, float)
    return float(np.mean(~np.isfinite(p) | (p < 0) | (p > 1)))


def predict_ratios(model, X, method):
    diagnostics = {}
    if method.startswith('DML-'):
        ratios = np.array([model.predict_ratio(
            X, t, 0., effect_features=X[:, :1]) for t in TARGETS])
    else:
        if method.startswith('Log-mean-'):
            degree = 2 if method.endswith('quadratic') else 1
            means = np.array([model.predict(log_mean_design(X, t, degree))
                              for t in GRID])
        else:
            means = np.array([model.predict_proba(np.column_stack(
                [X, np.full(len(X), t)]))[:, 1] for t in GRID])
            diagnostics['s_clipped_grid_fraction'] = float(np.mean(means < .001))
        diagnostics['invalid_grid_fraction'] = invalid_fraction(means)
        if method == 'S-learner':
            if diagnostics['s_clipped_grid_fraction']:
                warnings.warn('S-learner response clipped at 0.001, as in the '
                              'historical benchmark.', RuntimeWarning)
            means = np.maximum(means, .001)
        # Keep invalid log-mean predictions visible even when ratios fail.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            if method.startswith('Log-mean-'):
                # Cancel the baseline in the linear predictor. Dividing means
                # introduces X2-dependent rounding that breaks exact X1 ties.
                at_reference = log_mean_design(X, 0., degree)
                ratios = np.array([np.exp((log_mean_design(X, t, degree)
                                           - at_reference) @ model.coef_)
                                   for t in TARGETS])
            else:
                ratios = means[1:]/means[0]
    return ratios, diagnostics


def score_ratios(scenario, ratios, X):
    if not np.isfinite(ratios).all() or np.any(ratios <= 0):
        raise FloatingPointError('Nonpositive or nonfinite ratio predictions.')
    # Evaluate the known relative curve directly, preserving the two X1 ties.
    truths = np.array([np.exp(t*(scenario.effect + scenario.heterogeneity*X[:, 0])
                              + scenario.curvature*t*t) for t in TARGETS])
    ranking = np.nan
    # Spearman at +0.8, undefined for a constant true or predicted effect.
    if np.ptp(truths[-1]) > 1e-10 and np.ptp(ratios[-1]) > 1e-10:
        ranking = float(spearmanr(ratios[-1], truths[-1]).statistic)
    with np.errstate(over='raise', invalid='raise'):
        return dict(log_ratio_rmse=float(np.sqrt(np.mean(
                        (np.log(ratios)-np.log(truths))**2))),
                    lift_mae=float(np.mean(np.abs(ratios-truths))),
                    lift_bias=float(np.mean(ratios-truths)),
                    rank_spearman=ranking)


def replication_seeds(seed, rep):
    # Common random numbers across designs; independent train and test streams.
    return tuple(int(np.random.SeedSequence([seed, rep, stream]).generate_state(1)[0])
                 for stream in (0, 1))


def run_replication(scenario, rep, n, test_n, seed):
    train_seed, test_seed = replication_seeds(seed, rep)
    X, t, y = generate(scenario, n, np.random.default_rng(train_seed))
    Xt, _, _ = generate(scenario, test_n, np.random.default_rng(test_seed))
    rows = []
    for method, model in make_models(train_seed):
        row = dict(scenario=scenario.name, label=scenario.label,
                   curvature=scenario.curvature, confounding=scenario.confounding,
                   method=method, rep=rep, seed=train_seed, test_seed=test_seed,
                   n=n, test_n=test_n, converters=int(y.sum()), conversion=float(y.mean()),
                   success=False, fit_success=False, error='', warnings='',
                   convergence_warning=False, estimate_json='', coefficients_json='',
                   **{key: np.nan for key in (*METRICS, 'theta0', 'theta1', 'kappa',
                                              'se0', 'se1', 'se_kappa')})
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if method.startswith('DML-'):
                    model.fit(X, t, y, effect_features=X[:, :1])
                    row['fit_success'] = True
                    row.update(theta0=float(model.coef_[0]), theta1=float(model.coef_[1]),
                               se0=float(model.se_[0]), se1=float(model.se_[1]),
                               moment_norm=model.estimate_.moment_norm,
                               jac_condition=model.estimate_.jac_condition,
                               jac_min_singular=min(model.estimate_.jac_singular_values),
                               estimate_json=json.dumps(asdict(model.estimate_)))
                    if method == 'DML-quadratic':
                        row.update(kappa=float(model.coef_[-1]), se_kappa=float(model.se_[-1]))
                elif method.startswith('Log-mean-'):
                    degree = 2 if method.endswith('quadratic') else 1
                    design = log_mean_design(X, t, degree)
                    model.fit(design, y)
                    row['fit_success'] = True
                    means = model.predict(design)
                    row.update(invalid_train_fraction=invalid_fraction(means),
                               poisson_iterations=int(model.n_iter_),
                               poisson_gradient_max=float(np.max(np.abs(
                                   np.column_stack([np.ones(n), design]).T@(means-y)/n))),
                               coefficients_json=json.dumps(
                                   [float(model.intercept_), *model.coef_.tolist()]),
                               theta0=float(model.coef_[2]), theta1=float(model.coef_[3]))
                    if degree == 2:
                        row['kappa'] = float(model.coef_[4])
                else:
                    model.fit(np.column_stack([X, t]), y)
                    row['fit_success'] = True
                row['fit_seconds'] = time.perf_counter()-started
                ratios, diagnostics = predict_ratios(model, Xt, method)
                row.update(diagnostics)
                if any(row.get(key, 0) > 0 for key in
                       ('invalid_train_fraction', 'invalid_grid_fraction')):
                    warnings.warn('Predicted conversion means outside [0, 1] or '
                                  'nonfinite; see recorded fractions. Log-mean '
                                  'predictions are not clipped.', RuntimeWarning)
                row.update(score_ratios(scenario, ratios, Xt))
                row['success'] = True
            except Exception as exc:
                # Keep every method/replication, including unexpected fit failures.
                # Interrupts (KeyboardInterrupt/SystemExit) still stop the run.
                row['error'] = f'{type(exc).__name__}: {exc}'
            row.setdefault('fit_seconds', time.perf_counter()-started)
            row['convergence_warning'] = any(
                issubclass(w.category, ConvergenceWarning) for w in caught)
            if row['convergence_warning']:
                row['success'] = False
                row['error'] = row['error'] or 'ConvergenceWarning: fit did not converge.'
            row['warnings'] = ' | '.join(sorted({
                f'{w.category.__name__}: {w.message}' for w in caught}))
        rows.append(row)
    return rows


def mean_mcse(values):
    values = pd.Series(values, dtype=float).dropna()
    return (float(values.mean()),
            float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else np.nan,
            len(values))


def summarize(df):
    rows = []
    for (scenario, method), sub in df.groupby(['scenario', 'method'], sort=False):
        ok = sub[sub.success]
        row = dict(scenario=scenario, label=sub.label.iloc[0], method=method,
                   reps=len(sub), successes=len(ok), failures=len(sub)-len(ok),
                   warning_runs=int(sub.warnings.ne('').sum()),
                   convergence_failures=int(sub.convergence_warning.sum()))
        for metric in (*METRICS, *DIAGNOSTICS, 'theta0', 'theta1', 'kappa',
                       'se0', 'se1', 'se_kappa'):
            # Prediction/fit diagnostics include unsuccessful runs when available.
            source = sub if metric in DIAGNOSTICS else ok
            values = source[metric] if metric in source else []
            row[metric], row[f'{metric}_mcse'], row[f'{metric}_n'] = mean_mcse(values)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(df):
    rows = []
    for scenario, sub in df.groupby('scenario', sort=False):
        baseline = sub[sub.method.eq('DML-linear')].set_index('rep')
        for method in METHODS[1:]:
            candidate = sub[sub.method.eq(method)].set_index('rep')
            for metric in METRICS:
                good = baseline.success & candidate.success.reindex(baseline.index, fill_value=False)
                differences = (candidate[metric]-baseline[metric])[good]
                mean, mcse, count = mean_mcse(differences)
                rows.append(dict(scenario=scenario, method=method,
                                 comparator='DML-linear', metric=metric,
                                 difference=mean, difference_mcse=mcse,
                                 pairs=count, total_reps=len(baseline),
                                 excluded_pairs=len(baseline)-count))
    return pd.DataFrame(rows)


def compact(value, mcse):
    return f'{value:.4f} ({mcse:.4f})' if np.isfinite(value) else 'NaN'


def write_tables(summary, heterogeneity, out):
    lines = [
        '# Targeted benchmark', '',
        'Entries are mean (Monte Carlo SE). Error metrics and coefficients use '
        'successful runs; every failure remains in the raw data and counts below. '
        'Diagnostic fractions use all runs where available. NaN denotes an '
        'undefined quantity or unavailable Monte Carlo SE.', '',
        '| Design | Method | RMSE log-ratio | MAE lift | Spearman | Failures/reps | Warnings |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    tex = [r'\begin{tabular}{llrrr}', r'\toprule',
           r'Desenho & Método & RMSE log-ratio & MAE lift & Falhas \\', r'\midrule']
    design_codes = {scenario.name: f'D{j}' for j, scenario in enumerate(SCENARIOS, 1)}
    for row in summary.itertuples():
        rmse = compact(row.log_ratio_rmse, row.log_ratio_rmse_mcse)
        mae = compact(row.lift_mae, row.lift_mae_mcse)
        rank = compact(row.rank_spearman, row.rank_spearman_mcse)
        lines.append(f'| {row.label} | {row.method} | {rmse} | {mae} | {rank} '
                     f'| {row.failures}/{row.reps} | {row.warning_runs} |')
        label = design_codes[row.scenario]
        tex.append(f'{label} & {row.method} & {rmse} & {mae} & {row.failures}/{row.reps} '
                   + r'\\')
    tex.extend([r'\bottomrule', r'\end{tabular}', ''])
    lines.extend(['', '## Linear heterogeneity under curved responses', '',
                  'These are fitted coefficients of X1 in the linear dose term. '
                  'For a model omitting curvature they are allocation-dependent '
                  'summaries, not automatically the structural heterogeneity. '
                  'No structural-parameter coverage calculation is made.', '',
                  '| Design | Method | theta1 (MCSE) | Mean marginal SE | Successful runs |',
                  '|---|---|---:|---:|---:|'])
    for row in heterogeneity.itertuples():
        lines.append(f'| {row.label} | {row.method} | '
                     f'{compact(row.theta1, row.theta1_mcse)} | '
                     f'{row.se1:.4f} | {row.successes}/{row.reps} |')
    lines.extend(['', '## Log-mean conversion predictions', '',
                  'Poisson loss is a mean-fitting criterion for binary outcomes. '
                  'The log link does not constrain predictions to [0, 1]; no '
                  'clipping is applied. Grid = independent test X crossed with '
                  'doses {0, -0.8, -0.4, 0.4, 0.8}.', '',
                  '| Design | Method | Invalid train fraction (MCSE) | Invalid grid fraction (MCSE) |',
                  '|---|---|---:|---:|'])
    for row in summary[summary.method.str.startswith('Log-mean-')].itertuples():
        lines.append(f'| {row.label} | {row.method} | '
                     f'{compact(row.invalid_train_fraction, row.invalid_train_fraction_mcse)} | '
                     f'{compact(row.invalid_grid_fraction, row.invalid_grid_fraction_mcse)} |')
    lines.extend(['', 'The structured log-mean models restrict the baseline as '
                  'well as the relative curve, so this comparison does not '
                  'perfectly isolate orthogonalization. S-learner clipping at '
                  '0.001 follows the historical benchmark and is recorded. '
                  'Paired differences (candidate minus DML-linear) and their '
                  'actual pair counts are in targeted_paired.csv.', ''])
    (out/'targeted_tables.md').write_text('\n'.join(lines), encoding='utf-8')
    (out/'targeted_table.tex').write_text('\n'.join(tex), encoding='utf-8')


def write_results(rows, out):
    df = pd.DataFrame(rows)
    summary = summarize(df)
    curved_names = [s.name for s in SCENARIOS if s.curvature]
    heterogeneity = summary[summary.scenario.isin(curved_names) &
                           summary.method.isin(METHODS[:4])][
        ['scenario', 'label', 'method', 'reps', 'successes', 'failures',
         'theta0', 'theta0_mcse', 'theta1', 'theta1_mcse', 'theta1_n',
         'se1', 'kappa', 'kappa_mcse']]
    df.to_csv(out/'targeted_raw.csv', index=False)
    summary.to_csv(out/'targeted_summary.csv', index=False)
    paired_summary(df).to_csv(out/'targeted_paired.csv', index=False)
    heterogeneity.to_csv(out/'targeted_heterogeneity.csv', index=False)
    write_tables(summary, heterogeneity, out)
    return summary


def metadata(args):
    root = Path(__file__).resolve().parents[1]
    git = {}
    for key, command in [('commit', ['rev-parse', 'HEAD']),
                         ('status', ['status', '--porcelain=v1', '--untracked-files=normal'])]:
        try:
            result = subprocess.run(['git', *command], cwd=root, check=True,
                                    capture_output=True, text=True)
            git[key] = result.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            git[key] = None
            git[f'{key}_error'] = str(exc)
    git['dirty'] = bool(git['status']) if git['status'] is not None else None
    sources = [*sorted((root/'relative_dml').glob('*.py')),
               Path(__file__).resolve(), root/'continuous_q_paper/scenario_experiments.py']
    return dict(seed=args.seed, reps=args.reps, n=args.n, test_n=args.test_n,
                started_utc=datetime.now(timezone.utc).isoformat(), status='running',
                completed_rows=0, expected_rows=args.reps*len(SCENARIOS)*len(METHODS),
                scenarios=[asdict(s) for s in SCENARIOS], methods=list(METHODS),
                targets=list(TARGETS), reference_dose=0.,
                seed_scheme='uint32 SeedSequence([seed, rep, stream]); stream 0=train, '
                            '1=test; common random numbers across scenarios',
                dml=dict(n_splits=3, reference_dose=0., dose_degrees=[1, 2],
                         effect_features=['X1'], outcome='HistGradientBoostingClassifier',
                         treatment='HistGradientBoostingRegressor',
                         boosting_params=BOOSTING_PARAMS),
                log_mean=dict(class_name='sklearn.linear_model.PoissonRegressor',
                              parameters=POISSON_PARAMS,
                              columns=['X1', 'X2', 't', 't*X1', 't**2 (quadratic only)'],
                              coefficient_order=['intercept', 'X1', 'X2', 't', 't*X1',
                                                 't**2 (quadratic only)'],
                              invalid_definition='nonfinite or outside [0, 1]',
                              documentation='https://scikit-learn.org/stable/modules/'
                                            'generated/sklearn.linear_model.PoissonRegressor.html'),
                s_learner=dict(classifier='HistGradientBoostingClassifier',
                               parameters=BOOSTING_PARAMS, response_floor=.001),
                versions=dict(python=platform.python_version(), numpy=np.__version__,
                              scipy=scipy.__version__, sklearn=sklearn.__version__,
                              pandas=pd.__version__), platform=platform.platform(),
                threads={key: os.environ.get(key) for key in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                git=git, source_sha256={str(p.relative_to(root)).replace('\\', '/'):
                                       hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                notes=[
                    'Same samples per method, independent test covariates, no truth-based tuning.',
                    'Truth ratios use the causal log-relative curve directly; log-mean '
                    'ratios cancel the baseline in the linear predictor to preserve exact '
                    'X1 profile ties for Spearman. Conversion means still determine invalid fractions.',
                    'Convergence warnings count as failures; all failed rows and warnings remain.',
                    'Metric means/MCSE use successful runs; diagnostics use all available runs. '
                    'MCSE = replication SD / sqrt(number of available replications).',
                    'Poisson loss fits the mean of binary Y; it does not assert Poisson data. '
                    'Predictions outside [0,1] are counted and never clipped.',
                    'Log-mean regressions also restrict the baseline; this comparison does '
                    'not perfectly isolate orthogonalization.',
                    'Under omitted curvature the fitted linear coefficient can depend on '
                    'allocation. No structural coverage test is computed.',
                    'DML marginal SE/covariance require correct relative specification and '
                    'iid nuisance-rate conditions; no cluster inference is provided.'])


def prepare_output(out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output must be a new or empty directory; existing runs are not overwritten.')
    out.mkdir(parents=True, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--n', type=int, default=12000)
    parser.add_argument('--test-n', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=831905)
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).parent/'results/targeted_v1')
    args = parser.parse_args(argv)
    if args.reps < 2 or min(args.n, args.test_n) < 100 or args.seed < 0:
        parser.error('Require reps >= 2, n/test-n >= 100 and seed >= 0.')
    try:
        prepare_output(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    meta = metadata(args)
    metadata_path = args.out/'targeted_metadata.json'
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    for scenario in SCENARIOS:
        for rep in range(args.reps):
            rows.extend(run_replication(scenario, rep, args.n, args.test_n, args.seed))
            # Persist failures and progress even if a later fit is interrupted.
            pd.DataFrame(rows).to_csv(args.out/'targeted_raw.csv', index=False)
            meta['completed_rows'] = len(rows)
            metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        summary = write_results(rows, args.out)
        count = int(summary[summary.scenario.eq(scenario.name)].failures.sum())
        print(f'{scenario.name}: {args.reps} replicações concluídas, {count} falhas', flush=True)
    meta.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                failures=int(summary.failures.sum()), warning_runs=int(summary.warning_runs.sum()))
    metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary[['scenario', 'method', 'log_ratio_rmse', 'log_ratio_rmse_mcse',
                   'lift_mae', 'lift_mae_mcse', 'failures', 'warning_runs']].to_string(index=False))
    return summary


if __name__ == '__main__':
    main()
