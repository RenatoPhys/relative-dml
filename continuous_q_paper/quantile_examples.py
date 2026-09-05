"""Exact synthetic examples: quintiles of absolute CATE versus relative lift."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def number(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def write_table(path, columns, header, rows):
    lines = [rf'\begin{{tabular}}{{{columns}}}', r'\toprule',
             header + r' \\', r'\midrule']
    lines += [' & '.join(row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def group_metrics(frame):
    baseline = frame.mu0.mean()
    treated = frame.mu1.mean()
    return {'n': len(frame), 'mu0': baseline, 'mu1': treated,
            'cate': treated - baseline, 'lift': treated / baseline - 1,
            'mean_individual_lift': frame.lift.mean(),
            'incremental_per_1000': 1000 * (treated - baseline)}


def flat_lift_example(results, figures):
    """Cross absolute effects and relative lifts independently, with equal cells.

    The entire lift distribution is identical in all CATE quintiles; this
    purpose-built population isolates estimands, not estimation performance.
    """
    cells = pd.MultiIndex.from_product(
        [[.004, .008, .012, .016, .020], [.10, .20, .30, .40, .50]],
        names=['cate', 'lift']).to_frame(index=False)
    cells['n'] = 200
    cells['mu0'] = cells.cate / cells.lift
    cells['mu1'] = cells.mu0 + cells.cate
    population = cells.loc[cells.index.repeat(cells.n)].reset_index(drop=True)
    assert len(population) == 5000
    assert np.all((population.mu0 > 0) & (population.mu1 < 1))
    np.testing.assert_allclose(population.mu1/population.mu0 - 1, population.lift)
    records = []
    for score in ['cate', 'lift']:
        for quintile, (value, group) in enumerate(population.groupby(score, sort=True), 1):
            # Five equally frequent score levels; never split ties across bins.
            assert len(group) == 1000
            records.append({'score': score, 'quintile': quintile,
                            'score_value': value, **group_metrics(group)})
            if score == 'cate':
                assert group.groupby('lift').size().tolist() == [200] * 5
    quintiles = pd.DataFrame(records)
    by_cate = quintiles[quintiles.score == 'cate']
    by_lift = quintiles[quintiles.score == 'lift']
    np.testing.assert_allclose(by_cate.lift, 30/137)
    np.testing.assert_allclose(by_cate.mean_individual_lift, .30)
    np.testing.assert_allclose(by_lift.cate, .012)
    np.testing.assert_allclose(by_lift.lift, [.10, .20, .30, .40, .50])
    assert np.ptp(by_cate.lift) < 1e-12
    assert np.ptp(by_lift.lift) > .39
    total = group_metrics(population)
    np.testing.assert_allclose([total['mu0'], total['mu1']], [.0548, .0668])
    for _, groups in quintiles.groupby('score'):
        np.testing.assert_allclose(np.average(groups.cate, weights=groups.n), total['cate'])
        np.testing.assert_allclose(np.average(groups.lift, weights=groups.n*groups.mu0), total['lift'])
    cells.to_csv(results/'quantile_flat_cells.csv', index=False)
    quintiles.to_csv(results/'quantile_flat_rankings.csv', index=False)
    pd.DataFrame([total]).to_csv(results/'quantile_flat_population.csv', index=False)

    rows = [[number(100*delta)] + [number(100*delta/lift) for lift in [.1, .2, .3, .4, .5]]
            for delta in [.004, .008, .012, .016, .020]]
    write_table(results/'quantile_flat_cells.tex', 'rrrrrr',
                r'CATE (p.p.) & Lift 10\% & Lift 20\% & Lift 30\% & Lift 40\% & Lift 50\%', rows)
    rows = [['CATE' if r.score == 'cate' else 'Lift', f'Q{r.quintile}',
             number(100*r.mu0), number(100*r.mu1), number(100*r.cate),
             number(100*r.lift), number(r.incremental_per_1000, 1)]
            for r in quintiles.itertuples()]
    write_table(results/'quantile_flat_rankings.tex', 'llrrrrr',
                r'Score & Quantil & \shortstack{Base\\(\%)} & \shortstack{Com ação\\(\%)} & '
                r'\shortstack{CATE\\(p.p.)} & \shortstack{Lift do grupo\\(\%)} & '
                r'\shortstack{Extras por\\1\,000}', rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.4))
    for ax, metric, title, ylabel, upper in zip(
            axes, ['cate', 'lift'], ['O CATE separa o ganho absoluto', 'O CATE não separa o lift'],
            ['CATE do grupo (p.p.)', 'Lift do grupo (%)'], [2.3, 57]):
        for score, color, marker, label in [('cate', '#2467a5', 'o', 'Ordenação por CATE'),
                                           ('lift', '#c65d25', 's', 'Ordenação por lift')]:
            group = quintiles[quintiles.score == score]
            ax.plot(group.quintile, 100*group[metric], marker=marker,
                    color=color, linewidth=2, label=label)
        ax.set_xticks(range(1, 6), [f'Q{k}' for k in range(1, 6)])
        ax.set_xlabel('Do menor para o maior score')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, upper)
        ax.grid(axis='y', alpha=.2)
    axes[1].annotate('21,90% em todos os quintis de CATE', xy=(3, 100*30/137),
                     xytext=(1.15, 5), fontsize=9, color='#2467a5',
                     arrowprops={'arrowstyle': '->', 'color': '#2467a5'})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, .10, 1, 1])
    fig.savefig(figures/'quantile_flat_lift.pdf', bbox_inches='tight')
    fig.savefig(figures/'quantile_flat_lift.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('Flat-lift example verified: CATE quintiles all 21.8978% group lift; lift quintiles 10-50%.')


def main():
    folder = Path(__file__).resolve().parent
    results, figures = folder / 'results', folder / 'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    profiles = pd.DataFrame({'profile': list('ABCDE'), 'n': [1000] * 5,
                             'mu0': [.02, .04, .08, .16, .25],
                             'lift': [.80, .50, .30, .20, .15]})
    profiles['mu1'] = profiles.mu0 * (1 + profiles.lift)
    profiles['cate'] = profiles.mu1 - profiles.mu0
    profiles['incremental_per_1000'] = 1000 * profiles.cate
    population = profiles.loc[profiles.index.repeat(profiles.n)].reset_index(drop=True)
    assert population.shape[0] == 5000
    assert population.mu1.between(0, 1).all()
    assert np.all(np.diff(profiles.cate) > 0)
    assert np.all(np.diff(profiles.lift) < 0)

    records = []
    for score in ['cate', 'lift']:
        # Each distinct score has exactly 20% of the population; no tie is split.
        ordered = profiles.sort_values(score).profile.tolist()
        for quintile, profile in enumerate(ordered, start=1):
            group = population[population.profile == profile]
            records.append({'score': score, 'quintile': quintile,
                            'profile': profile, **group_metrics(group)})
    quintiles = pd.DataFrame(records)
    total = group_metrics(population)
    for score in ['cate', 'lift']:
        groups = quintiles[quintiles.score == score]
        np.testing.assert_allclose(np.average(groups.cate, weights=groups.n), total['cate'])
        np.testing.assert_allclose(np.average(groups.mu0, weights=groups.n), total['mu0'])
        np.testing.assert_allclose(np.average(groups.mu1, weights=groups.n), total['mu1'])
        # Aggregate lift is weighted by expected baseline conversions, not group size.
        np.testing.assert_allclose(np.average(groups.lift, weights=groups.n * groups.mu0),
                                   total['lift'])
    top = quintiles[quintiles.quintile == 5]
    assert top.set_index('score').profile.to_dict() == {'cate': 'E', 'lift': 'A'}
    assert population[population.profile == 'A'].shape[0] == 1000
    np.testing.assert_allclose(total['cate'], .0259)
    np.testing.assert_allclose(total['lift'], .23545454545454545)

    profiles.to_csv(results / 'quantile_profiles.csv', index=False)
    quintiles.to_csv(results / 'quantile_rankings.csv', index=False)
    pd.DataFrame([total]).to_csv(results / 'quantile_population.csv', index=False)
    rows = [[r.profile, str(r.n), number(100*r.mu0), number(100*r.mu1),
             number(100*r.cate), number(100*r.lift)] for r in profiles.itertuples()]
    write_table(results / 'quantile_profiles.tex', 'lrrrrr',
                r'Perfil & Pessoas & Base (\%) & Com ação (\%) & CATE (p.p.) & Lift (\%)', rows)
    rows = []
    for r in quintiles.itertuples():
        rows.append(['CATE' if r.score == 'cate' else 'Lift', f'Q{r.quintile}', r.profile,
                     number(100*r.mu0), number(100*r.mu1), number(100*r.cate),
                     number(100*r.lift), number(r.incremental_per_1000, 1)])
    write_table(results / 'quantile_rankings.tex', 'lllrrrrr',
                r'Score & Quantil & Perfil & \shortstack{Base\\(\%)} & '
                r'\shortstack{Com ação\\(\%)} & \shortstack{CATE\\(p.p.)} & '
                r'\shortstack{Lift\\(\%)} & \shortstack{Extras por\\1\,000}', rows)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.1, 3.35))
    colors = {'cate': '#2467a5', 'lift': '#c65d25'}
    labels = {'cate': 'Ordenação por CATE', 'lift': 'Ordenação por lift'}
    for ax, metric, title, ylabel in zip(
            axes, ['cate', 'lift'], ['Ganho absoluto em cada quintil', 'Ganho relativo em cada quintil'],
            ['CATE do grupo (p.p.)', 'Lift do grupo (%)']):
        for score in ['cate', 'lift']:
            values = quintiles[quintiles.score == score]
            ax.plot(values.quintile, 100*values[metric], marker='o' if score == 'cate' else 's',
                    color=colors[score], linewidth=2, label=labels[score])
        ax.set_xticks(range(1, 6), [f'Q{k}' for k in range(1, 6)])
        ax.set_xlabel('Do menor para o maior score')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0, top=4.2 if metric == 'cate' else 88)
        ax.grid(axis='y', alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=[0, .09, 1, 1])
    fig.savefig(figures / 'quantile_cate_vs_lift.pdf', bbox_inches='tight')
    fig.savefig(figures / 'quantile_cate_vs_lift.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    constant = profiles[['profile', 'n', 'mu0']].copy()
    constant['lift'] = .20
    constant['mu1'] = constant.mu0 * (1 + constant.lift)
    constant['cate'] = constant.mu1 - constant.mu0
    assert constant.lift.nunique() == 1
    assert np.all(np.diff(constant.cate) > 0)
    constant.to_csv(results / 'quantile_constant_lift.csv', index=False)
    rows = [[r.profile, number(100*r.mu0), number(100*r.mu1), number(100*r.cate),
             number(100*r.lift)] for r in constant.itertuples()]
    write_table(results / 'quantile_constant_lift.tex', 'lrrrr',
                r'Perfil & Base (\%) & Com ação (\%) & CATE (p.p.) & Lift (\%)', rows)

    mixture = pd.DataFrame({'profile': ['U', 'V'], 'mu0': [.01, .10], 'lift': [1., .10]})
    mixture['mu1'] = mixture.mu0 * (1 + mixture.lift)
    mixture['cate'] = mixture.mu1 - mixture.mu0
    aggregate = group_metrics(mixture)
    np.testing.assert_allclose(aggregate['mean_individual_lift'], .55)
    np.testing.assert_allclose(aggregate['lift'], 2/11)
    np.testing.assert_allclose(np.average(mixture.lift, weights=mixture.mu0), aggregate['lift'])
    assert not np.isclose(aggregate['lift'], aggregate['mean_individual_lift'])
    mixture.to_csv(results / 'quantile_mixture.csv', index=False)
    pd.DataFrame([aggregate]).to_csv(results / 'quantile_mixture_summary.csv', index=False)
    rows = [[r.profile, number(100*r.mu0), number(100*r.mu1), number(100*r.cate),
             number(100*r.lift)] for r in mixture.itertuples()]
    rows.append(['Grupo (médias)', number(100*aggregate['mu0']), number(100*aggregate['mu1']),
                 number(100*aggregate['cate']), number(100*aggregate['lift'])])
    write_table(results / 'quantile_mixture.tex', 'lrrrr',
                r'Perfil & Base (\%) & Com ação (\%) & CATE (p.p.) & Lift (\%)', rows)

    # A continuous curve on [0, 1] reproduces the same finite contrast 0 -> 1.
    dose = np.linspace(0, 1, 101)
    curve = profiles.mu0.to_numpy()[:, None] * np.exp(
        np.log1p(profiles.lift.to_numpy())[:, None] * dose)
    assert np.all((curve > 0) & (curve < 1))
    np.testing.assert_allclose(curve[:, 0], profiles.mu0)
    np.testing.assert_allclose(curve[:, -1], profiles.mu1)
    print('Exact examples verified: 5,000 people; opposite top quintiles E vs A.')
    print(f'Population CATE: {100*total["cate"]:.2f} pp; group lift: {100*total["lift"]:.2f}%.')
    print('Mixture: mean individual lift 55%; group lift 18.18%.')
    flat_lift_example(results, figures)
    print('Tables, CSVs and figure saved to continuous_q_paper/results and figures.')


if __name__ == '__main__':
    main()
