"""Generate the additional paper tables and figure from executed CSV results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    results = pd.read_csv(folder/'results/scenario_summary.csv')
    names = results.scenario.drop_duplicates().tolist()
    methods = ['Q', 'DML', 'S-learner']
    labels = results.drop_duplicates('scenario').set_index('scenario').label
    (folder/'figures').mkdir(exist_ok=True)
    for metric, filename in [('log_ratio_rmse', 'scenario_log_rmse.tex'),
                              ('lift_mae', 'scenario_lift_mae.tex')]:
        lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Cenário & Conv. (\%) & Q & DML & S-learner \\', r'\midrule']
        for name in names:
            sub = results[results.scenario == name].set_index('method')
            cells = [f'{sub.loc[m, metric]:.3f} ({sub.loc[m, metric + "_mcse"]:.3f})' for m in methods]
            lines.append(f'{labels[name]} & {100*sub.conversion.iloc[0]:.2f} & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (folder/'results'/filename).write_text('\n'.join(lines)+'\n', encoding='utf-8')

    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Cenário & Q & DML & S-learner \\', r'\midrule']
    for name in names:
        sub = results[results.scenario == name].set_index('method')
        cells = [f'{int(sub.loc[m, "warning_runs"])}/{int(sub.loc[m, "reps"])}' for m in methods]
        lines.append(labels[name] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (folder/'results/scenario_warnings.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    indexed = results.set_index(['scenario', 'method'])
    def rmse(scenario, method):
        return f'{indexed.loc[(scenario, method), "log_ratio_rmse"]:.3f}'
    explanation = (
        'No caso contínuo heterogêneo, o RMSE médio de log-ratio foi '
        f'{rmse("continuous_heterogeneous", "DML")} para DML, '
        f'{rmse("continuous_heterogeneous", "Q")} para Q e '
        f'{rmse("continuous_heterogeneous", "S-learner")} para S-learner. '
        'O DML utiliza aqui a forma relativa correta e apenas dois coeficientes. '
        'Seu RMSE aumenta para '
        f'{rmse("continuous_rare", "DML")} com conversão rara e '
        f'{rmse("continuous_weak", "DML")} com pouco overlap. '
        'Quando se omite curvatura verdadeira, o RMSE do DML chega a '
        f'{rmse("continuous_curved", "DML")}, enquanto o S-learner obtém '
        f'{rmse("continuous_curved", "S-learner")}. '
        'A restrição multiplicativa ajuda nos desenhos que a satisfazem e se torna uma fonte de erro no desenho curvo.\n\n'
        'No binário aleatorizado, o DML com estágio final regularizado apresenta RMSE '
        f'{rmse("binary_rct", "DML")}, frente a '
        f'{rmse("binary_rct", "Q")} do Q e '
        f'{rmse("binary_rct", "S-learner")} do S-learner. '
        'Essa regularização não resolve todos os regimes: com três braços, os RMSEs de DML e S-learner são '
        f'{rmse("three_arm", "DML")} e {rmse("three_arm", "S-learner")}; '
        'no binário raro, são '
        f'{rmse("binary_rare", "DML")} e {rmse("binary_rare", "S-learner")}. '
        'O S-learner é portanto uma referência relevante, e a garantia de consistência DR não implica menor erro em amostras finitas.\n\n'
        f'Foram concluídos {int(results.successes.sum())} dos {int(results.reps.sum())} ajustes principais; '
        f'{int(results.failures.sum())} falhas foram registradas. '
        'O DML discreto ainda apresentou clipping em '
        f'{int(indexed.loc[("three_arm", "DML"), "warning_runs"])}/30 replicações com três braços e '
        f'{int(indexed.loc[("binary_rare", "DML"), "warning_runs"])}/30 no binário raro. '
        'A ausência de avisos no Q não garante precisão: seus erros nos desenhos raros permanecem grandes. '
        'As diferenças observadas dependem destes desenhos, classes e parâmetros; não sustentam superioridade geral de uma família.\n'
    )
    (folder/'results/scenario_interpretation.tex').write_text(explanation, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(9.1, 5.9))
    colors = ['#2574a9', '#bd4f24', '#45854f']
    y = np.arange(len(names))
    for j, method in enumerate(methods):
        sub = results[results.method == method].set_index('scenario').loc[names]
        ax.errorbar(sub.log_ratio_rmse, y+(j-1)*.21,
                    xerr=1.96*sub.log_ratio_rmse_mcse, fmt='o', markersize=4,
                    capsize=2.5, color=colors[j], label=method)
    ax.set_yticks(y, labels.loc[names])
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel('RMSE de log-ratio na amostra de teste (menor é melhor)')
    ax.set_title('Dez cenários sintéticos com nuisances aprendidas')
    ax.grid(axis='x', alpha=.2)
    ax.legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(folder/'figures/scenario_comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
