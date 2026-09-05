"""Reproduz as figuras do manuscrito a partir dos resultados Monte Carlo."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    folder = Path(__file__).resolve().parent
    (folder / 'figures').mkdir(exist_ok=True)
    results = pd.read_csv(folder / 'results/mc_summary.csv')
    relative_change = np.linspace(-0.10, 0.10, 201)
    risk_ratio = (1 + relative_change) ** -2
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for baseline in [0.02, 0.06, 0.25]:
        ax.plot(100 * relative_change, 100 * baseline * (risk_ratio - 1),
                label=f'Conversão-base: {baseline:.0%}')
    ax.set_xlabel('Variação relativa da taxa (%)')
    ax.set_ylabel('Mudança de conversão (p.p.)')
    ax.set_title('Mesma elasticidade, diferentes efeitos absolutos')
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / 'figures/absolute_vs_relative.pdf')
    plt.close(fig)

    labels = ['CQ-DML: ambas corretas', 'CQ-DML: apenas m correto',
              'CQ-DML: apenas b correto', 'CQ-DML: ambas incorretas',
              'Q: densidade correta', 'Q: densidade incorreta']
    expected = ['CQ-DML: b e m corretos', 'CQ-DML: apenas m correto',
                'CQ-DML: apenas b correto', 'CQ-DML: ambos incorretos',
                'Q: densidade correta', 'Q: densidade incorreta']
    results = results.set_index('method').loc[expected]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    positions = np.arange(len(results))
    ax.errorbar(results['mean1'], positions,
                xerr=1.96 * results['mcse_bias1'], fmt='o', capsize=4)
    ax.axvline(-0.20, linestyle='--', label='Parâmetro verdadeiro')
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel('Média estimada da heterogeneidade relativa (θ₁)')
    ax.set_title('Verificação Monte Carlo da robustez')
    ax.legend(frameon=False)
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / 'figures/monte_carlo_robustness.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
