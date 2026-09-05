# Q-Learning contínuo e CQ-DML para conversão binária

Manuscrito metodológico em português, preparado em 5 de setembro de 2026.

## Conteúdo

- `paper.pdf`: artigo de trabalho com identificação, hipóteses, provas, limites, simulações e referências.
- `paper.tex`: fonte editável LaTeX.
- `experiments.py`: estimador multiplicativo, Q-Learner de densidade no desenho sintético, simulações e verificações algébricas.
- `make_figures.py`: reprodução das duas figuras a partir das saídas.
- `results/mc_raw.csv`: 600 registros (100 replicações × 6 estimadores).
- `results/mc_summary.csv`: viés, RMSE, erro Monte Carlo e cobertura.
- `results/ml_demo.json`: uma realização com nuisances aprendidas por boosting e cross-fitting em três folds.
- `results/stochastic_dr_checks.csv`: verificação por quadratura do resto DR para intervenções fixas.
- `results/metadata.json`: parâmetros e semente do desenho.
- `figures/`: figuras vetoriais incluídas no artigo.
- `package_benchmarks.tex`: seção sobre o pacote e os dez cenários adicionais.
- `scenario_experiments.py`: benchmark do pacote com treino/teste independentes e nuisances aprendidas.
- `make_scenario_figures.py`: geração das novas tabelas e figura diretamente dos CSVs.
- `results/scenario_*.csv`, `results/scenario_metadata.json`: avaliação principal, com 30 replicações por cenário e três métodos.
- `results/pilot/`: rodada inicial preservada; motivou regularização do estágio final do DML discreto.
- `quantile_appendix.tex`: apêndice didático comparando rankings por CATE absoluto, lift relativo e propensão à conversão sem ação como entrada.
- `quantile_examples.py`: reprodução determinística dos exemplos e verificações numéricas.
- `results/quantile_*` e `figures/quantile_*`: tabelas e gráficos com probabilidades sintéticas conhecidas.

## Pacote Python e novos cenários

O pacote instalável está na pasta superior. Consulte `../README.md` para os quatro estimadores e a convenção `lift = ratio - 1`. Antes de executar os experimentos, instale-o na raiz do projeto:

```powershell
python -m pip install -e ".[paper,test]"
$env:OMP_NUM_THREADS="1"
python continuous_q_paper/scenario_experiments.py --reps 30 --n 12000 --test-n 2000
python continuous_q_paper/make_scenario_figures.py
```

No diretório `continuous_q_paper`, compile `paper.tex` como antes. `package_benchmarks.tex` e as tabelas geradas são incluídos automaticamente. A API contínua usa a base linear na dose do manuscrito; a API discreta aceita múltiplos braços. Não foi implementado o aprendiz não paramétrico de políticas estocásticas.

Os ambientes das duas etapas diferem: as versões abaixo documentam os experimentos originais. O benchmark novo foi executado com Python 3.9.13, NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.5.1, pandas 2.2.2 e matplotlib 3.9.4, em ambiente virtual local. Os metadados registram as versões efetivas. O PDF atualizado foi compilado com Tectonic 0.17.0, mantendo a fonte LaTeX.

## Ambiente utilizado

Python 3.13.5; NumPy 2.3.5; SciPy 1.17.0; pandas 2.2.3; scikit-learn 1.8.0; matplotlib 3.10.8.
O arquivo `requirements.txt` fixa as versões das bibliotecas usadas. Pequenas diferenças numéricas entre plataformas são possíveis.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python experiments.py --reps 100 --n 30000 --demo-n 100000
python make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```

Para compilar o PDF são necessários TeX Live ou equivalente com suporte a português, `newtx`, `amsmath`, `amsthm`, `mathtools`, `booktabs`, `hyperref`, `fancyhdr` e os demais pacotes do preâmbulo. As figuras são PDFs vetoriais; não é necessário converter imagens.

## O que foi executado

A comparação Monte Carlo usa 100 replicações independentes de 30.000 observações, com conversão esperada de 6% e tratamento contínuo confundido por covariáveis observadas. As nuisances dessa comparação são funções verdadeiras ou deliberadamente incorretas **pré-fixadas**. Isso isola a identidade de robustez; não é um benchmark de diferentes algoritmos de ML.

A execução adicional com 100.000 observações estima a conversão-base e a média da dose por histogram gradient boosting, sempre fora do fold avaliado. É uma única realização ilustrativa, não uma avaliação de cobertura.

A extensão não paramétrica com intervenções estocásticas tem provas e verificações determinísticas de integração. Não foi executado um benchmark empírico completo de um aprendiz flexível de razões de política.

## Rotina reutilizável

```python
from experiments import fit_multiplicative_dml

estimate = fit_multiplicative_dml(
    a=a,               # vetor de doses, referência a=0
    y=y,               # vetor binário de conversão
    v=v,               # matriz (n,d) de modificadores do efeito, incluindo 1 se necessário
    b_hat=b_hat_oof,    # predições fora da amostra de P(Y(0)=1 | X)
    m_a_hat=m_hat_oof,  # predições fora da amostra de E[A | X]
)
print(estimate.theta, estimate.se)
```

O modelo implementado é `mu(a,x)=b(x)*exp(a*v(x)'theta)`. O código **não** implementa uma superfície dose–resposta completamente irrestrita. `b_hat` e `m_a_hat` devem ser predições cross-fitted ou funções pré-fixadas; não há como a função detectar treinamento incorreto por parte de quem a chama. Para clientes com múltiplas simulações, é necessário adaptar os folds e a inferência ao agrupamento; o código sintético usa observações i.i.d.

O estimador rejeita incompatibilidades dimensionais, valores não finitos, raízes numericamente inadequadas e identificação muito mal condicionada. Isso não substitui diagnósticos de confundimento, suporte, forma funcional ou validade probabilística. Não é uma solução pronta para produção ou decisão de crédito.

## Interpretação e limites

1. A identidade Q para densidades é exata sob identificação causal e suporte. Ajustar a densidade de alocação incorretamente pode enviesar o Q-Learner simples.
2. O momento CQ-DML tem dupla robustez de consistência **sob o modelo correto da resposta relativa**. Não protege contra uma curva log-relativa mal especificada.
3. A inferência DML demonstrada no artigo exige dimensão fixa, condições de regularidade e taxas das nuisances. Dupla robustez de consistência não implica automaticamente dupla robustez dos intervalos de confiança.
4. As taxas brutas podem parecer contínuas mesmo quando só há alguns braços condicionais a X. Nessa situação, uma densidade contínua artificial não cria identificação entre ofertas.
5. A figura de Monte Carlo mostra intervalos para a **média entre replicações**, usando o erro Monte Carlo; não são intervalos de um único ajuste.
6. Não há dados de clientes nem resultados empíricos do funil real neste pacote.
7. O artigo reconhece os antecedentes em modelos multiplicativos, G-estimação, DML, razões de densidade e intervenções estocásticas. Não estabelece prioridade científica ou superioridade geral de um novo estimador.
