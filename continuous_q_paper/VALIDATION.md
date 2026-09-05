# Validação da extensão de covariância e curvatura

Execução local em 5 de setembro de 2026. HEAD inicial confirmado: `5ec49ab35da48bc87fcf4af765e0a3475a236e56`, sem alterações locais. Nenhum `AGENTS.md` aplicável foi encontrado no projeto ou em seus diretórios ancestrais. Os estimadores discretos, geradores/protocolos históricos e resultados anteriores à rodada targeted foram preservados.

## Implementação e arquivos

- `relative_dml/_moment.py`: campos opcionais compatíveis `Estimate.cov` e `jac_singular_values`; solver/Jacobiano/sandwich compartilhados pela base geral interna e pelo wrapper linear público. A covariância já inclui `1/n`.
- `relative_dml/continuous.py`: `cov_`, `dose_degree=1/2`, curvatura comum, nuisance adicional do quadrado centrado fora do fold, momentos OOF inspecionáveis, ratios da mesma função e derivada na dose solicitada.
- `tests/test_learners.py`, `tests/test_continuous_quadratic.py`, `tests/test_targeted_benchmarks.py`: compatibilidade, covariância e contraste com termos cruzados, DR por células racionais exatas, ausência de vazamento, identificação, predição e protocolo de reporte.
- `continuous_q_paper/targeted_benchmarks.py`: cinco métodos, quatro desenhos fixos, médias/MCSE, diferenças pareadas, falhas/avisos e previsões inválidas preservadas; CSV, Markdown, LaTeX e metadados.
- `README.md`, `continuous_q_paper/README.md`, `continuous_q_paper/package_benchmarks.tex`, `continuous_q_paper/paper.tex` e `paper.pdf`: API, exemplos de contraste, benchmark efetivamente executado e limites.
- `continuous_q_paper/results/targeted_*`: rodadas finais, execuções anteriores identificadas e uma interrupção de escrita preservada. Este arquivo registra a verificação final.

Commits de implementação: `6e516e6` (covariância), `21f57b4` (quadrático), `c7617bc` (benchmark) e `05151d4` (correção de empates no ranking). A correção adicional surgiu na revisão; não houve seleção de configurações por desempenho. As rodadas finais usam `05151d4d5aee8060aecb50b2023cea64d887a68f`. Os metadados registram alterações de documentação ainda não commitadas e hashes dos fontes efetivamente utilizados.

## Ambiente e comandos executados

Python 3.9.13, NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.5.1 e pandas 2.2.2, usando `.venv`. As dependências já estavam disponíveis; a instalação editável evitou isolamento/downloads desnecessários:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[paper,test]" --no-build-isolation --no-deps

$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:PYTHONDONTWRITEBYTECODE="1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"
$env:PYTHONIOENCODING="utf-8"

# Antes de editar: 20 testes aprovados.
.\.venv\Scripts\python.exe -m pytest -q

# Validação final: 67 testes aprovados, zero falhas.
.\.venv\Scripts\python.exe -m pytest -q --basetemp tmp/pytest-final-rank
.\.venv\Scripts\python.exe examples/quickstart.py

.\.venv\Scripts\python.exe continuous_q_paper/targeted_benchmarks.py --reps 2 --n 3000 --test-n 1000 --seed 731905 --out continuous_q_paper/results/targeted_smoke
.\.venv\Scripts\python.exe continuous_q_paper/targeted_benchmarks.py --reps 30 --n 12000 --test-n 2000 --seed 831905 --out continuous_q_paper/results/targeted_v1
git diff --check
```

O quickstart executou as quatro classes e produziu previsões finitas. Antes da adaptação do ambiente, tentativas de pytest ficaram na criação de bytecode de SciPy fora da pasta gravável e foram interrompidas. `PYTHONDONTWRITEBYTECODE=1` resolveu o problema. A primeira suíte integrada também encontrou um erro no callback de limpeza do TEMP externo depois dos testes aprovados; o `--basetemp` local eliminou esse erro. A instalação emitiu avisos sobre distribuições antigas `-umpy`/`-cipy` do Anaconda, mas terminou com sucesso. Nenhum teste foi desativado por falha do projeto.

Houve também verificações por etapa: 21 testes após covariância, 37 testes específicos quadráticos e nove específicos do benchmark na versão final. Um smoke auxiliar de integração com `--reps 2 --n 300 --test-n 100 --seed 731905 --out tmp/targeted_agent_check` registrou seis falhas DML e dois avisos S-learner nessa amostra pequena; não foi usado como avaliação de referência nem para ajustar sementes ou parâmetros.

## Rodadas e auditoria dos resultados

| Saída | Configuração | Estado |
|---|---|---|
| `results/targeted_smoke` | 2 reps × 4 desenhos × 5 métodos; treino 3000, teste 1000; seed 731905 | 40 ajustes completos, zero falhas/avisos |
| `results/targeted_v1` | 30 reps × 4 desenhos × 5 métodos; treino 12000, teste 2000; seed 831905 | 600 ajustes completos, zero falhas/avisos |
| `results/targeted_smoke_before_rank_fix` | Mesma configuração do smoke; commit c7617bc | Completa, preservada; Spearman anterior não deve ser usado |
| `results/targeted_v1_before_rank_fix` | Mesma configuração da referência; commit c7617bc | Completa, preservada; Spearman anterior não deve ser usado |
| `results/targeted_smoke_interrupted_write` | Smoke corrigido; commit 05151d4 | Interrompido por OSError 22 ao gravar metadados; dados parciais preservados |

A revisão determinística descobriu que dividir médias para calcular a verdade e os ratios estruturados desfazia empates por X1 devido a arredondamento em X2. A correção usa diretamente a função log-relativa verdadeira e a diferença dos preditores lineares do comparador. Os testes exigem Spearman +1 e -1 nos exemplos exatos correspondentes. Repetimos os mesmos dados/sementes e configurações. Coeficientes e SEs ficaram idênticos; diferenças de RMSE/MAE entre versões ficaram abaixo de `2.23e-16`. Não alteramos o gerador, o ajuste nem o benchmark histórico.

A repetição do smoke após a correção teve uma interrupção de I/O (`OSError: [Errno 22] Invalid argument`) na regravação de `targeted_metadata.json`. A execução parcial foi preservada e repetida integralmente, com o mesmo código e sementes; completou. Isso é uma falha operacional registrada, não uma replicação removida para melhorar a tabela.

Auditoria independente conferiu todas as 640 linhas finais, ausência de duplicatas, seeds compartilhados entre métodos, streams de treino/teste independentes, hashes dos fontes, médias/MCSE e diferenças pareadas. A referência contém 64 comparações pareadas (4 desenhos × 4 candidatos × 4 métricas), todas com 30 pares. As 256 covariâncias DML das duas rodadas são finitas, simétricas e PSD na tolerância numérica, com diagonal igual a SE². As frações inválidas de log-média no treino/grid e o clipping S-learner foram zero nessas rodadas.

## Resultados e limites

No desenho linear, acrescentar curvatura elevou o RMSE DML de 0,0808 para 0,1067; a diferença pareada foi +0,0259, MCSE 0,0054. Nos três desenhos curvos, os RMSEs linear/quadrático foram 0,3782/0,0794, 0,3918/0,0754 e 0,5037/0,1478. A log-média quadrática teve RMSE médio 0,0736, 0,0705 e 0,1363: não há demonstração de superioridade geral de DML.

A heterogeneidade DML linear estimada foi −0,2029, −0,0415 e +0,3088 nas alocações curva aleatória/padrão/pouco overlap, mantendo a superfície causal. Sob curvatura omitida, esses valores são resumos dependentes da alocação. O modelo quadrático também perdeu precisão sob pouco overlap. No smoke desse desenho, o quadrático teve RMSE 0,9653, pior que os 0,6160 do linear; os resultados desfavoráveis foram mantidos.

Curvatura comum é uma restrição paramétrica, sem splines ou proteção geral contra misspecificação. O comparador log-média impõe também o baseline, correto nesta família sintética. A perda Poisson não afirma outcomes Poisson e o link log pode gerar médias inválidas em outros dados. O sandwich exige dados i.i.d., curva correta, identificação e taxas das nuisances; não cria inferência sob misspecificação arbitrária. Particionamento/inferência por cliente, políticas estocásticas e alocação conhecida direta continuam fora desta entrega. As métricas de erro de função não são teste de cobertura estrutural.

## PDF

O compilador local Tectonic 0.17.0 e seu cache estavam disponíveis. Comando efetivamente executado após atualizar o fonte:

```powershell
$env:TECTONIC_CACHE_DIR=(Resolve-Path -LiteralPath "tmp/tools/tex-cache").Path
.\tmp\tools\tectonic\tectonic.exe -X compile continuous_q_paper/paper.tex --only-cached --keep-logs --keep-intermediates --outdir tmp/pdfs/targeted-build
```

Uma compilação preliminar revelou uma fonte monoespaçada de 8pt ausente ao usar um nome de atributo dentro da fórmula. A notação foi corrigida e a compilação final terminou com sucesso. O PDF tem 26 páginas. As páginas 6, 13, 14, 17 e 18 foram renderizadas com Poppler e inspecionadas visualmente, incluindo API, covariância, distinção Q e tabela nova; não há texto/tabela cortado, sobreposição ou referência indefinida. O log final não contém overfull/underfull boxes, citações/referências indefinidas ou rótulos duplicados. Avisos do Tectonic sobre substituição de fontes do cache permanecem, sem defeito visual observado. O PDF recompilado substitui `continuous_q_paper/paper.pdf`.
