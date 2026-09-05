# Execução interrompida ao gravar metadados

O smoke do commit `05151d4`, seed=731905, foi interrompido por `OSError: [Errno 22] Invalid argument` ao regravar `targeted_metadata.json`. Os dados parciais foram preservados sem edição. Essa foi uma falha de execução, não uma replicação estatística descartada para melhorar métricas. A mesma configuração foi executada novamente em `../targeted_smoke`, sem mudança no código, nas sementes ou nos parâmetros. Não use esta tabela parcial como resultado final.
