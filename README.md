# Classificação Hiperespectral com MLOps

Projeto autocontido no notebook [HSI_MLOps.ipynb](HSI_MLOps.ipynb). Ele compara CNNs 1D, 2D e 3D para imagens hiperespectrais, segmentadores, HPO, paralelismo, batches na GPU, explicabilidade visual e rastreabilidade de experimentos.

## Executar

1. Prepare o ambiente: `uv venv --python 3.12 && uv pip install -r requirements.txt`.
2. Coloque os arquivos `.mat` das cenas em `data/dataHSI/` ou defina `HSI_DATA_DIR`.
3. Abra `HSI_MLOps.ipynb` e execute as seções 1 a 7 na ordem indicada pelo roteiro.
4. Na seção 8, habilite apenas a execução que deseja fazer (`RUN_EXPERIMENT`, `RUN_HPO`, `RUN_AUDITED_CAMPAIGN` ou `RUN_BATCH_SWEEP`).

As células de implementação definem cada nome uma única vez: rodar uma delas duas vezes, ou fora de ordem, não muda o comportamento do notebook.

## Política de treino

A receita padrão é explícita e auditável:

- Meta: `val_accuracy >= 0.95`.
- Parada por estagnação: 10 épocas sem ganho mínimo de `0.001` em validação.
- Pesos: o estado da melhor época é restaurado ao atingir a meta ou a patience.
- Teste: OA, AA, Kappa, F1 e IoU são somente relato final; a seleção de HPO usa validação.

Os parâmetros `early_stopping_target`, `patience` e `early_stopping_min_delta` ficam juntos na seção 2.2 do notebook e devem ser ajustados como uma única política experimental.

## Hiperparâmetros

A busca é um *random search* semeado, por CNN. `HPO_SPACE` (seção 5.4) define os eixos sorteados — learning rate, dropout, weight decay, batch, patch e componentes de PCA — e cada trial ainda sorteia um segmentador e um conjunto dos seus parâmetros internos. Os primeiros trials cobrem obrigatoriamente os quatro segmentadores.

O produto cartesiano desse espaço passa de mil combinações por arquitetura; sortear cobre mais eixos que uma grade pequena pelo mesmo número de treinos, e continua reprodutível porque o gerador é semeado pela seed da configuração e pela arquitetura. A seleção usa exclusivamente a validação. `plot_hpo_space` mostra a validação obtida em cada valor de cada eixo, o que permite descartar eixos irrelevantes na campanha seguinte.

Trocar a estratégia de busca (hipertunador bayesiano, algoritmo evolutivo) exige substituir apenas `hpo_candidates`: é o único ponto que decide quais configurações treinar.

## Batches

O benchmark usa uma escada uniforme `50, 100, 150, 200, 250, 300`. O incremento fixo de 50 isola o efeito do tamanho do batch, revela o ganho marginal de throughput e evita saltos arbitrários até o limite de memória. O vencedor é o maior throughput dentro de 0,005 da melhor validação.

## Rastreabilidade e PDFs

Cada execução auditada grava configuração, ambiente, fingerprint dos dados, métricas, histórico e checkpoint em `outputs/runs/`. As campanhas ficam em `outputs/campaigns/`.

Os relatórios PDF são gerados automaticamente para execuções e campanhas auditadas. Eles incluem métricas, curvas, metadados e a transcrição dos documentos textuais/estruturados gerados (`.json`, `.csv`, `.md` e `.txt`); artefatos binários também são inventariados. Os PDFs ficam no diretório da respectiva execução ou campanha.

Ao fim de campanhas e sweeps, o notebook atualiza `outputs/reports/all_runs_report.pdf`: uma visão inteiramente visual de **todo o conjunto de runs**, com qualidade, estabilidade, custo, Pareto, HPO, early stopping, batches, evolução das curvas de treino, linha do tempo, acurácia por classe, validação×teste, atingimento da meta e repetibilidade por seed. Cada gráfico é acompanhado de uma nota sobre origem, legenda e interpretação.

Na seção de diagnósticos, os quatro segmentadores (SEEDS, SLIC, LSC e Watershed) também são mostrados como regiões, contornos sobrepostos ao mesmo PCA/RGB e gráficos de quantidade, escala, distribuição de área, densidade de borda e pureza. Os dashboards das CNNs exibem curvas, métricas, matriz de confusão, camadas, MaxPooling, Grad-CAM, saliência, incerteza e a projeção inversa das decisões na cena.

## Estrutura

- `HSI_MLOps.ipynb`: implementação, documentação e execução.
- `requirements.txt`: versões validadas, conferidas por `check_versions()` na seção 2.
- `data/`: cenas de entrada.
- `outputs/`: artefatos, relatórios, checkpoints e campanhas.
- `tcc/`: material acadêmico de referência.
