# Proposta (cap. 3) comparada com o estado atual, em 05/11/2026

Os caminhos citados são relativos à raiz ``.

**Visão geral:**
- **Implementação.** Cobre os requisitos funcionais do cap. 3, menos a medição de memória do RF05. Estão no código a CNN do projeto, os três segmentadores, a fusão, a busca em grade, a bayesiana com processo gaussiano, a partição da cena e a fusão concorrente.
- **Medição.** Está atrasada. Só o experimento 01 está completo. O 03 tem apenas Indian Pines (IP) com SEEDS e LSC. Os experimentos 02, 04, 05, 06, 07 e 08 estão pendentes.
- **Tempo de máquina.** Pelas rodadas antigas e pelo 03 em IP, faltam umas 6 a 7 horas, o que ainda cabe em uma ou duas noites.
- **Maior risco até 15/11.** É a reescrita do texto, não o tempo de máquina. Ainda descrevem o estado antigo: os caps. 4, 5 e 6, os apêndices A, B e C, o resumo, o abstract, a lista de siglas e trechos dos caps. 1 e 2.

## 1. Requisitos funcionais e não funcionais

| Req. | O que o cap. 3 pede (e o critério de sucesso) | Situação | Evidência | O que falta |
|---|---|---|---|---|
| RF01 | CNN 1D extrai as características espectrais. Critério: dimensões de cada camada iguais às previstas. | Atendido | `src/hsi/modelo.py`, arquitetura `"projeto"`: Conv1D 16/32/64/128, kernels 9/3/3/3 do original, padding same, max-pooling 2 depois das três primeiras, densas 256 e 128 com dropout, saída com C classes (logits; o softmax fica na perda). Conferido em 05/11: em IP (30 componentes) o vetor achatado tem 384 posições, mais 1 id, total de 385 na primeira densa, e a rede tem 166.416 parâmetros. Em PU e SA (15 componentes) são 128 + 1, com 99.977 e 100.880 parâmetros. | Não há teste automático de dimensões. Existem só a trava `MIN_COMPONENTS` e a seção 3 do notebook, que imprime a rede e o tamanho da entrada das densas. A `tab:camadas` do cap. 4 ainda descreve a CNN do Sydney (entrada de 1.025). |
| RF02 | Segmentação espacial com SEEDS, LSC e Watershed. Critério: mapas gerados para os três. | Atendido | `src/hsi/segmentadores.py` e `config.toml` `[segmentadores]`. Conferido: em IP saem mapas 145x145 com 81, 81 e 82 regiões; em PU 45/45/46; em SA 96/96/97 (`experimentos/resultados/medicoes_avulsas.md`, 04/11). | Desvios menores para o Apêndice C. O SEEDS ajusta superpixels, iterações e prior, mas não os níveis. O LSC ajusta o tamanho da região e a razão, com 10 iterações fixas. O Watershed usa marcadores em grade, sem o parâmetro de compactação nem a escolha do método de marcação. `qua:paramSeg` e `tab:estruturaSeg` ainda trazem SLIC e os parâmetros antigos. A figura `segmentacoes_ip.pdf` já foi regerada em 05/11, com a calibração. |
| RF03 | Fusão espectral + espacial. Critério: dimensão fundida igual à soma das partes, e classificação com fusão melhor que a puramente espectral. | Atendido; a segunda parte do critério foi desviada | `src/hsi/modelo.py`, `CNN1D.fundir`: o id do superpixel é concatenado ao vetor achatado (fusão tardia, 384+1 em IP). Por decisão, nenhum braço é treinado sem fusão. | No Apêndice C, a linha do one-hot cita o embedding "implementado para análise", e ele foi removido. A linha do critério do RF03 (Ap. C) e o quadro de requisitos do cap. 5 citam a entrada de 1.025; agora é 385. A `tab:ablacao` foi medida com a CNN do Sydney e inclui o embedding: retirar a tabela ou marcá-la como medida antiga. |
| RF04 | Otimização automática de hiperparâmetros. Critério: OA da melhor configuração acima da OA da configuração de Souza, nas mesmas condições. | Implementado; resultado parcial | `src/hsi/hpo.py`: grade segmentador x taxa x dropout (27 pontos) e bayesiana com processo gaussiano (Optuna GPSampler). A bayesiana sorteia 4 trials iniciais e começa por um de cada segmentador. No 03, IP com SEEDS otimizado dá OA 0,9324 ± 0,0062 com 5 sementes. No 01, com uma semente só, a CNN do projeto dá 0,9142 e a do Sydney 0,8960. | Falta o 03 restante (Watershed em IP, e PU e SA inteiros). Falta o 02, cujo braço SEEDS é a linha de base com 5 sementes. Falta o 04, que compara as estratégias e mostra a convergência. A linha do Apêndice C sobre a busca ainda diz "busca aleatória, TPE e genético" e precisa passar a grade + bayesiana com GP, com o Optuna no lugar do Ray Tune. |
| RF05 | Métricas de desempenho e custo: OA, AA, Kappa, tempo de pré-processamento, de treino e de inferência (total e por pixel), pico de memória, nº de parâmetros, speedup. | Parcial | `src/hsi/treino.py`: a função `metrics` dá OA, AA e Kappa, e `fit` mede o tempo de treino. Esse tempo vai para os CSVs do 02, 03 e 04; o 01 e o 05 não o gravam. Tempo de segmentação: `medicoes_avulsas.md` (04/11, só o segmentador) e a coluna `segmentacao` do 08 (PCA de 3 componentes, RGB e segmentador). Inferência: arranjo `lote` do 08. Para o tempo por pixel, subtrair `segmentacao` de `parede` e dividir pelo nº de pixels rotulados. | O **pico de memória não é medido em lugar nenhum**; existe só o PSS de 481 MB por processo, medido à mão. O nº de parâmetros não vai para os CSVs; dá para usar os valores conferidos acima (com 15 componentes, como no vencedor do 03 em IP, a rede de IP também tem 100.880). O pré-processamento completo (PCA espectral + segmentação) só aparece em `RAMOS`, de 03/09, com o PCA antigo. Decidir entre medir a memória em `run_experiment` antes de rodar o 02 ou declarar o desvio. |
| RF06 | Paralelizar o processamento espacial e espectral. Critério: speedup maior que 1 na mesma máquina. | Implementado; medição pendente | `prepare_data(config, backend="thread")` em `src/hsi/dados.py`: o PCA espectral roda em paralelo com o ramo espacial (PCA de 3 componentes e segmentação). `src/hsi/paralelismo/particao.py`: N faixas; executores sequencial, threads e processos; pool frio e quente. `src/hsi/paralelismo/fusao_concorrente.py`: T0 segmenta enquanto T1..TN fazem as convoluções, com `threading.Condition`; tem duas variantes e duas extensões da seção crítica. | Rodar 06, 07 e 08 com a CNN do projeto. O paralelismo entre os ramos do pré-processamento é usado no 06 e no 07, mas nenhum experimento o mede; só existe a medição avulsa de 03/09. |
| RNF01 | CNN baseada em Souza (2023) e Viel. | Atendido | `ARQUITETURAS["projeto"]` segue a figura do cap. 3 com os kernels do original. `ARQUITETURAS["original"]` (16/32/64/64, sem pooling, densas 128/64) aparece no 01. | O Apêndice C ainda declara como desvio "16/32/64/64 sem pooling". Agora a CNN é a prevista: a linha deve sair. |
| RNF02 | Tudo em Python. | Atendido | Python 3.12, PyTorch 2.13, OpenCV-contrib 5.0, Optuna 4.9 (`requirements.txt`, `ambiente.check_versions`). | Os comentários de `requirements.txt` ainda citam SLIC e TPE. O `README.md` está vazio (0 bytes). |
| RNF03 | Google Colab ou HPC com GPU. | Desviado, com justificativa | GPU local (RTX 5070 Laptop, 32 threads, 15,8 GB de RAM). As partições do 06 e do 07 e a inferência do 08 rodam na CPU de propósito, para comparar executores no mesmo dispositivo. O monolítico do 06, o treino do modo replicado e o treino do modelo do 08 usam a GPU. | Manter a linha do Apêndice C. A RAM impede o pool de 32 processos no 07. |
| RNF04 | Fusão por concatenação. | Atendido, sem o one-hot previsto | Concatenação do id escalar, como em Souza. | O one-hot não foi implementado: manter no Apêndice C, sem a menção ao embedding. |
| RNF05 | A busca automática deve melhorar OA, AA e Kappa em relação ao ajuste manual. | Parcial, com indício positivo em IP | 03 em IP com SEEDS, contra o 01 (semente 42): OA +1,82 pp, AA +7,36 pp, Kappa +2,09 pp. | Aplicar o critério de duas vezes o desvio combinado: braço SEEDS do 02 (5 sementes) contra o 03, nas três cenas, e o 05 em IP. |
| RNF06 | Resultados reprodutíveis. Critério: desvio menor que 2% entre execuções. | Atendido até aqui | `config.toml` + `ExperimentConfig`. `ambiente.configure_determinism`: cuDNN determinístico, algoritmos determinísticos do PyTorch e OpenCV com 1 thread. Dados iguais aos do Colab: `dados.apply_pca` e a segunda divisão de `dados.split_indices`, estratificada pelo one-hot. Desvios no 03 em IP, SEEDS/LSC: OA 0,62/0,45 pp, AA 1,62/1,81 pp, Kappa 0,70/0,52 pp. | Conferir a AA nas outras cenas: o LSC em IP já está em 1,8 pp. Threads no modo independente, o executor do 06, não repetem a OA entre execuções iguais. No 07 antigo a amplitude foi de até 2 pp com N de 2 a 16 e de 3,5 pp com N = 32. Escrever um README de reprodução. |
| RNF07 | Speedup maior que 1 sobre a versão sequencial. | Pendente | Mesmo código do RF06. | Rodar 06, 07 e 08. |

## 2. Experimentos planejados no cap. 3 e experimentos atuais

### 2.1 Exp. 1 a 5 do projeto

| Exp. do projeto | O que pede | Coberto por | Situação | O que ficou de fora |
|---|---|---|---|---|
| 1 Linha de base | CNN + SEEDS com parâmetros fixos, sequencial, 3 cenas, 5 sementes | 01 (CNN do projeto e do Sydney, hiperparâmetros do Sydney, 3 cenas, só a semente 42) e o braço SEEDS do 02 (mesma configuração, 5 sementes) | 01 feito; 02 pendente | A CNN do Sydney tem uma semente só, então não há variância para comparar com o valor reportado |
| 2 Otimização da segmentação | SEEDS, LSC e Watershed com busca dos parâmetros do segmentador e CNN fixa | 02 (CNN fixa, sem busca, como contraexemplo) e 03 (busca conjunta para cada segmentador) | 02 pendente; 03 parcial | Nenhum experimento faz a busca só dos parâmetros do segmentador com a CNN fixa |
| 3 Otimização da CNN | Busca da CNN com o SEEDS fixo | Linha SEEDS do 03 (a busca também varia a configuração do SEEDS; em IP o vencedor manteve a padrão, 350/200/2) e a grade do 04 (segmentador x taxa x dropout, segmentadores na configuração padrão) | Parcial | Não há experimento isolado "só a CNN, SEEDS fixo" |
| 4 Otimização conjunta | Busca de CNN e segmentador juntos, com o melhor segmentador | 03: por segmentador, 3 cenas, bayesiana com 16 trials, vencedor retreinado com 5 sementes. 04: grade x bayesiana no espaço inteiro, IP, 10% do treino, meta de validação de 0,95, 27 trials, orçamentos de 10/40/200 épocas, 3 sementes. 05: otimizada x original, IP | 03 parcial; 04 e 05 pendentes | 04 só em IP e com 10% do treino. O espaço de busca não tem filtros, kernels nem otimizador; as épocas entram só como orçamento no 04. O weight decay entrou sem estar no projeto |
| 5 Paralelização | Sobre a melhor configuração, nas 3 cenas | 06 partição (IP, N = 2 a 16, 3 sementes); 07 pool frio x quente (IP, N = 2 a 32); 08 fusão concorrente (IP no `config.toml`; PU e SA pela linha de comando) | Pendentes | Usa a configuração original, não a otimizada. 06 e 07 rodam com 25 épocas e lote 64, só em IP. A segmentação roda em CPU com threads, e não em processo + GPU como previsto. O PCA em paralelo com a segmentação fica sem experimento |

### 2.2 Matriz de experimentos (formato do `tab:planejamentoExperimentos`)

Os tempos foram estimados pelas rodadas antigas. Os do 02 e do 03 partem do 03 em IP, escalado pelo tamanho das cenas e pelas épocas do 01. Confirmar na execução.

| Atual | Cenas | Segmentação | Busca na CNN | Busca no segmentador | Paralelo | Repetições | Situação | Tempo estimado |
|---|---|---|---|---|---|---|---|---|
| 01 | IP, PU, SA | SEEDS | – | – | – | 1 semente | Feito | – |
| 02 | IP, PU, SA | SEEDS, LSC, WS | – | – | – | 5 sementes | Pendente | ~1 h |
| 03 | IP, PU, SA | os 3, um por vez | X | X | – | 16 trials + 5 sementes | IP SEEDS e LSC feitos | ~2,5–3 h |
| 04 | IP | os 3 | X | X (só a bayesiana) | – | 27 trials x 3 orçamentos x 2 estratégias x 3 sementes | Pendente | ~30–45 min |
| 05 | IP | os 3 | X (vencedor do 03) | X | – | 5 sementes | Pendente; depende do 03 em IP | ~20 min |
| 06 | IP | SEEDS | – | – | X | 3 sementes | Pendente | ~15 min |
| 07 | IP | SEEDS | – | – | X | 2 frio / 3 quente | Pendente | ~45–60 min |
| 08 | IP (PU e SA pela linha de comando) | SEEDS | – | – | X | 5 (3 com bloco de 1 pixel) | Pendente | ~15 min em IP; ~45 min nas 3 cenas |

**Observações de código que mudam o plano:**
- **O 05 repete o 02 e o 03.** As configurações e as sementes são as mesmas (11, 29, 42, 73, 101) e a execução é determinística. O braço "herdada" do 05 é igual ao 02, e o braço "otimizada" é igual aos retreinos do 03. Por isso o quadro do 05 para PU e SA sai dos CSVs do 02 e do 03, sem rodar nada. Em IP, rodar o 05 confere o determinismo e recupera os valores por semente de SEEDS e LSC. Esses valores se perderam porque `03_por_seed_*.csv` só é gravado no fim da chamada, e a execução foi interrompida.
- **O 02 não retoma.** Ele grava o CSV só no final e não usa `carregar_feito`: uma interrupção perde tudo.
- **06, 07 e 08 medem tempo.** Precisam rodar sozinhos, com a máquina ociosa, depois das campanhas na GPU.

## 3. Metas, hipóteses e critérios de sucesso

### 3.1 Premissas do cap. 3

| Premissa | Como será respondida | Indício atual |
|---|---|---|
| Há gargalos que paralelismo e busca automática podem reduzir; a fração do tempo de cada etapa será medida | Segmentação (`medicoes_avulsas.md`, 08), treino (02, 03, 04), inferência (08) | Rodada antiga em IP: pré-processamento ≈ 0,9% e inferência ≈ 0,4% de uma execução. O treino domina |
| As 3 cenas são representativas e comparáveis com Souza | Dados idênticos aos do Colab | Atendida |
| Divisão 85/7,5/7,5 | `dados.split_indices` | Vale em todos os experimentos. O 04 treina com só 10% do treino, com validação e teste inalterados. O 06 e o 07 mudam épocas (25) e lote (64), não a divisão |
| A grade e a bayesiana chegam perto do ótimo | 04: melhor validação por busca, curva "melhor até agora", 3 sementes, 3 orçamentos | Sem número válido ainda |
| Paralelizar reduz significativamente o tempo total | 06, 07, 08 | Contra a premissa: os ramos e a fusão concorrente atuam sobre menos de 2% da execução (lei de Amdahl). Na rodada antiga, a fórmula do cap. 3, (Tseg+Tcnn)/(max+overhead), dá teto de ≈ 1,5x em IP, 1,2x em PU e 1,5x em SA. O lote levou 0,33/1,05/1,00 s. Desse tempo, 0,11/0,87/0,67 s são da segmentação de T0, que inclui o PCA de 3 componentes e o RGB, e não só os 33/599/343 ms do SEEDS. Em PU e SA a segmentação domina, e o teto é baixo. O ganho grande veio da partição com processos |

### 3.2 Metas por cena (`tab:validacaoDatasets`)

| Cena | Meta OA / AA / Kappa | OA reportada por Souza | Melhor valor medido hoje | Situação |
|---|---|---|---|---|
| IP | ≥ 95% / ≥ 93% / ≥ 0,94 | 0,9142 | 03, SEEDS: 0,9324 ± 0,0062 / 0,9361 / 0,9229 | Só a AA atinge a meta. A OA fica 1,76 pp abaixo e o Kappa 0,017 abaixo. É improvável fechar a diferença com o espaço atual: as rodadas antigas de SEEDS em IP ficaram entre 0,930 e 0,936. O texto precisa dizer isso |
| PU | ≥ 98% / ≥ 97% / ≥ 0,97 | 0,9928 | 01, CNN do projeto, semente 42: 0,9841 / 0,9771 / 0,9789 | Atingida já na linha de base. Confirmar com 5 sementes no 02 |
| SA | ≥ 97% / ≥ 96% / ≥ 0,96 | 0,9965 | 01: 0,9938 / 0,9931 / 0,9931 | Atingida já na linha de base |

As metas de PU e SA ficam abaixo do valor reportado por Souza (o próprio cap. 3 admite isso) e já são batidas pela linha de base. Só IP testa a otimização.

Reprodução da CNN do Sydney (01, semente 42) contra os valores reportados: IP −1,82 pp, PU −0,75 pp, SA −0,14 pp. A OA da CNN do projeto em IP (0,9142) coincide com o valor reportado por Souza; é coincidência de uma semente.

### 3.3 Hipóteses

O cap. 3 não lista hipóteses. Elas aparecem na seção "Hipóteses" do cap. 6, como as hipóteses originais.

| Hipótese | Como será respondida | Indício atual |
|---|---|---|
| H1: os segmentadores têm desempenhos diferentes, e há alternativas ao SEEDS para cenários específicos | 02, 03 e os tempos de segmentação | Em IP, o LSC fica 1,72 pp abaixo do SEEDS, acima de duas vezes o desvio combinado (1,52 pp): a diferença a favor do SEEDS é afirmável. O Watershed segmenta 28x (IP), 57x (PU) e 62x (SA) mais rápido que o SEEDS. Falta a acurácia dele |
| H2: a busca automática supera o ajuste manual de forma reprodutível | Braço SEEDS do 02 contra o 03, nas 3 cenas; 05 em IP | +1,82 pp de OA em IP, mas contra uma semente só |
| H3: a otimização conjunta supera a isolada | **Não testável como formulada**: os Exp. 2 e 3 isolados não existem | Resposta parcial possível no 04: a grade varia só taxa, dropout e segmentador; a bayesiana varia o espaço inteiro. Mas estratégia e espaço ficam confundidos. Alternativa, que o cap. 6 já adota: declarar a hipótese não testada |
| H4: paralelizar reduz o tempo de forma mensurável | 06, 07, 08 | Rodadas antigas, com a CNN do Sydney e os dados anteriores. Partição com processos e pool quente: de 1,86x a 4,47x (N de 2 a 16). Threads só ganham com N = 2 (1,61x com pool quente). Fusão concorrente na forma do esquema, sobre o sequencial de mesmo bloco: até 1,46x em IP, ~1,1x em PU e SA, e abaixo de 1 com bloco de 1 pixel |

### 3.4 Critérios gerais

- **Teste estatístico.** O projeto previa testes de significância. Foi adotada a regra de duas vezes o desvio combinado, que já está no Apêndice C e é calculada pelo código do 05. O 03 grava `ganho_OA_pp` mas não aplica a regra: ela precisa ser aplicada no texto.
- **Objetivos do cap. 1.** Os objetivos específicos 2 e 3 e a "Solução Proposta" ainda falam em SLIC, busca aleatória, algoritmo genético e "quatro algoritmos". A "Solução Proposta" ainda diz que a reimplementação preserva a arquitetura da rede do original. Esses trechos precisam ser alinhados sem mudar o tamanho do capítulo.

## 4. Cronograma e riscos até 15/11

O cap. 3 não tem cronograma. O único prazo é a entrega em 15/11, daqui a 10 dias.

### 4.1 Riscos declarados no cap. 3

| Risco (probabilidade / impacto declarados) | Situação em 05/11 | O que importa até 15/11 |
|---|---|---|
| Busca de hiperparâmetros excede o tempo (média / alto) | Mitigado como o projeto previa: grade de 27 pontos, 16 trials no 03, 10% do treino no 04, 04 e 05 só em IP. Faltam ~4–5 h de GPU para 02 a 05 | Rodar primeiro o 03 e o 02, que alimentam RF04, RNF05 e as metas. Se faltar tempo, cortar o orçamento de 200 épocas do 04 ou usar 2 sementes |
| Recursos computacionais insuficientes (baixa / médio) | Ocorreu em parte: com 15,8 GB de RAM, o pool de 32 processos do 07 é recusado | Já tratado no código (`MemoryError`). Citar como limitação |
| Dificuldade de reproduzir a linha de base (média / médio) | Dados iguais aos do Colab. CNN do Sydney a −1,82 pp em IP com uma semente | Sem 5 sementes da CNN do Sydney, não dá para dizer se a diferença em IP está dentro da variação. Nenhum experimento faz isso: rodar `cnn="original"` com 5 sementes à parte (~15 min) ou declarar a limitação |
| Variabilidade alta (média / médio) | Controlada: determinismo e sementes; desvios ≤ 1,8 pp em IP | Exceção: threads no modo independente (executor do 06) não repetem a OA. A OA reprodutível da partição vem do sequencial e dos processos do 07 |

### 4.2 Riscos que o cap. 3 não previa

1. **Reescrita do texto.** Vários trechos ainda descrevem SLIC, TPE, busca aleatória, algoritmo genético, CNN do Sydney, embedding, 10% do treino e a numeração antiga (05 a 14). Estão nos caps. 4, 5 e 6, nos apêndices A, B e C, no resumo e no abstract (`_info.tex`), na lista de siglas e em trechos dos caps. 1 e 2. SLIC aparece em 10 arquivos (15 linhas só no cap. 5); TPE também em 10. Os caps. 1 a 3 não podem mudar de tamanho, e o texto principal precisa ficar abaixo de 80 páginas.
2. **Figuras.** Das 7 de `tcc/TCC III - EDUARDO SAVIAN/src/assets/resultados`, 5 já foram regeradas em 05/11 por `src/hsi/figuras.py`, com a CNN do projeto e os segmentadores calibrados. Ainda não foram commitadas.
   - Já regeradas: `curvas_treino`, `mapa_classificacao`, `matriz_confusao`, `segmentacoes_ip` e `particao_faixas`.
   - Faltam duas. `eixos_hpo` (04/11) depende do 04. `fusao_linha_do_tempo` (02/11) depende do 08 em PU, porque `f_fusao_linha_do_tempo` lê `08_linha_do_tempo_PU_*.csv`, e o `config.toml` roda o 08 só em IP.
3. **Temas proibidos.**
   - O arranjo "independente · aleatória" do 06 é um controle que puxa um tema que fica fora da monografia. Na rodada antiga, as faixas ficaram muito acima da partição aleatória (OA de ~0,91 contra 0,59 a 0,84). Decidir se ele entra no texto, ou rodar o 06 sem ele e economizar uns 40% do tempo de partição.
   - O rótulo "herdada" no CSV do 05 e o "esquema do orientador" nos docstrings e no notebook não podem passar para a monografia.
4. **Interrupções.** O 02 não retoma. O `03_por_seed` perde as linhas já feitas.
5. **Medições de tempo contaminadas.** Rodar 06, 07 ou 08 ao mesmo tempo que as campanhas na GPU distorce os speedups.

### 4.3 Ordem sugerida

- **05–06/11:** 03 restante (~2,5–3 h), depois 02 (~1 h), depois 05 em IP (~20 min).
- **07/11:** 04 (~45 min). Em seguida 06, 07 e 08 com a máquina ociosa (~2 h); o 08 também em PU e SA (`src/executar.py 08 PU SA`).
- **07–08/11:** regerar `eixos_hpo` e `fusao_linha_do_tempo` e montar as tabelas. O 05 de PU e SA sai do 02 + 03.
- **08–13/11:** reescrever os caps. 4, 5 e 6, os apêndices A, B e C, o resumo, o abstract e a lista de siglas; alinhar os objetivos do cap. 1 e os trechos do cap. 2 sem mudar o tamanho.
- **14/11:** revisão, contagem de páginas e compilação.

### 4.4 Decisões para levar ao orientador

- Rodar ou não a CNN do Sydney com 5 sementes.
- Medir a memória ou declarar o desvio.
- Manter a H3 como não testada, como o cap. 6 já faz.
- Assumir que a meta de OA e Kappa em IP não será atingida.
- Manter ou tirar o controle aleatório do 06.
- Rodar o 08 nas 3 cenas (PU é necessário para a figura da linha do tempo).