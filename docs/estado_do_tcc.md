# Resumo para a reunião com o orientador (05/11/2026; entrega em 15/11/2026)

## 0. O recado em 30 segundos

- **Implementação:** os três pilares do cap. 3 estão implementados com os métodos que o projeto previa:
  - segmentadores SEEDS, LSC e Watershed;
  - busca em grade e otimização bayesiana com processo gaussiano;
  - partição da cena e fusão concorrente;
  - a CNN do projeto (16/32/64/128 filtros, com max-pooling).
  - O espaço de busca da HPO é diferente do espaço do cap. 3 (ver a decisão 11).
- **Dados válidos com o código atual:** a linha de base (exp. 01) está completa. O exp. 03 tem só Indian Pines (IP) com SEEDS e LSC. Para o resto faltam cerca de 5,5 h de máquina com o 08 nas 3 cenas (cerca de 7,5 h no pior caso). Isso cabe em uma noite e uma manhã.
- **O gargalo é o texto, não a máquina.** Os caps. 4, 5 e 6, os apêndices e o Resumo ainda descrevem a versão anterior: SLIC, TPE/aleatória/genético, CNN do Sydney, 10% do treino e a numeração 05–14.
- **O que pedir ao orientador:** aval para o plano (rodar tudo com a CNN do projeto) e as decisões de escopo da seção 6.

---

## 1. Situação em uma página

### O que mudou nos últimos dias (commits f84ead6 e 57f4b6d)

| Mudança | Antes | Agora | Por quê |
|---|---|---|---|
| Volta aos métodos do projeto | SLIC como 4º segmentador; busca aleatória, TPE e algoritmo genético | Só SEEDS, LSC e Watershed; só grade (27 pontos) e bayesiana com processo gaussiano (Optuna GPSampler) | Fidelidade ao cap. 3. O código removido fica no histórico do git; os resultados da rodada com TPE/SLIC ficaram em `tpe_e_slic/`, só como registro |
| CNN | A do notebook do Sydney (16/32/64/64, sem pooling, densas 128/64) | CNN do projeto em todos os experimentos (16/32/64/128, kernels 9/3/3/3, max-pooling depois das três primeiras, densas 256/128, fusão tardia do id). A do Sydney (`cnn="original"`) aparece só no 01 | É a arquitetura da figura do cap. 3 (RF01/RNF01) |
| Calibração dos segmentadores | LSC e Watershed com 273 a 1.428 regiões | LSC e Watershed calibrados para o mesmo número de regiões do SEEDS (IP 81, PU 45, SA 96). Para a busca, há variações de cerca de 0,7x e de 1,23x a 1,35x. O LSC de IP chega a 1,23x (100 regiões), mas o comentário do `config.toml` ainda diz 1,35x | A comparação mede a forma das regiões, não a quantidade |
| Dados | PCA do sklearn atual; divisão pelos rótulos inteiros | PCA que emula o sklearn < 1.5 e segunda divisão estratificada pelo one-hot: dados idênticos aos do Colab | O PCA atual invertia componentes e mudava a segmentação em IP e PU |
| Configuração | Espalhada pelo código | `config.toml` na raiz, lido por `src/hsi/config.py` | Um valor mudado ali vale para o notebook, para o terminal e para os experimentos |
| Organização | Código misturado, com ramos sem uso | Um arquivo por pilar: `segmentadores.py`, `hpo.py`, `paralelismo/particao.py`, `paralelismo/fusao_concorrente.py`. Embedding, ramo em processo e `medicoes.py` foram removidos | Código enxuto |
| Numeração | 01, 05, 06, 08, 10, 11, 13, 14 | 01 a 08 sem lacunas (`experimentos/resultados/NUMERACAO.md`) | Ordem de leitura |

### Status

| Item | Estado |
|---|---|
| Código dos três pilares e da linha de base | **Pronto**, com um ajuste pendente: o executor do modo independente no `e06` (P0, item 1) |
| Notebook (18 células, só chama o pacote) e `src/executar.py` | **Pronto**, exceto o texto da célula 16 sobre retomada (seção 4) |
| Exp. 01, linha de base nas 3 cenas e com as 2 CNNs (semente 42) | **Pronto** |
| Exp. 03, segmentadores otimizados | **Parcial:** só IP com SEEDS e LSC |
| Exp. 02, 04, 05, 06, 07, 08 | **Pendentes** (nada rodando agora; GPU ociosa) |
| Figuras da monografia (7) | **Desatualizadas** (geradas de 02 a 04/11) |
| Monografia (`monografia.pdf` de 04/11, texto principal com 74 páginas) | Caps. 1 a 3 quase prontos; caps. 4 a 6, apêndices e Resumo descrevem o estado antigo |
| `README.md` | **Vazio** (0 bytes) |

### Requisitos do cap. 3

| Req. | Situação | Falta |
|---|---|---|
| RF01 CNN 1D | Atendido. 166.416 parâmetros em IP; vetor achatado de 384 + 1 id = 385 na primeira densa | `tab:camadas` do cap. 4 ainda mostra a rede do Sydney |
| RF02 Segmentação | Atendido. IP 81/81/82, PU 45/45/46 e SA 96/96/97 regiões | `qua:paramSeg` com os valores calibrados |
| RF03 Fusão | Atendido (concatenação do id). Por decisão, nenhum braço é treinado sem fusão | Tirar o embedding e a `tab:ablacao` |
| RF04 HPO | Implementado; resultado parcial | 03 completo, 04, 05 |
| RF05 Métricas | Parcial | **O pico de memória não é medido**; o nº de parâmetros não vai para os CSVs |
| RF06 / RNF07 Paralelismo | Implementado; medição pendente | 06, 07, 08 |
| RNF03 Colab/HPC | Atendido em parte: GPU local (RTX 5070 Laptop, 32 threads, 15 GB de RAM) em vez de Colab/HPC | Manter no Apêndice C |
| RNF04 Fusão por concatenação | Atendido (concatenação do id) | O id não é codificado em one-hot. Isso desvia do texto do cap. 3, não do requisito; registrar à parte no Apêndice C |
| RNF05 Busca melhora a manual | Indício positivo em IP | 02 + 05 com o critério de 2 desvios |
| RNF06 Reprodutibilidade | Atendido: repetir a linha de base de IP reproduz OA/AA/Kappa e épocas exatamente; desvio entre sementes até 1,81 pp (AA) | Threads no modo independente não repetem a OA |

---

## 2. Os pilares e a arquitetura

Critério usado em todo o trabalho: uma diferença é **afirmável** quando |dif| > 2 × hypot(desvio_a, desvio_b), com o desvio populacional entre sementes.

### 2.0 Arquitetura e linha de base (exp. 01)

**Pergunta.** A reimplementação reproduz Souza (2023)? A CNN do projeto é pelo menos tão boa quanto a do Sydney?

**Implementação.** `src/hsi/modelo.py` (`ARQUITETURAS["projeto"]` e `["original"]`), `pipeline.run_experiment`, `dados.py`.

**Dados** (`01_linha_base.csv`; hiperparâmetros do Sydney, SEEDS, treino completo, **só a semente 42**):

| Cena | CNN | OA | AA | Kappa | Épocas (melhor) | Reportado por Souza (OA / AA / Kappa) |
|---|---|---|---|---|---|---|
| IP | projeto | **0,9142** | 0,8625 | 0,9020 | 52 (42) | 0,9142 / 0,8956 / 0,9019 |
| IP | Sydney | 0,8960 | 0,8717 | 0,8813 | 63 (53) | |
| PU | projeto | 0,9841 | 0,9771 | 0,9789 | 26 (16) | 0,9928 / 0,9866 / 0,9905 |
| PU | Sydney | 0,9853 | 0,9781 | 0,9806 | 30 (20) | |
| SA | projeto | 0,9938 | 0,9931 | 0,9931 | 29 (19) | 0,9965 / 0,9970 / 0,9962 |
| SA | Sydney | 0,9951 | 0,9959 | 0,9945 | 25 (15) | |

**Leitura**
- Em IP, a CNN do projeto ficou +1,82 pp de OA acima da do Sydney: 66 contra 80 erros em 769 pixels de teste.
- A AA ficou −0,92 pp só por causa da classe Oats, que tem 2 pixels de teste e tira 3,13 pp da AA: a CNN do projeto acertou 0 de 2 e a do Sydney 1 de 2. As outras 15 classes somam +2,21 pp a favor da CNN do projeto.
- Em PU e SA a diferença é de −0,12 pp, dentro de qualquer ruído plausível.
- A CNN do projeto iguala a OA reportada em IP (0,9142). Com uma semente só, isso é coincidência.
- Distância da CNN do Sydney ao valor reportado: IP −1,82, PU −0,75 e SA −0,14 pp de OA.
- Indício pós-busca: otimizadas, as duas CNNs empatam em IP com SEEDS. A do Sydney deu 0,9355 ± 0,0103 (arquivo `cnn_do_sydney/`) e a do projeto 0,9324 ± 0,0062. A diferença é −0,31 pp contra um limiar de 2,40.

**Ressalvas**
- Uma semente só, então o critério não se aplica ao 01.
- O +1,82 em IP fica no limite. Os limiares de 0,45 e 2,91 pp citados antes vêm de desvios de configurações antigas:
  - 0,45 pp vem do desvio da OA com 100% do treino em FRACAO_TREINO (10/09, CNN do Sydney, 3 sementes, teto de 60 épocas, antes da fidelidade);
  - 2,91 pp vem do desvio do SEEDS otimizado com a CNN do Sydney (`cnn_do_sydney/06`).
- O único desvio atual é o do 03 em IP (0,62 pp). Ele dá limiar de 1,74 pp, e o +1,82 passa por 0,08 pp.

**Falta**
- 5 sementes da linha de base: saem do braço SEEDS do 02 e do 05.
- **Nenhum experimento roda a CNN do Sydney com 5 sementes.** É ela a referência formal "configuração do Sydney com fusão SEEDS". Ver a decisão 2 da seção 6.

### 2.1 Segmentadores (exp. 02 e 03)

**Pergunta.** Com o mesmo número de regiões, algum segmentador supera o SEEDS? A que custo de tempo? (H1)

**Implementação**
- `src/hsi/segmentadores.py`, com as seções `[segmentadores]` e `[hpo.*]` do `config.toml`.
- O 02 usa a configuração original fixa, 5 sementes, nas 3 cenas.
- O 03 faz, para cada segmentador, uma busca bayesiana de 16 trials restrita a ele; o vencedor é retreinado com 5 sementes.

**Dados válidos** (medição de 04/11 com o código atual, `medicoes_avulsas.md`). Cada célula traz regiões / pureza / ms por segmentação. O tempo é só o da chamada do segmentador, sem o PCA(3) e a composição RGB.

| Cena | SEEDS | LSC | Watershed |
|---|---|---|---|
| IP | 81 / **0,8919** / 33,3 | 81 / 0,8825 / 37,7 | 82 / 0,8330 / **1,2** |
| PU | 45 / **0,9272** / 599,1 | 45 / 0,8194 / 280,2 | 46 / 0,7918 / **10,6** |
| SA | 96 / **0,9886** / 343,1 | 96 / 0,9241 / 151,0 | 97 / 0,9563 / **5,5** |

Exp. 03 parcial (IP, treino completo, 5 sementes):

| Segmentador | OA | AA | Kappa | Vencedor |
|---|---|---|---|---|
| SEEDS | **0,9324 ± 0,0062** | 0,9361 ± 0,0162 | 0,9229 ± 0,0070 | taxa 3e-4, dropout 0,2, wd 0, lote 64, PCA 15; SEEDS no padrão (350/200/2) |
| LSC | 0,9152 ± 0,0045 | 0,8895 ± 0,0181 | 0,9031 ± 0,0052 | os mesmos hiperparâmetros; LSC no padrão (16 px, razão 0,075) |

**Leitura**
- **SEEDS contra LSC em IP:**
  - OA −1,72 pp contra limiar de 1,52: **afirmável**.
  - Kappa −1,98 contra 1,74: **afirmável**.
  - AA −4,67 contra 4,85: ruído.
- Com o mesmo número de regiões, o SEEDS tem a maior pureza nas 3 cenas.
- O Watershed segmenta 28x (IP), 57x (PU) e 62x (SA) mais rápido que o SEEDS.
- A busca manteve a configuração padrão de cada segmentador e escolheu os mesmos hiperparâmetros da CNN para os dois. O ganho vem da CNN, não da segmentação.

**Indícios de arquivo** (outra configuração)
- Em nenhuma rodada antiga uma alternativa superou o SEEDS de forma afirmável.
- Com a CNN do Sydney e o LSC sem calibração (cerca de 2x as regiões do SEEDS), o LSC empatava: −0,26 pp.
- **O resultado novo vale "com número de regiões comparável".** O texto precisa apresentar isso como escolha de desenho.

**Falta**
- Watershed em IP.
- PU e SA inteiros no 03.
- O 02 inteiro. Ele mostra se a configuração fixa ainda prejudica algum segmentador. O contraexemplo antigo, do SLIC com Kappa 0,03 em PU, dependia das 1.428 regiões e provavelmente desaparece.

### 2.2 Otimização de hiperparâmetros (exp. 03, 04 e 05)

**Pergunta.** A busca automática supera a configuração do Sydney de forma reprodutível (RF04, RNF05, H2)? Grade e bayesiana diferem? (RF04)

**Implementação** (`src/hsi/hpo.py`)
- Espaço com 7 eixos de 3 valores (2.187 combinações): segmentador, configuração do segmentador, taxa, dropout, weight decay, lote e PCA. É diferente do espaço do cap. 3 (decisão 11).
- **Grade:** segmentador × taxa × dropout, 27 pontos, com os demais eixos no valor original.
- **Bayesiana:** Optuna GPSampler no espaço inteiro, com 4 trials iniciais sorteados, os 3 primeiros com um segmentador cada. No 03, que é restrito a um segmentador, há 1 trial enfileirado e 3 sorteados.
- Ray Tune foi trocado por Optuna porque o BayesOptSearch do Ray só aceita eixos contínuos.

**Dados válidos**
- SEEDS otimizado em IP (03): 0,9324 / 0,9361 / 0,9229.
- Diferença para a linha de base do 01, que tem uma semente só: OA +1,82, AA +7,36 e Kappa +2,09 pp sobre a CNN do projeto, e OA +3,64 pp sobre a CNN do Sydney.
- Se o desvio da linha de base for igual ao do 03, os limiares seriam 1,74 (OA), 4,57 (AA) e 1,98 (Kappa). A AA passaria com folga; a OA passaria por 0,08 pp e o Kappa por 0,11 pp. **Quem decide é o 05.**
- A validação do vencedor fica de 1,6 a 2,4 pp acima do teste, o efeito esperado de escolher o melhor de 16.

**Indícios de arquivo**
- Com 10% do treino (CNN do Sydney, TPE), o SEEDS foi de 0,7441 para 0,8174 (+7,33 pp, afirmável). **É o número que está hoje no Resumo e vai sair.**
- Com o treino completo, o ganho encolhe para cerca de 2 a 4 pp e se concentra em IP. PU e SA já estão no teto: +0,06 e +0,01 em `tpe_e_slic`. Isso é só registro e fica fora do texto.
- Estratégias antigas (aleatória, TPE, genético): empate. O resultado provável do 04 também é empate.
- O orçamento pesou mais: melhor validação de cerca de 0,72 → 0,79 → 0,82 com 10, 40 e 200 épocas.
- Peso dos eixos na validação: dropout 18,1 pp, taxa 15,3, PCA 8,4, weight decay 4,6, lote 1,8. É o que justifica a grade em taxa × dropout.

**Metas do cap. 3**

| Cena | Meta OA / AA / Kappa | CNN do projeto (fonte) | Situação |
|---|---|---|---|
| IP | 0,95 / 0,93 / 0,94 | 0,9324 / 0,9361 / 0,9229 (03) | Só a AA. OA a 1,76 pp e Kappa a 1,71 pp; rodadas antigas ficaram entre 0,930 e 0,936 |
| PU | 0,98 / 0,97 / 0,97 | 0,9841 / 0,9771 / 0,9789 (01, semente 42) | Atingida na linha de base |
| SA | 0,97 / 0,96 / 0,96 | 0,9938 / 0,9931 / 0,9931 (01, semente 42) | Atingida na linha de base |

**Falta**
- **05:** ganho afirmável sobre a configuração original, em IP, com treino completo e 5 sementes. Ele também precisa repetir 0,9324 e 0,9152, o que confere o determinismo.
- **04:** grade contra bayesiana, com 27 trials, orçamentos de 10/40/200 épocas, 3 sementes e 10% do treino.
- Resto do 03: se a busca acrescenta algo em PU e SA.
- **H3 (conjunta melhor que isolada) não é testável como formulada:** os Exp. 2 e 3 isolados do projeto não existem.

### 2.3 Paralelismo (exp. 06, 07 e 08)

**Pergunta.** Paralelizar reduz o tempo de forma mensurável na mesma máquina (RF06, RNF07, H4)?

**Implementação**
- `dados.prepare_data(backend="thread")`: PCA e segmentação em paralelo.
- `paralelismo/particao.py`: cena em N faixas, com executores sequencial, threads e processos, e pool frio ou quente.
- `paralelismo/fusao_concorrente.py`: T0 executa o ramo espacial inteiro (PCA(3), composição RGB e segmentação) enquanto T1..TN fazem as convoluções. A concatenação espera o mapa (`threading.Condition`).

**Dados válidos:** nenhum ainda. Todos os números abaixo vêm de **rodadas antigas, com a CNN do Sydney e os dados de antes da correção de fidelidade**.

Partição, speedup contra o sequencial nas mesmas partições (`cnn_do_sydney/13`):

| Executor | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|
| Processos, pool quente | 1,86 | 2,87 | 3,68 | **4,47** |
| Processos, pool frio | 1,20 | 1,46 | 1,37 | 1,05 |
| Threads, pool frio | 1,63 | 0,61 | 0,23 | 0,24 |

- Com threads no modo independente, a OA variou entre execuções iguais de 0,91 a 1,95 pp com N=2 a 16, e até 3,51 pp com N=32. Com sequencial e processos variou 0,00 pp, mas em N=32 não há braço de processos para comparar.
- Processos com N=32 foram recusados por memória (481 MB cada).

Fusão concorrente (`antes_da_fidelidade/14`):

| Cena | Contra o lote com 1 thread | Contra o lote com 16 threads |
|---|---|---|
| IP | até 2,01x | 1,20x |
| PU | 1,21x | 1,01x |
| SA | 1,50x | 1,11x |

- Com bloco de 1 pixel, foi de 7 a 35 vezes mais lento.
- As 723 execuções repetiram os ids e as classes do lote.

Ramos do pré-processamento em threads (03/09): IP 1,05x, PU 0,95x e SA 1,22x.

**Leitura (lei de Amdahl)**
- Pré-processamento (cerca de 0,9%) e inferência (cerca de 0,4%) são uma fração pequena de uma execução em IP (rodada antiga). O treino domina.
- Pela definição da monografia (lote / T0), os tetos da sobreposição na rodada antiga eram 2,98x em IP (T0 de 0,111 s), 1,21x em PU e 1,49x em SA.
- Os speedups medidos foram 2,01x, 1,21x e 1,50x. PU e SA atingiram o teto, e IP teve o maior ganho medido. Os dois conjuntos de números vêm da CNN do Sydney, antes da fidelidade.
- Os 33/599/343 ms da seção 2.1 medem só a segmentação, sem o PCA(3) e a composição RGB que o T0 executa. Na rodada antiga, o T0 de IP levou cerca de 0,1 s, contra 31,9 ms do SEEDS sozinho. O teto com a CNN do projeto só se conhece depois do 08.
- **O ganho grande vem da partição com processos e pool quente.**

**Falta**
- 06, 07 e 08 com a CNN do projeto. Espera-se a mesma forma qualitativa e números diferentes.
- O 08 está configurado só para IP. Levá-lo às 3 cenas dá cobertura: compara cenas com tempos de segmentação diferentes e gera a figura `fusao_linha_do_tempo`, que é feita a partir de PU.
- Os ramos não têm experimento atual; só existe a medição avulsa de 03/09.

### 2.4 Organização do código e reprodutibilidade

- **Execução determinística** (`ambiente.configure_determinism`: cuDNN determinístico, OpenCV com 1 thread).
  - Repetir a célula 9 (IP, CNN do projeto, semente 42) reproduziu o 01 exatamente: OA 0,9142, AA 0,8625, Kappa 0,9020, 52 épocas, melhor época 42.
  - No 03 em IP (SEEDS/LSC), o desvio padrão entre sementes foi de 0,62/0,45 pp na OA e de 1,62/1,81 pp na AA. Isso mede a sensibilidade à semente, não a reprodutibilidade. A amplitude não é conhecida, porque o `03_por_seed` não foi gravado.
- **Defeitos encontrados e corrigidos:**
  - **LSC não determinístico** com 32 threads: muda de 188 a 322 pixels entre chamadas iguais em IP. Com 1 thread fica estável, mas 2,4x mais lento (35,6 contra 14,8 ms).
  - **Piso de componentes do PCA:** 15 na CNN do Sydney e 8 na do projeto.
- **Defeito encontrado e contornado, não corrigido:** o gerador aleatório global do PyTorch é compartilhado entre threads, porque cada partição chama `seed_everything` no gerador global.
  - Efeito: com threads no modo independente, a OA varia até 3,51 pp (N=32), e de 0,91 a 1,95 pp com N=2 a 16.
  - Contorno: as acurácias de partição vêm só do executor sequencial ou de processos, e as threads ficam só na medição de tempo (07).
  - O 06 ainda usa `executor="thread"`; a troca é o P0, item 1.
- **Fidelidade aos dados do Colab:** PCA do sklearn < 1.5 e segunda divisão pelo one-hot.

---

## 3. Demonstração (5 a 10 min)

**Antes da reunião:** rodar as células 1 a 9 uma vez, porque a célula 9 leva cerca de 1 min e a 13 usa o modelo dela. Deixar abertos `config.toml`, `experimentos/resultados/01_linha_base.csv` e `03_resumo.csv`.

| # | Seção do notebook (célula) | O que mostrar | O que prova |
|---|---|---|---|
| 0 | Estrutura do repositório (30 s) | `src/hsi/` com um arquivo por etapa ou pilar; `config.toml`; `experimentos/resultados/` com os CSVs novos e as pastas de arquivo | Organização (pilar de código) |
| 1 | Índice (célula 0) | A tabela "etapa → o que a célula mostra → arquivo" | Que o notebook só chama o pacote |
| 2 | 1. Configuração (célula 3) | `check_versions`, dispositivo, `configure_determinism`; os padrões vêm de `[execucao]` | RNF02, RNF06; config única |
| 3 | 2. Dados e segmentação (célula 5) | RGB de IP e as regiões de SEEDS, LSC e Watershed lado a lado | RF02; a calibração (81/81/82 regiões) |
| 4 | 3. CNN (célula 7) | `print(modelo)`: 166.416 parâmetros, densas com 384 características + 1 id | RF01, RF03, RNF01 (CNN do projeto) |
| 5 | 4. Treino (célula 9) | Curvas e métricas: OA 0,9142, 52 épocas, melhor época 42, **iguais ao CSV do 01** | Linha de base e determinismo |
| 6 | 5. Busca (célula 11, `BUSCAR=True`, cerca de 1 min) | 8 trials bayesianos curtos com 10% do treino: segmentador, taxa, dropout, lote, PCA, validação | RF04 funcionando; o objetivo é a validação e o teste só é relatado |
| 7 | 6. Fusão concorrente (célula 13) | Tempo do lote contra o concorrente; **classes iguais às do lote** | RF06; o esquema discutido com o orientador está implementado e correto |
| 8 | 7. Partição (célula 15, opcional, cerca de 3 min) | 4 faixas com executor sequencial, threads e processos, uma execução cada, pool frio: tempo do bloco e OA | Com N=4: processos mais rápidos que o sequencial e com a mesma OA; threads mais lentas. A escala e a repetibilidade ficam para o 07 |
| 9 | 8. Experimentos (células 16 e 17) | Tabela 01 a 08; `.venv/bin/python src/executar.py 03 IP`; abrir `01_linha_base.csv` e `03_resumo.csv` | Como cada número da monografia é produzido |

**Para fechar a demonstração:** a tabela da seção 2.0 e a tabela do 03 (seção 2.1) são os dois resultados válidos. O resto do plano está na seção 5.

---

## 4. Documentação existente

| Documento | Onde | Estado |
|---|---|---|
| Monografia | `tcc/TCC III - EDUARDO SAVIAN/src/*.tex`; `monografia.pdf` de 04/11 | Desatualizada a partir do cap. 4 (tabela abaixo) |
| Guia do código | `docs/guia_do_codigo.md` | Cobre o fluxo de uma execução (config → dados → modelo → treino) e os pilares e experimentos |
| Notebook | `Eduardo_Savian_2026_CNN1D_SEEDS_otimizado.ipynb` (18 células) | Atual, exceto o texto da célula 16 sobre retomada. Ele diz que cada campanha grava o CSV a cada medição e continua de onde parou, mas o 01 e o 02 só gravam no fim; a docstring de `infra.py` repete o mesmo erro. As células 0 e 12 e os docstrings de `src/hsi/paralelismo/` dizem "o esquema do orientador" (aceitável no notebook e no código, não no texto) |
| Configuração | `config.toml` | Atual: todas as variáveis de execução e de experimento |
| Numeração | `experimentos/resultados/NUMERACAO.md` | Atual: novo ← antigo |
| Arquivo de resultados | `LEIA-ME.md` em `cnn_do_sydney/`, `tpe_e_slic/`, `antes_da_fidelidade/`, `fracao_10/`, `invalidados_lsc/` | Dizem com que configuração cada rodada antiga foi medida |
| Medições avulsas | `experimentos/resultados/medicoes_avulsas.md` | Segmentadores e LSC/threads válidos (04/11); ramos, ablação, fração do treino, GIL e threads são antigos |
| README | `README.md` | **Vazio**. Falta um README de reprodução (cerca de 1 h) |
| `requirements.txt` | raiz | Os comentários ainda citam SLIC e TPE |

### Estado da monografia por capítulo

| Parte | Estado | O que fazer | Depende de |
|---|---|---|---|
| Capa | Data "Dezembro de 2026" | Conferir se é o mês da entrega (novembro) | — |
| Resumo / Abstract (481 palavras) | Quase tudo antigo | Reescrever, sem passar de 500 palavras; a frase da linha de base já pode entrar | 03, 04, 05, 07, 08 |
| Siglas (pré-textual) | Pequenos ajustes | Tirar SLIC, TPE e siglas não usadas; incluir CUDA, MB e GB | — |
| Cap. 1 | Quase pronto | Objetivos 2 e 3, "quatro algoritmos", "arquitetura preservada", citações SLIC/TPE, sem mudar o tamanho | — |
| Cap. 2 | Quase pronto | 4 pontos: parágrafo do SLIC, parágrafo da HPO (aleatória/TPE/genético → grade + processo gaussiano), quadro comparativo, l. 489/501 | Só a l. 419 depende do 04 |
| Cap. 3 | Pronto (é a proposta) | Não mexer | — |
| Cap. 4 | **Reescrever o método** | `tab:camadas` (166.416), `qua:paramSeg` calibrado, HPO com grade e processo gaussiano, `qua:metodosHPO`, espaço de 2.187, quadro de experimentos 01–08, parada com teto de 200, 10% só no 04 | Quase nada; só os números de determinismo (07/08) |
| Cap. 5 | **Depende de resultado** | Já dá para fazer: linha de base (`tab:linhaBase`, por classe), `tab:estruturaSeg`, LSC com threads, remover SLIC, embedding, ablação e fração do treino | 02 a 08 |
| Cap. 6 | Depende de resultado | Objetivos e hipóteses com os números novos; H3 "não testada" | 03, 04, 05, 07, 08 |
| Apêndice A | Depende do 03 | Tabela de vencedores; IP SEEDS/LSC já disponíveis | 03 |
| Apêndice B | Misto | GIL e divisão continuam válidos; por classe e LSC dá para fazer já; o resto espera | 03, 04, 05, 07, 08 |
| Apêndice C | Reescrever os quadros | Saem 4 desvios (PyTorch, CNN, SLIC, aleatória/TPE/genético). Entram Optuna/Ray, grade restrita, calibração, sementes por experimento, linha de base com 2 CNNs e o id sem one-hot (desvio do texto do cap. 3, não do RNF04). A linha "Épocas, filtros, tamanho dos kernels e otimizador..." precisa de motivo novo, porque a CNN agora é a do projeto | Pouco |
| Figuras (7) | Antigas | Dá para regerar já: `curvas_treino`, `mapa_classificacao`, `matriz_confusao`, `segmentacoes_ip`, `particao_faixas` | `eixos_hpo` depende do 04; `fusao_linha_do_tempo` do 08 em PU |

---

## 5. O que falta até 15/11

### Prioridades

**P0, código, antes de rodar (cerca de 1 h, hoje)**
1. **06:** trocar o executor do modo independente de `thread` para `sequencial` ou `process`. Com threads a OA não se repete; o tempo de threads já sai do 07.
2. **08:** `cenas = ["IP", "PU", "SA"]` no `config.toml` (+0,3 h de máquina).
3. Mover os logs soltos com a numeração antiga (`05_*`, `06_*`, `08_IP.log`, `10_*`, `14_*`, `09_*`, `12_*`) para as pastas de arquivo. Eles dividem prefixo com as saídas novas.
4. Opcionais:
   - tornar o 02 retomável (hoje ele só grava no fim) ou corrigir o texto da célula 16 e da docstring de `infra.py`;
   - gravar o `03_por_seed` a cada segmentador;
   - braço da CNN do Sydney com 5 sementes (decisão 2).

**P1, máquina na GPU, em fila e sem nada mais rodando**

| Exp. | Estimativa |
|---|---|
| 03 IP (retoma no Watershed) | ~10 min |
| 05 (IP) | ~20 min |
| 04 | ~25 min (0,4 h) |
| 03 PU SA | ~2 h (pessimista ~4 h) |
| 02 | ~1,1 h |

Os três primeiros somam cerca de 1 h e dão um resultado válido por pilar de HPO e segmentadores.

**P2, máquina ociosa, medições de tempo**

| Exp. | Estimativa |
|---|---|
| 08 nas 3 cenas | ~0,55 h |
| 06 | ~0,25 h |
| 07 | ~0,65 h |

**P3, extras**
- CNN do Sydney com 5 sementes: ~20 min nas 3 cenas, mais código novo.
- Repetir o 01 em IP para conferir histórico e pesos: ~2 min.
- Remedir os ramos ou cortá-los.
- Regerar as figuras.

**Totais de máquina:** cerca de **5,5 h** com o 08 nas 3 cenas (5,2 h se ficar só em IP); pior caso cerca de 7,5 h. Somam-se 0,6 h de extras.

**P4, escrita** (estimativa minha, sem base em medição)

| Bloco | Horas |
|---|---|
| Cap. 4 | ~6 |
| Cap. 5 | ~12 |
| Cap. 6 | ~3 |
| Apêndices A, B e C | ~6 |
| Resumo / Abstract | ~2 |
| Caps. 1 e 2 e siglas | ~3 |
| Figuras e tabelas | ~3 |
| Revisão final (páginas, palavras proibidas, referências) | ~5 |
| **Total** | **~40 h**, cerca de 5 h por dia por 8 dias |

### Cronograma dia a dia

| Dia | Máquina | Escrita / outros |
|---|---|---|
| **05/11 qui** | P0 (~1 h). À noite, fila na GPU: 03 IP → 05 → 04 → 03 PU SA → 02 (~4–6 h) | **Reunião.** Se sobrar tempo: siglas e cap. 1 |
| **06/11 sex** | Manhã: conferir os CSVs. O 05 tem que repetir 0,9324 e 0,9152. Máquina ociosa: 08 → 06 → 07 (~1,5 h). Regerar as figuras | Cap. 4 (não depende de resultado) |
| **07/11 sáb** | Folga para refazer o que falhar; extras (CNN do Sydney, 01 repetido, ramos) | Fim do cap. 4 + Apêndice C |
| **08/11 dom** | — | Cap. 5: protocolo, linha de base, organização do código, segmentadores (02/03) |
| **09/11 seg** | — | Cap. 5: HPO (04/05) + Apêndice A |
| **10/11 ter** | — | Cap. 5: paralelismo (06–08), verificação de requisitos e metas + Apêndice B |
| **11/11 qua** | — | Cap. 6 + os 4 pontos do cap. 2 |
| **12/11 qui** | — | Resumo e Abstract, README de reprodução, compilar. **Enviar a versão completa ao orientador** |
| **13/11 sex** | — | Revisão: < 80 páginas, ≤ 500 palavras, termos proibidos, numeração 01–08, referências cruzadas |
| **14/11 sáb** | — | Incorporar os comentários do orientador; compilação final |
| **15/11 dom** | — | **Entrega** |

---

## 6. Decisões e perguntas para o orientador

| # | Decisão | Opções | Minha sugestão |
|---|---|---|---|
| 1 | Rodar tudo de novo com a CNN do projeto ou reduzir o escopo | Tudo (~5,5 h) ou só IP / só alguns experimentos | **Tudo.** A máquina não é o gargalo; os resultados antigos não são comparáveis (outra CNN, outros dados, 10% do treino) |
| 2 | Referência dos ganhos (configuração do Sydney com fusão SEEDS) | Acrescentar a CNN do Sydney com 5 sementes (~20 min, código novo) ou declarar a limitação | **Acrescentar.** Sem isso, a cadeia "Sydney → CNN do projeto → otimizada" fica com uma semente no primeiro elo |
| 3 | Calibração dos segmentadores pelo nº de regiões do SEEDS | Manter como desenho, ou também mostrar o LSC sem calibração | Manter e escrever como escolha de desenho. Sem calibração o LSC empatava; calibrado, perde |
| 4 | H3 (conjunta melhor que isolada) | Declarar não testada, ou responder parcialmente pelo 04 (estratégia e espaço ficam confundidos) | Declarar não testada |
| 5 | Meta de OA/Kappa em IP (0,95 / 0,94) | Assumir que não será atingida | Assumir: a meta está acima do próprio valor reportado (0,9142) |
| 6 | Controle por partição aleatória no 06 | Manter ou tirar | Tirar: economiza cerca de 40% do tempo do 06 (indício da rodada antiga) e evita um tema fora do escopo da monografia |
| 7 | 08 nas 3 cenas | Só IP ou IP, PU e SA (+0,3 h) | 3 cenas, pela cobertura: compara cenas com tempos de segmentação diferentes e gera a figura `fusao_linha_do_tempo`, feita a partir de PU |
| 8 | Pico de memória (RF05) | Instrumentar o `run_experiment` antes do 02 ou declarar o desvio | Declarar o desvio e usar a memória por processo (481 MB) e por trial |
| 9 | Medições avulsas antigas (ramos, ablação da fusão, fração do treino, threads do PyTorch) | Refazer ou cortar | Cortar a ablação e a fração do treino; os ramos, refazer rápido ou tirar do Resumo |
| 10 | Notebook como fonte de verdade | Notebook ou pacote | O pacote é a fonte de verdade; o notebook é a vitrine executável; mais um README de reprodução |
| 11 | Desvios do projeto | Optuna no lugar do Ray Tune. Exp. 2 e 3 isolados não executados. Grade restrita a 3 eixos. Espaço de busca sem filtros, kernels, épocas e otimizador, e com weight decay, PCA e configuração do segmentador (motivo novo: arquitetura fixa na do cap. 3 e orçamento de épocas estudado no 04). Id sem one-hot (desvio do texto do cap. 3) | Confirmar que basta registrar no Apêndice C |
| 12 | Se faltar tempo, o que cortar (nesta ordem) | (a) orçamento de 200 épocas do 04 ou 2 sementes; (b) N=32 do 07; (c) braço Watershed do 03 em PU e SA, ou o 03 inteiro de SA; (d) 08 só em IP | — |
| 13 | Calendário | Quando ele lê a versão completa? Capa com novembro ou dezembro? | Propor 12/11 para a leitura |

---

## 7. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Reescrita não fecha até 15/11 (SLIC, TPE e CNN antiga espalhados por 9+ arquivos) | Alta | Alto | Começar o cap. 4 já, porque não depende de resultado. Cap. 5 por blocos. Cortar as tabelas do Ap. B se apertar |
| 03 em PU e SA no cenário pessimista (~4 h) ou interrupção | Média | Médio | Rodar à noite; o 03 retoma. **O 02 não retoma:** rodar inteiro numa janela só |
| O 05 não repetir 0,9324 / 0,9152 | Baixa | Alto | Indicaria quebra de determinismo; conferir na manhã de 06/11 |
| Medições de tempo contaminadas por campanhas na GPU | Média | Médio | 06, 07 e 08 sozinhos, depois da fila na GPU |
| O 02 não mostrar colapso, e a tese da "CNN fixa inválida" cai | Alta | Baixo | Reescrever como "a CNN fixa mede quanto a configuração original combina com cada segmentador" |
| Ganho em IP não afirmável no 05 (OA/Kappa no limite) | Média | Médio | Reportar como está; RF04 "atendido em parte" |
| Meta de IP não atingida | Quase certa | Baixo | Explicar: a meta está acima do valor reportado por Souza |
| Texto principal > 80 páginas ou Resumo > 500 palavras | Média | Médio | Cortar ablação, fração do treino, ramos e a seção do embedding |
| Termos proibidos passarem para o texto: o rótulo `herdada` do CSV do 05, a atribuição do esquema no notebook e nos docstrings, o controle aleatório do 06 | Média | Médio | Busca textual (grep) final no `.tex` antes de compilar |
| Threads no 06 com OA não repetível | Certa, se não mudar o executor | Médio | P0, item 1 |
| 32 processos no 07 recusados por RAM (15 GB) | Certa | Baixo | Já tratado no código; citar como limitação |
| Figuras antigas no PDF | Certa | Médio | Regerar 5 já; `eixos_hpo` depois do 04 e `fusao_linha_do_tempo` depois do 08 em PU |

**Arquivos citados:**
- Configuração: `config.toml`
- Resultados: `experimentos/resultados/01_linha_base.csv`, `experimentos/resultados/03_resumo.csv`, `experimentos/resultados/NUMERACAO.md`, `experimentos/resultados/medicoes_avulsas.md`
- Notebook: `Eduardo_Savian_2026_CNN1D_SEEDS_otimizado.ipynb`
- Monografia: `tcc/TCC III - EDUARDO SAVIAN/src/` e `tcc/TCC III - EDUARDO SAVIAN/monografia.pdf`