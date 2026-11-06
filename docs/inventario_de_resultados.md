# Inventário de resultados e leitura por pilar (05/11/2026)

Li tudo o que está em `experimentos/resultados/`: os CSVs e logs da raiz, `NUMERACAO.md`, `medicoes_avulsas.md` e as cinco pastas de arquivo com seus `LEIA-ME.md`. Também conferi no código (`src/hsi/experimentos/e0*.py`, `hpo.py`, `config.toml`) como cada experimento roda, para estimar os tempos.

Para dizer se uma diferença conta, uso o critério do trabalho: ela é **afirmável** quando |dif| > 2 × hypot(desvio_a, desvio_b), com desvio populacional entre sementes. As diferenças estão em pontos percentuais (pp).

---

## 0. O que está na pasta e quanto vale cada coisa

| Situação | Arquivos |
|---|---|
| **Válido com o código atual** | `01_linha_base.csv`, `01_linha_base_por_classe.csv`, `01_projeto.log`; `03_resumo.csv` e `03_projeto.log` (só IP com SEEDS e LSC); a tabela `SEGMENTADORES` e a `LSC_THREADS_MS` de `medicoes_avulsas.md`, medidas em 04/11 com LSC e Watershed já calibrados |
| Serve de indício: CNN do Sydney com dados fiéis ao Colab | `cnn_do_sydney/01_*` (mesmos números das linhas `original` do 01 atual), `cnn_do_sydney/06_*` (busca bayesiana, IP com SEEDS e LSC; o LEIA-ME diz "interrompido depois do SEEDS", mas o CSV e o log também têm o LSC) |
| Serve de indício: CNN do Sydney com dados de antes da correção de fidelidade | `cnn_do_sydney/11_*` (partição), `cnn_do_sydney/13_*` (pool), `antes_da_fidelidade/*` (06 só IP, 08 com estratégias, 14 fusão concorrente), `08_IP.log`, `14_IP_PU_SA.log`, `06_treino_completo.log` |
| Indício com TPE e SLIC, que **não entra na monografia** | `tpe_e_slic/*` |
| Indício com 10% do treino | `fracao_10/*`, `05_*.log`, `06_IP.log`, `06_PU_SA.log`, `06_lsc_corrigido.log`, `10_*.log`. As linhas do LSC de `05_cnn_fixa_contraexemplo.log`, `06_IP.log`, `06_PU_SA.log`, `10_IP.log` e `10_IP_herdada_batch32.log` são de antes da correção do LSC ou usam o vencedor antigo; as corrigidas estão em `05_IP_PU.log`, `06_lsc_corrigido.log` e `10_lsc_corrigido.log` |
| Não usar | `invalidados_lsc/*` (LSC ainda não determinístico); `14_IP_PU_SA_invalidado_omp.log` |
| **Fora do escopo da monografia** (divisão por região) | `09_IP.log`, `09_validacao_regiao.csv`, `12_IP.log`, `12_particao_split.csv`. Não existem na numeração nova e não devem ser citados |
| Medições avulsas antigas | `medicoes_avulsas.md`: ramos e serialização (03/09), ablação e modos da fusão (03/09), fração do treino e memória por trial (10/09), threads do PyTorch, executores com pool frio e GIL (08/10). As que treinam a rede usaram a CNN do Sydney; ramos e GIL não dependem dela |

---

## 1. Resultados válidos com o código atual

### 1.1 Experimento 01: linha de base (hiperparâmetros do Sydney, SEEDS, treino completo, **uma execução, semente 42**)

| Cena | CNN | OA | AA | Kappa | Épocas (melhor) | Reportado por Souza (OA / AA / Kappa) |
|---|---|---|---|---|---|---|
| IP | projeto | **0,9142** | 0,8625 | 0,9020 | 52 (42) | 0,9142 / 0,8956 / 0,9019 |
| IP | Sydney (no CSV, `original`) | 0,8960 | 0,8717 | 0,8813 | 63 (53) | |
| PU | projeto | 0,9841 | 0,9771 | 0,9789 | 26 (16) | 0,9928 / 0,9866 / 0,9905 |
| PU | Sydney | 0,9853 | 0,9781 | 0,9806 | 30 (20) | |
| SA | projeto | 0,9938 | 0,9931 | 0,9931 | 29 (19) | 0,9965 / 0,9970 / 0,9962 |
| SA | Sydney | 0,9951 | 0,9959 | 0,9945 | 25 (15) | |

**CNN do projeto menos CNN do Sydney (OA / AA / Kappa, em pp)**
- IP: +1,82 / −0,92 / +2,07. São 66 contra 80 erros num teste de 769 pixels, e cada pixel vale 0,13 pp.
- PU: −0,12 / −0,10 / −0,17.
- SA: −0,12 / −0,28 / −0,14.

**O critério não se aplica ao 01**, porque é uma semente só e não há desvio. Para ter ordem de grandeza, usei os desvios de IP medidos em outras rodadas:

| Desvio usado nos dois braços | Limiar em IP |
|---|---|
| 0,0016 (fração do treino de 100%, Sydney) | 0,45 pp |
| 0,0062 (03, SEEDS) | 1,74 pp |
| 0,0103 (Sydney com busca bayesiana) | 2,91 pp |

Com isso, o +1,82 de IP fica no limite. PU e SA ficam dentro de qualquer desvio plausível (limiares de 0,40 e 0,16 pp com os desvios de `tpe_e_slic`).

**AA de IP:** a queda de −0,92 pp se explica pela classe 9 (Oats, 2 pixels de teste). A CNN do projeto acertou 0 de 2 e a do Sydney acertou 1 de 2, o que sozinho tira 3,1 pp da AA; sem essa classe, a CNN do projeto ficaria 2,2 pp à frente. Ela ganha nas classes 3, 11, 12 e 14 (+4, +8, +2 e +2 acertos) e em 4, 5 e 16 (+1 cada), e perde nas classes 2 (−3) e 6 (−1).

**Distância ao valor reportado por Souza (OA / AA / Kappa, em pp)**

| Cena | Reimplementação da CNN do Sydney | CNN do projeto |
|---|---|---|
| IP | −1,82 / −2,39 / −2,06 | 0,00 / −3,31 / +0,01 |
| PU | −0,75 / −0,85 / −0,99 | −0,87 / −0,95 / −1,16 |
| SA | −0,14 / −0,11 / −0,17 | −0,27 / −0,39 / −0,31 |

Isso muda a frase atual da seção de resultados ("menos de 1 ponto em IP e SA, 1,6 em PU, 7,2 de AA"), que vem dos números antigos (IP 0,9051 / 0,8237 / 0,8914). O parágrafo seguinte (Alfalfa com 3 amostras de teste e nenhum acerto, 60 dos 73 erros entre milho e soja) também vem deles: no 01 atual, Alfalfa tem 4 amostras de teste e as duas CNNs acertam as 4.

### 1.2 Experimento 03 parcial: IP, busca bayesiana com 16 trials restrita a cada segmentador, vencedor retreinado com 5 sementes

| Segmentador | Validação na busca | OA | AA | Kappa | Épocas | Treino por execução | Minutos no total | Vencedor |
|---|---|---|---|---|---|---|---|---|
| SEEDS | 0,9480 | 0,9324 ± 0,0062 | 0,9361 ± 0,0162 | 0,9229 ± 0,0070 | 69,6 | 24,9 s | 8,2 | taxa 3e-4, dropout 0,2, wd 0, lote 64, PCA 15; SEEDS no padrão (350/200/2) |
| LSC | 0,9389 | 0,9152 ± 0,0045 | 0,8895 ± 0,0181 | 0,9031 ± 0,0052 | 82,6 | 37,7 s | 12,2 | taxa 3e-4, dropout 0,2, wd 0, lote 64, PCA 15; LSC no padrão (region_size 16, ratio 0,075) |

**SEEDS contra LSC, os dois otimizados**

| Métrica | Diferença (LSC − SEEDS) | Limiar | Resultado |
|---|---|---|---|
| OA | −1,72 pp | 1,52 | **afirmável**, com folga pequena |
| AA | −4,67 pp | 4,85 | dentro do ruído |
| Kappa | −1,98 pp | 1,74 | **afirmável** |

**Otimizado contra a linha de base (indicativo, porque o 01 tem uma semente)**
- SEEDS otimizado menos a CNN do projeto no 01: OA +1,82, AA +7,36, Kappa +2,09.
- SEEDS otimizado menos a CNN do Sydney no 01: OA +3,64, AA +6,45, Kappa +4,16.
- Se o desvio da linha de base for igual ao do 03 SEEDS, os limiares seriam 1,74 (OA), 4,57 (AA) e 1,98 (Kappa). A AA passaria com folga; OA e Kappa passariam por cerca de 0,1 pp (0,08 e 0,11). Quem decide é o 05.
- O LSC otimizado (0,9152) empata na prática com a linha de base com SEEDS (0,9142).

**Metas do projeto (Quadro de validação do cap. 3)**
- IP: a meta de AA ≥ 0,93 é **atingida** pelo SEEDS otimizado (0,9361). As de OA ≥ 0,95 e Kappa ≥ 0,94 não são: faltam 1,76 e 1,71 pp.
- PU e SA: as metas já são atingidas na linha de base com a CNN do projeto. PU tem OA 0,9841 / AA 0,9771 / Kappa 0,9789 contra metas de 0,98 / 0,97 / 0,97. SA tem 0,9938 / 0,9931 / 0,9931 contra 0,97 / 0,96 / 0,96.

### 1.3 Medições válidas de 04/11: segmentação com LSC e Watershed calibrados para o número de regiões do SEEDS

OpenCV em uma thread, mediana de 14 chamadas, só a segmentação (sem o PCA de 3 componentes e a composição RGB).

| Cena | SEEDS (regiões / pureza / ms) | LSC | Watershed |
|---|---|---|---|
| IP | 81 / **0,8919** / 33,3 | 81 / 0,8825 / 37,7 | 82 / 0,8330 / **1,2** |
| PU | 45 / **0,9272** / 599,1 | 45 / 0,8194 / 280,2 | 46 / 0,7918 / **10,6** |
| SA | 96 / **0,9886** / 343,1 | 96 / 0,9241 / 151,0 | 97 / 0,9563 / **5,5** |

**LSC e threads do OpenCV:** com 32 threads, o LSC muda de 188 a 322 pixels entre chamadas iguais em IP (de 142 a 330 em PU). Com 1 thread não muda nada, ao custo de ser 2,4 vezes mais lento em IP (35,6 contra 14,8 ms) e 3,5 vezes em PU (289,7 contra 83,6 ms).

### 1.4 O que já dá para afirmar

1. A CNN do projeto reproduz o OA e o Kappa reportados por Souza em IP (0,9142 / 0,9020 contra 0,9142 / 0,9019) e fica abaixo dele por menos de 1,2 pp em PU e SA.
2. Com o mesmo número de regiões, o SEEDS gera as regiões mais puras nas três cenas, e o Watershed é de 28 a 62 vezes mais rápido que o SEEDS (e de 26 a 31 vezes mais que o LSC).
3. Em IP, com a CNN ajustada para cada segmentador, o SEEDS supera o LSC de forma afirmável em OA e Kappa.
4. A busca escolheu os mesmos hiperparâmetros da CNN para os dois segmentadores (taxa 3e-4, dropout 0,2, wd 0, lote 64, PCA 15 em vez de 30) e manteve a configuração padrão de cada segmentador. O ganho vem da CNN, não dos parâmetros da segmentação.
5. A validação do vencedor fica 1,6 a 2,4 pp acima do teste. É o efeito esperado de escolher o melhor de 16 trials.

---

## 2. Leitura por pilar

### 2.1 Segmentadores

**O que os resultados de arquivo indicavam**
- `tpe_e_slic` (CNN do Sydney, dados fiéis, 85% do treino, TPE com 16 trials, LSC e Watershed **sem calibração**):
  - IP: SEEDS 0,9300 ± 0,0029; LSC 0,9186 ± 0,0067 (−1,14, limiar 1,45, ruído); Watershed 0,8736 ± 0,0079 (−5,64, limiar 1,69, **afirmável**).
  - PU: SEEDS 0,9859 ± 0,0014; LSC 0,9609 ± 0,0086 (−2,50, limiar 1,75, **afirmável**); Watershed 0,9847 ± 0,0016 (−0,12, ruído).
  - SA: só SEEDS (0,9952 ± 0,0006) e SLIC foram medidos; a rodada parou no LSC. Não entra na monografia.
- `antes_da_fidelidade/06` (IP, CNN do Sydney, PCA antigo, teto de 60 épocas): SEEDS 0,9334 ± 0,0053; LSC 0,8978 ± 0,0189 (−3,56, limiar 3,92, ruído); Watershed 0,8622 ± 0,0123 (−7,13, **afirmável**).
- `cnn_do_sydney/06` (IP, busca bayesiana, LSC sem calibração com region_size 11, cerca de 2 vezes as regiões do SEEDS): SEEDS 0,9355 ± 0,0103 contra LSC 0,9329 ± 0,0090 (−0,26, limiar 2,73, empate).
- `fracao_10` (10% do treino, TPE):
  - IP: SEEDS 0,8174 ± 0,0112 contra LSC 0,7657 ± 0,0220 (−5,18, limiar 4,95, afirmável). O Watershed foi instável (0,5646 ± 0,2255; 2 das 5 sementes caíram para cerca de 0,26–0,32).
  - PU e SA: tudo dentro do ruído.
- Experimento 02 antigo (10% do treino, CNN fixa, só IP e PU, segmentadores **sem calibração**, de 273 a 1.428 regiões): o LSC colapsou em PU (OA 0,4508 ± 0,0114, com 1.428 regiões).
- **Resumo:** em nenhuma configuração uma alternativa superou o SEEDS de forma afirmável. O Watershed é o pior em IP. O LSC fica no meio.

**O que os novos já confirmam ou contradizem**
- **Confirmam** que o SEEDS não é superado. Em IP, o LSC agora perde de forma afirmável (−1,72 contra limiar 1,52).
- **Nuance a discutir:** sem calibração, com mais regiões e com a CNN do Sydney, o LSC empatava (−0,26). Com a calibração, limitado a 0,7–1,35 vezes as regiões do SEEDS, ele perde. O resultado novo vale "com número de regiões comparável", e isso precisa estar escrito no texto como escolha de desenho.

**O que só os experimentos pendentes respondem**
- Watershed em IP, e os três segmentadores em PU e SA (resto do 03).
- Se a calibração elimina os colapsos com a CNN fixa (02). Se eliminar, o 02 deixa de ser contraexemplo, e o texto atual sobre isso precisa mudar: as considerações finais citam o Kappa de 0,03 do SLIC em PU, e a seção de resultados, 0,0346 do SLIC e 0,0369 do LSC.

### 2.2 HPO

**O que os resultados de arquivo indicavam**
- `fracao_10/10_impacto_hpo` (10% do treino, CNN do Sydney, TPE, configuração original com lote 32):
  - SEEDS: de 0,7441 ± 0,0169 para 0,8174 ± 0,0112, +7,33 pp contra limiar 4,06, **afirmável**. É o número que a monografia cita hoje no RF04, no RNF05 e na discussão.
  - LSC: +8,43 contra limiar 8,80, ruído.
  - Watershed: −2,00, instável.
- Com treino completo, mas contra uma linha de base de uma semente só:
  - SEEDS IP otimizado com a CNN do Sydney (busca bayesiana) menos a linha de base do Sydney: +3,95 pp.
  - Em `tpe_e_slic`: IP +3,41, PU +0,06, SA +0,01. **O ganho se concentra em IP; PU e SA já estão no teto.**
- Estratégias (`antes_da_fidelidade/08`, 10% do treino, CNN do Sydney, 20 trials): melhor validação com 200 épocas foi genético 0,8288 ± 0,0346, TPE 0,8266 ± 0,0175 e aleatória 0,8067 ± 0,0149.
  - Genético menos aleatória: +2,21 contra limiar 7,53. TPE menos aleatória: +1,99 contra limiar 4,60. **Empate.**
  - Orçamento de 10 → 40 → 200 épocas: melhor validação de cerca de 0,72 → 0,79 → 0,82.
  - Peso de cada eixo na validação: dropout 18,1 pp, taxa 15,3, PCA 8,4, weight decay 4,6, lote 1,8. É o que justifica a grade com segmentador × taxa × dropout.

**O que os novos já confirmam ou contradizem**
- Confirmam a direção do ganho (+1,82 OA e +7,36 AA sobre a CNN do projeto; +3,64 OA sobre a CNN do Sydney).
- Confirmam que o ganho encolhe com o treino completo: de cerca de 7 pp com 10% para cerca de 2–4 pp com 85%.
- Os vencedores se repetem entre segmentadores e entre as duas CNNs: taxa 3e-4, dropout 0,2 e PCA 15.

**O que só os experimentos pendentes respondem**
- **05:** se o ganho é afirmável. O braço com a configuração original e SEEDS (no código, o rótulo é `herdada`) dá as 5 sementes da linha de base. A semente 42 dele é igual ao 01.
- **04:** grade contra bayesiana. Ainda não há nada válido; pelo empate antigo entre estratégias, o resultado provável é empate.
- **Resto do 03:** se a busca acrescenta algo em PU e SA. O 05 só roda IP.

### 2.3 Paralelismo

Tudo o que existe é de antes de 04/11, com os dados de antes da correção de fidelidade e, quando há treino, com a CNN do Sydney.

**O que os resultados de arquivo indicavam**
- **Ramos do pré-processamento (03/09):** threads contra sequencial deu IP 0,166 → 0,158 s (1,05 vezes), PU 0,971 → 1,027 s (0,95 vezes) e SA 0,822 → 0,675 s (1,22 vezes). Processos foram sempre piores (0,739 / 1,296 / 1,118 s), por enviar 8,4 / 42,7 / 45,3 MB ao processo filho. Esta medição não depende da CNN, mas não está em nenhum experimento novo.
- **GIL (08/10):** laço em Python puro foi de 1,01 para 0,91 s com 4 threads; multiplicação de matrizes em C, de 0,80 para 0,28 s.
- **Threads do PyTorch (08/10):** o melhor foi com 8 threads (7,81 s); com 16, 11,67 s.
- **Partição (`cnn_do_sydney/11`, executor de threads, 25 épocas, lote 64):**
  - Tempo total contra o monolítico, que treina na GPU: 0,97 vezes (N=2), 0,89 (4), 0,36 (8) e 0,32 (16). Na CPU, a partição perde para a GPU.
  - Modo replicado: 1,06–1,10 vezes.
  - OA com faixas: +2,82 / +3,25 / +3,47 / +2,69 pp sobre o monolítico (0,8825 ± 0,0101), todos afirmáveis.
  - Partição aleatória: de −4,07 a −28,83 pp. Replicado: OA idêntica.
  - Ao explicar a diferença entre faixas e aleatória na monografia, ficar só na especialização de cada modelo local às classes da sua faixa, sem entrar no tema excluído.
- **Pool frio contra quente (`cnn_do_sydney/13`, modo independente), speedup contra o sequencial nas mesmas partições:**

  | Executor | N=2 | N=4 | N=8 | N=16 |
  |---|---|---|---|---|
  | Processos, pool quente | 1,86 | 2,87 | 3,68 | 4,47 |
  | Processos, pool frio | 1,20 | 1,46 | 1,37 | 1,05 |
  | Threads, frio (quente quase igual) | 1,63 | 0,61 | 0,23 | 0,24 |

  - Threads com N=32: 0,21 vezes no frio e 0,22 no quente.
  - Partida dos processos: 3,8 a 10,1 s.
  - Processos com N=32 foram recusados por memória (481 MB cada, 11,3 GB livres).
  - Com threads no modo independente, a OA variou 3,51 pp entre execuções iguais. Com sequencial e processos, variou 0,00 pp.
- **Fusão concorrente (`antes_da_fidelidade/14`, `14_IP_PU_SA.log`):**
  - IP: lote com 1 thread do PyTorch levou 0,330 s; com 16, 0,196 s. O melhor concorrente (N=8, bloco 1.024) levou 0,164 s: 2,01 vezes contra o lote de 1 thread e 1,20 contra o de 16 (a variante desacoplada chegou a 2,17 e 1,29). Com bloco 1, foi de 7 a 35 vezes mais lento.
  - PU: melhor caso 1,21 vezes contra o lote de 1 thread e 1,01 contra o de 16.
  - SA: 1,50 e 1,11 vezes (a variante desacoplada chegou a 1,58 e 1,18).
  - Todas as 723 execuções repetiram os ids do treino e as classes do lote.

**O que os novos já confirmam ou contradizem:** nada ainda. A expectativa é que as conclusões qualitativas se mantenham: processos com pool quente escalam, threads não escalam, e a fusão concorrente só compensa com blocos grandes (1.024 posições). Os números vão mudar, porque a CNN do projeto, com pooling, muda o custo das convoluções e das densas.

**O que só os experimentos pendentes respondem:** 06, 07 e 08 com a CNN do projeto.

### 2.4 Troca de arquitetura (CNN do projeto no lugar da CNN do Sydney)

- **Novo (01, uma semente):** IP +1,82 OA, +2,07 Kappa e −0,92 AA (o efeito da classe Oats); PU e SA −0,12 pp. A CNN do projeto treina menos épocas em IP e PU (52 contra 63 e 26 contra 30) e mais em SA (29 contra 25).
- **Arquivo contra novo, os dois otimizados em IP:**
  - SEEDS: CNN do Sydney 0,9355 ± 0,0103 contra CNN do projeto 0,9324 ± 0,0062, −0,31 pp contra limiar 2,40. **Empate.**
  - LSC: −1,77 contra limiar 2,01. Empate, mas o conjunto de opções do LSC era diferente.
- **Leitura:** a CNN do projeto é pelo menos equivalente. Antes da busca ela parece ganhar em IP, sem prova estatística; depois da busca, as duas empatam.
- **Falta:** 5 sementes das duas CNNs com a configuração original e SEEDS. O 05 dá as da CNN do projeto. **Nenhum experimento atual roda a CNN do Sydney com 5 sementes**, e é ela a referência "configuração do Sydney com fusão SEEDS" contra a qual o trabalho mede os ganhos. Acrescentar esse braço custa cerca de 4 min em IP e 20 min nas três cenas.

### 2.5 Organização do código (evidências que já existem)

- **Determinismo:** sequencial e processos repetem a OA (0,00 pp em 20–25 execuções), e as 723 execuções da fusão concorrente reproduzem ids e classes.
- **Defeitos encontrados e corrigidos:**
  - LSC não determinístico: antes da correção, a mesma campanha deu 0,8713 numa execução e 0,7711 em outra (`invalidados_lsc`).
  - Gerador aleatório compartilhado entre threads (variação de 3,51 pp).
  - Piso de componentes do PCA: está no traceback de `06_IP.log`, "Kernel size can't be greater than actual input size".

---

## 3. Tempo dos experimentos pendentes e ordem de execução

### Base das estimativas

- **IP, por segmentador no 03:** os valores medidos no `03_resumo.csv` são 8,2 e 12,2 min, com 16 trials e 5 retreinos. A premissa de cerca de 0,5 h por segmentador é pessimista para IP; ela só serve de teto para PU e SA.
- **PU e SA, por segmentador:** em `tpe_e_slic` (mesmo desenho, CNN do Sydney, custo parecido) foram 15,6 a 28,8 min em PU e 15,2 a 19,9 min em SA.
- **Custo por época:** em IP com lote 64 e SEEDS, 0,36 s na CNN do projeto e 0,34 s na do Sydney (com LSC, 0,46 s na do projeto). Com lote 32 na configuração original, cerca de 37 s por execução em IP e 97–100 s em PU e SA (tempos de treino da CNN do Sydney no `14_IP_PU_SA.log`: 37,5 / 97,1 / 100,4 s).

### Estimativa por experimento

| Exp. | O que roda | Base da estimativa | Estimativa |
|---|---|---|---|
| 03 (resto) | Watershed em IP; PU e SA com 3 segmentadores (16 trials + 5 retreinos cada) | IP 8–12 min por segmentador; PU cerca de 64 min (15,6 + 28,8 + 19,2); SA cerca de 55 min | **~2,2 h** (pessimista ~4 h, se a busca escolher lote 32 em PU e SA) |
| 02 | 3 cenas × 3 segmentadores × 5 sementes = 45 treinos com lote 32 | IP ~40–60 s; PU e SA ~100–120 s por treino | **~1,1 h** |
| 04 | IP, 2 estratégias × 3 orçamentos × 3 sementes × 27 trials = 486 trials com 10% do treino | `08_IP.log`: 20 trials em ~19 / 42 / 72 s com 10 / 40 / 200 épocas, mais o ajuste do processo gaussiano | **~0,4 h** (com 85% do treino seriam ~2 h) |
| 05 | IP, 3 segmentadores × 2 configurações × 5 sementes = 30 treinos | configuração original com ~37 s por treino (`14_IP_PU_SA.log`); retreinos do 03 com 25–38 s | **~0,35 h** (estender a PU e SA: +1,7 h) |
| 06 | IP, monolítico + 3 arranjos × 4 valores de N × 3 sementes = 39 execuções | `cnn_do_sydney/11`: soma de `tempo_total` = 9,9 min | **~0,25 h** |
| 07 | IP, 5 arranjos × 5 valores de N × (2 frio + 3 quente) = 125 execuções | `cnn_do_sydney/13`: blocos e aquecimentos = 25,6 min, mais o treino do modo replicado | **~0,65 h** |
| 08 | IP: um treino + 283 execuções | `14_IP_PU_SA.log`: bloco 1 de 1–12 s × 63 execuções ≈ 7,7 min; o resto até 0,5 s cada (~1 min no total) | **~0,25 h** (+0,3 h com PU e SA) |
| **Total, como está configurado** | | | **~5,2 h** (pessimista ~7 h) |
| Extras recomendados | CNN do Sydney com 5 sementes (precisa de código novo); 08 com PU e SA (`src/executar.py 08 PU SA`, sem mudar código) | | +0,6 h |

A máquina aguenta: RTX 5070 Laptop (8 GB), 32 threads e 15 GB de RAM, sem nenhum experimento rodando agora. Dá para terminar tudo até a manhã de 06/11.

### Ordem proposta: primeiro um resultado válido por pilar, depois as outras cenas

1. **`03 IP`** (retoma no Watershed, ~10 min). Fecha IP no pilar dos segmentadores. O 05 precisa dessa linha do `03_resumo.csv`.
2. **`05`** (IP, ~20 min). Dá o ganho da busca sobre a configuração original com o critério aplicado e o desvio da linha de base.
3. **`04`** (~25 min). Grade contra bayesiana, o RF04 do projeto.

   Os passos 1 a 3 somam cerca de 1 h. Se começarem agora, podem estar prontos para a conversa de hoje.

4. **`08`, depois `06`, depois `07`** (~1,2 h). São medições de tempo, então a máquina precisa estar sem outro treino rodando.
5. **`03 PU SA`** (~2–4 h, à noite).
6. **`02`** (~1,1 h, à noite).
7. Se sobrar tempo: o braço da CNN do Sydney com 5 sementes, `08` com PU e SA, e `05` com PU e SA.

Com isso, 07 a 13/11 ficam para a escrita e 14/11 para a revisão.

### Antes de rodar

- **06:** o código (`e06_particao.py`) usa `executor="thread"` em todos os arranjos, inclusive no modo independente. Nesse modo a OA não se repete (variação de até 3,51 pp), e o `speedup_bloco` (`soma_partes ÷ bloco_paralelo`) sai inflado. Decidir antes de rodar: trocar para `process` ou `sequencial` na OA, e tirar o speedup do 07. A monografia já diz que toda OA de partição vem de execuções sequenciais ou em processos.
- **08:** está configurado só para IP (`cenas = ["IP"]` em `config.toml`), mas a seção de resultados cita PU e SA (1,21× e 1,50×), onde o esquema atinge o teto dado pela segmentação: o SEEDS leva 599 e 343 ms, contra 33 ms em IP. Para incluí-las, basta `src/executar.py 08 PU SA`; o bloco 1 continua só em IP.
- **Detalhes por segmentador do 03:** as linhas por semente de IP com SEEDS e LSC não foram gravadas, porque o CSV `03_por_seed_*` só é escrito no fim da execução (ao retomar, `03_por_seed_IP.csv` terá só o Watershed). O 05 retreina os mesmos vencedores com as mesmas sementes e as recupera. Isso também confere o determinismo: o 05 tem que repetir 0,9324 e 0,9152.
- **Logs antigos na raiz:** os logs soltos com a numeração antiga (`05_*`, `06_*`, `08_IP.log`, `10_*`, `14_*`) e os de divisão por região (`09_*`, `12_*`) ficam misturados com as saídas novas. Os de 05, 06 e 08 dividem prefixo com `05_impacto_hpo.csv`, `06_paralelismo_particao.csv` e `08_fusao_concorrente.csv`; os de 09 a 14 usam números que não existem mais. Nenhum nome coincide, então nada é sobrescrito, mas vale movê-los para as pastas de arquivo antes de rodar.
- **Medições avulsas citadas na monografia:** a ablação da fusão (−10,83 pp no RF03), a fração do treino e as threads do PyTorch foram medidas com a CNN do Sydney; ramos e GIL não dependem da CNN, mas também são de antes de 04/11. A ablação não tem experimento no código atual: ou precisa ser refeita (~10 min) ou citada com essa ressalva.