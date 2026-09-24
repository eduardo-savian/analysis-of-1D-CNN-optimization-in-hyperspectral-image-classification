# Diário de experimentos

Registro informal do que foi testado, **por que foi testado daquele jeito**, e o que
deu. Não é texto de TCC — é a memória de percurso. O que o Felipe pediu: daqui a um
mês ninguém lembra o critério de cada teste, e é justamente o critério que dá sentido
ao número.

Formato de cada entrada: **pergunta → critério → configuração → resultado → o que ficou
aberto**. O "critério" é a parte que se perde, então é a que está mais detalhada.

Scripts em `experimentos/`, saídas em `experimentos/resultados/`.
Para rodar qualquer um: `.venv/bin/python experimentos/<script>.py`.

---

## Lição zero: onde as coisas moram

Perdi os scripts e logs de experimento **três vezes** (03/09, 10/09, 17/09), porque
viviam num diretório temporário que é limpo entre sessões. O notebook sempre
sobreviveu, porque está no repositório.

**Regra: script de experimento e resultado vão para `experimentos/`, versionados.**
Nada de valor mora em `/tmp`. Isso vale mais que qualquer otimização feita até aqui —
um resultado que não se pode reproduzir amanhã não é resultado.

`experimentos/_bootstrap.py` carrega as definições do notebook (só as células que
definem coisas, ignorando as que executam). É o que evita ter duas verdades: o
notebook continua sendo a implementação, e os scripts só o dirigem.

---

## 1. Paralelismo — 03/09

**Pergunta.** O paralelismo dos dois ramos do pré-processamento entrega ganho real?

**Critério.** Medir os três backends (`sequential`, `thread`, `process`) nas três cenas,
5 repetições cada, e comparar o speedup com o **teto teórico de Amdahl** — que se
calcula como `soma dos ramos ÷ maior ramo`. Sem o teto, um speedup de 1,05× parece
fracasso de implementação; com ele, pode ser o limite do desenho. Essa é a distinção
que o experimento existe para fazer.

**Resultado.**

| Cena | sequential | thread | process | ramo espectral | ramo espacial | teto |
|---|---|---|---|---|---|---|
| Indian Pines | 0,166 s | 1,05× | 0,22× | 0,037 s | 0,129 s (78%) | 1,29× |
| Pavia | 0,971 s | 0,95× | 0,75× | 0,072 s | 0,899 s (93%) | 1,08× |
| Salinas | 0,822 s | 1,22× | 0,74× | 0,142 s | 0,680 s (83%) | 1,21× |

**Conclusão.** O ganho é estruturalmente limitado: a segmentação é 78–93% do
pré-processamento, então dois ramos nunca passam de ~1,1–1,3×. Em Salinas chegamos a
1,22× contra teto de 1,21× — **a implementação está no limite do que o desenho
permite**. Não é problema de código.

O `process` é pior que o sequencial em todos os casos. Três custos somados: criar e
destruir o pool a cada chamada, serializar os arrays (8,4 MB em IP, 42,7 em PU, 45,3
em SA) e sobreassinatura de CPU — pai e filho usam bibliotecas multithread e brigam
pelos mesmos núcleos.

**O número que encerra a discussão:** numa execução real, pré-processamento 0,193 s
contra treino de 20,13 s. O pré-processamento é **0,9% do tempo total**. Speedup
infinito ali mudaria menos de 1% da execução.

**Aberto.** O paralelismo que valeria a pena é no nível dos *trials* de HPO, que são
independentes entre si — ganho linear no número de núcleos. Não implementado.

---

## 2. Fusão CNN + segmentador — 03/09

**Pergunta.** Quanto a segunda entrada (o id do superpixel) acrescenta, e o modo de
codificá-la importa?

**Critério.** Três condições sobre exatamente o mesmo split e a mesma arquitetura:
sem fusão (piso), fusão escalar (o id cru como uma coluna) e fusão por embedding (um
vetor aprendido por região). 3 seeds. O piso é obrigatório — sem ele, "0,89 de OA"
não é atribuível a nada.

**Resultado (split por pixel, protocolo padrão).**

| Cena | Condição | OA | AA | Kappa |
|---|---|---|---|---|
| Indian Pines | sem fusão | 0,8101 | 0,7558 | 0,7834 |
| | escalar | 0,8981 (+8,80 pp) | 0,8288 | 0,8836 |
| | **embedding** | **0,9471 (+13,70 pp)** | 0,9181 | 0,9397 |
| Pavia | sem fusão | 0,9480 | 0,9247 | 0,9308 |
| | escalar | 0,9785 (+3,04 pp) | 0,9673 | 0,9715 |
| | **embedding** | **0,9868 (+3,88 pp)** | 0,9806 | 0,9825 |

**Conclusão.** A fusão funciona, e **o modo de codificar importa mais do que eu
esperava**: trocar o id cru por `nn.Embedding` rende +4,9 pontos em IP com a mesma
informação de entrada. O id é um rótulo nominal; tratá-lo como número contínuo obriga
a rede a fingir que a região 78 é "78 vezes" a região 1.

**Ablação que explica o mecanismo** (IP, split por pixel):

| Manipulação | O que preserva | OA |
|---|---|---|
| id original | tudo | 0,8981 |
| id permutado | agrupamento e escala; perde a ordem | 0,8552 |
| id normalizado [0,1] | tudo menos a magnitude | 0,8223 |
| sem fusão | — | 0,8101 |
| id aleatório por pixel | só a escala; perde o agrupamento | 0,7898 |

Três coisas saem daí: a informação é real (ruído com a mesma escala deixa **pior** que
não ter fusão); metade do ganho vem da numeração funcionar como coordenada espacial
grosseira (os ids crescem na varredura raster); e a magnitude bruta importa — normalizar
preserva toda a informação e mesmo assim derruba o ganho de +8,80 para +1,22 pp.

Medida de alavancagem: a coluna do id gera **1,79× mais variação** entre pixels na
primeira camada densa do que as 1.024 colunas do espectro somadas.

---

## 3. Vazamento espacial — 03/09

**Pergunta.** O split aleatório por pixel deixa o mesmo superpixel no treino e no teste?

**Critério.** Medir, não supor. `segment_leakage` conta quantas amostras de teste
compartilham região com o treino.

**Resultado.** **100% das amostras de teste** compartilham superpixel com o treino, nas
duas cenas. Implementei `split_mode="region"` (cada superpixel inteiro numa partição
só), que zera isso por construção.

**Erro de percurso que vale registrar.** Minha primeira tentativa foi separar por
**blocos espaciais**. O `segment_leakage` mediu **100% de vazamento mesmo assim** — em
Indian Pines os superpixels do SEEDS têm ~16×16 px, do tamanho do bloco que eu tinha
escolhido. Bloco maior baixou para 66,5%, ainda longe de disjunto. Só o split por
região resolve. **A intuição de "basta separar espacialmente" falha quando o superpixel
é grande em relação à cena.**

**Resultado sob split por região (IP).** O baseline sem fusão cai de 0,8101 para
**0,3724**. Isso não é sobre a fusão — é sobre o protocolo: pixels vizinhos são quase
idênticos, e o split por pixel coloca quase-duplicatas em treino e teste, inflando
**todos** os números da literatura de HSI.

**Aberto — e importante.** Com `test_ratio=0,15` o split por região deixa só **3 ou 4
regiões** no teste, e o desvio entre seeds chega a ±0,25 em Pavia. **Os valores absolutos
não são confiáveis ainda.** Precisa de validação cruzada por região (dezenas de
partições, média com intervalo) antes de virar tabela do TCC. Não apresentar como
resultado fechado.

---

## 4. Estratégias de HPO — 03/09 e 10/09

**Pergunta.** Vale um hipertunador, ou random search basta?

**Critério.** As três estratégias percorrem **o mesmo espaço** (2.916 combinações para
CNN1D), com **o mesmo orçamento de trials** e **a mesma cobertura inicial** dos quatro
segmentadores. Só muda como o candidato é escolhido. Sem isso a comparação não vale.

A representação comum é uma **tupla de índices**, um por gene — é ela que permite ao
genético fazer crossover sem codificação binária e ao TPE tratar o segmentador como
categórico e os demais eixos como ordinais.

- `random` — sorteio uniforme; o `tell()` vazio *é* a definição.
- `tpe` — Optuna 4.9.0, `TPESampler`. Separa os trials em bons e ruins, estima uma
  densidade para cada grupo e propõe o ponto que maximiza a razão entre elas.
- `genetic` — implementado no notebook: torneio (3), crossover uniforme, mutação 25%,
  elitismo 2 (os melhores passam **sem retreinar**, economizando orçamento).

**Resultado com 100% de treino (03/09), IP, 20 trials:**

| Orçamento | genetic | random | tpe | dispersão |
|---|---|---|---|---|
| 10 épocas | 0,9220 | 0,8986 | 0,9194 | 2,34 pp |
| 40 épocas | 0,9402 | 0,9363 | 0,9337 | 0,65 pp |

**Resultado com 10% de treino (10/09), IP, 20 trials:**

| Orçamento | genetic | random | tpe | dispersão |
|---|---|---|---|---|
| 10 épocas | 0,7347 | 0,7204 | **0,7399** | 1,95 pp |
| 40 épocas | **0,8427** | 0,7880 | 0,7880 | 5,46 pp |
| 200 épocas | 0,8336 | 0,8075 | **0,8713** | 6,37 pp |

OA de teste do vencedor em 200 épocas: tpe **0,8557**, genetic 0,8440, random 0,7984.

**Conclusão — e uma correção que fiz no caminho.** Com 100% dos dados a dispersão
**cai** com o orçamento (2,34 → 0,65 pp): treinando até convergir, até configurações
medianas chegam lá, e a busca guiada perde vantagem. Eu cheguei a concluir que "a
vantagem do hipertunador some com orçamento alto".

**Com 10% dos dados o efeito se inverte**: a dispersão **cresce** (1,95 → 5,46 → 6,37
pp). Com dados escassos nenhuma configuração satura, então escolher bem importa mais,
não menos. A conclusão correta é mais específica: *a vantagem da busca guiada depende de
haver diferença real entre as configurações — e dados escassos preservam essa diferença.*

Isso significa que **o ranking das estratégias não é estável entre protocolos.** É um
resultado metodológico legítimo sobre HPO multi-fidelidade, não um defeito.

**Épocas realmente executadas** (teto de 200): média 63,8, mín 12, máx 157. O early
stopping governa bem antes do teto — **não há motivo para testar 500 épocas**.

**Quais eixos importam** (200 épocas, validação média por valor):

| Eixo | amplitude | melhor valor |
|---|---|---|
| dropout | 20,7 pp | 0,20 |
| learning rate | 21,5 pp | 3e-3 |
| weight decay | 5,1 pp | 0 |
| PCA | 3,3 pp | 15 |
| batch | 2,7 pp | 64 |

Dropout e learning rate dominam; batch e weight decay são quase irrelevantes. Com 871
amostras, regularizar demais é o pior erro — o modelo já tem pouca informação.

**Aberto.** Uma execução por estratégia, sem barras de erro. A diferença entre TPE e
genético está dentro do que outra semente inverteria. Falta repetir com várias sementes.

---

## 5. Memória: a técnica do orientador — 10/09

**Pergunta.** Por que a campanha completa não cabe na máquina?

**Critério.** Medir quanto **um** trial retém, por cena e arquitetura, antes de propor
qualquer solução.

**Diagnóstico.** Cada trial cria um `ExperimentResult` que carrega **sua própria cópia
dos dados preparados**. O `automated_hpo` guardava todos numa coluna da tabela.

| Cena | Arquitetura | Array dominante | Por trial | 20 trials |
|---|---|---|---|---|
| IP | cnn1d | `spectra` | 2,5 MB | 49 MB |
| PU | cnn2d | `patches_2d` | 25,7 MB | 513 MB |
| IP | cnn3d | `patches_3d` | **410 MB** | **8,2 GB** |
| SA | cnn3d | `patches_3d` | **2,21 GB** | **44,2 GB** |

Em CNN1D sobre IP — o caso que eu vinha rodando — o problema não aparece. Por isso
passou despercebido. Mas o plano de verificação inclui CNN3D em Salinas.

**A técnica.** Instanciar → treinar → **extrair as métricas** → **liberar**. A
assimetria que a justifica: o que se precisa de um trial são uns kilobytes (OA, AA,
Kappa, config); o que ele segura são centenas de megabytes.

Implementado como `release_result()` e `automated_hpo(retain="winner")`, agora o padrão.
Só o campeão corrente sobrevive; quando um trial melhor aparece, o anterior é liberado.
Como a tabela passou a guardar a `config` de cada trial, **liberar não perde
rastreabilidade** — qualquer execução pode ser reconstruída.

**Resultado (CNN3D em IP, 4 trials).**

| | memória acumulada | tempo |
|---|---|---|
| `retain="all"` | +2.475 MB | 12,1 s |
| `retain="winner"` | **+228 MB** | **6,2 s** |

11× menos memória e 2× mais rápido — liberar cedo evita que o alocador vá buscar
páginas novas a cada trial.

**Prova em escala.** No run de 180 trials: `695 → 1894 MB` (inicialização do PyTorch,
uma vez), depois `1894 → 1902 → 1905 → 1906 → 1910 → 1910 → 1910 → 1910`.
**16 MB de crescimento ao longo de 180 treinos.**

**Bug encontrado no caminho.** `validate_prepared_data` exigia que os splits cobrissem
**todas** as amostras. A subamostragem quebra isso legitimamente. O invariante que de
fato protege o experimento é a **disjunção** (nenhuma amostra em duas partições), não a
cobertura. Troquei a checagem; o contrato agora reporta `coverage` (1,0 no protocolo
cheio, ~0,19 com 10% de treino).

**Aberto.** `train_fraction` não reduz memória — `patches_3d` é construído para todos
os pixels antes do split. Se apertar, o próximo passo é construir patches **sob demanda
no DataLoader** em vez de materializar o array. Mudança maior, não feita.

---

## 6. Fração de treino — 10/09

**Pergunta.** Quanto custa treinar com 10% das amostras?

**Critério.** Amostragem **estratificada por classe**, não uniforme: com 10% uniforme
uma classe de 20 pixels teria em média 2 amostras e frequentemente zero, e a acurácia
dela ficaria indefinida, contaminando a AA. **Validação e teste ficam intactos** —
comparar frações só faz sentido com a régua idêntica.

**Resultado** (CNN1D + SEEDS, IP, política completa, 3 seeds):

| Fração | Amostras | OA | AA | Kappa | Tempo/run | Épocas |
|---|---|---|---|---|---|---|
| 1,00 | 8.711 | 0,9016 ± 0,0016 | 0,8308 ± 0,0087 | 0,8877 | 34,3 s | 44 |
| 0,50 | 4.356 | 0,8643 ± 0,0140 | 0,7310 ± 0,0405 | 0,8448 | 18,8 s | 46 |
| 0,25 | 2.179 | 0,8270 ± 0,0105 | 0,6755 ± 0,0201 | 0,8021 | 11,4 s | 58 |
| **0,10** | 871 | **0,7586 ± 0,0109** | **0,6006 ± 0,0111** | 0,7227 | **6,0 s** | 59 |
| 0,05 | 435 | 0,6810 ± 0,0131 | 0,5322 ± 0,0074 | 0,6334 | 4,0 s | 60 |

**Conclusão.** Os 10% custam **14,3 pontos de OA e 23,0 de AA**. Não é ajuste barato.
A AA cai quase o dobro porque subamostrar tira desproporcionalmente das classes
minoritárias — a menor passa a ter 2 amostras de treino.

Ganho de tempo é 5,7×, não 10× — há custo fixo por run (carga da cena, PCA,
segmentação) que não escala com as amostras.

**As épocas usadas sobem quando a fração cai** (44 → 59 → 60): com 10% o modelo nunca
bate a meta de 0,95 e o early stopping deixa rodar até o teto. É por isso que testar
200 épocas faz sentido **nesse** protocolo e não fazia no de 100%.

**Recomendação de protocolo.** Os 10% são ótimos para a **busca** (o HPO só precisa
ranquear), mas os números do capítulo de resultados não deveriam sair de 10% sem
declaração explícita. O defensável é dois estágios: **buscar com 10%, retreinar o
vencedor com 100%** — mesma lógica de buscar com 10 épocas e retreinar com 60, só que
no eixo dos dados.

---

## 7. Segmentadores: estrutura — 10/09

**Pergunta.** O que cada um dos quatro produz?

**Critério.** Métricas estruturais são determinísticas (não dependem de seed nem de
treino), então uma medição basta. Pureza = fração da classe dominante dentro de cada
região. **Pureza não é acurácia** — é diagnóstico espacial, e a docstring do código
avisa isso.

**Resultado.**

| Cena | Segmentador | Regiões | Área média | Dens. borda | Pureza | Tempo |
|---|---|---|---|---|---|---|
| IP | seeds | 81 | 259,6 | 0,2051 | 0,8831 | 0,149 s |
| IP | slic | 324 | 64,9 | 0,2920 | 0,9581 | 0,051 s |
| IP | **lsc** | 324 | 64,9 | 0,3503 | **0,9699** | 0,031 s |
| IP | watershed | 273 | 77,0 | 0,5650 | 0,8543 | **0,004 s** |
| PU | seeds | 45 | 4.608,9 | 0,1063 | 0,9212 | 1,854 s |
| PU | slic | 1.428 | 145,2 | 0,2132 | 0,9857 | 0,263 s |
| PU | **lsc** | 1.428 | 145,2 | 0,2440 | **0,9955** | 0,203 s |
| PU | watershed | 421 | 492,6 | 0,2797 | 0,9313 | 0,030 s |

**Observações.** SEEDS entrega muito menos regiões do que se pede (pedi 350 em IP,
vieram 81; pedi 150 em PU, vieram 45) — `num_superpixels` é alvo, não garantia. Suas
regiões são enormes: 4.609 pixels em PU contra 145 do LSC, diferença de 32×.
Watershed fragmenta: densidade de borda de 0,5650 significa que 56% dos pares de
pixels vizinhos estão em regiões diferentes.

**A tensão a resolver.** O LSC tem a maior pureza nas duas cenas e é 5× mais rápido,
mas o TPE concentrou 10 dos 20 trials em **SEEDS**. Hipótese: regiões grandes tornam
"região → classe" fácil de memorizar, que é o mecanismo do vazamento. **O SEEDS pode
estar vencendo por facilitar a memorização, não por ajudar a classificar.**

---

## 8. Segmentadores com CNN fixa — 17/09 — **DESENHO INVÁLIDO, guardar como contraexemplo**

**O que eu fiz de errado.** Comparei os quatro segmentadores mantendo os
hiperparâmetros da CNN fixos, achando que isso "isolava a variável". Isola a variável
errada: mede o quanto aquele conjunto fixo por acaso serve para cada segmentador, não
o que cada segmentador consegue entregar.

O Felipe cortou na hora: *"não faz sentido neste trabalho analisar ele separado"*. E
está certo — neste trabalho o segmentador é **um eixo da otimização conjunta**, não uma
variável independente. Ele nunca vai ser usado com hiperparâmetros que não foram
escolhidos para ele.

**Os dados que mostram por que o desenho falha** (10% de treino, 5 seeds, CNN fixa):

| Cena | Condição | OA | Kappa | Regiões |
|---|---|---|---|---|
| IP | sem fusão | 0,6661 ± 0,0140 | 0,6152 | — |
| IP | seeds | 0,7441 ± 0,0169 | 0,7058 | 81 |
| IP | lsc | 0,6993 ± 0,0380 | 0,6514 | 324 |
| IP | slic | 0,6060 ± **0,1603** | 0,5342 | 324 |
| IP | watershed | 0,5847 ± **0,1586** | 0,5056 | 273 |
| PU | sem fusão | 0,9158 ± 0,0039 | 0,8880 | — |
| PU | seeds | 0,9519 ± 0,0026 | 0,9360 | 45 |
| PU | lsc | 0,5398 ± **0,1886** | 0,1970 | 1.428 |
| PU | **slic** | **0,4485** | **0,0346** | 1.428 |

Kappa de 0,0346 é **praticamente aleatório**. E o padrão é evidente: quanto mais
regiões o segmentador produz, pior o resultado e maior o desvio. Não é qualidade do
segmentador — é que um id escalar indo até 1.428, contra um espectro de desvio 1,0,
exige learning rate e dropout completamente diferentes dos que foram fixados.

O desvio gigante (±0,16 a ±0,19) é a assinatura: algumas sementes treinam, outras não.
A configuração fixa está no limiar da instabilidade para esses casos.

**Por que guardei em vez de apagar.** É a evidência de que a otimização conjunta não é
capricho metodológico: sem ela, a conclusão seria "SLIC e Watershed são piores que não
usar segmentação", que é falso. Arquivo: `05_cnn_fixa_contraexemplo.py` e o `.log`.

---

## 9. Segmentadores sob otimização conjunta — 17/09

**Pergunta certa.** Qual é o melhor resultado que **cada segmentador alcança quando a
CNN é ajustada para ele**?

**Critério.** Duas etapas por condição:

1. **HPO restrito àquele segmentador** — busca learning rate, dropout, weight decay,
   batch, componentes de PCA e os parâmetros internos do próprio segmentador. 16 trials,
   seleção só por validação.
2. **Retreino do vencedor com 5 sementes** — para o número reportado ter barra de erro.

Três decisões que sustentam a comparação:

- **A condição "sem fusão" também passa por HPO.** Um piso com hiperparâmetros
  arbitrários não é piso, é sorte. Ela busca sobre os mesmos eixos da CNN.
- **A estratégia de busca é fixada em TPE** nas cinco condições, para que a diferença
  medida seja do segmentador e não da estratégia.
- **O orçamento de busca é idêntico** (16 trials) para todos.

**Resultado — Indian Pines** (`06_IP.log`, `06_resumo_IP.csv`):

| Condição | val (busca) | OA | AA | Kappa | ganho | lr | dropout | batch | PCA |
|---|---|---|---|---|---|---|---|---|---|
| **seeds** | 0,8544 | **0,8174 ± 0,0112** | 0,6985 | 0,7910 | **+14,02 pp** | 1e-3 | 0,35 | 64 | 15 |
| lsc | 0,8140 | 0,7446 ± 0,0284 | 0,5570 | 0,7059 | +6,74 pp | 1e-3 | 0,35 | 64 | 30 |
| sem fusão | 0,7035 | 0,6772 ± 0,0168 | 0,5482 | 0,6299 | — | 1e-3 | 0,20 | 32 | 45 |
| slic | 0,7685 | 0,6411 ± **0,1856** | 0,4835 | 0,5839 | −3,62 pp | **3e-4** | 0,20 | **128** | 30 |
| watershed | 0,7711 | 0,5646 ± **0,2255** | 0,4224 | 0,4777 | **−11,26 pp** | 1e-3 | 0,35 | 64 | 15 |

**Conclusões.**

**1. O segmentador é a decisão mais impactante do pipeline.** 25,3 pontos de OA
separam o melhor do pior, com todos partindo da mesma busca e do mesmo orçamento. É
mais amplitude do que qualquer hiperparâmetro isolado produziu (learning rate deu
21,5 pp; dropout, 20,7 pp — ver entrada 4).

**2. Dois segmentadores pioram o resultado, mesmo otimizados para si.** SLIC
(−3,62 pp) e Watershed (−11,26 pp) ficam abaixo de não usar segmentação. Cada um teve
16 trials só para ele, então não é falta de ajuste.

**3. A pureza não prevê a acurácia — o número de amostras por região prevê.**

| Segmentador | Regiões | Pureza | Amostras de treino por região | Ganho |
|---|---|---|---|---|
| seeds | 81 | 0,8831 | **10,8** | +14,02 pp |
| lsc | 324 | **0,9710** | 2,7 | +6,74 pp |
| slic | 324 | 0,9581 | 2,7 | −3,62 pp |
| watershed | 273 | 0,8543 | 3,2 | −11,26 pp |

O LSC tem a maior pureza e fica 7 pontos atrás do SEEDS, que tem a menor entre os
superpixels. Com ~2,7 pixels de treino por região, o modelo não estima
"região → classe" e o id vira ruído de alta magnitude entrando na cabeça densa —
mesmo mecanismo do "id aleatório por pixel" da ablação da entrada 2, que também
ficava abaixo do piso. Os desvios de ±0,19 e ±0,23 são a assinatura: algumas
sementes treinam, outras colapsam.

**Métrica de projeto que falta implementar:** *amostras de treino por região*. É o
número que prevê, antes de treinar, se aquele segmentador vai ajudar ou atrapalhar
naquele protocolo. Hoje só sabemos depois de rodar.

**4. Cada segmentador escolheu hiperparâmetros diferentes.** O SLIC foi o único a
pedir learning rate três vezes menor (3e-4) e batch 128 — exatamente o que se espera
de quem precisa domar um id que vai até 324. **O segmentador reconfigura o treino; não
é um módulo plugável.** Esse achado é invisível para a comparação com CNN fixa, e é o
melhor argumento a favor do desenho conjunto.

**Status de PU e SA:** em execução (`06_PU_SA.log`), relançado após a correção do
piso de componentes de PCA (ver Bugs, abaixo).

**O que também sai de graça:** os hiperparâmetros vencedores de cada segmentador ficam
registrados no CSV. Se o SLIC precisar de um learning rate muito diferente do SEEDS,
isso é um resultado por si — mostra que o segmentador não é um módulo plugável, e sim
algo que reconfigura o treino inteiro.

**Ressalva que permanece.** Roda sob split por pixel, com 100% de vazamento (ver
entrada 3). O ranking pode refletir "qual segmentador facilita memorizar" tanto quanto
"qual ajuda a classificar". Repetir sob `split_mode="region"` resolveria o vazamento,
mas ali a partição passa a depender do segmentador, o que confunde a comparação *entre*
segmentadores. Não há saída limpa ainda — registrar como limitação no TCC.

---

## 10. Arquitetura × segmentador — 17/09

**Pergunta.** Qual combinação de arquitetura e segmentador entrega o melhor
resultado, quando cada combinação recebe seus próprios hiperparâmetros?

**Critério.** O mesmo princípio da entrada 9, agora em duas dimensões. Não faz
sentido comparar arquiteturas com hiperparâmetros fixos pelo mesmo motivo que não
fazia para segmentadores: mede-se o quanto aquele conjunto por acaso serve, não o que
a arquitetura consegue entregar. **Cada célula da matriz (3 arquiteturas × 5
condições) recebe sua própria busca**, com orçamento idêntico de 16 trials, e só
depois os vencedores são comparados.

Dois eixos entram automaticamente: `patch_size` volta ao espaço para CNN2D e CNN3D
(a CNN1D não usa contexto espacial e o `hpo_search_space` já o remove de lá), e o
piso de componentes de PCA só se aplica à CNN1D.

**Custo medido antes de lançar** (IP, 10% de treino, 30 épocas, 1 trial):

| Arquitetura | Prep | Treino | Retido por trial | OA |
|---|---|---|---|---|
| cnn1d | 0,19 s | 3,65 s | 3,3 MB | 0,7490 |
| cnn2d | 0,24 s | 3,19 s | 5,8 MB | 0,6515 |
| cnn3d | 0,16 s | 5,18 s | **207,7 MB** | 0,3927 |

**Status:** em execução (`07_arquiteturas_x_segmentadores.py IP`), 315 execuções.

**Hipótese a verificar.** A CNN3D deu 0,3927 com 30 épocas. Ela tem muito mais
capacidade, e 871 amostras de treino provavelmente não a sustentam. Se a matriz
confirmar, o resultado não é "CNN1D é melhor" e sim **"CNN1D é a única compatível com
10% de treino"** — e aí vale repetir a comparação de arquiteturas com
`train_fraction=1.0`, para separar as duas afirmações.

**Registro de escopo.** Em 17/09 o escopo foi restrito a CNN1D e depois reaberto para
as três arquiteturas, na mesma sessão. Consequência importante: **a justificativa da
técnica de memória (entrada 5) depende do escopo.** Com CNN1D só, o ganho é marginal
(3,3 MB por trial). Com CNN3D, `release_result` volta a ser requisito — 16 trials
reteriam 3,3 GB em Indian Pines, e em Salinas o array é 2,21 GB por trial. Ao citar o
número, dizer em qual regime ele vale.

---

## 11. Arquiteturas 2D e 3D: medidas, e depois retiradas do escopo — 24/09

**Resultado completo da matriz em Indian Pines** (`07_arquiteturas.csv`, 15 células,
cada uma com HPO próprio de 16 trials + 5 sementes):

| Condição | CNN1D | CNN2D | CNN3D |
|---|---|---|---|
| sem fusão | 0,6772 | **0,8713** | 0,2393 |
| seeds | **0,8174** | 0,8611 | **0,5204** |
| slic | 0,6411 | **0,8806** | 0,4187 |
| lsc | 0,7316 | 0,8629 | 0,4229 |
| watershed | 0,5646 | 0,8364 | 0,3059 |

Ganho da fusão sobre o piso de cada arquitetura, em pontos percentuais:

| Condição | CNN1D | CNN2D | CNN3D |
|---|---|---|---|
| seeds | **+14,02** | −1,01 | +28,11 |
| lsc | +5,44 | −0,83 | +18,36 |
| slic | −3,62 | +0,94 | +17,95 |
| watershed | −11,26 | −3,49 | +6,66 |

**O achado, e por que ele incomoda.** A fusão por superpixel substitui contexto
espacial que a arquitetura não tem. Para a CNN2D, que já recebe patches espaciais,
**a fusão não acrescenta nada** — três dos quatro segmentadores pioram e o melhor dá
+0,94 pp. Para a CNN1D, que vê um pixel isolado, o mesmo SEEDS dá +14,02 pp.

Consequência: **a CNN2D sem segmentação nenhuma (0,8713) supera a CNN1D com o melhor
segmentador (0,8174)**, e com metade dos parâmetros (18.736 contra 38.032). O melhor
absoluto da matriz foi CNN2D + SLIC, OA 0,8806 ± 0,0200.

**A CNN3D sem fusão está degenerada**: OA 0,2393, AA 0,0625 (exatamente 1/16), Kappa
0,0000, desvio **zero** nas cinco sementes — ela prevê uma única classe. Com 871
amostras de treino não sai do lugar. O ganho de +28,11 pp do SEEDS parte desse piso
degenerado e **não é comparável** aos +14,02 da CNN1D, que parte de um modelo que
funciona.

**Decisão de escopo (24/09).** O trabalho voltou a ser só CNN1D — é o que está no
título do TCC. O código de CNN2D e CNN3D foi removido do notebook. Esta entrada existe
para que o resultado não se perca junto com o código: **se a banca perguntar "por que
não uma CNN 2D?", a resposta medida é que ela é melhor nesta cena, e que a fusão por
superpixel perde sentido nela.** Isso é limitação a declarar, não resultado a esconder.

Reproduzir exige recuperar as classes do histórico do git (commit anterior a 24/09) —
os números estão em `resultados/07_arquiteturas.csv`.

---

## Bugs encontrados nos próprios experimentos

Registrados porque cada um quase contaminou um resultado em silêncio.

**Piso de componentes de PCA (17/09).** `_pca_options` gerava opções escalando a
receita da cena por 0,5/1,0/1,5, com piso de 3. O piso 3 vinha da **composição RGB da
segmentação**, que é um PCA diferente. O que alimenta as convoluções da CNN1D precisa
de pelo menos 15: os kernels (9, 3, 3, 3) consomem 14 posições, e abaixo disso a saída
fica vazia. Cenas com receita 15 (PaviaU, Salinas) geravam a opção 8 e o treino
quebrava dentro da primeira convolução, com a mensagem do PyTorch apontando para a
convolução em vez da causa.

Corrigido derivando o piso dos próprios kernels (`CNN1D_MIN_COMPONENTS`) e fazendo a
CNN1D levantar um erro que diz a causa. **Indian Pines não foi afetada** — a receita 30
nunca gerou opção abaixo de 15. PU e SA foram relançadas.

**Contrato de dados recusando subamostragem (10/09).** `validate_prepared_data` exigia
que os splits cobrissem todas as amostras. Com `train_fraction < 1` isso é falso de
propósito. O invariante que protege o experimento é a **disjunção** (nenhuma amostra em
duas partições), não a cobertura. Trocada a checagem; o contrato agora reporta
`coverage`.

**`pkill -f` matando o próprio shell (17/09).** `pkill -9 -f "06_segmentadores"` casou
com a linha de comando do bash que o executava, e o comando morreu antes de relançar o
experimento. Usar `setsid` para lançar e padrões específicos para matar.

---

## Estado dos itens do plano de verificação

| Experimento do plano | Estado |
|---|---|
| Exp1 — Baseline | sai da condição "sem fusão" do experimento 8 |
| Exp2 — Otimização de segmentação | experimento 8, em execução |
| Exp3 — Otimização de CNN | feito em IP/CNN1D; PU e SA em execução |
| Exp4 — Otimização conjunta | entradas 9 e 10 — é exatamente isto |
| Exp5 — Paralelização | feito nas três cenas |
| Comparação de arquiteturas | entrada 10, em execução (IP) |

**O que falta:** PU e SA para as três arquiteturas, e a decisão sobre repetir a
comparação de arquiteturas com 100% de treino (ver hipótese da entrada 10).

---

## Coisas que aprendi e não quero reaprender

1. **Medir antes de concluir.** O split por blocos "obviamente" resolveria o vazamento.
   Media 100%. Duas vezes a intuição errou e a medição salvou.
2. **Sempre incluir um piso.** Número sem baseline não é atribuível a nada.
3. **Conclusão vale no protocolo em que foi medida.** "A vantagem do hipertunador some
   com orçamento alto" valia com 100% dos dados e se inverteu com 10%.
4. **O teto teórico transforma um resultado fraco em resultado.** 1,22× parece pouco;
   1,22× contra teto de 1,21× é a implementação no limite.
5. **Scripts e resultados no repositório.** Três perdas em três sessões.
6. **Nunca comparar B mantendo A fixo, quando A e B são otimizados juntos no uso
   real.** Foi o corte do Felipe em 17/09, e os dados provaram: com CNN fixa o SLIC em
   Pavia dava Kappa 0,0346 (aleatório); com hiperparâmetros próprios, sobe. A pergunta
   certa é sempre "qual é o melhor resultado que X alcança quando o resto é ajustado
   para ele".
7. **Toda afirmação carrega o regime em que foi medida.** O ganho da técnica de memória
   é marginal em CNN1D e decisivo em CNN3D; a vantagem do hipertunador aparece com 10%
   de treino e some com 100%. Citar o número sem o regime é meio-erro.
