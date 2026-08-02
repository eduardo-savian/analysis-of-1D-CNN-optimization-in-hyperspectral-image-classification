#import "univali.typ": *

= Considerações Finais <cap-consid>

Durante a pesquisa bibliográfica, foi possível observar uma variedade de técnicas
aplicadas à classificação de imagens hiperespectrais utilizando DL. Das abordagens
estudadas, optou-se por focar em CNNs 1D combinadas com segmentação espacial para
fusão de características espectrais e espaciais.

A fundamentação teórica evidenciou que trabalhos recentes, particularmente
#cite(<Souza2023>), demonstram ganhos significativos com fusão de dados espaciais
e espectrais. Entretanto, foram identificadas limitações importantes:
processamento sequencial consumindo tempo de execução desnecessário, configuração
manual de hiperparâmetros comprometendo reprodutibilidade, e potencial para
exploração de algoritmos de segmentação alternativos.

A revisão sobre processamento paralelo em GPUs demonstrou que as operações de
convolução presentes nas CNNs apresentam características intrínsecas de
independência de dados que permitem paralelização eficiente @Jorda_cuDNN @cuConv.
Cada elemento do mapa de características de saída pode ser calculado de forma
independente, o que viabiliza a execução simultânea em milhares de CUDA cores.
Adicionalmente, as etapas de segmentação espacial e extração de características
pela CNN operam sobre dados diferentes, permitindo execução concorrente dessas
operações através de threads paralelas.

O projeto de pesquisa visa endereçar as limitações através de paralelização do
processamento, otimização automática de hiperparâmetros, e comparação de
diferentes algoritmos de segmentação. A metodologia proposta, baseada em
experimentos sobre datasets de referência, permitindo a análise incremental dos
impactos de cada otimização.

Algumas das hipóteses levantadas foram:

+ Diferentes algoritmos de segmentação apresentarão desempenhos variados em termos
  de acurácia e tempo de processamento, sendo possível identificar alternativas ao
  SEEDS para cenários específicos.

+ A otimização automática de hiperparâmetros identificará configurações superiores
  ao ajuste manual, melhorando métricas de acurácia (OA, AA, Kappa) de forma
  reprodutível.

+ A otimização conjunta de segmentação e CNN superará a otimização isolada de cada
  componente.

+ A paralelização do processamento espacial e espectral resultará em redução
  mensurável do tempo de execução em relação à implementação sequencial,
  explorando a independência de dados entre as operações e a execução concorrente
  via threads.

A validação experimental em três datasets de referência (Indian Pines, Salinas
Valley e University of Pavia) permitirá comparação direta com resultados da
literatura, assegurando a validade das conclusões. A coleta de métricas tanto de
acurácia quanto de custo computacional fornecerá uma análise abrangente do impacto
das técnicas propostas.

Ao realizar uma análise sistemática do impacto das otimizações propostas, espera-se
demonstrar que a combinação de seleção adequada de algoritmos de segmentação,
otimização automática de hiperparâmetros e paralelização do processamento pode
transformar os gargalos identificados em um fluxo eficiente, estabelecendo uma
metodologia reprodutível para classificação de imagens hiperespectrais com
arquiteturas leves.
