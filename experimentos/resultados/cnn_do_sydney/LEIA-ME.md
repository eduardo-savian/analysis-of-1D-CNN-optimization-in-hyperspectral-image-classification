# Resultados com a CNN do notebook de Souza (2023)

Medidos antes de o trabalho adotar, em todos os experimentos, a CNN prevista no projeto
(capítulo 3: convoluções de 16, 32, 64 e 128 filtros com max-pooling e densas de 256 e
128). A CNN do Sydney continua no código (`cnn="original"`) e aparece no experimento 01
novo como linha de comparação.

- `01_linha_base*.csv`, `01_fiel.log`: linha de base com os dados fiéis ao Colab.
- `06_resumo_bayes_interrompido.csv`, `06_bayes_interrompido.log`: o experimento 06 com
  a otimização bayesiana, interrompido depois do SEEDS de Indian Pines.
- `11_paralelismo_particao.csv`, `13_pool_quente.csv` e logs: partição, ainda com os
  dados de antes da correção de fidelidade.
