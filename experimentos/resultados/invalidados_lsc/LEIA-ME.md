# Medições invalidadas pelo não-determinismo do LSC

Tudo aqui foi medido antes da correção de 24/09 (`cv.setNumThreads(1)`), quando
`createSuperpixelLSC` do OpenCV ainda rodava multithread e devolvia rótulos diferentes
a cada chamada (~390 pixels de diferença em Indian Pines), ou depende de uma busca
feita antes dela:

- `08_trials.csv`, `08_IP.log`: a campanha do experimento 08 antes da correção;
- `06_resumo_lsc_antes_da_correcao.csv`, `06_por_seed_lsc_antes_da_correcao.csv`: as
  linhas do LSC do experimento 06 (IP em 17/09; PU e SA na manhã de 24/09, antes da
  correção), refeitas em 04/11;
- `10_lsc_otimizada_vencedor_antigo.csv`: o braço otimizado do LSC no experimento 10,
  treinado depois da correção, mas com o vencedor da busca antiga do 06 (OA 0,7566 ±
  0,0219); refeito em 04/11 com o vencedor novo.

O efeito: o score de um mesmo trial variava entre execuções, e as estratégias
adaptativas (TPE e algoritmo genético) amplificavam a variação até trocarem de
vencedor. A mesma campanha deu 0,8713 numa execução e 0,7711 em outra.

**Não use estes números.** Ficam guardados só como evidência de que o problema
existia e do quanto ele deslocava o resultado. Ver a seção 9 ("Bugs e não-determinismos") do
Context.md.
