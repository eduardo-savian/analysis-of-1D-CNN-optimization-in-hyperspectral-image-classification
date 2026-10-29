# Medições invalidadas pelo não-determinismo do LSC

Tudo aqui foi medido **antes** de 24/09, quando `createSuperpixelLSC` do OpenCV
ainda rodava multithread e devolvia rótulos diferentes a cada chamada (~390 pixels
de diferença em Indian Pines).

O efeito: o score de um mesmo trial variava entre execuções, e as estratégias
adaptativas (TPE e algoritmo genético) amplificavam a variação até trocarem de
vencedor. A mesma campanha deu 0,8713 numa execução e 0,7711 em outra.

**Não use estes números.** Ficam guardados só como evidência de que o problema
existia e do quanto ele deslocava o resultado. Ver a seção "Bugs encontrados nos
próprios experimentos" do DIARIO.md.
