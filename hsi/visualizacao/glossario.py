"""Definições curtas exibidas junto aos gráficos do notebook.

As métricas de teste são sempre de relato; HPO e escolha de receitas usam somente
validação. Isso evita vazamento do conjunto de teste para a seleção do modelo.
"""

import pandas as pd
from IPython.display import Markdown, display

VISUAL_GLOSSARY = {
    "dados": {
        "título": "Dados, PCA e segmentação",
        "termos": [
            ("RGB de visualização", "Composição de três bandas/componentes apenas para leitura humana; não equivale à imagem hiperespectral inteira."),
            ("PCA / componente principal", "Projeção que concentra variância espectral. Componentes não são classes e não possuem cores físicas fixas."),
            ("Superpixel", "Região de pixels visualmente/espectralmente semelhantes tratada como contexto espacial."),
            ("Densidade de borda", "Proporção de pixels de contorno; valores altos significam segmentação mais fragmentada."),
            ("Pureza rotulada", "Fração da classe dominante em cada superpixel; perto de 1 indica segmentos que respeitam melhor as fronteiras de classe."),
        ],
    },
    "avaliacao": {
        "título": "Qualidade preditiva e convergência",
        "termos": [
            ("Loss / cross-entropy", "Erro de classificação probabilístico; menor é melhor. É diferente de acurácia e penaliza previsões confiantes erradas."),
            ("OA", "Overall Accuracy: proporção total de pixels corretamente classificados; pode favorecer classes muito frequentes."),
            ("AA", "Average Accuracy: média das acurácias por classe. É mais representativa quando as classes são desbalanceadas."),
            ("Kappa", "Concordância além do acaso; 1 é concordância perfeita, 0 é aproximadamente acaso."),
            ("Precisão / recall", "Precisão mede se as predições de uma classe estavam corretas; recall mede quanto da classe real foi recuperado."),
            ("F1 macro", "Média entre classes do equilíbrio precisão–recall; cada classe tem o mesmo peso."),
            ("IoU macro", "Interseção sobre união média por classe; mede sobreposição entre classe prevista e real."),
            ("Melhor época / early stopping", "Linha tracejada: melhor validação. Linha pontilhada: fim antecipado após a janela sem ganho mínimo configurado."),
        ],
    },
    "explicabilidade": {
        "título": "O que a CNN usa para decidir",
        "termos": [
            ("Mapa de características / ativação", "Resposta de um filtro convolucional. Regiões intensas são padrões aos quais aquele filtro respondeu mais."),
            ("MaxPooling", "Mantém a maior ativação de cada vizinhança; realça evidências fortes, mas reduz resolução espacial."),
            ("Grad-CAM", "Mapa de importância por gradiente para a classe prevista. Mostra onde a última camada convolucional sustentou a decisão; não é segmentação real."),
            ("Saliência", "Sensibilidade da saída a cada entrada. Valores altos indicam bandas ou pixels cuja pequena alteração afeta mais a predição."),
            ("Confiança", "Maior probabilidade softmax entre as classes. Alta confiança pode estar errada e não representa calibração estatística por si só."),
            ("Entropia normalizada", "Incerteza da distribuição softmax: 0 significa uma classe dominante; 1 significa probabilidades muito divididas entre as classes."),
        ],
    },
    "comparacao": {
        "título": "Arquitetura da CNN 1D",
        "termos": [
            ("CNN1D", "Opera principalmente sobre a assinatura espectral de cada pixel."),
            ("Parâmetros", "Quantidade de pesos treináveis; mais parâmetros não implica automaticamente melhor generalização."),
            ("Tempo / memória", "Custo medido no ambiente da execução. Compare apenas receitas com mesma cena, split e orçamento de treino."),
        ],
    },
    "hpo": {
        "título": "HPO e seleção de modelos",
        "termos": [
            ("Trial", "Uma receita completa: CNN, segmentador e seus parâmetros, taxa de aprendizado, dropout e demais configurações."),
            ("Val accuracy (objetivo)", "Melhor acurácia no conjunto de validação; é o único critério de escolha do HPO."),
            ("OA/AA/Kappa de teste", "Relato final após o trial. Não participam da escolha, para não vazar informação do teste."),
            ("Vencedor", "Melhor trial de validação por CNN; empates usam menor tempo e número do trial como regra auditável."),
        ],
    },
    "campanha": {
        "título": "Campanha MLOps auditável",
        "termos": [
            ("Seed / repetição", "Inicialização e embaralhamento controlados. Repetições verificam se um resultado não foi apenas sorte."),
            ("Média e desvio", "Resumo da variabilidade entre seeds; desvio menor indica resultado mais estável."),
            ("IC 95%", "Intervalo de confiança aproximado da média; intervalos sobrepostos sugerem cautela ao declarar superioridade."),
            ("Artefatos", "Plano, configuração, ambiente, dados, histórico, métricas e checkpoint preservados para reprodução e auditoria."),
        ],
    },
    "espacial": {
        "título": "Explicação inversa na imagem",
        "termos": [
            ("Classe prevista", "Classe de maior probabilidade para cada pixel rotulado."),
            ("Mapa de erro", "Divergência entre previsão e rótulo de referência; mostra onde o modelo errou no conjunto avaliado."),
            ("Confiança por pixel", "Probabilidade máxima por pixel; leia sempre junto do mapa de erro e da entropia."),
            ("Entropia por pixel", "Incerteza normalizada: tons altos destacam pixels ambíguos, mesmo quando a classe prevista parece confiante."),
            ("Resumo por segmento", "Agrega acurácia, confiança, entropia e F1 dentro de cada superpixel para explicar regiões, não só pixels isolados."),
        ],
    },
    "batch": {
        "título": "Batch, throughput e uso de GPU",
        "termos": [
            ("Batch físico", "Número de amostras enviadas de uma vez à GPU."),
            ("Batch efetivo", "Amostras que contribuem para uma atualização após acumulação de gradientes."),
            ("Throughput", "Amostras de treino por segundo; maior é melhor, desde que a validação permaneça comparável."),
            ("OOM", "Out of Memory: batch não coube na GPU. A falha é mantida no relatório para tornar o limite auditável."),
            ("Batch recomendado", "Maior throughput dentro de 0,5% da melhor validação, sem trocar qualidade por velocidade silenciosamente."),
        ],
    },
}


def glossary_table(*sections):
    """Os termos de `show_glossary` como `DataFrame`, só das seções pedidas."""
    rows = []
    for section in sections:
        block = VISUAL_GLOSSARY[section]
        rows.extend({"painel": block["título"], "termo": term, "como interpretar": meaning}
                    for term, meaning in block["termos"])
    return pd.DataFrame(rows)


def show_glossary(*sections):
    """Exibe legendas contextuais em cartões Markdown após cada grupo de gráficos."""
    labels = " • ".join(VISUAL_GLOSSARY[section]["título"] for section in sections)
    cards = []
    for section in sections:
        for term, meaning in VISUAL_GLOSSARY[section]["termos"]:
            cards.append(f"- **{term}:** {meaning}")
    display(Markdown(f"#### Como ler os gráficos — {labels}\n\n" + "\n".join(cards)))
