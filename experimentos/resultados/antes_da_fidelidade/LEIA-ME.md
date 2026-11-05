# Resultados de antes de igualar os dados aos do Colab

Medidos até 04/11 com o PCA do sklearn atual (que inverte o sinal de algumas
componentes e, no PCA de 3 componentes, um canal da composição RGB do SEEDS em Indian
Pines e University of Pavia), a segunda divisão estratificada pelos rótulos inteiros (o
notebook estratifica pelo one-hot), ε = 10⁻⁷ no Adam e teto de 60 épocas.

- `06_resumo.csv`: a rodada com treino completo interrompida em 04/11, só Indian Pines;
- `08_trials.csv`: a campanha de estratégias de busca (10% do treino);
- `14_*`: a fusão concorrente.

Foram refeitos com os dados idênticos aos do notebook, ε = 10⁻⁸ e teto de 200 épocas.
