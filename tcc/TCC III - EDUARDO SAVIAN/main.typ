// =============================================================================
// TCC III — Eduardo Savian
// Análise da Otimização de CNNs 1D na Classificação de Imagens Hiperespectrais
//
// Compilação:  typst compile main.typ
// Edição ao vivo:  typst watch main.typ
// =============================================================================

#import "typst/univali.typ": *

#show: tcc.with(
  // ------------------------------------------------------------ src/_info.tex
  universidade: [UNIVERSIDADE DO VALE DO ITAJAÍ],
  escola: [ESCOLA POLITÉCNICA],
  curso: [CURSO DE CIÊNCIA DA COMPUTAÇÃO],
  titulo: [Análise da Otimização de CNNs 1D na Classificação de Imagens Hiperespectrais],
  autor: [Eduardo Savian],
  local-data: [Itajaí (SC), Dezembro de 2025],
  area: [Visão Computacional],
  orientador: [Felipe Viel, PhD.],

  epigrafe: [Todos os Homens, por natureza, desejam conhecer — Aristóteles],

  resumo: [
    A classificação de imagens hiperespectrais (HSI) apresenta desafios relacionados
    à alta dimensionalidade dos dados e à complexidade da extração de informações
    espectrais discriminativas. Abordagens recentes que combinam redes neurais
    convolucionais unidimensionais (CNNs 1D) com técnicas de segmentação espacial
    têm demonstrado ganhos significativos de acurácia por meio da fusão
    espectral-espacial. No entanto, arquiteturas existentes apresentam limitações
    que incluem gargalos de processamento sequencial, ajuste manual de
    hiperparâmetros e exploração restrita de algoritmos de segmentação.

    Este estudo experimental e quantitativo investiga o impacto de estratégias de
    otimização na classificação de HSI utilizando CNNs 1D combinadas com
    segmentação espacial. Experimentos sistemáticos são conduzidos em três
    conjuntos de dados de referência: Indian Pines, Salinas Valley e University of
    Pavia. A pesquisa avalia a paralelização das etapas de processamento,
    otimização automática de hiperparâmetros por meio de Grid Search e Otimização
    Bayesiana, e o desempenho comparativo dos algoritmos de segmentação SEEDS, LSC
    e Watershed.

    As métricas de avaliação incluem Overall Accuracy, Average Accuracy,
    coeficiente Kappa, tempo de execução, consumo de memória e speedup. Os
    resultados esperados visam fornecer evidências empíricas sobre como essas
    estratégias de otimização afetam o desempenho de classificação e a eficiência
    computacional.
  ],
  palavras-chave: [
    Imagem Hiperespectral. Aprendizado de Máquina. Aprendizado Profundo.
    Segmentação. Rede Neural Convolucional Unidimensional. Otimização de
    Hiperparâmetros.
  ],

  abstract: [
    Hyperspectral image (HSI) classification presents challenges related to high
    data dimensionality and the complexity of extracting discriminative spectral
    information. Recent approaches combining one-dimensional convolutional neural
    networks (1D CNNs) with spatial segmentation techniques have demonstrated
    significant accuracy gains through spectral-spatial fusion. However, existing
    architectures exhibit limitations including sequential processing bottlenecks,
    manual hyperparameter tuning, and restricted exploration of segmentation
    algorithms.

    This experimental and quantitative study investigates the impact of
    optimization strategies on HSI classification using 1D CNNs combined with
    spatial segmentation. Systematic experiments are conducted on three benchmark
    datasets: Indian Pines, Salinas Valley, and University of Pavia. The research
    evaluates parallelization of processing stages, automatic hyperparameter
    optimization through Grid Search and Bayesian Optimization, and comparative
    performance of SEEDS, LSC, and Watershed segmentation algorithms.

    Evaluation metrics include Overall Accuracy, Average Accuracy, Kappa
    coefficient, execution time, memory usage, and speedup. The expected findings
    aim to provide empirical evidence on how these optimization strategies affect
    classification performance and computational efficiency.
  ],
  keywords: [
    Hyperspectral Image. Machine Learning. Deep Learning. Segmentation.
    One-Dimensional Convolutional Neural Network. Hyperparameter Optimization.
  ],

  // ------------------------------------------------------ src/0_pre_textual.tex
  siglas: (
    ([1D], [Unidimensional (_One-Dimensional_)]),
    ([2D], [Bidimensional (_Two-Dimensional_)]),
    ([3D], [Tridimensional (_Three-Dimensional_)]),
    ([AA], [Acurácia Média (_Average Accuracy_)]),
    ([AI], [Inteligência Artificial (_Artificial Intelligence_)]),
    ([AM], [Manufatura Aditiva (_Additive Manufacturing_)]),
    ([ASHA], [Asynchronous Successive Halving Algorithm]),
    ([CFD], [Fluidodinâmica Computacional (_Computational Fluid Dynamics_)]),
    ([CNN], [Rede Neural Convolucional (_Convolutional Neural Network_)]),
    ([CPU], [Unidade Central de Processamento (_Central Processing Unit_)]),
    ([CRC], [Verificação de Redundância Cíclica (_Cyclic Redundancy Check_)]),
    ([DL], [Aprendizado Profundo (_Deep Learning_)]),
    ([DMP], [Derivada do Perfil Morfológico (_Derivative Morphological Profile_)]),
    ([FC], [Totalmente Conectada (_Fully Connected_)]),
    ([GPU], [Unidade de Processamento Gráfico (_Graphics Processing Unit_)]),
    ([GT], [Ground Truth]),
    ([HPC], [Computação de Alto Desempenho (_High Performance Computing_)]),
    ([HPO], [Otimização de Hiperparâmetros (_Hyperparameter Optimization_)]),
    ([HSI], [Imagem Hiperespectral (_Hyperspectral Imaging_)]),
    ([ML], [Aprendizado de Máquina (_Machine Learning_)]),
    ([MLP], [Perceptron Multicamadas (_Multilayer Perceptron_)]),
    ([MSE], [Erro Quadrático Médio (_Mean Squared Error_)]),
    ([NN], [Rede Neural (_Neural Network_)]),
    ([OA], [Acurácia Geral (_Overall Accuracy_)]),
    ([OF], [Sobreajuste (_Overfitting_)]),
    ([PCA], [Análise de Componentes Principais (_Principal Component Analysis_)]),
    ([RASDA], [Resource-Adaptive Successive Doubling Algorithm]),
    ([ReLU], [Unidade Linear Retificada (_Rectified Linear Unit_)]),
    ([RGB], [Vermelho, Verde e Azul (_Red, Green, Blue_)]),
    ([RNA], [Rede Neural Artificial]),
    ([SEEDS], [Superpixels Extracted via Energy-Driven Sampling]),
    ([SGD], [Descida de Gradiente Estocástica (_Stochastic Gradient Descent_)]),
    ([SSL], [Aprendizado Semissupervisionado (_Semi-supervised Learning_)]),
    ([TCC], [Trabalho de Conclusão de Curso]),
    ([TF], [Transformer]),
  ),

  simbolos: (
    ([$kappa$], [Coeficiente Kappa]),
    ([$E$], [Experiência (no contexto de aprendizado de máquina)]),
    ([$L$], [Função de perda (_Loss function_)]),
    ([$P$], [Medida de desempenho ou Polinômio gerador]),
    ([$T$], [Classe de tarefas]),
    ([$theta$], [Parâmetros do modelo]),
    ([$nabla$], [Operador gradiente]),
    ([GiB], [Gibibyte]),
    ([kB], [Kilobyte]),
    ([m], [Metro]),
    ([MiB], [Mebibyte]),
    ([ms], [Milissegundo]),
    ([nm], [Nanômetro]),
    ([s], [Segundo]),
    ([µm], [Micrômetro]),
  ),

  // ------------------------------------------------------ elementos pós-textuais
  bibliografia: bibliography(
    "typst/refs.bib",
    title: [Referências],
    style: "associacao-brasileira-de-normas-tecnicas",
  ),
)

// -------------------------------------------------------- elementos textuais
#include "typst/1-introducao.typ"
#include "typst/2-fundamentacao.typ"
#include "typst/3-projeto.typ"
#include "typst/4-consideracoes-finais.typ"
