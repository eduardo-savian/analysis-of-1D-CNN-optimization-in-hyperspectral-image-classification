# Setup do ambiente

O código vive em [src/hsi/](src/hsi/) e tem dois frontends, que chamam exatamente as mesmas
funções e por isso produzem resultados idênticos:

- notebook: [src/TCC3_SydneySouza_PyTorch.ipynb](src/TCC3_SydneySouza_PyTorch.ipynb);
- CLI: [scripts/run_experiment.py](scripts/run_experiment.py), para rodar sem interface.

Sem Google Colab e sem Google Drive: os dados ficam em disco e tudo roda igual na máquina
local (RTX 5070) e em instância AWS.

## Versões fixadas

Todas em [requirements.txt](requirements.txt). Foram as usadas para validar o notebook:

| pacote | versão |
| --- | --- |
| Python | 3.12.13 |
| torch | 2.13.0 (+cu130) |
| numpy | 2.5.1 |
| scipy | 1.18.0 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| matplotlib | 3.11.1 |
| seaborn | 0.13.2 |
| opencv-contrib-python-headless | 5.0.0.93 |
| jupyterlab / ipykernel | 4.6.2 / 7.3.0 |

O **build CUDA do torch não é pinável** no `requirements.txt` — ele vem do índice escolhido
na instalação. A célula *Versões* do notebook reporta e valida isso em tempo de execução.

Dois detalhes que quebram o ambiente se ignorados:

- **`opencv-contrib`, não `opencv-python`.** A segmentação usa
  `cv.ximgproc.createSuperpixelSEEDS`, que só existe no pacote *contrib*. Os dois pacotes
  colidem no mesmo namespace `cv2`; instale apenas um. Use `-headless` em servidor/AWS
  (sem `libGL`); em desktop com GUI, `opencv-contrib-python==5.0.0.93`.
- **RTX 5070 é Blackwell, `sm_120`.** Wheels cu126 e anteriores **não** têm kernels para
  essa arquitetura: importam sem erro e falham no primeiro kernel com
  *"no kernel image is available for execution on the device"*. Use **cu129 ou cu130**.

## Local — RTX 5070 (WSL2)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.13.0 \
    --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
```

Verificar que a GPU realmente executa kernels:

```bash
.venv/bin/python -c "
import torch, torch.nn as nn
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('cap', torch.cuda.get_device_capability(0), '| build', torch.cuda.get_arch_list())
print(nn.Conv1d(1,16,9).cuda()(torch.zeros(2,1,32,device='cuda')).shape)"
```

Saída esperada nesta máquina:

```text
2.13.0+cu130 13.0 NVIDIA GeForce RTX 5070 Laptop GPU
cap (12, 0) | build ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
torch.Size([2, 16, 24])
```

## AWS

O mesmo `requirements.txt` serve. Instâncias com GPU e suas arquiteturas:

| instância | GPU | compute cap. |
| --- | --- | --- |
| g5 | A10G | 8.6 (`sm_86`) |
| g6 | L4 | 8.9 (`sm_89`) |
| p4d | A100 | 8.0 (`sm_80`) |
| p5 | H100 | 9.0 (`sm_90`) |

O wheel cu130 traz cubins para `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`, o que atende todas
essas instâncias. A L4 (`sm_89`) não tem cubin exato, mas roda o de `sm_86`: a
compatibilidade binária do CUDA vale entre minors do mesmo major (nunca entre majors, e é
por isso que `sm_100` não socorre a `sm_120`). A célula de GPU do notebook reconhece esses
três casos — cubin exato, compatibilidade binária e JIT de PTX — e só acusa erro quando
nenhum se aplica.

Se o driver da instância for mais antigo que o runtime do wheel, `nvidia-smi` mostra uma
"CUDA Version" menor que a do build — nesse caso use `cu129`, ou uma AMI com driver mais
novo (as Deep Learning AMIs já vêm adequadas).

Para dados num volume separado (EBS/EFS), aponte o notebook sem editar código:

```bash
export HSI_DATA_DIR=/mnt/dados/dataHSI
```

Como este modelo é pequeno (37 mil parâmetros, batches de 32×15 floats), o gargalo é
lançamento de kernel, não computação nem I/O — uma g5.xlarge já satura o que a arquitetura
consegue usar; instâncias maiores não reduzem o tempo por época.

## Dados

```bash
.venv/bin/python scripts/download_data.py            # os três datasets (~67 MB)
.venv/bin/python scripts/download_data.py -d PU      # somente PaviaU
.venv/bin/python scripts/download_data.py --check    # verificar integridade
```

Vão para `data/dataHSI/` (ou `--out` / `$HSI_DATA_DIR`). Cada arquivo tem sha256 pinado e
é conferido também por conteúdo (shape e número de classes contra a especificação
publicada da cena), então um mirror que devolva página de erro é rejeitado.

Fonte canônica: [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes),
Grupo de Inteligencia Computacional, UPV/EHU. A EHU costuma responder HTTP 403 a download
automatizado; o script tenta a EHU primeiro e cai para um mirror no GitHub.

| dataset | cena | dimensões | classes |
| --- | --- | --- | --- |
| `IP` | Indian Pines (corrected) | 145×145×200 | 16 |
| `PU` | Pavia University | 610×340×103 | 9 |
| `SA` | Salinas (corrected) | 512×217×204 | 16 |

O notebook baixa o dataset selecionado automaticamente se os arquivos faltarem.

## Rodar

Pela linha de comando:

```bash
.venv/bin/python scripts/run_experiment.py                       # PaviaU, duas entradas
.venv/bin/python scripts/run_experiment.py -d IP --no-two-input  # Indian Pines, só espectro
.venv/bin/python scripts/run_experiment.py --json outputs/pu.json
```

Ou no notebook:

```bash
.venv/bin/python -m jupyterlab            # e abrir src/TCC3_SydneySouza_PyTorch.ipynb
```

Sem interface, executando o notebook inteiro:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=2400 \
    --output-dir outputs --output executed.ipynb \
    src/TCC3_SydneySouza_PyTorch.ipynb
```

Dataset e modo saem da célula *Configuração do experimento*, num objeto só:

```python
config = ExperimentConfig(dataset='PU', two_input=True)   # 'IP', 'PU' ou 'SA'
```

Saídas (curvas em PDF, checkpoint `.pt`) vão para `outputs/`.

## Testes

```bash
.venv/bin/python tests/test_hsi.py
```

Rodam em segundos, sem GPU e sem os `.mat` baixados — usam cenas sintéticas e cobrem o filtro
de background, o split estratificado, os shapes que atravessam o `Conv1d` e a parada
antecipada.

## Resultado de referência

Execução completa nesta máquina (RTX 5070 Laptop, 36 SMs) — PaviaU, `two_input=True`,
`SEED=42`, `DETERMINISTIC=True`:

```text
Early stopping na época 21 de 60 (melhor val_loss: 0.0561)
Tempo total de treino: 59.32s | média por época: 2.82s
OA:    0.9763
AA:    0.9579
kappa: 0.9685
```

Duas execuções independentes desse notebook na mesma máquina dão resultado **bit a bit
idêntico** — verificado comparando histórico de perda por época, OA, AA, kappa e pesos.
Isso depende de três coisas configuradas no notebook:

- `random_state=42` no PCA e nos splits, `SEED` em NumPy e PyTorch;
- `generator` semeado no `DataLoader` de treino, para a ordem dos batches não depender do
  estado global do RNG. O `fit` repõe essa semente antes da primeira época, então inspecionar
  um batch antes de treinar não altera o resultado;
- `DETERMINISTIC = True` na célula de GPU: `cudnn.deterministic=True`,
  `cudnn.benchmark=False` e TF32 desligado. Com `benchmark=True` o cuDNN mede e escolhe o
  kernel mais rápido por shape, e a escolha pode variar entre execuções; TF32 é
  determinístico, mas trunca a mantissa e dá números diferentes de FP32 puro. Troque para
  `False` apenas ao medir desempenho, sabendo que aí os últimos dígitos podem oscilar.

Entre GPUs de arquiteturas diferentes (a RTX 5070 e uma instância AWS, por exemplo) ainda há
variação nas últimas casas decimais, porque a ordem de redução dos kernels muda. Ao comparar
execuções, mantenha fixos dataset, semente, `two_input` e a máquina.
