import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from hsi.config import CONFIG, ExperimentConfig
from hsi.experimentos.infra import caminho_resultado
from hsi.pipeline import run_experiment


def experimento_01(*cenas: str) -> pd.DataFrame:
    """Hiperparâmetros originais nas três cenas, com as duas CNNs, semente 42."""
    cenas = cenas or tuple(CONFIG["experimentos"]["01"]["cenas"])
    linhas, classes = [], []
    for cena, cnn in ((c, a) for c in cenas for a in ("projeto", "original")):
        r = run_experiment(ExperimentConfig(dataset=cena, cnn=cnn))
        linhas.append({"cena": cena, "cnn": cnn, "OA": r["OA"], "AA": r["AA"], "Kappa": r["Kappa"],
                       "épocas": r["epocas"], "melhor_época": r["melhor_epoca"],
                       "motivo": r["motivo_parada"]})
        d = r["dados"]
        verdade = d["rotulos"][d["teste"]]
        confusao = confusion_matrix(verdade, r["previsto"], labels=np.arange(d["n_classes"]))
        for k, linha in enumerate(confusao):
            classes.append({"cena": cena, "cnn": cnn, "classe": k + 1, "teste": int(linha.sum()),
                            "acertos": int(linha[k])})
        print(f"{cena} {cnn:<8}: OA {r['OA']:.4f}  AA {r['AA']:.4f}  Kappa {r['Kappa']:.4f}  "
              f"épocas {r['epocas']} (melhor {r['melhor_epoca']}, {r['motivo_parada']})", flush=True)
    quadro = pd.DataFrame(linhas)
    quadro.to_csv(caminho_resultado("01_linha_base.csv"), index=False)
    pd.DataFrame(classes).to_csv(caminho_resultado("01_linha_base_por_classe.csv"), index=False)
    print(f"CSV: {caminho_resultado('01_linha_base.csv')}")
    return quadro
