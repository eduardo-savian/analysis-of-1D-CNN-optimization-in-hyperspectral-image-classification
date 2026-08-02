#!/usr/bin/env python3
"""Baixa os datasets hiperespectrais usados no TCC para um diretório local.

Substitui o Google Drive do notebook original: funciona igual na máquina local
e numa instância AWS, sem depender de montagem de drive.

Uso:
    python scripts/download_data.py                 # baixa os três datasets
    python scripts/download_data.py --dataset PU    # só PaviaU
    python scripts/download_data.py --out /mnt/data/dataHSI
    python scripts/download_data.py --check         # só verifica o que já existe

Cada arquivo tem sha256 pinado. Se o download vier corrompido ou um mirror
devolver uma página de erro, a verificação falha e o próximo mirror é tentado.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

# Fonte canônica dos dados: Grupo de Inteligencia Computacional (GIC), UPV/EHU
# https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
#
# A EHU costuma bloquear download automatizado (HTTP 403), então cada arquivo
# tem uma lista de origens tentadas em ordem. O sha256 pinado é o que garante
# que qualquer mirror entregue exatamente o arquivo esperado.
EHU = "https://www.ehu.eus/ccwintco/uploads"
GH = "https://github.com/gokriznastic/HybridSN/raw/master/data"

FILES = {
    "IP": [
        {
            "name": "Indian_pines_corrected.mat",
            "sha256": "ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939",
            "size": 5953527,
            "urls": [f"{EHU}/6/67/Indian_pines_corrected.mat", f"{GH}/Indian_pines_corrected.mat"],
        },
        {
            "name": "Indian_pines_gt.mat",
            "sha256": "65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c",
            "size": 1125,
            "urls": [f"{EHU}/c/c4/Indian_pines_gt.mat", f"{GH}/Indian_pines_gt.mat"],
        },
    ],
    "PU": [
        {
            "name": "PaviaU.mat",
            "sha256": "28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb",
            "size": 34806917,
            "urls": [f"{EHU}/e/ee/PaviaU.mat", f"{GH}/PaviaU.mat"],
        },
        {
            "name": "PaviaU_gt.mat",
            "sha256": "23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829",
            "size": 11005,
            "urls": [f"{EHU}/5/50/PaviaU_gt.mat", f"{GH}/PaviaU_gt.mat"],
        },
    ],
    "SA": [
        {
            "name": "Salinas_corrected.mat",
            "sha256": "5ec1c0d22f56d18ecd336f8e35735863c0f160682e04e0c18ef3f89a3334d87d",
            "size": 26552770,
            "urls": [f"{EHU}/a/a3/Salinas_corrected.mat", f"{GH}/Salinas_corrected.mat"],
        },
        {
            "name": "Salinas_gt.mat",
            "sha256": "ecfab4d31ef5553f097943235d8ea502038eb4a2067b2ad10b33e37c949955e2",
            "size": 4277,
            "urls": [f"{EHU}/f/fa/Salinas_gt.mat", f"{GH}/Salinas_gt.mat"],
        },
    ],
}

# Especificação publicada de cada cena — usada para conferir que o .mat baixado
# é de fato o dataset certo, e não outro arquivo com o nome correto.
SPEC = {
    "Indian_pines_corrected.mat": ("indian_pines_corrected", (145, 145, 200), None),
    "Indian_pines_gt.mat": ("indian_pines_gt", (145, 145), 16),
    "PaviaU.mat": ("paviaU", (610, 340, 103), None),
    "PaviaU_gt.mat": ("paviaU_gt", (610, 340), 9),
    "Salinas_corrected.mat": ("salinas_corrected", (512, 217, 204), None),
    "Salinas_gt.mat": ("salinas_gt", (512, 217), 16),
}

DEFAULT_OUT = os.environ.get(
    "HSI_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataHSI"),
)

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def fetch(url: str, dest: str, expected_sha: str, timeout: int = 120) -> None:
    """Baixa url para dest e valida o sha256. Levanta exceção se não bater."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Referer": "https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes",
        },
    )
    tmp_fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix=".part")
    os.close(tmp_fd)
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as out:
            total = int(r.headers.get("Content-Length") or 0)
            got = 0
            while True:
                block = r.read(1 << 20)
                if not block:
                    break
                out.write(block)
                got += len(block)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {human(got)} / {human(total)} ({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r    {human(got)}", end="", flush=True)
        dt = time.perf_counter() - t0
        print(f"\r    {human(got)} em {dt:.1f}s ({human(got / max(dt, 1e-9))}/s)      ")

        actual = sha256_of(tmp)
        if actual != expected_sha:
            raise ValueError(f"sha256 divergente\n      esperado: {expected_sha}\n      obtido:   {actual}")
        shutil.move(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def verify_content(path: str) -> str:
    """Confere shape/classes do .mat contra a especificação publicada da cena.

    Retorna uma descrição legível. Requer scipy; se ausente, avisa e não falha.
    """
    try:
        import numpy as np
        import scipy.io as sio
    except ImportError:
        return "conteúdo não verificado (scipy/numpy ausentes)"

    key, shape, nclass = SPEC[os.path.basename(path)]
    arr = sio.loadmat(path)[key]
    if arr.shape != shape:
        raise ValueError(f"shape {arr.shape} != esperado {shape}")
    desc = f"{key} {arr.shape} {arr.dtype}"
    if nclass is not None:
        found = len(np.unique(arr)) - 1  # descontando o background (0)
        if found != nclass:
            raise ValueError(f"{found} classes != esperado {nclass}")
        desc += f", {found} classes + background"
    return desc


def ensure(entry: dict, out_dir: str, force: bool = False, check_only: bool = False) -> bool:
    name = entry["name"]
    dest = os.path.join(out_dir, name)

    if os.path.exists(dest) and not force:
        if sha256_of(dest) == entry["sha256"]:
            print(f"  [ok]   {name} já presente e íntegro")
            print(f"         {verify_content(dest)}")
            return True
        if check_only:
            print(f"  [!]    {name} existe mas o sha256 não bate — rode sem --check para rebaixar")
            return False
        print(f"  [!]    {name} existe mas o sha256 não bate — baixando de novo")
    elif check_only:
        print(f"  [--]   {name} ausente")
        return False

    for i, url in enumerate(entry["urls"], 1):
        print(f"  [get]  {name}  ({human(entry['size'])})")
        print(f"         origem {i}/{len(entry['urls'])}: {url}")
        try:
            fetch(url, dest, entry["sha256"])
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError, OSError) as e:
            print(f"         falhou: {e}")
            continue
        print(f"         {verify_content(dest)}")
        return True

    print(f"  [FAIL] não foi possível obter {name} de nenhuma origem")
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", "-d", action="append", choices=sorted(FILES), metavar="{IP,PU,SA}",
                   help="dataset a baixar (repetível); padrão: todos")
    p.add_argument("--out", "-o", default=DEFAULT_OUT, help=f"diretório de saída (padrão: {DEFAULT_OUT})")
    p.add_argument("--force", "-f", action="store_true", help="rebaixar mesmo se já íntegro")
    p.add_argument("--check", action="store_true", help="apenas verificar, sem baixar")
    args = p.parse_args()

    targets = args.dataset or sorted(FILES)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Diretório de dados: {out_dir}\n")

    all_ok = True
    for ds in targets:
        label = {"IP": "Indian Pines", "PU": "Pavia University", "SA": "Salinas Valley"}[ds]
        print(f"{ds} — {label}")
        for entry in FILES[ds]:
            all_ok &= ensure(entry, out_dir, force=args.force, check_only=args.check)
        print()

    if all_ok:
        print("Tudo pronto. Aponte o notebook para este diretório com:")
        print(f'  export HSI_DATA_DIR="{out_dir}"')
        return 0
    print("Alguns arquivos não foram obtidos/validados.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
