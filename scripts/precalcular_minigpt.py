"""
Entrena el MiniGPT (scripts/minigpt.py) con el Quijote y guarda la curva de pérdida
y muestras de texto en distintos momentos del entrenamiento.

Genera diapositivas/datos/minigpt_quijote.json

Uso:
    python scripts/precalcular_minigpt.py --texto ruta/a/pg2000.txt
    (el texto se descarga de https://www.gutenberg.org/cache/epub/2000/pg2000.txt)
"""

import argparse
import json
import time
from pathlib import Path

import torch

from minigpt import MiniGPT, generar, limpiar_gutenberg

RAIZ = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--texto", required=True)
parser.add_argument("--pasos", type=int, default=2000)
args = parser.parse_args()

torch.manual_seed(1)
texto = limpiar_gutenberg(open(args.texto, encoding="utf-8").read())
vocab = sorted(set(texto))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
datos = torch.tensor([stoi[c] for c in texto], dtype=torch.long)
n = int(0.9 * len(datos))
entreno, valida = datos[:n], datos[n:]
CONTEXTO, LOTE = 128, 32


def lote(d):
    ix = torch.randint(len(d) - CONTEXTO - 1, (LOTE,))
    return torch.stack([d[i:i + CONTEXTO] for i in ix]), torch.stack([d[i + 1:i + CONTEXTO + 1] for i in ix])


modelo = MiniGPT(len(vocab), d=128, n_capas=4, n_cabezas=4, contexto=CONTEXTO)
n_param = sum(p.numel() for p in modelo.parameters())
opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3)
inicio_txt = torch.tensor([[stoi[c] for c in "En un lugar de la Mancha"]])
muestras_en = {0, 50, 200, 500, 1000, args.pasos}
curva, muestras = [], {}
t0 = time.time()
for paso in range(args.pasos + 1):
    if paso in muestras_en:
        torch.manual_seed(7)
        txt = "".join(itos[int(i)] for i in generar(modelo, inicio_txt, 220, temperatura=0.8)[0])
        muestras[paso] = txt
        print(f"--- paso {paso} ---\n{txt}\n")
    if paso == args.pasos:
        break
    modelo.train()
    x, y = lote(entreno)
    _, perdida = modelo(x, y)
    opt.zero_grad()
    perdida.backward()
    opt.step()
    if paso % 25 == 0:
        modelo.eval()
        with torch.no_grad():
            pv = sum(modelo(*lote(valida))[1].item() for _ in range(4)) / 4
        curva.append([paso, round(perdida.item(), 4), round(pv, 4)])
print(f"{n_param} parámetros, {time.time() - t0:.0f} s")
json.dump({"parametros": n_param, "vocabulario": len(vocab), "caracteres": len(texto), "curva": curva,
           "muestras": {str(k): v for k, v in muestras.items()}},
          open(RAIZ / "diapositivas" / "datos" / "minigpt_quijote.json", "w"), ensure_ascii=False, indent=1)
print("✓ minigpt_quijote.json")
