"""
Precalcula los datos de las gráficas de redes convolucionales y autocodificadores
que aparecen en las diapositivas (entrena modelos pequeños con MNIST).

Genera en diapositivas/datos/:
  - cnn_mnist.json          filtros aprendidos, mapas de activación, exactitud, matriz de confusión
  - autocodificador.json    espacio latente 2D de un autocodificador convolucional + decodificaciones

Uso (necesita torch y torchvision; tarda ~3–5 min en CPU):
    python scripts/precalcular_cnn.py --datos /ruta/para/descargar/mnist
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "diapositivas" / "datos"
SALIDA.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--datos", default=str(Path.home() / ".cache" / "mnist"))
parser.add_argument("--epocas", type=int, default=2)
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"  # (MPS de Apple falla con algunas operaciones; usamos CPU o CUDA)
print("Dispositivo:", dispositivo)

tf = transforms.ToTensor()
entreno = datasets.MNIST(args.datos, train=True, download=True, transform=tf)
prueba = datasets.MNIST(args.datos, train=False, download=True, transform=tf)
cargador = torch.utils.data.DataLoader(entreno, batch_size=128, shuffle=True)
cargador_prueba = torch.utils.data.DataLoader(prueba, batch_size=1000)


def r(x, n=3):
    return np.round(np.asarray(x, dtype=float), n).tolist()


def entrenar(modelo, epocas, perdida_fn):
    opt = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    modelo.to(dispositivo).train()
    for ep in range(epocas):
        total = 0.0
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad()
            loss = perdida_fn(modelo, x, y)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        print(f"  época {ep + 1}: pérdida {total / len(entreno):.4f}")
    return modelo.eval()


def exactitud(modelo):
    aciertos, preds = 0, []
    with torch.no_grad():
        for x, y in cargador_prueba:
            p = modelo(x.to(dispositivo)).argmax(1).cpu()
            aciertos += (p == y).sum().item()
            preds.append(p)
    return aciertos / len(prueba), torch.cat(preds).numpy()


def n_param(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# 1. Perceptrón multicapa (línea base) vs red convolucional pequeña
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.red = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.red(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)    # 28 → 24
        self.conv2 = nn.Conv2d(8, 16, 5)   # 12 → 8
        self.fc = nn.Linear(16 * 4 * 4, 10)

    def forward(self, x, devolver_mapas=False):
        a1 = F.relu(self.conv1(x))
        p1 = F.max_pool2d(a1, 2)
        a2 = F.relu(self.conv2(p1))
        p2 = F.max_pool2d(a2, 2)
        out = self.fc(p2.flatten(1))
        return (out, a1, a2) if devolver_mapas else out


ce = lambda m, x, y: F.cross_entropy(m(x), y)  # noqa: E731
print("Entrenando MLP…")
mlp = entrenar(MLP(), args.epocas, ce)
acc_mlp, _ = exactitud(mlp)
print("Entrenando CNN…")
cnn = entrenar(CNN(), args.epocas, ce)
acc_cnn, preds = exactitud(cnn)
print(f"  exactitud MLP {acc_mlp:.4f} ({n_param(mlp)} parámetros), CNN {acc_cnn:.4f} ({n_param(cnn)} parámetros)")

y_prueba = prueba.targets.numpy()
confusion = np.zeros((10, 10), dtype=int)
for t, p in zip(y_prueba, preds):
    confusion[t, p] += 1
errores = np.where(preds != y_prueba)[0][:8]

ejemplos = [0, 1, 2]  # índices de prueba: dígitos 7, 2, 1
mapas = []
with torch.no_grad():
    for i in ejemplos:
        x = prueba[i][0].unsqueeze(0).to(dispositivo)
        _, a1, a2 = cnn(x, devolver_mapas=True)
        mapas.append({"imagen": r(prueba[i][0][0].numpy(), 2), "etiqueta": int(y_prueba[i]),
                      "capa1": r(a1[0].cpu().numpy(), 2), "capa2": r(a2[0].cpu().numpy(), 2)})

json.dump({
    "mlp": {"exactitud": acc_mlp, "parametros": n_param(mlp)},
    "cnn": {"exactitud": acc_cnn, "parametros": n_param(cnn)},
    "filtros1": r(cnn.conv1.weight.detach().cpu().numpy()[:, 0], 3),
    "mapas": mapas,
    "confusion": confusion.tolist(),
    "errores": [{"imagen": r(prueba[int(i)][0][0].numpy(), 2), "real": int(y_prueba[i]),
                 "predicho": int(preds[i])} for i in errores],
}, open(SALIDA / "cnn_mnist.json", "w"), separators=(",", ":"))
print("✓ cnn_mnist.json")


# ---------------------------------------------------------------------------
# 2. Autocodificador convolucional con espacio latente de 2 dimensiones
# ---------------------------------------------------------------------------
class Autocodificador(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.codificador = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 14×14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 7×7
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, d))
        self.decodificador = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.decodificador(self.codificador(x))


print("Entrenando autocodificador…")
ae = entrenar(Autocodificador(), args.epocas + 3, lambda m, x, y: F.binary_cross_entropy(m(x), x))
with torch.no_grad():
    xs = torch.stack([prueba[i][0] for i in range(2000)]).to(dispositivo)
    z = ae.codificador(xs).cpu().numpy()
    rec = ae(xs[:8]).cpu().numpy()[:, 0]
    # Rejilla en el espacio latente (percentiles 2–98 para evitar regiones vacías)
    lo, hi = np.percentile(z, 2, axis=0), np.percentile(z, 98, axis=0)
    n = 11
    g0, g1 = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    zz = torch.tensor([[a, b] for b in g1 for a in g0], dtype=torch.float32).to(dispositivo)
    rejilla = (ae.decodificador(zz).cpu().numpy()[:, 0] * 255).round().astype(int)

json.dump({
    "z": r(z, 3), "etiquetas": y_prueba[:2000].tolist(),
    "originales": r(xs[:8].cpu().numpy()[:, 0], 2), "reconstrucciones": r(rec, 2),
    "rejilla_ejes": [r(g0, 3), r(g1, 3)], "rejilla": rejilla.tolist(), "n": n,
}, open(SALIDA / "autocodificador.json", "w"), separators=(",", ":"))
print("✓ autocodificador.json")
