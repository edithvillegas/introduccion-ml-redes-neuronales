"""
Genera un ejemplo adversario (método FGSM de Goodfellow et al., 2015) sobre la foto
de la guacamaya con una ResNet-18 preentrenada en ImageNet.

Genera diapositivas/datos/adversario.json y diapositivas/img/adversario_*.png

Uso:
    python scripts/precalcular_adversario.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

RAIZ = Path(__file__).resolve().parents[1]
torch.manual_seed(0)

pesos = ResNet18_Weights.IMAGENET1K_V1
modelo = resnet18(weights=pesos).eval()
categorias = pesos.meta["categories"]

img = Image.open(RAIZ / "diapositivas" / "img" / "foto_convolucion.jpg").convert("RGB")
lado = min(img.size)
izq, arriba = (img.size[0] - lado) // 2, (img.size[1] - lado) // 2
img = img.crop((izq, arriba, izq + lado, arriba + lado)).resize((224, 224))
x = torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255
media = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
desv = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def predecir(imagen):
    with torch.no_grad():
        p = F.softmax(modelo((imagen - media) / desv), dim=1)[0]
    top = torch.topk(p, 3)
    return [[categorias[int(i)], float(v)] for v, i in zip(top.values, top.indices)]


clase_original = int(torch.argmax(modelo((x - media) / desv)))
# FGSM iterado y dirigido: empujar la imagen hacia la clase "hongo" (mushroom, 947) con cambios minúsculos
objetivo = torch.tensor([947])
x_adv = x.clone()
epsilon = 2 / 255
for _ in range(10):
    x_adv.requires_grad_(True)
    perdida = F.cross_entropy(modelo((x_adv - media) / desv), objetivo)
    grad, = torch.autograd.grad(perdida, x_adv)
    x_adv = (x_adv - (0.5 / 255) * grad.sign()).detach()
    x_adv = torch.clamp(torch.min(torch.max(x_adv, x - epsilon), x + epsilon), 0, 1)

pert = (x_adv - x)[0].permute(1, 2, 0).numpy()
IMG = RAIZ / "diapositivas" / "img"
a_png = lambda arr, nombre: Image.fromarray(np.clip(arr * 255, 0, 255).round().astype(np.uint8)).save(IMG / nombre)  # noqa: E731
a_png(x[0].permute(1, 2, 0).numpy(), "adversario_original.png")
a_png(x_adv[0].permute(1, 2, 0).numpy(), "adversario_modificada.png")
a_png(pert / epsilon * 0.5 + 0.5, "adversario_perturbacion.png")
salida = {"pred_original": predecir(x), "pred_adversaria": predecir(x_adv), "epsilon_255": 2}
print("Original:", salida["pred_original"])
print("Adversaria:", salida["pred_adversaria"])
json.dump(salida, open(RAIZ / "diapositivas" / "datos" / "adversario.json", "w"), indent=1)
print("✓ adversario.json y 3 imágenes PNG")
