"""Notebook 05 · Redes convolucionales y autocodificadores."""

from utilidades import ESTILO_GRAFICAS, FUNCION_REVISAR, construir

celdas = [
    ("md", """
# 05 · Redes convolucionales y autocodificadores

**Notebook de práctica** · Escuela de Machine Learning Honduras 2026

En este notebook vas a:

1. Programar una **convolución a mano** con NumPy y aplicarla a una foto
2. Clasificar ropa (**Fashion-MNIST**) con un perceptrón multicapa
3. Construir y entrenar una **red convolucional** en PyTorch
4. Mirar **qué aprendió**: filtros y mapas de características
5. Analizar los **errores** con una matriz de confusión
6. Entrenar un **autocodificador** que quita ruido
7. Reto: superar el 90 % de exactitud

**Antes de empezar:** en Colab, *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*. Funciona también sin GPU, pero más lento (y con menos datos).

En cada **Ejercicio** escribe tu código donde dice `# TU CÓDIGO AQUÍ`; la celda siguiente revisa tu respuesta (✅ / ❌). Si te atoras, abre la **Pista**.
"""),
    ("code", f"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)
np.random.seed(0)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", dispositivo)
plt.rcParams["figure.figsize"] = (6, 4)

{ESTILO_GRAFICAS}

{FUNCION_REVISAR}

print("¡Todo listo!")
"""),
    ("md", """
## 1. La convolución a mano

En una red convolucional, un **filtro** pequeño (por ejemplo 3×3) recorre la imagen. En cada posición multiplicamos la ventana de la imagen por el filtro **elemento a elemento** y **sumamos**:

$$
\\text{salida}[i, j] = \\sum_{a=0}^{k-1}\\sum_{b=0}^{k-1} \\text{imagen}[i+a,\\, j+b]\\cdot \\text{filtro}[a, b]
$$

(Técnicamente esto es una *correlación cruzada*; en aprendizaje profundo se le llama convolución.)

Primero descargamos una foto de una guacamaya roja, el ave nacional de Honduras.
"""),
    ("code", """
import io, urllib.request
from PIL import Image

URL = ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
       "Scarlet_macaw_%28Ara_macao_cyanopterus%29_Copan.jpg/960px-Scarlet_macaw_%28Ara_macao_cyanopterus%29_Copan.jpg")
try:
    pedido = urllib.request.Request(URL, headers={"User-Agent": "EscuelaMLHonduras/1.0 (uso educativo)"})
    foto = Image.open(io.BytesIO(urllib.request.urlopen(pedido, timeout=20).read())).convert("L")
except Exception as e:
    print("No se pudo descargar la foto; usamos una imagen sintética.", e)
    xx, yy = np.meshgrid(np.linspace(-1, 1, 320), np.linspace(-1, 1, 240))
    foto = Image.fromarray(((np.sin(8 * xx) * np.cos(6 * yy) > 0) * 255).astype(np.uint8))
foto = np.asarray(foto.resize((320, 213)), dtype=float) / 255
plt.imshow(foto, cmap="gray"); plt.axis("off")
plt.title("Foto: Charles J. Sharp, CC BY-SA 4.0 (Wikimedia Commons)", fontsize=8)
plt.show()
print("Tamaño:", foto.shape)
"""),
    ("md", """
### Ejercicio 1 · Programa la convolución

Completa la función `convolucion2d(imagen, filtro)` (sin relleno y con paso 1: la salida mide `(H − k + 1) × (W − k + 1)`).

<details><summary>Pista</summary>

Dentro de los dos bucles: `salida[i, j] = np.sum(imagen[i:i+k, j:j+k] * filtro)`
</details>
"""),
    ("ej", """
def convolucion2d(imagen, filtro):
    H, W = imagen.shape
    k = filtro.shape[0]
    salida = np.zeros((H - k + 1, W - k + 1))
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            # TU CÓDIGO AQUÍ
            pass
    return salida
""", """
def convolucion2d(imagen, filtro):
    H, W = imagen.shape
    k = filtro.shape[0]
    salida = np.zeros((H - k + 1, W - k + 1))
    for i in range(H - k + 1):
        for j in range(W - k + 1):
            salida[i, j] = np.sum(imagen[i:i + k, j:j + k] * filtro)
    return salida
"""),
    ("code", """
from scipy.signal import correlate2d
A, K = np.random.rand(9, 11), np.random.rand(3, 3)
revisar(lambda: np.allclose(convolucion2d(A, K), correlate2d(A, K, mode="valid")),
        "¡Tu convolución coincide con la de SciPy!", "Compara tu resultado con correlate2d(A, K, mode='valid').")
"""),
    ("md", "Ahora apliquemos algunos filtros clásicos a la foto:"),
    ("code", """
filtros = {
    "bordes verticales (Sobel x)": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    "bordes horizontales (Sobel y)": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "desenfoque 5×5": np.ones((5, 5)) / 25,
}
fig, ejes = plt.subplots(1, 3, figsize=(15, 4))
for eje, (nombre, k) in zip(ejes, filtros.items()):
    res = convolucion2d(foto, k)
    eje.imshow(np.abs(res) if "bordes" in nombre else res, cmap="gray")
    eje.set_title(nombre); eje.axis("off")
plt.show()
"""),
    ("md", """
### Ejercicio 2 · Diseña tu propio filtro

Inventa un filtro 3×3 que detecte **bordes diagonales** (de arriba-izquierda a abajo-derecha) y muéstralo aplicado a la foto. No hay una única respuesta correcta: ¡experimenta!

<details><summary>Pista</summary>

Prueba con `np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])`: positivo arriba a la izquierda, negativo abajo a la derecha.
</details>
"""),
    ("ej", """
mi_filtro = np.zeros((3, 3))   # TU CÓDIGO AQUÍ

plt.imshow(np.abs(convolucion2d(foto, mi_filtro)), cmap="gray"); plt.axis("off"); plt.show()
""", """
mi_filtro = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])

plt.imshow(np.abs(convolucion2d(foto, mi_filtro)), cmap="gray"); plt.axis("off"); plt.show()
"""),
    ("code", """
revisar(lambda: mi_filtro.shape == (3, 3) and np.abs(mi_filtro).sum() > 0,
        "Tienes un filtro 3×3. ¿Qué bordes resalta?", "Define un arreglo 3×3 que no sea todo ceros.")
"""),
    ("md", """
## 2. Fashion-MNIST: 70 000 imágenes de ropa

Cada imagen es de 28×28 píxeles en escala de grises, de una de 10 clases. Es un poco más difícil que los dígitos de MNIST.
"""),
    ("code", """
CLASES = ["camiseta", "pantalón", "suéter", "vestido", "abrigo", "sandalia", "camisa", "tenis", "bolso", "botín"]

transformacion = transforms.ToTensor()          # convierte a tensor con valores entre 0 y 1
entreno = datasets.FashionMNIST("datos", train=True, download=True, transform=transformacion)
prueba = datasets.FashionMNIST("datos", train=False, download=True, transform=transformacion)

# Sin GPU usamos solo una parte de los datos para que sea rápido
if dispositivo == "cpu":
    entreno = torch.utils.data.Subset(entreno, range(15000))
cargador = torch.utils.data.DataLoader(entreno, batch_size=128, shuffle=True)
cargador_prueba = torch.utils.data.DataLoader(prueba, batch_size=1000)
print(len(entreno), "imágenes de entrenamiento,", len(prueba), "de prueba")

fig, ejes = plt.subplots(2, 8, figsize=(14, 4))
for eje, i in zip(ejes.ravel(), range(16)):
    img, etiqueta = prueba[i]
    eje.imshow(img[0], cmap="gray_r"); eje.set_title(CLASES[etiqueta], fontsize=9); eje.axis("off")
plt.show()
"""),
    ("md", """
## 3. Línea base: perceptrón multicapa

Estas dos funciones entrenan y evalúan **cualquier** modelo. Fíjate en el ciclo de entrenamiento: hacia adelante, pérdida, `backward()`, `step()`.
"""),
    ("code", """
def entrenar(modelo, epocas=3, lr=1e-3):
    modelo.to(dispositivo)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    for epoca in range(epocas):
        modelo.train()
        total = 0.0
        for x, y in cargador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            perdida = F.cross_entropy(modelo(x), y)
            opt.zero_grad()
            perdida.backward()
            opt.step()
            total += perdida.item() * len(x)
        print(f"época {epoca + 1}: pérdida = {total / len(entreno):.4f} · exactitud en prueba = {evaluar(modelo):.2%}")
    return modelo

@torch.no_grad()
def evaluar(modelo, devolver_predicciones=False):
    modelo.eval()
    preds = torch.cat([modelo(x.to(dispositivo)).argmax(1).cpu() for x, _ in cargador_prueba])
    exactitud = (preds == prueba.targets).float().mean().item()
    return (exactitud, preds) if devolver_predicciones else exactitud

def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())

mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))
print("Parámetros del perceptrón multicapa:", contar_parametros(mlp))
entrenar(mlp, epocas=3)
exactitud_mlp = evaluar(mlp)
"""),
    ("md", """
## 4. Tu red convolucional

### Ejercicio 3 · Construye la red

Completa la clase con esta arquitectura:

| Capa | Detalle | Tamaño de salida |
|---|---|---|
| `conv1` | `nn.Conv2d(1, 16, kernel_size=3, padding=1)` + ReLU | 16 × 28 × 28 |
| *pooling* | `F.max_pool2d(x, 2)` | 16 × 14 × 14 |
| `conv2` | `nn.Conv2d(16, 32, kernel_size=3, padding=1)` + ReLU | 32 × 14 × 14 |
| *pooling* | `F.max_pool2d(x, 2)` | 32 × 7 × 7 |
| aplanar | `x.flatten(1)` | 1568 |
| `fc` | `nn.Linear(32 * 7 * 7, 10)` | 10 |

<details><summary>Pista</summary>

```python
def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    return self.fc(x.flatten(1))
```
</details>
"""),
    ("ej", """
class RedConvolucional(nn.Module):
    def __init__(self):
        super().__init__()
        # TU CÓDIGO AQUÍ: define self.conv1, self.conv2 y self.fc

    def forward(self, x):
        # TU CÓDIGO AQUÍ
        return x
""", """
class RedConvolucional(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc(x.flatten(1))
"""),
    ("code", """
cnn = RedConvolucional()
revisar(lambda: cnn(torch.zeros(5, 1, 28, 28)).shape == (5, 10), "La salida tiene forma (lote, 10).",
        "La salida para un lote de 5 imágenes debe tener forma (5, 10).")
revisar(lambda: contar_parametros(cnn) == 20490, "20 490 parámetros: ¡arquitectura correcta!",
        f"Se esperaban 20 490 parámetros. ¿Revisaste los tamaños de las capas?")
"""),
    ("md", """
**Pregunta:** el perceptrón multicapa tiene ~100 000 parámetros y la red convolucional ~20 000. ¿Cuál crees que será mejor? Entrénala y compruébalo (con GPU tarda menos de un minuto).
"""),
    ("code", """
entrenar(cnn, epocas=3)
exactitud_cnn = evaluar(cnn)
print(f"\\nPerceptrón multicapa: {exactitud_mlp:.2%} con {contar_parametros(mlp):,} parámetros")
print(f"Red convolucional:    {exactitud_cnn:.2%} con {contar_parametros(cnn):,} parámetros")
revisar(exactitud_cnn > 0.83, "¡Tu red convolucional supera el 83 %! (con GPU y todos los datos llega a ~90 %)", "Entrena un poco más (más épocas).")
"""),
    ("md", """
## 5. ¿Qué aprendió la red?

Los **filtros** de la primera capa son matrices 3×3 de pesos. Los **mapas de características** son el resultado de aplicar cada filtro a una imagen.
"""),
    ("code", """
filtros_aprendidos = cnn.conv1.weight.detach().cpu()[:, 0]
fig, ejes = plt.subplots(2, 8, figsize=(12, 3.2))
lim = filtros_aprendidos.abs().max()
for k, eje in enumerate(ejes.ravel()):
    eje.imshow(filtros_aprendidos[k], cmap=DIVERGENTE, vmin=-lim, vmax=lim); eje.axis("off")
fig.suptitle("Los 16 filtros de la primera capa (coral = peso positivo, verde oscuro = negativo)")
plt.show()

img, etiqueta = prueba[0]
with torch.no_grad():
    mapas = F.relu(cnn.conv1(img.unsqueeze(0).to(dispositivo)))[0].cpu()
fig, ejes = plt.subplots(2, 9, figsize=(14, 3.4))
ejes[0, 0].imshow(img[0], cmap="gray_r"); ejes[0, 0].set_title(CLASES[etiqueta])
ejes[1, 0].axis("off")
for k in range(16):
    eje = ejes.ravel()[k + 1 + (k >= 8)]
    eje.imshow(mapas[k], cmap=SECUENCIAL)
for eje in ejes.ravel():
    eje.axis("off")
fig.suptitle("Mapas de características de la primera capa")
plt.show()
"""),
    ("md", "## 6. ¿En qué se equivoca?"),
    ("code", """
from sklearn.metrics import ConfusionMatrixDisplay
_, preds = evaluar(cnn, devolver_predicciones=True)
fig, eje = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay.from_predictions(prueba.targets, preds, display_labels=CLASES, xticks_rotation=60,
                                        ax=eje, colorbar=False, cmap=SECUENCIAL)
plt.show()
"""),
    ("md", """
### Ejercicio 4 · La confusión más común

Usa la matriz de confusión para encontrar el **par de clases** (real, predicha) **distintas** que la red confunde más veces. Guarda los nombres en `clase_real` y `clase_predicha`.

<details><summary>Pista</summary>

```python
from sklearn.metrics import confusion_matrix
C = confusion_matrix(prueba.targets, preds)
np.fill_diagonal(C, 0)          # ignorar los aciertos
i, j = np.unravel_index(C.argmax(), C.shape)
```
</details>
"""),
    ("ej", """
clase_real, clase_predicha = None, None   # TU CÓDIGO AQUÍ
print(clase_real, "→", clase_predicha)
""", """
from sklearn.metrics import confusion_matrix
C = confusion_matrix(prueba.targets, preds)
np.fill_diagonal(C, 0)
i, j = np.unravel_index(C.argmax(), C.shape)
clase_real, clase_predicha = CLASES[i], CLASES[j]
print(clase_real, "→", clase_predicha)
"""),
    ("code", """
from sklearn.metrics import confusion_matrix
_C = confusion_matrix(prueba.targets, preds); np.fill_diagonal(_C, 0)
_i, _j = np.unravel_index(_C.argmax(), _C.shape)
revisar(lambda: (clase_real, clase_predicha) == (CLASES[_i], CLASES[_j]),
        f"La confusión más común es {CLASES[_i]} → {CLASES[_j]}. ¿Tiene sentido?", "Revisa la pista.")
"""),
    ("md", """
## 7. Un autocodificador que quita ruido

Un **autocodificador** comprime la imagen en un código pequeño (**codificador**) y la reconstruye (**decodificador**). Si le damos imágenes **con ruido** y le pedimos reconstruir la imagen **limpia**, aprende a quitar el ruido. ¡No necesita etiquetas!

El decodificador usa `nn.ConvTranspose2d`, que hace lo contrario de una convolución con paso 2: **duplica** el tamaño.

### Ejercicio 5 · Completa el decodificador

El codificador va de 1×28×28 → 16×14×14 → 32×7×7. Escribe un decodificador **simétrico**:

- `nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)` + ReLU → 16×14×14
- `nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)` + Sigmoid → 1×28×28
"""),
    ("ej", """
class Autocodificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.codificador = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # → 16×14×14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())   # → 32×7×7
        self.decodificador = nn.Sequential(
            # TU CÓDIGO AQUÍ
        )

    def forward(self, x):
        return self.decodificador(self.codificador(x))
""", """
class Autocodificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.codificador = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),    # → 16×14×14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())   # → 32×7×7
        self.decodificador = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Sigmoid())

    def forward(self, x):
        return self.decodificador(self.codificador(x))
"""),
    ("code", """
ae = Autocodificador()
revisar(lambda: ae(torch.zeros(2, 1, 28, 28)).shape == (2, 1, 28, 28),
        "La reconstrucción tiene el mismo tamaño que la entrada.", "La salida debe medir (lote, 1, 28, 28).")
"""),
    ("code", """
RUIDO = 0.4
ae.to(dispositivo)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
for epoca in range(3):
    total = 0.0
    for x, _ in cargador:                       # ¡no usamos las etiquetas!
        x = x.to(dispositivo)
        x_ruido = (x + RUIDO * torch.randn_like(x)).clamp(0, 1)
        perdida = F.mse_loss(ae(x_ruido), x)     # reconstruir la imagen LIMPIA
        opt.zero_grad(); perdida.backward(); opt.step()
        total += perdida.item() * len(x)
    print(f"época {epoca + 1}: error de reconstrucción = {total / len(entreno):.4f}")

x = torch.stack([prueba[i][0] for i in range(8)]).to(dispositivo)
x_ruido = (x + RUIDO * torch.randn_like(x)).clamp(0, 1)
with torch.no_grad():
    limpia = ae(x_ruido)
fig, ejes = plt.subplots(3, 8, figsize=(14, 5.5))
for k in range(8):
    for fila, (imgs, titulo) in enumerate([(x_ruido, "con ruido"), (limpia, "reconstruida"), (x, "original")]):
        ejes[fila, k].imshow(imgs[k, 0].cpu(), cmap="gray_r"); ejes[fila, k].axis("off")
        if k == 0:
            ejes[fila, k].set_title(titulo, loc="left")
plt.show()
"""),
    ("md", """
## 8. Reto final: supera el 90 %

Modifica la red convolucional para superar **90 %** de exactitud en el conjunto de prueba. Ideas:

- Más filtros (32 y 64) o una tercera capa convolucional
- `nn.BatchNorm2d` después de cada convolución
- `nn.Dropout(0.3)` antes de la capa final (regularización)
- Más épocas de entrenamiento (¡pero vigila el sobreajuste!)

Guarda tu exactitud final en `exactitud_reto`.
"""),
    ("ej", """
class MiMejorRed(nn.Module):
    def __init__(self):
        super().__init__()
        # TU CÓDIGO AQUÍ

    def forward(self, x):
        # TU CÓDIGO AQUÍ
        return x

mejor = MiMejorRed()
# entrenar(mejor, epocas=...)
exactitud_reto = 0.0   # evaluar(mejor)
""", """
class MiMejorRed(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloque1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.bloque2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.salida = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 10))

    def forward(self, x):
        return self.salida(self.bloque2(self.bloque1(x)))

mejor = MiMejorRed()
entrenar(mejor, epocas=5 if dispositivo == "cuda" else 2)
exactitud_reto = evaluar(mejor)
"""),
    ("code", """
revisar(exactitud_reto > 0.90, f"¡{exactitud_reto:.2%}! Superaste el reto.",
        f"Llevas {exactitud_reto:.2%}. ¡Sigue intentando! (sin GPU entrenamos con menos datos, así que es más difícil)")
"""),
    ("md", """
---

## Resumen

- Una **convolución** es un filtro pequeño que recorre la imagen: detecta el mismo patrón en cualquier posición.
- Las redes convolucionales tienen **muchos menos parámetros** que un perceptrón multicapa y funcionan **mejor** con imágenes.
- Los filtros **se aprenden**: en la primera capa suelen parecer detectores de bordes y texturas.
- Un **autocodificador** aprende sin etiquetas a comprimir y reconstruir; si entrenamos con ruido, aprende a **quitarlo**.

**Para seguir:** Distill, *Feature Visualization* (2017) y *Zoom In: An Introduction to Circuits* (2020) · CNN Explainer (poloclub.github.io/cnn-explainer)
"""),
]

if __name__ == "__main__":
    construir("05_redes_convolucionales", celdas, gpu=True)
