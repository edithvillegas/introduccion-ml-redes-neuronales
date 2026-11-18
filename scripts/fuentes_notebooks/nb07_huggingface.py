"""Notebook 07 · Python, PyTorch y Hugging Face: usa, ajusta y crea tus propios modelos."""

from pathlib import Path

from utilidades import ESTILO_GRAFICAS, FUNCION_REVISAR, construir

# El código del mini-GPT es el mismo que el de scripts/minigpt.py
_fuente = (Path(__file__).resolve().parents[1] / "minigpt.py").read_text(encoding="utf-8")
CLASES_MINIGPT = _fuente[_fuente.index("class AutoAtencion"):_fuente.index("@torch.no_grad()")].strip()

celdas = [
    ("md", """
# 07 · Python, PyTorch y Hugging Face

**Notebook de práctica** · Escuela de Machine Learning Honduras 2026

Hoy vas a **usar**, **ajustar** y **crear** modelos de aprendizaje profundo:

1. Python en 5 minutos (repaso)
2. PyTorch: tensores y derivadas automáticas
3. Modelos listos para usar con `pipeline` de Hugging Face
4. ¿Qué hace `pipeline` por dentro?
5. **Ajuste fino**: entrena tu propio clasificador de reseñas en español con BETO
6. **Desde cero**: un GPT diminuto que aprende a escribir como Cervantes
7. (Opcional) Comparte tu modelo con una demo de Gradio

**Antes de empezar:** *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*. Las secciones 5 y 6 necesitan GPU para ir rápido (sin GPU usamos menos datos y menos pasos).
"""),
    ("code", f"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", dispositivo)

{ESTILO_GRAFICAS}

{FUNCION_REVISAR}

print("¡Todo listo!")
"""),
    ("md", """
## 1. Python en 5 minutos

```python
texto = "Murciélago"                 # str
for letra in texto.lower():          # recorrer letras
    if letra in "aeiou":             # condición
        ...
departamentos = ["Copán", "Lempira", "Olancho"]       # lista
largos = {d: len(d) for d in departamentos}         # comprensión de diccionario
```

### Ejercicio 1 · Contar vocales

Escribe `contar_vocales(texto)` que cuente las vocales (con y sin tilde: `aeiouáéíóú`), sin importar mayúsculas.
"""),
    ("ej", """
def contar_vocales(texto):
    # TU CÓDIGO AQUÍ
    return 0

contar_vocales("Murciélago")
""", """
def contar_vocales(texto):
    return sum(1 for letra in texto.lower() if letra in "aeiouáéíóú")

contar_vocales("Murciélago")
"""),
    ("code", """
revisar(lambda: contar_vocales("Murciélago") == 5 and contar_vocales("HONDURAS") == 3,
        "¡Murciélago tiene las 5 vocales!", "¿Consideras mayúsculas y tildes?")
"""),
    ("md", """
### Ejercicio 2 · Comprensión de diccionarios

Crea el diccionario `largos` que asocie cada departamento de la lista con su número de letras, **solo** para los departamentos con más de 6 letras.
"""),
    ("ej", """
departamentos = ["Atlántida", "Choluteca", "Colón", "Copán", "Cortés", "Gracias a Dios", "Lempira", "Olancho", "Valle", "Yoro"]
largos = {}   # TU CÓDIGO AQUÍ
largos
""", """
departamentos = ["Atlántida", "Choluteca", "Colón", "Copán", "Cortés", "Gracias a Dios", "Lempira", "Olancho", "Valle", "Yoro"]
largos = {d: len(d) for d in departamentos if len(d) > 6}
largos
"""),
    ("code", """
revisar(lambda: largos == {d: len(d) for d in departamentos if len(d) > 6}, "¡Perfecto!", "Usa {d: len(d) for d in ... if ...}")
"""),
    ("md", """
## 2. PyTorch: tensores y derivadas automáticas

Un **tensor** es como un arreglo de NumPy, pero puede vivir en la **GPU** y **recordar las operaciones** para calcular derivadas automáticamente (*autograd*): ¡es la retropropagación que vimos en la sesión de redes neuronales!
"""),
    ("code", """
x = torch.tensor([[1., 2.], [3., 4.]])
print("forma:", x.shape)
print("x @ x.T =\\n", x @ x.T)
print("en", x.to(dispositivo).device)

w = torch.tensor(1.0, requires_grad=True)
f = (3 * w - 6) ** 2
f.backward()                 # calcula df/dw
print("df/dw en w = 1:", w.grad.item(), "   (a mano: 6·(3w − 6) = −18)")
"""),
    ("md", """
### Ejercicio 3 · Tu primera derivada automática

Usa autograd para calcular la derivada de $g(w) = w^3 + 2w$ en $w = 2$. Guarda el resultado (un número) en `derivada`.
"""),
    ("ej", """
derivada = None   # TU CÓDIGO AQUÍ
derivada
""", """
w = torch.tensor(2.0, requires_grad=True)
g = w ** 3 + 2 * w
g.backward()
derivada = w.grad.item()
derivada
"""),
    ("code", """
revisar(lambda: abs(derivada - 14) < 1e-6, "g'(2) = 3·2² + 2 = 14", "Crea w con requires_grad=True, calcula g y llama g.backward().")
"""),
    ("md", """
### Ejercicio 4 · Descenso de gradiente con autograd

Generamos datos con $y = 2x + 1 + \\text{ruido}$. Completa el bucle para encontrar `w` y `b` minimizando el error cuadrático medio. En cada paso:

1. predicción `y_pred = w * x + b`, pérdida `((y_pred - y) ** 2).mean()`
2. `perdida.backward()`
3. dentro de `with torch.no_grad():` actualiza `w -= lr * w.grad` y `b -= lr * b.grad`
4. pon los gradientes en cero: `w.grad.zero_()` y `b.grad.zero_()`
"""),
    ("ej", """
x = torch.rand(100) * 10
y = 2 * x + 1 + torch.randn(100) * 0.5
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

for paso in range(2000):
    # TU CÓDIGO AQUÍ
    pass

print(f"w = {w.item():.3f}, b = {b.item():.3f}")
""", """
x = torch.rand(100) * 10
y = 2 * x + 1 + torch.randn(100) * 0.5
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01

for paso in range(2000):
    y_pred = w * x + b
    perdida = ((y_pred - y) ** 2).mean()
    perdida.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"w = {w.item():.3f}, b = {b.item():.3f}")
"""),
    ("code", """
revisar(lambda: abs(w.item() - 2) < 0.1 and abs(b.item() - 1) < 0.35, "¡Encontraste la recta y = 2x + 1!",
        "Revisa el orden: predicción → pérdida → backward → actualizar → poner gradientes en cero.")
"""),
    ("md", """
## 3. Modelos listos para usar: `pipeline`

[Hugging Face](https://huggingface.co) es una plataforma con cientos de miles de modelos abiertos. Con `pipeline` se usan en una línea. La primera vez cada modelo se **descarga** (puede tardar un poco).
"""),
    ("code", """
from transformers import pipeline

sentimiento = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=dispositivo)
for t in ["¡Me encantó la clase de redes neuronales!", "El servicio fue lento y la comida llegó fría.", "Está bien, nada especial."]:
    print(sentimiento(t)[0], "←", t)
"""),
    ("code", """
clasificador = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=dispositivo)
clasificador("El Olimpia ganó el clásico 2 a 1 en el Estadio Nacional.",
             candidate_labels=["deportes", "política", "ciencia", "economía"])
"""),
    ("md", """
### Ejercicio 5 · Clasifica titulares sin entrenar nada

La clasificación *zero-shot* usa **etiquetas que tú inventas**. Escribe 3 titulares de noticias (inventados o reales) y clasifícalos con tus propias etiquetas. Guarda los resultados en la lista `resultados`.
"""),
    ("ej", """
titulares = [
    # TU CÓDIGO AQUÍ
]
etiquetas = ["deportes", "política", "ciencia", "economía", "cultura"]
resultados = [clasificador(t, candidate_labels=etiquetas) for t in titulares]
for r in resultados:
    print(f"{r['labels'][0]:>10s} ({r['scores'][0]:.0%}) ← {r['sequence']}")
""", """
titulares = [
    "Estudiantes de la UNAH participan en una escuela internacional de inteligencia artificial.",
    "El precio del café hondureño sube en el mercado internacional.",
    "Nuevo festival de música en San Pedro Sula reúne a miles de personas.",
]
etiquetas = ["deportes", "política", "ciencia", "economía", "cultura"]
resultados = [clasificador(t, candidate_labels=etiquetas) for t in titulares]
for r in resultados:
    print(f"{r['labels'][0]:>10s} ({r['scores'][0]:.0%}) ← {r['sequence']}")
"""),
    ("code", """
revisar(lambda: len(resultados) >= 3, "¡Clasificaste tus titulares sin entrenar ningún modelo!", "Escribe al menos 3 titulares.")
"""),
    ("md", "Un modelo de **visión** (Vision Transformer) clasifica una foto de la guacamaya roja de Copán:"),
    ("code", """
import io, urllib.request
from PIL import Image
URL = ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
       "Scarlet_macaw_%28Ara_macao_cyanopterus%29_Copan.jpg/960px-Scarlet_macaw_%28Ara_macao_cyanopterus%29_Copan.jpg")
try:
    pedido = urllib.request.Request(URL, headers={"User-Agent": "EscuelaMLHonduras/1.0 (uso educativo)"})
    foto = Image.open(io.BytesIO(urllib.request.urlopen(pedido, timeout=20).read())).convert("RGB")
    vision = pipeline("image-classification", model="google/vit-base-patch16-224", device=dispositivo)
    plt.imshow(foto); plt.axis("off"); plt.title("Foto: Charles J. Sharp, CC BY-SA 4.0", fontsize=8); plt.show()
    for r in vision(foto)[:3]:
        print(f"{r['label']:>40s}  {r['score']:.1%}")
except Exception as e:
    print("No se pudo descargar la foto:", e)
"""),
    ("md", """
Y un pequeño **asistente conversacional** (Qwen2.5-0.5B-Instruct). Con los modelos de chat se puede traducir, resumir, explicar… (¡con errores! es un modelo muy pequeño).
"""),
    ("code", """
chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device=dispositivo)
mensajes = [{"role": "system", "content": "Eres un traductor. Responde solo con la traducción."},
            {"role": "user", "content": "Traduce al inglés: La inteligencia artificial está cambiando la ciencia."}]
respuesta = chat(mensajes, max_new_tokens=60, do_sample=False)
print(respuesta[0]["generated_text"][-1]["content"])
"""),
    ("md", """
## 4. ¿Qué hace `pipeline` por dentro?

1. El **tokenizador** convierte el texto en números (ids)
2. El **modelo** produce **logits** (un número por clase)
3. Una **softmax** los convierte en probabilidades
"""),
    ("code", """
from transformers import AutoTokenizer, AutoModelForSequenceClassification
nombre = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizador = AutoTokenizer.from_pretrained(nombre)
modelo_sent = AutoModelForSequenceClassification.from_pretrained(nombre).to(dispositivo)

frase = "¡Me encantó la clase de redes neuronales!"
entrada = tokenizador(frase, return_tensors="pt").to(dispositivo)
print("ids:", entrada["input_ids"][0].tolist())
print("tokens:", tokenizador.convert_ids_to_tokens(entrada["input_ids"][0]))
with torch.no_grad():
    logits = modelo_sent(**entrada).logits[0].cpu()
print("logits:", logits.numpy().round(2))
print("etiquetas:", modelo_sent.config.id2label)
"""),
    ("md", """
### Ejercicio 6 · De logits a probabilidades

Convierte `logits` en probabilidades con `torch.softmax` y guarda la **probabilidad de la clase más probable** en `prob_max`. Debe coincidir con el `score` que dio el `pipeline`.
"""),
    ("ej", """
prob_max = None   # TU CÓDIGO AQUÍ
prob_max
""", """
probs = torch.softmax(logits, dim=-1)
prob_max = probs.max().item()
prob_max
"""),
    ("code", """
revisar(lambda: abs(prob_max - sentimiento(frase)[0]["score"]) < 1e-3, "¡Igual que el pipeline!", "Usa torch.softmax(logits, dim=-1).max()")
"""),
    ("md", """
## 5. Ajuste fino: tu propio clasificador de reseñas en español

Vamos a **ajustar** BETO (un BERT entrenado en español en la Universidad de Chile) para decidir si una reseña de producto es **positiva o negativa**. Usamos reseñas de Amazon en español con 1 a 5 estrellas: 1–2 estrellas → negativa (0), 4–5 → positiva (1); descartamos las de 3.

> **Aprendizaje por transferencia:** BETO ya "sabe" español; solo le agregamos una capa nueva y ajustamos con pocos ejemplos.
"""),
    ("code", """
from datasets import load_dataset
resenas = load_dataset("SetFit/amazon_reviews_multi_es")
print(resenas)

N_ENTRENO = 4000 if dispositivo == "cuda" else 400
N_PRUEBA = 1000 if dispositivo == "cuda" else 200

def binarizar(ds, n):
    ds = ds.filter(lambda e: e["label"] != 2).shuffle(seed=0).select(range(n))
    return ds.map(lambda e: {"label": int(e["label"] >= 3)})

entreno_ds = binarizar(resenas["train"], N_ENTRENO)
prueba_ds = binarizar(resenas["test"], N_PRUEBA)
for e in entreno_ds.select(range(3)):
    print("positiva" if e["label"] else "negativa", "←", e["text"][:100])
"""),
    ("md", """
### Ejercicio 7 · Tokenizar el conjunto de datos

Completa `tokenizar(lote)` para que devuelva el resultado de `tokenizador_beto(lote["text"], truncation=True, max_length=128)`. La aplicamos a todo el conjunto con `.map(..., batched=True)`.
"""),
    ("ej", """
from transformers import AutoTokenizer
tokenizador_beto = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def tokenizar(lote):
    # TU CÓDIGO AQUÍ
    return {}

entreno_tok = entreno_ds.map(tokenizar, batched=True)
prueba_tok = prueba_ds.map(tokenizar, batched=True)
""", """
from transformers import AutoTokenizer
tokenizador_beto = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

def tokenizar(lote):
    return tokenizador_beto(lote["text"], truncation=True, max_length=128)

entreno_tok = entreno_ds.map(tokenizar, batched=True)
prueba_tok = prueba_ds.map(tokenizar, batched=True)
"""),
    ("code", """
revisar(lambda: "input_ids" in entreno_tok.column_names and max(len(i) for i in entreno_tok["input_ids"]) <= 128,
        "Datos tokenizados (máximo 128 tokens por reseña).", "¿Devuelves tokenizador_beto(lote['text'], truncation=True, max_length=128)?")
"""),
    ("code", """
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

beto_clasif = AutoModelForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", num_labels=2,
    id2label={0: "negativa", 1: "positiva"}, label2id={"negativa": 0, "positiva": 1})

def exactitud(evaluacion):
    logits, etiquetas = evaluacion
    return {"exactitud": float((np.argmax(logits, axis=-1) == etiquetas).mean())}

argumentos = TrainingArguments(
    output_dir="beto-resenas", num_train_epochs=2 if dispositivo == "cuda" else 1,
    per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=3e-5,
    eval_strategy="epoch", save_strategy="no", logging_steps=20, report_to="none",
    fp16=(dispositivo == "cuda"), use_cpu=(dispositivo == "cpu"))

entrenador = Trainer(model=beto_clasif, args=argumentos, train_dataset=entreno_tok, eval_dataset=prueba_tok,
                     data_collator=DataCollatorWithPadding(tokenizador_beto), compute_metrics=exactitud)
entrenador.train()
resultado = entrenador.evaluate()
print(f"Exactitud en {N_PRUEBA} reseñas de prueba: {resultado['eval_exactitud']:.1%}")
"""),
    ("code", """
revisar(resultado["eval_exactitud"] > (0.85 if dispositivo == "cuda" else 0.7),
        "¡Tu clasificador funciona!", "La exactitud es baja: ¿entrenaste con GPU?")
"""),
    ("md", """
Pruébalo con tus propias reseñas. ¿Entiende el español de Honduras?
"""),
    ("code", """
mi_clasificador = pipeline("text-classification", model=beto_clasif, tokenizer=tokenizador_beto, device=dispositivo)
for t in ["¡Qué buena onda este producto, maje! Lo recomiendo.",
          "Una gran decepción, se arruinó a los dos días.",
          "Llegó tarde pero funciona bien.",
          "No está mal, pero esperaba más por ese precio."]:
    r = mi_clasificador(t)[0]
    print(f"{r['label']:>9s} ({r['score']:.0%}) ← {t}")
"""),
    ("md", """
> **Para pensar:** ¿de dónde vienen estas reseñas? (Amazon, mayormente España y México). ¿Funcionará igual con regionalismos hondureños, con sarcasmo, con reseñas de otros temas? Para saberlo hay que **evaluar** con datos del contexto donde se usará el modelo.

## 6. Desde cero: un GPT diminuto que escribe como Cervantes

Ahora entrenamos un **transformer** completo desde cero, a nivel de **caracteres**, con el texto de *Don Quijote de la Mancha* (dominio público, Proyecto Gutenberg). Es la misma arquitectura de GPT, pero con ~0.8 millones de parámetros en vez de cientos de miles de millones.
"""),
    ("code", """
import urllib.request
crudo = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/2000/pg2000.txt", timeout=60).read().decode("utf-8")
texto = crudo[crudo.index("En un lugar de la Mancha"):crudo.index("*** END OF THE PROJECT GUTENBERG")].replace("\\r\\n", "\\n")
print(f"{len(texto):,} caracteres")
print(texto[:300])

vocab = sorted(set(texto))
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
codificar = lambda s: [stoi[c] for c in s]
decodificar = lambda ids: "".join(itos[int(i)] for i in ids)
datos = torch.tensor(codificar(texto), dtype=torch.long)
n = int(0.9 * len(datos))
datos_entreno, datos_valida = datos[:n], datos[n:]
print("Tamaño del vocabulario:", len(vocab), "caracteres distintos")
"""),
    ("md", """
El modelo completo: **atención causal** (cada carácter solo mira a los anteriores), **bloques** con flujo residual, y una cabeza que da probabilidades para el siguiente carácter. ¡Compáralo con el diagrama de las diapositivas!
"""),
    ("code", "import math\n\n" + CLASES_MINIGPT + """


CONTEXTO, LOTE = 128, 64
modelo = MiniGPT(len(vocab), d=128, n_capas=4, n_cabezas=4, contexto=CONTEXTO).to(dispositivo)
print(f"{sum(p.numel() for p in modelo.parameters()):,} parámetros")
"""),
    ("md", """
### Ejercicio 8 · Generar texto

Completa la función `generar`. En cada paso:

1. Calcula los logits del modelo con el contexto (los últimos `modelo.contexto` caracteres): `logits, _ = modelo(idx[:, -modelo.contexto:])`
2. Quédate con el **último** paso y divide por la temperatura: `logits[:, -1, :] / temperatura`
3. Convierte a probabilidades con `F.softmax(..., dim=-1)`
4. **Muestrea** un carácter con `torch.multinomial(probs, num_samples=1)`
5. Agrégalo: `idx = torch.cat([idx, siguiente], dim=1)`
"""),
    ("ej", """
@torch.no_grad()
def generar(modelo, idx, n_nuevos, temperatura=1.0):
    modelo.eval()
    for _ in range(n_nuevos):
        # TU CÓDIGO AQUÍ
        pass
    return idx
""", """
@torch.no_grad()
def generar(modelo, idx, n_nuevos, temperatura=1.0):
    modelo.eval()
    for _ in range(n_nuevos):
        logits, _ = modelo(idx[:, -modelo.contexto:])
        probs = F.softmax(logits[:, -1, :] / temperatura, dim=-1)
        siguiente = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, siguiente], dim=1)
    return idx
"""),
    ("code", """
inicio = torch.tensor([codificar("En un lugar de la Mancha")], device=dispositivo)
revisar(lambda: generar(modelo, inicio, 50).shape == (1, inicio.shape[1] + 50), "Tu función genera 50 caracteres nuevos.",
        "La salida debe tener 50 caracteres más que la entrada.")
print("ANTES de entrenar:\\n", decodificar(generar(modelo, inicio, 200)[0]))
"""),
    ("md", "Ahora a entrenar. Con GPU, 2000 pasos tardan ~1–2 minutos. Mira cómo baja la pérdida."),
    ("code", """
def lote(d):
    ix = torch.randint(len(d) - CONTEXTO - 1, (LOTE,))
    x = torch.stack([d[i:i + CONTEXTO] for i in ix])
    y = torch.stack([d[i + 1:i + CONTEXTO + 1] for i in ix])
    return x.to(dispositivo), y.to(dispositivo)

PASOS = 2000 if dispositivo == "cuda" else 150
opt = torch.optim.AdamW(modelo.parameters(), lr=2e-3)
historia = []
for paso in range(PASOS + 1):
    modelo.train()
    x, y = lote(datos_entreno)
    _, perdida = modelo(x, y)
    opt.zero_grad(); perdida.backward(); opt.step()
    if paso % 100 == 0:
        modelo.eval()
        with torch.no_grad():
            pv = modelo(*lote(datos_valida))[1].item()
        historia.append((paso, perdida.item(), pv))
        print(f"paso {paso:5d} · pérdida entrenamiento {perdida.item():.3f} · validación {pv:.3f}")

h = np.array(historia)
plt.plot(h[:, 0], h[:, 1], label="entrenamiento"); plt.plot(h[:, 0], h[:, 2], label="validación")
plt.xlabel("paso"); plt.ylabel("entropía cruzada"); plt.legend(); plt.show()
"""),
    ("code", """
for T in [0.5, 0.8, 1.2]:
    print(f"--- temperatura {T} ---")
    print(decodificar(generar(modelo, inicio, 300, temperatura=T)[0]), "\\n")
"""),
    ("md", """
### Reto

- Entrena más pasos o usa un modelo más grande (`d=256`, `n_capas=6`). ¿Mejora el texto? ¿Cuánto tarda?
- Entrena con **otro texto**: poemas de Rubén Darío, *Cien años de soledad* no (¡derechos de autor!), letras de canciones tradicionales, tus propios apuntes…
- Escribe un inicio distinto, por ejemplo `"Sancho"` o `"Dulcinea"`.

## 7. (Opcional) Comparte tu modelo con una demo de Gradio

Cambia `USAR_GRADIO = True` y ejecuta la celda. Colab te dará un **enlace público** temporal para que cualquiera pruebe tu clasificador. Así se construyen los *Spaces* de Hugging Face.
"""),
    ("code", """
USAR_GRADIO = False

if USAR_GRADIO:
    !pip install -q gradio
    import gradio as gr

    def clasificar(texto):
        return {r["label"]: r["score"] for r in mi_clasificador(texto, top_k=None)}

    gr.Interface(fn=clasificar, inputs=gr.Textbox(label="Escribe una reseña"), outputs=gr.Label(label="Sentimiento"),
                 title="Clasificador de reseñas — Escuela de ML Honduras 2026").launch(share=True)
"""),
    ("md", """
**Subir tu modelo a Hugging Face (opcional):** crea una cuenta gratuita en huggingface.co, genera un *token* con permiso de escritura y ejecuta:

```python
from huggingface_hub import notebook_login
notebook_login()                               # pega tu token cuando lo pida (¡nunca lo escribas en el código!)
entrenador.push_to_hub("beto-resenas-honduras")
```

Después, escribe una **tarjeta del modelo** explicando con qué datos se entrenó, para qué sirve y sus limitaciones.

---

## Resumen

- **PyTorch** = tensores + GPU + derivadas automáticas.
- `pipeline` de **Hugging Face** da acceso a miles de modelos en una línea: tokenizador → modelo → softmax.
- **Ajuste fino**: con un modelo preentrenado y pocos datos etiquetados se obtiene un buen clasificador en minutos.
- Un **GPT** es "solo" atención causal + bloques residuales + predecir el siguiente token… ¡y lo entrenaste desde cero!

**Para seguir:** Curso de Hugging Face (huggingface.co/learn) · Andrej Karpathy, *Neural Networks: Zero to Hero* · pytorch.org/tutorials
"""),
]

if __name__ == "__main__":
    construir("07_huggingface", celdas, gpu=True)
