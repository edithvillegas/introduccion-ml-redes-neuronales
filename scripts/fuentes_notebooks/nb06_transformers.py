"""Notebook 06 · Transformers y modelos de lenguaje."""

from utilidades import FUNCION_REVISAR, construir

celdas = [
    ("md", """
# 06 · Transformers y modelos de lenguaje

**Notebook de práctica** · Escuela de Machine Learning Honduras 2026 · Sesión del lunes 30 de noviembre

En este notebook vas a abrir modelos de lenguaje reales y mirar cómo funcionan por dentro:

1. **Tokens**: cómo se corta el texto (y por qué el español "cuesta" más)
2. **Vectores de palabras**: rey − hombre + mujer ≈ reina… y los sesgos que aprenden
3. **Palabra enmascarada** con BETO (un BERT en español)
4. **Siguiente token** con Qwen2.5: probabilidades, temperatura y generación
5. **Atención**: ¿a qué palabras mira cada palabra?
6. 🏆 Reto: encontrar el **círculo** de los días de la semana dentro de un modelo

**Recomendado:** *Entorno de ejecución → Cambiar tipo de entorno → GPU T4* (también funciona sin GPU, un poco más lento). Los modelos se descargan la primera vez (~1.5 GB en total).
"""),
    ("code", """
# gensim sirve para cargar vectores de palabras (GloVe); Colab no lo trae instalado
!pip install -q gensim
"""),
    ("code", f"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline

torch.manual_seed(0)
dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando:", dispositivo)

{FUNCION_REVISAR}

print("¡Todo listo!")
"""),
    ("md", """
## 1. Tokens

Un modelo de lenguaje no ve letras ni palabras: ve **tokens**, pedazos de texto que aparecen con frecuencia. Cada tokenizador se entrena con un corpus distinto. Comparemos tres:

- **GPT-2** (OpenAI, 2019): entrenado casi solo con texto en inglés
- **Qwen2.5** (Alibaba, 2024): multilingüe
- **BETO** (Universidad de Chile, 2020): entrenado con texto en español
"""),
    ("code", """
tokenizadores = {
    "GPT-2": AutoTokenizer.from_pretrained("gpt2"),
    "Qwen2.5": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
    "BETO": AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased"),
}
frase = "La inteligencia artificial está transformando la investigación científica en Honduras."
for nombre, tk in tokenizadores.items():
    ids = tk(frase, add_special_tokens=False)["input_ids"]
    piezas = [tk.decode([i]) for i in ids]
    print(f"{nombre:8s} ({len(ids):2d} tokens): {piezas}")
"""),
    ("md", """
### ✏️ Ejercicio 1 · El "impuesto" del español

Escribe la función `contar_tokens(tokenizador, texto)` que devuelva cuántos tokens produce un texto (**sin** tokens especiales). Luego compara la frase en español con su traducción al inglés para cada tokenizador.

<details><summary>💡 Pista</summary>

`len(tokenizador(texto, add_special_tokens=False)["input_ids"])`
</details>
"""),
    ("ej", """
def contar_tokens(tokenizador, texto):
    # TU CÓDIGO AQUÍ
    return 0

en_ingles = "Artificial intelligence is transforming scientific research in Honduras."
for nombre, tk in tokenizadores.items():
    print(f"{nombre:8s} español: {contar_tokens(tk, frase):2d} · inglés: {contar_tokens(tk, en_ingles):2d}")
""", """
def contar_tokens(tokenizador, texto):
    return len(tokenizador(texto, add_special_tokens=False)["input_ids"])

en_ingles = "Artificial intelligence is transforming scientific research in Honduras."
for nombre, tk in tokenizadores.items():
    print(f"{nombre:8s} español: {contar_tokens(tk, frase):2d} · inglés: {contar_tokens(tk, en_ingles):2d}")
"""),
    ("code", """
revisar(lambda: all(contar_tokens(tk, frase) == len(tk(frase, add_special_tokens=False)["input_ids"]) for tk in tokenizadores.values()),
        "¡Bien! ¿Qué tokenizador es más «justo» con el español?", "Revisa la pista.")
"""),
    ("md", """
**Para pensar:** ¿cómo se tokeniza "ferrocarril"? ¿Por qué a los modelos de lenguaje les cuesta contar cuántas letras "r" tiene una palabra?
"""),
    ("code", """
for nombre, tk in tokenizadores.items():
    print(nombre, [tk.decode([i]) for i in tk("ferrocarril", add_special_tokens=False)["input_ids"]])
"""),
    ("md", """
## 2. Vectores de palabras

Cargamos **GloVe** (Pennington *et al.*, 2014): 400 000 palabras en inglés, cada una representada por un vector de 100 números, aprendidos a partir de qué palabras aparecen juntas en Wikipedia y noticias. (Descarga de ~130 MB.)
"""),
    ("code", """
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")
print("Dimensión de cada vector:", glove["king"].shape)
print("Palabras más parecidas a 'coffee':", [w for w, _ in glove.most_similar("coffee", topn=8)])
print("Similitud coseno(cat, dog) =", round(float(glove.similarity("cat", "dog")), 3))
print("Similitud coseno(cat, physics) =", round(float(glove.similarity("cat", "physics")), 3))
"""),
    ("md", """
### ✏️ Ejercicio 2 · Analogías

"*a* es a *b* como *c* es a ___". La respuesta es la palabra cuyo vector está más cerca de $\\vec{b} - \\vec{a} + \\vec{c}$.

Completa `analogia(a, b, c)` para que devuelva **la palabra más cercana** usando `glove.most_similar(positive=[...], negative=[...], topn=1)`.

<details><summary>💡 Pista</summary>

`glove.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]`
</details>
"""),
    ("ej", """
def analogia(a, b, c):
    # TU CÓDIGO AQUÍ
    return None

print("man : king :: woman :", analogia("man", "king", "woman"))
print("france : paris :: honduras :", analogia("france", "paris", "honduras"))
""", """
def analogia(a, b, c):
    return glove.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]

print("man : king :: woman :", analogia("man", "king", "woman"))
print("france : paris :: honduras :", analogia("france", "paris", "honduras"))
"""),
    ("code", """
revisar(lambda: analogia("man", "king", "woman") == "queen" and analogia("walk", "walked", "swim") == "swam",
        "king − man + woman = queen 👑", "Revisa el orden de positive y negative.")
"""),
    ("md", """
### Los vectores también aprenden estereotipos

Prueben estas analogías. Nadie programó estos resultados: el modelo los aprendió de los textos que escribimos los humanos (Bolukbasi *et al.*, 2016; Caliskan *et al.*, 2017).
"""),
    ("code", """
for a, b, c in [("man", "doctor", "woman"), ("woman", "nurse", "man"), ("man", "programmer", "woman"),
                ("man", "engineer", "woman"), ("man", "boss", "woman")]:
    print(f"{a} : {b} :: {c} : {analogia(a, b, c)}")
"""),
    ("md", """
**✏️ Ejercicio 3 (libre):** inventa tus propias analogías (en inglés). ¿Encuentras alguna que funcione muy bien? ¿Alguna que muestre un sesgo?
"""),
    ("ej", """
# TU CÓDIGO AQUÍ
""", """
for a, b, c in [("spain", "spanish", "brazil"), ("small", "smaller", "big"), ("japan", "sushi", "mexico")]:
    print(f"{a} : {b} :: {c} : {analogia(a, b, c)}")
"""),
    ("md", """
## 3. Palabra enmascarada con BETO

BETO se entrenó **tapando** palabras y adivinándolas. El *pipeline* `fill-mask` nos da sus predicciones.
"""),
    ("code", """
rellenar = pipeline("fill-mask", model="dccuchile/bert-base-spanish-wwm-cased", device=dispositivo)
for r in rellenar("Tegucigalpa es la [MASK] de Honduras.", top_k=5):
    print(f"{r['token_str']:>12s}  {r['score']:.1%}")
"""),
    ("md", """
### ✏️ Ejercicio 4 · Pon a prueba a BETO

1. Escribe tres oraciones con `[MASK]` sobre Honduras, la física o tu vida diaria y mira las predicciones.
2. Compara `"Él es muy [MASK]."` con `"Ella es muy [MASK]."`. ¿Qué diferencias ves?

Guarda tus oraciones en la lista `mis_oraciones`.
"""),
    ("ej", """
mis_oraciones = [
    # TU CÓDIGO AQUÍ
]
for o in mis_oraciones:
    print(o, "→", [r["token_str"] for r in rellenar(o, top_k=5)])
""", """
mis_oraciones = [
    "El café de Honduras es muy [MASK].",
    "La [MASK] es la ciencia que estudia la materia y la energía.",
    "Él es muy [MASK].",
    "Ella es muy [MASK].",
]
for o in mis_oraciones:
    print(o, "→", [r["token_str"] for r in rellenar(o, top_k=5)])
"""),
    ("code", """
revisar(lambda: len(mis_oraciones) >= 3 and all("[MASK]" in o for o in mis_oraciones),
        "¡Bien! ¿BETO acertó lo que esperabas?", "Escribe al menos 3 oraciones con [MASK].")
"""),
    ("md", """
## 4. Predecir el siguiente token con Qwen2.5

Ahora un modelo **generativo** (decodificador) pequeño: Qwen2.5 con 500 millones de parámetros. Para cada posible token, el modelo produce un número llamado **logit**; la función **softmax** los convierte en probabilidades.
"""),
    ("code", """
tok_qwen = tokenizadores["Qwen2.5"]
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B").to(dispositivo).eval()
print(f"Parámetros: {sum(p.numel() for p in qwen.parameters()) / 1e6:.0f} millones")

def logits_siguiente(texto):
    ids = tok_qwen(texto, return_tensors="pt").to(dispositivo)
    with torch.no_grad():
        return qwen(**ids).logits[0, -1].float().cpu()

texto = "Había una vez un"
logits = logits_siguiente(texto)
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 10)
for p, i in zip(top.values, top.indices):
    print(f"{tok_qwen.decode([int(i)])!r:>14}  {p:.1%}")
"""),
    ("md", """
### ✏️ Ejercicio 5 · Softmax con temperatura

$$
p_i = \\frac{e^{z_i / T}}{\\sum_j e^{z_j / T}}
$$

Completa `softmax_con_temperatura(logits, T)`. Para que sea numéricamente estable, resta el máximo antes de exponenciar: $e^{(z_i - \\max z)/T}$.

<details><summary>💡 Pista</summary>

```python
z = (logits - logits.max()) / T
e = torch.exp(z)
return e / e.sum()
```
</details>
"""),
    ("ej", """
def softmax_con_temperatura(logits, T=1.0):
    # TU CÓDIGO AQUÍ
    return logits
""", """
def softmax_con_temperatura(logits, T=1.0):
    z = (logits - logits.max()) / T
    e = torch.exp(z)
    return e / e.sum()
"""),
    ("code", """
revisar(lambda: torch.allclose(softmax_con_temperatura(logits, 1.0), torch.softmax(logits, -1), atol=1e-5)
        and torch.allclose(softmax_con_temperatura(logits, 0.5), torch.softmax(logits / 0.5, -1), atol=1e-5),
        "¡Tu softmax funciona con cualquier temperatura!", "Compara con torch.softmax(logits / T, -1).")

fig, ejes = plt.subplots(1, 3, figsize=(15, 3.5), sharey=True)
for eje, T in zip(ejes, [0.3, 1.0, 2.0]):
    p = softmax_con_temperatura(logits, T)
    top = torch.topk(p, 10)
    eje.bar([tok_qwen.decode([int(i)]) for i in top.indices], top.values)
    eje.set_title(f"T = {T}"); eje.tick_params(axis="x", rotation=60)
plt.suptitle(f"«{texto} ___»"); plt.show()
"""),
    ("md", """
### ✏️ Ejercicio 6 · Genera texto token por token

Escribe un bucle que, partiendo de `"Honduras es un país"`, agregue **15 tokens**, eligiendo cada vez el **más probable** (decodificación voraz o *greedy*).

<details><summary>💡 Pista</summary>

```python
for _ in range(15):
    logits = logits_siguiente(texto)
    siguiente = int(logits.argmax())
    texto = texto + tok_qwen.decode([siguiente])
```
</details>
"""),
    ("ej", """
texto = "Honduras es un país"
# TU CÓDIGO AQUÍ

print(texto)
""", """
texto = "Honduras es un país"
for _ in range(15):
    logits = logits_siguiente(texto)
    siguiente = int(logits.argmax())
    texto = texto + tok_qwen.decode([siguiente])

print(texto)
"""),
    ("code", """
revisar(lambda: texto.startswith("Honduras es un país") and contar_tokens(tok_qwen, texto) >= contar_tokens(tok_qwen, "Honduras es un país") + 12,
        "¡Generaste texto como un modelo de lenguaje!", "¿Agregaste 15 tokens al texto?")
"""),
    ("md", """
Ahora con **muestreo** (eligiendo al azar según las probabilidades) a distintas temperaturas. Ejecuta la celda varias veces: ¿qué cambia?
"""),
    ("code", """
for T in [0.2, 0.8, 1.5]:
    ids = tok_qwen("Había una vez un", return_tensors="pt").to(dispositivo)
    salida = qwen.generate(**ids, max_new_tokens=30, do_sample=True, temperature=T, top_k=0,
                           pad_token_id=tok_qwen.eos_token_id)
    print(f"T = {T}:", tok_qwen.decode(salida[0], skip_special_tokens=True), "\\n")
"""),
    ("md", """
## 5. Atención

Pedimos a BETO que nos devuelva sus **pesos de atención**: para cada capa (12) y cada cabeza (12), una matriz que dice cuánto "mira" cada token a cada otro token.
"""),
    ("code", """
tok_beto = tokenizadores["BETO"]
beto = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", attn_implementation="eager").to(dispositivo).eval()
oracion = "El gato no cruzó la calle porque estaba muy cansado."
entrada = tok_beto(oracion, return_tensors="pt").to(dispositivo)
with torch.no_grad():
    atencion = torch.stack(beto(**entrada, output_attentions=True).attentions)[:, 0].cpu()   # capas × cabezas × n × n
tokens = tok_beto.convert_ids_to_tokens(entrada["input_ids"][0])
print("Forma:", tuple(atencion.shape), "\\nTokens:", tokens)

def dibujar_atencion(capa, cabeza):
    plt.figure(figsize=(6, 5))
    plt.imshow(atencion[capa, cabeza], cmap="Blues")
    plt.xticks(range(len(tokens)), tokens, rotation=70); plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Capa {capa + 1}, cabeza {cabeza + 1}"); plt.colorbar(); plt.show()

dibujar_atencion(0, 0)
"""),
    ("md", """
### ✏️ Ejercicio 7 · ¿Quién estaba cansado?

Encuentra la **capa** y la **cabeza** (índices desde 0) donde el token `"cansado"` presta **más atención** al token `"gato"`. Guarda los índices en `mejor_capa` y `mejor_cabeza` y dibuja esa cabeza.

<details><summary>💡 Pista</summary>

```python
i, j = tokens.index("cansado"), tokens.index("gato")
puntaje = atencion[:, :, i, j]          # matriz capas × cabezas
mejor_capa, mejor_cabeza = np.unravel_index(int(puntaje.argmax()), puntaje.shape)
```
</details>
"""),
    ("ej", """
mejor_capa, mejor_cabeza = 0, 0   # TU CÓDIGO AQUÍ

dibujar_atencion(mejor_capa, mejor_cabeza)
""", """
i, j = tokens.index("cansado"), tokens.index("gato")
puntaje = atencion[:, :, i, j]
mejor_capa, mejor_cabeza = np.unravel_index(int(puntaje.argmax()), puntaje.shape)
print("Capa", mejor_capa + 1, "cabeza", mejor_cabeza + 1, "→ peso", float(puntaje.max()))
dibujar_atencion(mejor_capa, mejor_cabeza)
"""),
    ("code", """
_p = atencion[:, :, tokens.index("cansado"), tokens.index("gato")]
revisar(lambda: (int(mejor_capa), int(mejor_cabeza)) == tuple(int(v) for v in np.unravel_index(int(_p.argmax()), _p.shape)),
        "¡Encontraste la cabeza que conecta «cansado» con «gato»!", "Busca el máximo de atencion[:, :, i, j].")
"""),
    ("md", """
## 6. 🏆 Reto: los días de la semana forman un círculo

Engels *et al.* (ICLR 2025) descubrieron que los modelos de lenguaje representan los días de la semana y los meses en **círculos**. ¡Vamos a comprobarlo con Qwen2.5!

Idea: para cada día, ponemos la palabra en muchas frases distintas, tomamos el **vector interno** del modelo (el *flujo residual*) en esa palabra, y promediamos. Luego proyectamos los 7 promedios en 2D con PCA.
"""),
    ("code", """
DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
PLANTILLAS = ["Nos vemos el {}.", "El examen es el {} por la mañana.", "Nací un {}.", "¡Hasta el próximo {}!",
              "Cada {} vamos al mercado.", "Pasó el {} pasado.", "La tienda cierra el {}.", "El {} voy a viajar.",
              "Mi día favorito es el {}.", "La fiesta es este {} en la noche.", "Llegamos el {} por la tarde.",
              "La tarea es para el {}."]

def vector_palabra(plantilla, palabra, capa):
    \"\"\"Estado oculto de la capa `capa` en el último token de `palabra` dentro de la frase.\"\"\"
    prefijo = plantilla.split("{}")[0]
    posicion = len(tok_qwen(prefijo + palabra)["input_ids"]) - 1
    ids = tok_qwen(plantilla.format(palabra), return_tensors="pt").to(dispositivo)
    with torch.no_grad():
        estados = qwen(**ids, output_hidden_states=True).hidden_states
    return estados[capa][0, posicion].float().cpu().numpy()

print("Número de capas (+ embeddings):", qwen.config.num_hidden_layers + 1)
"""),
    ("md", """
### ✏️ Ejercicio 8 · Encuentra el círculo

1. Para cada día, calcula el **promedio** de `vector_palabra(p, dia, CAPA)` sobre todas las `PLANTILLAS`. Guarda el resultado en una matriz `medias` de forma (7, d).
2. Proyecta `medias` a 2 dimensiones con `PCA(2)` de scikit-learn → `proyeccion` (forma 7 × 2).
3. Dibuja los puntos con el nombre de cada día. ¿Forman un círculo en orden?

<details><summary>💡 Pista</summary>

```python
medias = np.array([np.mean([vector_palabra(p, d, CAPA) for p in PLANTILLAS], axis=0) for d in DIAS])
proyeccion = PCA(2).fit_transform(medias - medias.mean(0))
```
</details>
"""),
    ("ej", """
from sklearn.decomposition import PCA
CAPA = 13

medias = None        # TU CÓDIGO AQUÍ
proyeccion = None    # TU CÓDIGO AQUÍ

if proyeccion is not None:
    plt.figure(figsize=(5, 5))
    plt.plot(*np.vstack([proyeccion, proyeccion[:1]]).T, ":", color="gray")
    plt.scatter(proyeccion[:, 0], proyeccion[:, 1], s=80)
    for (x, y), d in zip(proyeccion, DIAS):
        plt.annotate(d, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.axis("equal"); plt.title(f"Días de la semana en la capa {CAPA}"); plt.show()
""", """
from sklearn.decomposition import PCA
CAPA = 13

medias = np.array([np.mean([vector_palabra(p, d, CAPA) for p in PLANTILLAS], axis=0) for d in DIAS])
proyeccion = PCA(2).fit_transform(medias - medias.mean(0))

if proyeccion is not None:
    plt.figure(figsize=(5, 5))
    plt.plot(*np.vstack([proyeccion, proyeccion[:1]]).T, ":", color="gray")
    plt.scatter(proyeccion[:, 0], proyeccion[:, 1], s=80)
    for (x, y), d in zip(proyeccion, DIAS):
        plt.annotate(d, (x, y), textcoords="offset points", xytext=(5, 5))
    plt.axis("equal"); plt.title(f"Días de la semana en la capa {CAPA}"); plt.show()
"""),
    ("code", """
def es_ciclo(puntos):
    \"\"\"¿Los puntos, ordenados por ángulo, aparecen en el orden de la semana (o al revés)?\"\"\"
    angulos = np.arctan2(puntos[:, 1], puntos[:, 0])
    orden = np.argsort(angulos)
    pos = np.empty(len(orden), int); pos[orden] = np.arange(len(orden))
    n = len(orden)
    return all((pos[(k + 1) % n] - pos[k]) % n in (1, n - 1) for k in range(n))

revisar(lambda: proyeccion.shape == (7, 2) and es_ciclo(proyeccion),
        "¡Los días forman un ciclo en orden! El modelo aprendió que después del domingo viene el lunes. 🔄",
        "Revisa el promedio y la PCA (o prueba otra capa).")
"""),
    ("md", """
**Para explorar:**

- ¿En qué capas aparece el círculo? Prueba con `CAPA = 1, 5, 20`.
- Haz lo mismo con los **meses del año** (necesitarás otras plantillas, p. ej. `"Nací en {}."`).
- ¿Pasa lo mismo con los números del 1 al 12? ¿Con los colores?

---

## Resumen

- Los modelos ven **tokens**; los tokenizadores entrenados en inglés "cobran más" al español.
- Las palabras se representan como **vectores**: las relaciones son direcciones (rey − hombre + mujer ≈ reina), y también aparecen **sesgos**.
- BETO aprende adivinando palabras **tapadas**; Qwen, prediciendo la **siguiente**. La **temperatura** controla qué tan aleatoria es la generación.
- La **atención** conecta palabras entre sí (como "cansado" con "gato").
- Dentro de los modelos hay **geometría** interpretable: ¡los días de la semana forman un círculo!

**Para seguir:** *The Illustrated Transformer* (Jay Alammar) · Transformer Explainer (poloclub.github.io/transformer-explainer) · transformer-circuits.pub
"""),
]

if __name__ == "__main__":
    construir("06_transformers_lenguaje", celdas, gpu=True)
