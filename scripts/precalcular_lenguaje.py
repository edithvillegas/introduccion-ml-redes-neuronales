"""
Precalcula los datos de las gráficas sobre modelos de lenguaje que aparecen en
las diapositivas (para que GitHub Actions no tenga que descargar modelos).

Genera archivos JSON en diapositivas/datos/:
  - circulos_dias_meses.json   días y meses en círculo (Qwen2.5-0.5B, español; GPT-2, inglés)
  - analogias_glove.json       vectores de palabras (GloVe) proyectados con PCA
  - mlm_beto.json              predicciones de [MASK] con BETO (BERT en español)
  - siguiente_token.json       probabilidades del siguiente token (Qwen2.5-0.5B)
  - tokenizacion.json          cómo distintos tokenizadores parten una frase
  - atencion_beto.json         pesos de atención de BETO para una frase

Uso (necesita torch, transformers, gensim, scikit-learn):
    python scripts/precalcular_lenguaje.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM,
                          AutoTokenizer)

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "diapositivas" / "datos"
SALIDA.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


def guardar(nombre, datos):
    with open(SALIDA / nombre, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"))
    print("✓", nombre)


def r(x, n=4):
    """Redondea arreglos para que los JSON sean pequeños."""
    return np.round(np.asarray(x, dtype=float), n).tolist()


# ---------------------------------------------------------------------------
# 1. Días de la semana y meses del año en círculo
#    (idea: Engels et al. 2024, "Not All Language Model Features Are Linear")
# ---------------------------------------------------------------------------
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
            "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
PLANTILLAS_DIAS_ES = [
    "Nos vemos el {}.", "El examen es el {} por la mañana.", "Nací un {}.",
    "¡Hasta el próximo {}!", "Cada {} vamos al mercado.", "Pasó el {} pasado.",
    "La tienda cierra el {}.", "El {} voy a viajar.", "Mi día favorito es el {}.",
    "La fiesta es este {} en la noche.", "Llegamos el {} por la tarde.",
    "La tarea es para el {}.", "El partido de fútbol es el {}.",
    "Mi abuela siempre cocina el {}.", "No hay clases el {}.", "El {} hace calor.",
]
PLANTILLAS_MESES_ES = [
    "Nací en {}.", "El festival es en {}.", "Llovió mucho en {}.",
    "¡Nos vemos en {}!", "El próximo {} viajamos.", "El curso empieza en {}.",
    "Mi cumpleaños es en {}.", "El {} pasado hizo frío.",
    "En {} empieza la cosecha.", "El informe de {} está listo.",
    "Cada {} visitamos a la familia.", "Las clases terminan en {}.",
    "La feria del pueblo es en {}.", "Pagamos la renta en {}.",
    "El huracán llegó en {}.", "Me gradúo en {}.",
]
DIAS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MESES_EN = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]
PLANTILLAS_DIAS_EN = [
    "Let's meet on {}.", "The exam is on {} morning.", "I was born on a {}.",
    "See you next {}!", "Every {} we go to the market.", "It happened last {}.",
    "The store is closed on {}.", "On {} I will travel.", "My favorite day is {}.",
    "The party is this {} night.", "We arrived on {} afternoon.", "Report due {}.",
]
PLANTILLAS_MESES_EN = [
    "I was born in {}.", "The festival is in {}.", "It rained a lot in {}.",
    "See you in {}!", "Next {} we travel.", "The course starts in {}.",
    "My birthday is in {}.", "Last {} was cold.", "In {} the harvest begins.",
    "The report for {} is ready.", "Every {} we visit family.", "Classes end in {}.",
]


def estados_ocultos(tok, modelo, palabras, plantillas, capa):
    """Vector de la capa `capa` en el último token de cada palabra, en cada contexto."""
    salida = []
    for p in palabras:
        vecs = []
        for t in plantillas:
            prefijo = t.split("{}")[0]
            n = len(tok(prefijo + p)["input_ids"])  # posición del último token de la palabra
            ids = tok(t.format(p), return_tensors="pt")
            with torch.no_grad():
                hs = modelo(**ids, output_hidden_states=True).hidden_states
            vecs.append(hs[capa][0, n - 1].float().numpy())
        salida.append(np.array(vecs))
    return np.array(salida)  # palabras × contextos × d


def circulo(tok, modelo, palabras, plantillas, capa):
    H = estados_ocultos(tok, modelo, palabras, plantillas, capa)
    medias = H.mean(axis=1)
    pca = PCA(2).fit(medias - medias.mean(0))
    centro = medias.mean(0)
    medias_2d = pca.transform(medias - centro)
    puntos_2d = pca.transform(H.reshape(-1, H.shape[-1]) - centro).reshape(H.shape[0], H.shape[1], 2)
    return {"palabras": palabras, "capa": capa, "medias": r(medias_2d),
            "puntos": r(puntos_2d), "varianza": r(pca.explained_variance_ratio_, 3)}


print("Cargando Qwen2.5-0.5B…")
tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
lm_qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
print("Cargando GPT-2…")
tok_gpt2 = AutoTokenizer.from_pretrained("gpt2")
m_gpt2 = AutoModel.from_pretrained("gpt2").eval()

guardar("circulos_dias_meses.json", {
    "modelo_es": "Qwen/Qwen2.5-0.5B",
    "dias_es": circulo(tok_qwen, lm_qwen, DIAS_ES, PLANTILLAS_DIAS_ES, 13),
    "meses_es": circulo(tok_qwen, lm_qwen, MESES_ES, PLANTILLAS_MESES_ES, 21),
    "modelo_en": "gpt2",
    "dias_en": circulo(tok_gpt2, m_gpt2, DIAS_EN, PLANTILLAS_DIAS_EN, 7),
    "meses_en": circulo(tok_gpt2, m_gpt2, MESES_EN, PLANTILLAS_MESES_EN, 7),
})

# ---------------------------------------------------------------------------
# 2. Siguiente token (Qwen2.5-0.5B) — para el control deslizante de temperatura
# ---------------------------------------------------------------------------
FRASES_SIGUIENTE = [
    "La capital de Honduras es la ciudad de",
    "Había una vez un",
    "El gato se subió al",
    "Mi comida favorita son las",
    "La inteligencia artificial es una",
]
siguiente = []
for frase in FRASES_SIGUIENTE:
    ids = tok_qwen(frase, return_tensors="pt")
    with torch.no_grad():
        logits = lm_qwen(**ids).logits[0, -1].float()
    top = torch.topk(logits, 15)
    siguiente.append({
        "frase": frase,
        "tokens": [tok_qwen.decode([int(i)]) for i in top.indices],
        "logits": r(top.values.numpy(), 3),
    })
guardar("siguiente_token.json", {"modelo": "Qwen/Qwen2.5-0.5B", "ejemplos": siguiente})

# Generación paso a paso (decodificación voraz) para mostrar el "autocompletado"
frase = "Honduras es un país"
pasos = []
ids = tok_qwen(frase, return_tensors="pt")["input_ids"]
for _ in range(10):
    with torch.no_grad():
        logits = lm_qwen(input_ids=ids).logits[0, -1].float()
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, 5)
    pasos.append({"contexto": tok_qwen.decode(ids[0]),
                  "candidatos": [tok_qwen.decode([int(i)]) for i in top.indices],
                  "probs": r(top.values.numpy(), 4)})
    ids = torch.cat([ids, top.indices[:1].view(1, 1)], dim=1)
guardar("generacion_paso_a_paso.json", {"modelo": "Qwen/Qwen2.5-0.5B", "pasos": pasos,
                                         "final": tok_qwen.decode(ids[0])})

# ---------------------------------------------------------------------------
# 3. Tokenización
# ---------------------------------------------------------------------------
tok_beto = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
FRASES_TOK = {
    "es": "La inteligencia artificial está transformando la investigación científica en Honduras.",
    "en": "Artificial intelligence is transforming scientific research in Honduras.",
}
tokenizacion = {}
for nombre, tk in [("GPT-2 (OpenAI, 2019; entrenado casi solo en inglés)", tok_gpt2),
                   ("Qwen2.5 (Alibaba, 2024; multilingüe)", tok_qwen),
                   ("BETO (U. de Chile, 2020; entrenado en español)", tok_beto)]:
    tokenizacion[nombre] = {}
    for idioma, frase in FRASES_TOK.items():
        ids = tk(frase, add_special_tokens=False)["input_ids"]
        piezas = [tk.decode([i]) for i in ids]
        tokenizacion[nombre][idioma] = {"piezas": piezas, "ids": [int(i) for i in ids]}
guardar("tokenizacion.json", {"frases": FRASES_TOK, "tokenizadores": tokenizacion})

# ---------------------------------------------------------------------------
# 4. Modelado de lenguaje enmascarado con BETO
# ---------------------------------------------------------------------------
print("Cargando BETO…")
mlm = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-cased").eval()
FRASES_MLM = [
    "Tegucigalpa es la [MASK] de Honduras.",
    "El perro [MASK] toda la noche.",
    "Voy a la [MASK] a estudiar física.",
    "Él trabaja como [MASK] en el hospital.",
    "Ella trabaja como [MASK] en el hospital.",
]
mlm_res = []
for frase in FRASES_MLM:
    ids = tok_beto(frase, return_tensors="pt")
    pos = (ids["input_ids"][0] == tok_beto.mask_token_id).nonzero()[0, 0]
    with torch.no_grad():
        logits = mlm(**ids).logits[0, pos]
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, 8)
    mlm_res.append({"frase": frase,
                    "tokens": [tok_beto.decode([int(i)]).strip() for i in top.indices],
                    "probs": r(top.values.numpy(), 4)})
    print("  ", frase, mlm_res[-1]["tokens"][:5])
guardar("mlm_beto.json", {"modelo": "dccuchile/bert-base-spanish-wwm-cased", "ejemplos": mlm_res})

# ---------------------------------------------------------------------------
# 5. Atención en BETO
# ---------------------------------------------------------------------------
enc = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased", attn_implementation="eager").eval()
frase_att = "El gato no cruzó la calle porque estaba muy cansado."
ids = tok_beto(frase_att, return_tensors="pt")
with torch.no_grad():
    att = enc(**ids, output_attentions=True).attentions  # capas × (1, cabezas, n, n)
att = torch.stack([a[0] for a in att]).numpy()  # capas × cabezas × n × n
tokens = tok_beto.convert_ids_to_tokens(ids["input_ids"][0])
i_cans = tokens.index("cansado")
i_gato = tokens.index("gato")
i_calle = tokens.index("calle")
# Cabeza donde "cansado" mira más a "gato" que a "calle"
puntaje = att[:, :, i_cans, i_gato] - att[:, :, i_cans, i_calle]
capa_c, cabeza_c = np.unravel_index(np.argmax(puntaje), puntaje.shape)
# Cabeza que mira al token anterior
n = len(tokens)
prev = np.array([[np.mean([att[l, h, i, i - 1] for i in range(1, n)]) for h in range(att.shape[1])]
                 for l in range(att.shape[0])])
capa_p, cabeza_p = np.unravel_index(np.argmax(prev), prev.shape)
# Cabeza que mira al token siguiente
sig = np.array([[np.mean([att[l, h, i, i + 1] for i in range(0, n - 1)]) for h in range(att.shape[1])]
                for l in range(att.shape[0])])
capa_s, cabeza_s = np.unravel_index(np.argmax(sig), sig.shape)
seleccion = [
    {"nombre": f"Capa {capa_c + 1}, cabeza {cabeza_c + 1}: «cansado» → «gato»",
     "pesos": r(att[capa_c, cabeza_c], 3)},
    {"nombre": f"Capa {capa_p + 1}, cabeza {cabeza_p + 1}: mira la palabra anterior",
     "pesos": r(att[capa_p, cabeza_p], 3)},
    {"nombre": f"Capa {capa_s + 1}, cabeza {cabeza_s + 1}: mira la palabra siguiente",
     "pesos": r(att[capa_s, cabeza_s], 3)},
    {"nombre": "Promedio de todas las capas y cabezas", "pesos": r(att.mean(axis=(0, 1)), 3)},
]
print("  cabeza 'cansado→gato':", capa_c + 1, cabeza_c + 1, att[capa_c, cabeza_c, i_cans, i_gato])
guardar("atencion_beto.json", {"modelo": "dccuchile/bert-base-spanish-wwm-cased",
                                "frase": frase_att, "tokens": tokens, "cabezas": seleccion})

# ---------------------------------------------------------------------------
# 6. Vectores de palabras (GloVe) — rey/reina, hombre/mujer, países/capitales…
# ---------------------------------------------------------------------------
import gensim.downloader as api  # noqa: E402

print("Cargando GloVe (glove-wiki-gigaword-100)…")
kv = api.load("glove-wiki-gigaword-100")
TRADUCCION = {
    "king": "rey", "queen": "reina", "man": "hombre", "woman": "mujer",
    "prince": "príncipe", "princess": "princesa", "uncle": "tío", "aunt": "tía",
    "brother": "hermano", "sister": "hermana", "actor": "actor", "actress": "actriz",
    "father": "padre", "mother": "madre", "boy": "niño", "girl": "niña",
    "spain": "España", "madrid": "Madrid", "france": "Francia", "paris": "París",
    "italy": "Italia", "rome": "Roma", "japan": "Japón", "tokyo": "Tokio",
    "honduras": "Honduras", "tegucigalpa": "Tegucigalpa", "mexico": "México",
    "germany": "Alemania", "berlin": "Berlín", "peru": "Perú", "lima": "Lima",
    "walk": "caminar", "walked": "caminó", "swim": "nadar", "swam": "nadó",
    "run": "correr", "ran": "corrió", "fly": "volar", "flew": "voló",
}
GRUPOS = {
    "género": [("king", "queen"), ("man", "woman"), ("prince", "princess"), ("uncle", "aunt"),
               ("brother", "sister"), ("actor", "actress"), ("father", "mother"), ("boy", "girl")],
    "país–capital": [("spain", "madrid"), ("france", "paris"), ("italy", "rome"), ("japan", "tokyo"),
                     ("germany", "berlin"), ("peru", "lima"), ("honduras", "tegucigalpa")],
    "verbo–pasado": [("walk", "walked"), ("swim", "swam"), ("run", "ran"), ("fly", "flew")],
}
analogias = {}
for grupo, pares in GRUPOS.items():
    palabras = [w for par in pares for w in par]
    X = np.array([kv[w] for w in palabras])
    pca = PCA(3).fit(X - X.mean(0))
    Y = pca.transform(X - X.mean(0))
    analogias[grupo] = {"palabras": palabras, "es": [TRADUCCION[w] for w in palabras],
                        "pares": [[a, b] for a, b in pares], "xyz": r(Y, 3)}

consultas = [
    (["king", "woman"], ["man"]), (["paris", "italy"], ["france"]),
    (["tokyo", "spain"], ["japan"]), (["walked", "swim"], ["walk"]),
    (["doctor", "woman"], ["man"]), (["programmer", "woman"], ["man"]),
    (["nurse", "man"], ["woman"]),
]
resultados = []
for pos, neg in consultas:
    sim = kv.most_similar(positive=pos, negative=neg, topn=5)
    resultados.append({"positivo": pos, "negativo": neg,
                       "resultado": [[w, round(float(s), 3)] for w, s in sim]})
    print("  ", pos, "-", neg, "→", [w for w, _ in sim])

# Nube de palabras por categorías (para una gráfica 3D)
CATEGORIAS = {
    "animales": ["dog", "cat", "horse", "cow", "lion", "tiger", "eagle", "fish", "snake", "monkey"],
    "frutas": ["apple", "banana", "mango", "orange", "pineapple", "grape", "lemon", "coconut", "papaya", "strawberry"],
    "países": ["honduras", "guatemala", "mexico", "brazil", "spain", "france", "japan", "india", "nigeria", "canada"],
    "profesiones": ["doctor", "teacher", "engineer", "lawyer", "nurse", "farmer", "scientist", "programmer", "physicist", "chef"],
    "números": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    "colores": ["red", "blue", "green", "yellow", "black", "white", "purple", "pink", "brown", "gray"],
}
ES_CAT = {
    "dog": "perro", "cat": "gato", "horse": "caballo", "cow": "vaca", "lion": "león", "tiger": "tigre",
    "eagle": "águila", "fish": "pez", "snake": "serpiente", "monkey": "mono", "apple": "manzana",
    "banana": "banano", "mango": "mango", "orange": "naranja", "pineapple": "piña", "grape": "uva",
    "lemon": "limón", "coconut": "coco", "papaya": "papaya", "strawberry": "fresa",
    "honduras": "Honduras", "guatemala": "Guatemala", "mexico": "México", "brazil": "Brasil",
    "spain": "España", "france": "Francia", "japan": "Japón", "india": "India", "nigeria": "Nigeria",
    "canada": "Canadá", "doctor": "médico/a", "teacher": "docente", "engineer": "ingeniero/a",
    "lawyer": "abogado/a", "nurse": "enfermero/a", "farmer": "agricultor/a", "scientist": "científico/a",
    "programmer": "programador/a", "physicist": "físico/a", "chef": "cocinero/a", "one": "uno",
    "two": "dos", "three": "tres", "four": "cuatro", "five": "cinco", "six": "seis", "seven": "siete",
    "eight": "ocho", "nine": "nueve", "ten": "diez", "red": "rojo", "blue": "azul", "green": "verde",
    "yellow": "amarillo", "black": "negro", "white": "blanco", "purple": "morado", "pink": "rosado",
    "brown": "café", "gray": "gris",
}
todas = [w for ws in CATEGORIAS.values() for w in ws]
X = np.array([kv[w] for w in todas])
Y = PCA(3).fit_transform(X - X.mean(0))
nube = {"palabras": todas, "es": [ES_CAT[w] for w in todas],
        "categoria": [c for c, ws in CATEGORIAS.items() for _ in ws], "xyz": r(Y, 3)}

guardar("analogias_glove.json", {"modelo": "GloVe 6B, 100 dimensiones (Pennington et al., 2014)",
                                  "grupos": analogias, "consultas": resultados, "nube": nube})
print("Listo.")
