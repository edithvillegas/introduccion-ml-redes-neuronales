# Inteligencia artificial y redes neuronales · Escuela de Machine Learning Honduras 2026

**ICTP – Physics Without Frontiers · Machine Learning: Theory and Applications**
23 de noviembre al 4 de diciembre de 2026 · UNAH, Tegucigalpa, Honduras · [machine-learning-honduras.org](https://machine-learning-honduras.org/)

Materiales de las sesiones de **Edith Natalia Villegas García** (ICTP): introducción a la inteligencia artificial y al aprendizaje automático, Linux y la terminal, teoría de redes neuronales, redes convolucionales y autocodificadores, transformers y modelos de lenguaje, y una práctica con PyTorch y Hugging Face.

- **Diapositivas interactivas** hechas con [Quarto](https://quarto.org) + [reveal.js](https://revealjs.com), con gráficas en **Python** ([Plotly](https://plotly.com/python/)) que se pueden explorar en vivo: deslizadores, botones, gráficas 3D y demos incrustadas.
- **Notebooks de práctica** para Google Colab, con ejercicios que se revisan solos (✅ / ❌) y sus **soluciones**.

**Diapositivas en línea:** https://edithvillegas.github.io/introduccion-ml-redes-neuronales/

## Contenido

| # | Sesión | Diapositivas | Notebook | Soluciones |
|---|---|---|---|---|
| 1 | Introducción a la IA y al aprendizaje automático (parte 1) | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/01_introduccion_ia.html) | — | — |
| 2 | Introducción a la IA y al aprendizaje automático (parte 2) | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/02_fundamentos_ml.html) | — | — |
| 3 | Linux, la terminal y git · compilado vs. interpretado | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/03_linux_terminal.html) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/notebooks/03_terminal_linux.ipynb) | [Colab](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/soluciones/03_terminal_linux_soluciones.ipynb) |
| 4 | Teoría de redes neuronales | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/04_redes_neuronales.html) | — | — |
| 5 | Redes convolucionales y autocodificadores | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/05_redes_convolucionales.html) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/notebooks/05_redes_convolucionales.ipynb) | [Colab](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/soluciones/05_redes_convolucionales_soluciones.ipynb) |
| 6 | Transformers y modelos de lenguaje | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/06_transformers_lenguaje.html) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/notebooks/06_transformers_lenguaje.ipynb) | [Colab](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/soluciones/06_transformers_lenguaje_soluciones.ipynb) |
| 7 | Práctica: Python, PyTorch y Hugging Face | [ver](https://edithvillegas.github.io/introduccion-ml-redes-neuronales/diapositivas/07_practica_huggingface.html) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/notebooks/07_huggingface.ipynb) | [Colab](https://colab.research.google.com/github/edithvillegas/introduccion-ml-redes-neuronales/blob/main/soluciones/07_huggingface_soluciones.ipynb) |

### Qué se ve en cada sesión

- **IA y aprendizaje automático (1):** bienvenida y programa de la escuela; definiciones de IA, aprendizaje automático, aprendizaje profundo y ciencia de datos; historia (de McCulloch y Pitts a los Nobel de 2024); por qué ahora (datos, cómputo, algoritmos); la IA en la vida diaria (asistentes, búsqueda, imágenes y video generados, con demostraciones en vivo); la IA en la ciencia (AlphaFold, diseño de proteínas, cáncer de mama, huracanes, física, materiales, matemáticas, artículos escritos por IA).
- **IA y aprendizaje automático (2):** aprendizaje supervisado, no supervisado (agrupamiento, reducción de dimensionalidad), autosupervisado y semisupervisado; clasificación vs. regresión; sobreajuste y generalización; preprocesamiento, escalado, fuga de datos e ingeniería de características; adelanto de redes neuronales, convolucionales y transformers; ética: sesgos por nombre en contratación y créditos, sesgos en medicina y genómica, paridad de género en los datos, algoritmos de mitigación, leyes (Ley de IA de la UE, UNESCO), medio ambiente, trabajo, artistas, vigilancia y deepfakes; interpretabilidad.
- **Linux:** sistema operativo, sistema de archivos, comandos esenciales, tuberías, permisos, git, y C vs. Python vs. NumPy.
- **Redes neuronales:** neurona artificial, funciones de activación, perceptrón y XOR, perceptrón multicapa, aproximación universal, funciones de pérdida, descenso de gradiente, retropropagación, sobreajuste y regularización, ImageNet, visualización de características, redes de Hopfield.
- **Redes convolucionales:** convolución paso a paso, filtros, *pooling*, arquitecturas (LeNet → ResNet → ViT), filtros y mapas aprendidos, interpretabilidad (Distill), ejemplos adversarios, autocodificadores y espacio latente.
- **Transformers:** modelos de lenguaje, tokens y el "impuesto" del español, vectores de palabras (rey − hombre + mujer ≈ reina; días de la semana y meses en círculo), BERT/BETO y GPT, atención, la arquitectura transformer, leyes de escala, de modelo a asistente, interpretabilidad (superposición, autocodificadores dispersos, circuitos, emociones y adulación), RNN/LSTM y alternativas (Mamba, RWKV, xLSTM, híbridos, difusión).
- **Hugging Face:** Python en 5 minutos, PyTorch y *autograd*, `pipeline`, qué hay dentro de un *pipeline*, ajuste fino de BETO con reseñas en español, un mini-GPT entrenado desde cero con el Quijote, demos con Gradio.

## Para estudiantes

1. Hagan clic en **Abrir en Colab** en la sesión correspondiente.
2. En Colab: **Archivo → Guardar una copia en Drive** para guardar sus cambios.
3. Para las sesiones 5–7: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
4. Ejecuten las celdas en orden con `Shift + Enter`. En cada **Ejercicio** escriban su código donde dice `# TU CÓDIGO AQUÍ`; la celda siguiente revisa la respuesta (✅ / ❌). Si se atoran, abran la **Pista**.

## Estructura del repositorio

```
├── index.qmd                  # página de inicio del sitio (lista de sesiones)
├── _quarto.yml                # configuración del sitio web
├── diapositivas/
│   ├── 01_introduccion_ia.qmd … 07_practica_huggingface.qmd
│   ├── _metadata.yml          # opciones comunes de reveal.js
│   ├── graficas.py            # estilo y utilidades de Plotly compartidas
│   ├── programa.py            # programa de la escuela (gráfica del calendario)
│   ├── datos/                 # datos precalculados para las gráficas (JSON)
│   └── img/                   # imágenes y videos (ver img/CREDITOS.md)
├── notebooks/                 # notebooks para estudiantes (Colab)
├── soluciones/                # notebooks resueltos
├── scripts/
│   ├── fuentes_notebooks/     # código Python que genera los notebooks (versión estudiante + soluciones)
│   ├── precalcular_*.py       # regeneran diapositivas/datos/ (necesitan torch, transformers…)
│   ├── minigpt.py             # el mini-GPT del notebook 07
│   └── configurar_repo.py     # pone su usuario/repositorio de GitHub en los enlaces
├── assets/                    # tema visual (SCSS) y colores del código (paleta.theme)
├── requirements.txt           # para construir las diapositivas
├── requirements-datos.txt     # para regenerar datos y probar notebooks
└── .github/workflows/publicar.yml   # publica el sitio en GitHub Pages
```

## Construir las diapositivas en su computadora

Necesitan [Quarto](https://quarto.org/docs/get-started/) (≥ 1.4) y Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
quarto preview                      # sitio completo con recarga automática
quarto render diapositivas/04_redes_neuronales.qmd   # una sola presentación
```

El sitio queda en `_site/`. Las gráficas se calculan con Python al construir el sitio; los datos que requieren modelos grandes (vectores de palabras, BETO, Qwen, redes entrenadas con MNIST, el ejemplo adversario, el mini-GPT) ya están precalculados en `diapositivas/datos/`. Para regenerarlos:

```bash
pip install -r requirements-datos.txt
python scripts/precalcular_lenguaje.py
python scripts/precalcular_cnn.py
python scripts/precalcular_adversario.py
python scripts/precalcular_minigpt.py --texto pg2000.txt   # https://www.gutenberg.org/cache/epub/2000/pg2000.txt
```

Para regenerar los notebooks (versión estudiante y soluciones) después de editar sus fuentes:

```bash
cd scripts/fuentes_notebooks
for f in nb*.py; do python "$f"; done
```

## Publicar en GitHub Pages

1. Cambien los marcadores `edithvillegas` / `introduccion-ml-redes-neuronales` por los reales:
   ```bash
   python scripts/configurar_repo.py su-usuario su-repositorio
   ```
2. En GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Cada `git push` a `main` reconstruye y publica el sitio (`.github/workflows/publicar.yml`).

## Notas para presentar

- Atajos: `→`/`←` avanzar, `F` pantalla completa, `O` vista general, `S` notas de la ponente (incluyen sugerencias para las demos en vivo), `C` pizarra, `M` menú.
- Las demos incrustadas (chat de Hugging Face, generador de imágenes, TensorFlow Playground, CNN Explainer, Transformer Explainer, JSLinux, AlphaFold) necesitan **internet**. Revisen el día anterior que los *Spaces* de Hugging Face estén funcionando. El generador de imágenes usa *ZeroGPU*: inicien sesión en Hugging Face en el navegador para tener más cuota.
- Los videos de YouTube no se reproducen si abren el HTML directamente como archivo (`file://`); usen `quarto preview` o el sitio publicado.

## Créditos

**Materiales:** Edith Natalia Villegas García (ICTP), con la colaboración de Joshua Lemus en la sesión de Linux. Preparados con ayuda de Claude (Anthropic).

**Organizan la escuela:** Edith Natalia Villegas García (ICTP), Carlos Soto (OGS), Joshua Lemus (Universidad de Módena y Reggio Emilia), Daniela Pinto Veizaga (NYU); coordinación local: Lucio Villanueva y Melissa Cruz (UNAH). Con el apoyo de ICTP – Physics Without Frontiers, la UNAH y el Tecnológico de Monterrey.

Materiales de otras sesiones (Python, visualización, clasificación, agentes): [dapivei/escuela-machine-learning-honduras-2026](https://github.com/dapivei/escuela-machine-learning-honduras-2026).

Las imágenes de terceros conservan sus licencias originales (ver [`diapositivas/img/CREDITOS.md`](diapositivas/img/CREDITOS.md)).
