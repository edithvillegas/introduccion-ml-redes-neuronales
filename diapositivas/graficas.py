"""
Utilidades compartidas para las gráficas interactivas (plotly) de las diapositivas.

Cada presentación hace `from graficas import *` en una celda oculta.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, display
from plotly.subplots import make_subplots

DATOS = Path(__file__).resolve().parent / "datos"

# Paleta de la escuela (la misma de las diapositivas, el sitio y los notebooks) ---
ZAFIRO = "#72B0AB"      # Sapphire
ARTICO = "#BCDDDC"      # Arctic
ENCAJE = "#FFEDD1"      # Lace
CHICLE = "#FDC1B4"      # Bubblegum
CORAL = "#FE9179"       # Ballet Slipper
SALVIA = "#CFB97E"      # Sage
PISTACHO = "#B89D47"    # Pistachio
ABETO = "#355E58"       # Spruce
PAVO_REAL = "#053229"   # Peacock

# Tonos derivados de la paleta, solo para que el texto y las líneas se lean bien
CORAL_OSCURO = "#A84A35"   # texto «coral» con buen contraste sobre fondo claro
GRIS = "#5E7A75"           # texto y líneas secundarias (abeto apagado)
FONDO = "#FFFCF7"          # fondo de las diapositivas (encaje muy claro)
REJILLA = "#E2EFEE"        # cuadrícula de los ejes (ártico muy claro)

# Orden fijo de colores categóricos: primero los que más contrastan entre sí
CATEGORICOS = [ABETO, CORAL, ZAFIRO, PISTACHO, PAVO_REAL, SALVIA, CHICLE, ARTICO]

# Escala secuencial (claro → oscuro) y divergente (abeto ↔ coral)
SECUENCIAL = [[0, ENCAJE], [0.2, ARTICO], [0.45, ZAFIRO], [0.75, ABETO], [1, PAVO_REAL]]
DIVERGENTE = [[0, ABETO], [0.25, ZAFIRO], [0.5, ENCAJE], [0.75, CORAL], [1, CORAL_OSCURO]]
GRISES = [[0, "#000000"], [1, "#ffffff"]]

# Con comillas: «Source Sans 3» sin comillas no es un nombre de fuente válido en CSS
FUENTE = '"Source Sans 3", "Source Sans Pro", Helvetica, Arial, sans-serif'


def _rgb(color):
    color = color.lstrip("#")
    return [int(color[i:i + 2], 16) for i in (0, 2, 4)]


def mezclar(color, blanco=0.5):
    """Aclara un color mezclándolo con blanco (blanco=0: igual; blanco=1: blanco)."""
    return "#" + "".join(f"{round(c + (255 - c) * blanco):02x}" for c in _rgb(color))


def oscurecer(color, negro=0.35):
    """Oscurece un color mezclándolo con negro (útil para texto del mismo tono que una línea)."""
    return "#" + "".join(f"{round(c * (1 - negro)):02x}" for c in _rgb(color))


def texto_sobre(color):
    """Color de texto legible sobre un relleno: blanco sobre colores oscuros, pavo real sobre claros."""
    lineal = [(c / 255) / 12.92 if c / 255 <= 0.03928 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in _rgb(color)]
    luminancia = 0.2126 * lineal[0] + 0.7152 * lineal[1] + 0.0722 * lineal[2]
    return "white" if luminancia < 0.18 else PAVO_REAL


pio.templates["escuela"] = go.layout.Template(
    layout=dict(
        font=dict(family=FUENTE, size=20, color=PAVO_REAL),
        title=dict(font=dict(size=24, color=PAVO_REAL), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=FONDO,
        colorway=CATEGORICOS,
        xaxis=dict(gridcolor=REJILLA, zerolinecolor=ARTICO, linecolor=ARTICO,
                   ticks="outside", tickcolor=ARTICO, title=dict(standoff=10)),
        yaxis=dict(gridcolor=REJILLA, zerolinecolor=ARTICO, linecolor=ARTICO,
                   ticks="outside", tickcolor=ARTICO, title=dict(standoff=10)),
        legend=dict(bgcolor="rgba(255,252,247,0.85)", bordercolor=ARTICO, borderwidth=1),
        # Plotly usa el color del borde como color del texto si no se indica otro: por eso
        # el texto se fija explícitamente en oscuro (antes salía gris muy claro).
        hoverlabel=dict(font=dict(family=FUENTE, size=18, color=PAVO_REAL), bgcolor="white",
                        bordercolor=ZAFIRO, align="left"),
        margin=dict(l=70, r=30, t=60, b=60),
        sliderdefaults=dict(currentvalue=dict(font=dict(size=20, color=PAVO_REAL)),
                            font=dict(size=16, color=ABETO), bgcolor=ARTICO,
                            activebgcolor=ABETO, bordercolor=ZAFIRO),
        updatemenudefaults=dict(font=dict(color=PAVO_REAL), bordercolor=ZAFIRO),
    )
)
pio.templates.default = "plotly_white+escuela"

# plotly.js se incrusta una sola vez por página (funciona sin internet)
pio.renderers.default = "notebook"

CONFIG = {"displaylogo": False, "responsive": False,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "toImage"],
          "locale": "es"}


def mostrar(fig, ancho=1400, alto=650):
    """Muestra una figura con tamaño fijo (las diapositivas miden 1600×900)."""
    fig.update_layout(width=ancho, height=alto, autosize=False)
    fig.show(config=CONFIG)


def cargar(nombre):
    with open(DATOS / nombre, encoding="utf-8") as f:
        return json.load(f)


def deslizador(pasos, prefijo="", y=-0.08, x=0.05, largo=0.9, activo=0):
    """Crea un deslizador de plotly a partir de una lista de pasos ya construidos."""
    return [dict(active=activo, pad=dict(t=40, b=10), x=x, y=y, len=largo,
                 currentvalue=dict(prefix=prefijo, visible=True, xanchor="left"),
                 steps=pasos)]


def pasos_visibilidad(n_trazas_por_paso, etiquetas, fijas=0, extras=None):
    """
    Pasos de deslizador que muestran solo un grupo de trazas a la vez.

    - n_trazas_por_paso: cuántas trazas tiene cada paso (todas del mismo tamaño)
    - fijas: número de trazas al inicio que siempre se ven
    - extras: lista opcional de dicts de layout (p. ej. títulos) para cada paso
    """
    n = len(etiquetas)
    pasos = []
    for i, et in enumerate(etiquetas):
        vis = [True] * fijas + [False] * (n * n_trazas_por_paso)
        for k in range(n_trazas_por_paso):
            vis[fijas + i * n_trazas_por_paso + k] = True
        args = [{"visible": vis}]
        if extras is not None:
            args.append(extras[i])
        pasos.append(dict(method="update", label=str(et), args=args))
    return pasos


def imagen_heatmap(img, escala=GRISES, zmin=0, zmax=1, mostrar_escala=False, **kw):
    """Imagen en escala de grises como heatmap (con el eje y invertido)."""
    return go.Heatmap(z=np.asarray(img)[::-1], colorscale=escala, zmin=zmin, zmax=zmax,
                      showscale=mostrar_escala, hovertemplate="valor: %{z:.2f}<extra></extra>", **kw)


def ejes_imagen(fig, filas=1, columnas=1):
    for i in range(1, filas * columnas + 1):
        sufijo = "" if i == 1 else str(i)
        fig.layout[f"xaxis{sufijo}"].update(visible=False, constrain="domain")
        fig.layout[f"yaxis{sufijo}"].update(visible=False, scaleanchor=f"x{sufijo}", constrain="domain")
    return fig


def a_uint8(z, zmin=0.0, zmax=1.0):
    """Convierte una imagen a enteros 0–255 (plotly la guarda 8 veces más compacta que en float64)."""
    return np.clip((np.asarray(z, dtype=float) - zmin) / (zmax - zmin) * 255, 0, 255).astype(np.uint8)


def frontera_decision(predecir, x_rango, y_rango, n=150):
    """Evalúa una función de predicción en una rejilla 2D."""
    xx, yy = np.meshgrid(np.linspace(*x_rango, n), np.linspace(*y_rango, n))
    zz = predecir(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return np.round(xx[0], 4), np.round(yy[:, 0], 4), np.round(zz, 3)


def tokens_coloreados(piezas, colores=None):
    """HTML con cada token en una cajita de color (para mostrar la tokenización)."""
    colores = colores or [ARTICO, CHICLE, ENCAJE, mezclar(SALVIA, 0.45), mezclar(ZAFIRO, 0.45), mezclar(CORAL, 0.4)]
    spans = []
    for i, p in enumerate(piezas):
        texto = p.replace(" ", "␣").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<span style="background:{colores[i % len(colores)]};border-radius:6px;'
                     f'padding:2px 6px;margin:2px;display:inline-block;font-family:JetBrains Mono,monospace;'
                     f'font-size:0.62em">{texto}</span>')
    return "".join(spans)


__all__ = [n for n in dir() if not n.startswith("_")]
