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

# Paleta categórica (orden fijo, validada para daltonismo) ---------------------
AZUL = "#2a78d6"
NARANJA = "#eb6834"
AQUA = "#1baf7a"
AMARILLO = "#eda100"
MAGENTA = "#e87ba4"
VERDE = "#008300"
VIOLETA = "#6250d6"
ROJO = "#e34948"
CATEGORICOS = [AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA, ROJO]

MARINO = "#0b2545"
TURQUESA = "#0095b6"
GRIS = "#52514e"
GRIS_CLARO = "#e3e2dd"
FONDO = "#fcfcfb"

# Escala secuencial (un solo tono, claro → oscuro) y divergente
AZULES = [[0, "#f4f8fd"], [0.15, "#cde2fb"], [0.35, "#86b6ef"], [0.55, "#3987e5"],
          [0.75, "#1c5cab"], [1, "#0d366b"]]
DIVERGENTE = [[0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f2f1ee"], [0.75, "#f4a582"], [1, "#b4441c"]]
GRISES = [[0, "#000000"], [1, "#ffffff"]]

FUENTE = "Source Sans 3, Source Sans Pro, Helvetica, Arial, sans-serif"

pio.templates["escuela"] = go.layout.Template(
    layout=dict(
        font=dict(family=FUENTE, size=20, color=MARINO),
        title=dict(font=dict(size=24, color=MARINO), x=0.0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=FONDO,
        colorway=CATEGORICOS,
        xaxis=dict(gridcolor="#ebeae6", zerolinecolor="#cfcec9", linecolor="#cfcec9",
                   ticks="outside", tickcolor="#cfcec9", title=dict(standoff=10)),
        yaxis=dict(gridcolor="#ebeae6", zerolinecolor="#cfcec9", linecolor="#cfcec9",
                   ticks="outside", tickcolor="#cfcec9", title=dict(standoff=10)),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=GRIS_CLARO, borderwidth=1),
        hoverlabel=dict(font=dict(family=FUENTE, size=18), bgcolor="white", bordercolor=GRIS_CLARO),
        margin=dict(l=70, r=30, t=60, b=60),
        sliders=[dict(currentvalue=dict(font=dict(size=20, color=MARINO)),
                      font=dict(size=16, color=GRIS), bgcolor=GRIS_CLARO,
                      activebgcolor=TURQUESA, bordercolor=GRIS_CLARO)],
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
    colores = colores or ["#cde2fb", "#fde0d1", "#c9eedd", "#fbe7b0", "#f6d3e1", "#dcd7f7"]
    spans = []
    for i, p in enumerate(piezas):
        texto = p.replace(" ", "␣").replace("<", "&lt;").replace(">", "&gt;")
        spans.append(f'<span style="background:{colores[i % len(colores)]};border-radius:6px;'
                     f'padding:2px 6px;margin:2px;display:inline-block;font-family:JetBrains Mono,monospace;'
                     f'font-size:0.62em">{texto}</span>')
    return "".join(spans)


__all__ = [n for n in dir() if not n.startswith("_")]
