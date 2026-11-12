"""Programa de la Escuela de Machine Learning Honduras 2026 (hora de Honduras, UTC−6)."""

import plotly.graph_objects as go

from graficas import CATEGORICOS, GRIS, mostrar

DIAS = ["Lun 23 nov", "Mar 24 nov", "Mié 25 nov", "Jue 26 nov", "Vie 27 nov",
        "Lun 30 nov", "Mar 1 dic", "Mié 2 dic", "Jue 3 dic", "Vie 4 dic"]

AREAS = {
    "Escuela": CATEGORICOS[3],
    "Programación y herramientas": CATEGORICOS[0],
    "Matemáticas y optimización": CATEGORICOS[6],
    "Aprendizaje supervisado": CATEGORICOS[1],
    "Aprendizaje no supervisado": CATEGORICOS[2],
    "Redes neuronales e IA": CATEGORICOS[7],
    "Aplicaciones y charlas invitadas": CATEGORICOS[4],
    "Ética y sociedad": CATEGORICOS[5],
}

# (día, inicio, fin, título corto, título completo, ponente, área)
SESIONES = [
    (0, "09:00", "09:50", "Bienvenida", "Bienvenida a la escuela", "Comité organizador", "Escuela"),
    (0, "10:00", "10:50", "Intro ML e IA", "Introducción al aprendizaje automático y la IA (1)", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (0, "11:30", "12:20", "Intro ML e IA", "Introducción al aprendizaje automático y la IA (2)", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (0, "13:30", "14:30", "Linux / Bash", "Linux y Bash; lenguajes compilados vs. interpretados", "Edith Natalia Villegas García y Joshua Lemus", "Programación y herramientas"),
    (0, "14:50", "16:00", "Python", "Introducción a Python", "Daniela Pinto Veizaga", "Programación y herramientas"),
    (1, "09:00", "09:50", "Clustering", "Aprendizaje no supervisado (agrupamiento)", "Alex Rodriguez", "Aprendizaje no supervisado"),
    (1, "10:00", "10:50", "Python", "Introducción a Python", "Daniela Pinto Veizaga", "Programación y herramientas"),
    (1, "11:30", "12:20", "Probabilidad", "Probabilidad", "Javier Aguilar", "Matemáticas y optimización"),
    (1, "13:30", "16:00", "Python y clasificación", "Python y clasificación", "Daniela Pinto Veizaga", "Aprendizaje supervisado"),
    (2, "09:00", "10:50", "PCA", "Análisis de componentes principales (PCA)", "Carlos Soto", "Aprendizaje no supervisado"),
    (2, "11:30", "12:20", "Probabilidad", "Probabilidad", "Javier Aguilar", "Matemáticas y optimización"),
    (2, "13:30", "14:30", "Regresión lineal", "Regresión lineal", "Carlos Soto", "Aprendizaje supervisado"),
    (2, "14:50", "16:00", "Regresión múltiple", "Regresión lineal múltiple", "Carlos Soto", "Aprendizaje supervisado"),
    (3, "09:00", "09:50", "No supervisado", "Aprendizaje no supervisado", "Alex Rodriguez", "Aprendizaje no supervisado"),
    (3, "10:00", "12:20", "Optimización", "Métodos de optimización iterativos", "Carlos Soto", "Matemáticas y optimización"),
    (3, "13:30", "16:00", "Optimización (práctica)", "Métodos de optimización: sesión práctica", "Carlos Soto", "Matemáticas y optimización"),
    (4, "09:00", "09:50", "Invitada", "Charla invitada", "Eleonora Bergamin (SISSA)", "Aplicaciones y charlas invitadas"),
    (4, "10:00", "10:50", "Invitado", "Charla invitada: ecología y comportamiento colectivo", "Arshed Nabeel (IISc)", "Aplicaciones y charlas invitadas"),
    (4, "11:30", "12:20", "Redes neuronales", "Teoría de redes neuronales", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (4, "13:30", "14:30", "Datos médicos", "Análisis de datos médicos", "Juan Calderón Santamaría (UNAH)", "Aplicaciones y charlas invitadas"),
    (4, "14:50", "16:00", "Datos demográficos", "Datos demográficos", "Luis Benjamín Rodríguez Aguilar (INE Guatemala)", "Aplicaciones y charlas invitadas"),
    (5, "09:00", "09:50", "Invitado", "Charla invitada: biología y ML", "Manuel Vasquez (IGC)", "Aplicaciones y charlas invitadas"),
    (5, "10:00", "10:50", "Ética de la IA", "Ética de la IA", "Matteo Marsili (ICTP)", "Ética y sociedad"),
    (5, "11:30", "12:20", "ML y políticas", "Taller: ML para políticas públicas", "Elena Villalobos Nolasco (Tec de Monterrey)", "Ética y sociedad"),
    (5, "13:30", "14:30", "Redes conv.", "Redes neuronales convolucionales", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (5, "14:50", "16:00", "Transformers", "Transformers y modelos de lenguaje", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (6, "09:00", "09:50", "No supervisado 2", "Aprendizaje no supervisado 2", "Alex Rodriguez", "Aprendizaje no supervisado"),
    (6, "10:00", "10:50", "Invitado", "Charla invitada", "Alessio Ansuini (U. de Trieste)", "Aplicaciones y charlas invitadas"),
    (6, "11:30", "12:20", "ML y políticas", "Taller: ML para políticas públicas", "Elena Villalobos Nolasco (Tec de Monterrey)", "Ética y sociedad"),
    (6, "13:30", "14:30", "Invitada", "Charla invitada", "Irene Rubia-Rodríguez (DTU)", "Aplicaciones y charlas invitadas"),
    (6, "14:50", "16:00", "Práctica RN", "Práctica: Python, PyTorch y Hugging Face", "Edith Natalia Villegas García", "Redes neuronales e IA"),
    (7, "09:00", "12:20", "Grafos", "Aprendizaje en grafos", "Alvaro Carbonero y Ana Rivera Him (MIT)", "Redes neuronales e IA"),
    (7, "13:30", "14:30", "Invitado", "Charla invitada", "Vora Arhan Jesal (CNRS)", "Aplicaciones y charlas invitadas"),
    (7, "14:50", "16:00", "Agentes de IA", "Agentes de IA", "Daniela Pinto Veizaga (NYU)", "Redes neuronales e IA"),
    (8, "09:00", "12:20", "Grafos", "Aprendizaje en grafos", "Alvaro Carbonero y Ana Rivera Him (MIT)", "Redes neuronales e IA"),
    (8, "13:30", "15:50", "Neurociencia", "Introducción a la neurociencia", "Joshua Lemus (U. de Módena y Reggio Emilia)", "Aplicaciones y charlas invitadas"),
    (9, "09:00", "11:20", "Física de partículas", "Ciencia de datos aplicada a la física de partículas", "Edgar Fernando Carrera Jarrín (USFQ)", "Aplicaciones y charlas invitadas"),
    (9, "11:30", "12:20", "Carreras", "Sesión de carreras y oportunidades", "Comité organizador", "Escuela"),
    (9, "13:30", "14:30", "Diplomas", "Ceremonia de entrega de diplomas", "Comité organizador", "Escuela"),
]


def horas(h):
    hh, mm = h.split(":")
    return int(hh) + int(mm) / 60


def grafica_programa(resaltar=None, ancho=1450, alto=690):
    """Calendario interactivo de la escuela. `resaltar`: texto para resaltar sesiones."""
    fig = go.Figure()
    for area, color in AREAS.items():
        ses = [s for s in SESIONES if s[6] == area]
        if not ses:
            continue
        opac = [1.0 if (resaltar is None or resaltar in s[5]) else 0.25 for s in ses]
        fig.add_trace(go.Bar(
            x=[DIAS[s[0]] for s in ses],
            base=[horas(s[1]) for s in ses],
            y=[horas(s[2]) - horas(s[1]) for s in ses],
            name=area, marker=dict(color=color, opacity=opac, line=dict(color="white", width=2)),
            text=[s[3] for s in ses], textposition="inside", insidetextanchor="middle", textangle=0,
            textfont=dict(size=13, color="white"),
            customdata=[[s[4], s[5], s[1], s[2]] for s in ses],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{x}, %{customdata[2]}–%{customdata[3]}<extra></extra>",
            width=0.92,
        ))
    fig.update_layout(
        barmode="overlay",
        yaxis=dict(autorange="reversed", tickvals=list(range(9, 17)),
                   ticktext=[f"{h}:00" for h in range(9, 17)], range=[16.1, 8.9], title=None),
        xaxis=dict(categoryorder="array", categoryarray=DIAS, side="top", tickfont=dict(size=15)),
        legend=dict(orientation="h", y=-0.04, x=0, font=dict(size=15)),
        margin=dict(l=60, r=10, t=40, b=10),
        bargap=0.05,
    )
    fig.add_vline(x=4.5, line=dict(color=GRIS, width=2, dash="dot"))
    mostrar(fig, ancho, alto)
