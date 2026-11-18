"""
Utilidades para construir los notebooks de práctica.

Cada notebook se define como una lista de celdas:
    ("md", "texto en markdown")
    ("code", "código que se ejecuta igual en ambas versiones")
    ("ej", "código para estudiantes (con # TU CÓDIGO AQUÍ)", "código resuelto")

y se generan dos versiones:
    notebooks/<nombre>.ipynb   (para estudiantes)
    soluciones/<nombre>_soluciones.ipynb   (resuelto, para docentes)
"""

import hashlib
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[2]
REPO_GITHUB = "edithvillegas/introduccion-ml-redes-neuronales"


def insignia_colab(carpeta, archivo):
    url = f"https://colab.research.google.com/github/{REPO_GITHUB}/blob/main/{carpeta}/{archivo}"
    return (f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" '
            f'alt="Abrir en Colab"/></a>')


FUNCION_REVISAR = '''
# Función para revisar tus ejercicios: imprime ✅ o ❌ (nunca rompe el notebook)
def revisar(condicion, mensaje_ok="¡Correcto!", pista="Revisa tu código e inténtalo otra vez."):
    try:
        ok = bool(condicion() if callable(condicion) else condicion)
    except Exception as e:
        ok, pista = False, f"{pista}  (error: {type(e).__name__}: {e})"
    print(("✅ " + mensaje_ok) if ok else ("❌ Aún no. " + pista))
'''.strip()


ESTILO_GRAFICAS = '''
# Colores de la escuela para las gráficas (los mismos de las diapositivas)
from matplotlib.colors import LinearSegmentedColormap
PALETA = {"zafiro": "#72B0AB", "artico": "#BCDDDC", "encaje": "#FFEDD1", "chicle": "#FDC1B4",
          "coral": "#FE9179", "salvia": "#CFB97E", "pistacho": "#B89D47", "abeto": "#355E58",
          "pavo_real": "#053229", "gris": "#5E7A75"}
SECUENCIAL = LinearSegmentedColormap.from_list("escuela", ["#FFEDD1", "#BCDDDC", "#72B0AB", "#355E58", "#053229"])
DIVERGENTE = LinearSegmentedColormap.from_list("escuela_div", ["#355E58", "#72B0AB", "#FFEDD1", "#FE9179", "#A84A35"])
plt.rcParams.update({
    "axes.prop_cycle": plt.cycler(color=["#355E58", "#FE9179", "#72B0AB", "#B89D47", "#053229", "#CFB97E"]),
    "text.color": "#053229", "axes.labelcolor": "#053229", "axes.titlecolor": "#053229",
    "axes.edgecolor": "#5E7A75", "xtick.color": "#053229", "ytick.color": "#053229",
})
'''.strip()


def construir(nombre, celdas, gpu=False):
    for version in ("notebooks", "soluciones"):
        archivo = f"{nombre}.ipynb" if version == "notebooks" else f"{nombre}_soluciones.ipynb"
        nb = nbf.v4.new_notebook()
        nb.metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        }
        if gpu:
            nb.metadata["accelerator"] = "GPU"
            nb.metadata["colab"]["gpuType"] = "T4"
        nb.cells.append(nbf.v4.new_markdown_cell(insignia_colab(version, archivo)))
        for celda in celdas:
            tipo = celda[0]
            if tipo == "md":
                texto = celda[1].strip()
                if version == "soluciones" and texto.startswith("# "):
                    primera, _, resto = texto.partition("\n")
                    texto = primera + " · ✅ SOLUCIONES\n" + resto
                nb.cells.append(nbf.v4.new_markdown_cell(texto))
            elif tipo == "code":
                nb.cells.append(nbf.v4.new_code_cell(celda[1].strip()))
            elif tipo == "ej":
                codigo = celda[1] if version == "notebooks" else celda[2]
                nb.cells.append(nbf.v4.new_code_cell(codigo.strip()))
            else:
                raise ValueError(tipo)
        # Identificadores de celda fijos: así regenerar los notebooks no cambia nada si las fuentes no cambian
        for i, celda_nb in enumerate(nb.cells):
            celda_nb.id = hashlib.sha1(f"{nombre}-{i}".encode()).hexdigest()[:8]
        salida = RAIZ / version / archivo
        salida.parent.mkdir(exist_ok=True)
        nbf.write(nb, salida)
        print("✓", salida.relative_to(RAIZ))
