"""
Reemplaza los marcadores TU_USUARIO / TU_REPOSITORIO por el usuario y el nombre
reales del repositorio de GitHub en todos los archivos (enlaces de Colab, GitHub Pages…).

Uso (desde la raíz del repositorio):
    python scripts/configurar_repo.py mi-usuario mi-repositorio

Después vuelve a generar los notebooks si cambias sus fuentes:
    cd scripts/fuentes_notebooks && for f in nb*.py; do python "$f"; done
"""

import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(__doc__)
usuario, repo = sys.argv[1], sys.argv[2]
raiz = Path(__file__).resolve().parents[1]
archivos = ([raiz / "_quarto.yml", raiz / "index.qmd", raiz / "README.md"]
            + sorted((raiz / "notebooks").glob("*.ipynb")) + sorted((raiz / "soluciones").glob("*.ipynb"))
            + sorted((raiz / "scripts" / "fuentes_notebooks").glob("*.py")))
for archivo in archivos:
    texto = archivo.read_text(encoding="utf-8")
    nuevo = texto.replace("TU_USUARIO", usuario).replace("TU_REPOSITORIO", repo)
    if nuevo != texto:
        archivo.write_text(nuevo, encoding="utf-8")
        print("✓", archivo.relative_to(raiz))
print(f"Listo: https://github.com/{usuario}/{repo} · https://{usuario}.github.io/{repo}/")
