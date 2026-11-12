"""Notebook 03 · La terminal de Linux, git y lenguajes compilados vs. interpretados."""

from utilidades import FUNCION_REVISAR, construir

FINCAS = """finca,departamento,altitud_m,variedad,produccion_qq
El Mirador,Copán,1350,Catuaí,120
La Esperanza,Copán,1420,Obatá,95
Los Pinos,Ocotepeque,1600,Pacas,80
San Rafael,Lempira,1280,Parainema,150
Buena Vista,La Paz,1150,IHCAFE 90,200
El Cedro,Comayagua,1300,Catuaí,130
Las Nubes,Copán,1550,Geisha,40
Cumbre Celaque,Lempira,1750,Bourbon,35
La Montañita,Santa Bárbara,1250,Obatá,160
El Paraíso Alto,El Paraíso,1450,Parainema,110
Santa Elena,La Paz,1200,Catuaí,175
Agua Fría,Intibucá,1650,Pacas,60
El Roble,Copán,1100,IHCAFE 90,210
Los Laureles,Comayagua,1380,Caturra,90
Monte Verde,Ocotepeque,1500,Geisha,45
La Cascada,Santa Bárbara,1330,Catuaí,140
San José,El Paraíso,1220,Obatá,185
Piedra Azul,Lempira,1480,Parainema,100
El Zapote,Copán,1000,Catuaí,230
Cerro Bonito,Intibucá,1580,Bourbon,55
La Quebrada,Comayagua,1260,IHCAFE 90,170
Tres Ríos,La Paz,1400,Caturra,85
Las Flores,El Paraíso,1360,Catuaí,125
Montecillos,Comayagua,1520,Pacas,70
El Pital,Ocotepeque,1700,Geisha,30
Santa Rita,Copán,1240,Obatá,145
Los Naranjos,Santa Bárbara,1180,Parainema,190
Guarita,Lempira,1390,Catuaí,115
Lepaterique,El Paraíso,1300,Caturra,105
Gualcinse,Lempira,1610,Pacas,65"""

celdas = [
    ("md", """
# 03 · La terminal de Linux, git y C vs. Python

**Notebook de práctica** · Escuela de Machine Learning Honduras 2026 · Sesión del lunes 23 de noviembre

Google Colab corre en una **máquina virtual con Linux**. En este notebook usaremos esa máquina para practicar la terminal sin instalar nada.

**Contenido**

1. Preparar el espacio de trabajo
2. ¿Dónde estoy? Explorar el sistema
3. Crear, copiar, mover y borrar archivos
4. Ver y analizar archivos de texto
5. Tuberías y redirección: analizar fincas de café ☕
6. Bucles y *scripts* de bash
7. git paso a paso
8. Compilado vs. interpretado: C contra Python
9. 🏆 Retos finales

**Cómo usar este notebook**

- Ejecuta las celdas en orden con `Shift + Enter`.
- Las celdas que empiezan con `%%bash` se ejecutan **en la terminal** (bash), no en Python.
- En cada **✏️ Ejercicio** escribe tu solución donde dice `# TU CÓDIGO AQUÍ`; la celda siguiente revisa tu respuesta y responde ✅ o ❌.
- ⚠️ Detalle de Colab: cada celda `%%bash` abre una terminal **nueva**, así que un `cd` no "dura" de una celda a otra. Por eso, en cada celda partimos desde la carpeta de trabajo.
"""),
    ("md", "## 1. Preparar el espacio de trabajo\n\nEsta celda (de Python) crea una carpeta de trabajo y define la función `revisar`."),
    ("code", f"""
import os, subprocess

BASE = "/content" if os.path.isdir("/content") else os.path.expanduser("~")
TRABAJO = os.path.join(BASE, "terminal_escuela")
os.makedirs(TRABAJO, exist_ok=True)
os.chdir(TRABAJO)          # todas las celdas %%bash empiezan aquí

def bash(comando):
    \"\"\"Ejecuta un comando de bash desde Python y devuelve su salida como texto.\"\"\"
    return subprocess.run(comando, shell=True, capture_output=True, text=True, executable="/bin/bash").stdout.strip()

{FUNCION_REVISAR}

print("Carpeta de trabajo:", TRABAJO)
print("¡Todo listo!")
"""),
    ("md", """
Ahora creamos un archivo de datos **inventados** sobre fincas de café en Honduras. Usamos un *heredoc* de bash: todo lo que está entre `<< 'FIN'` y `FIN` se escribe en el archivo.
"""),
    ("code", f"""
%%bash
cat > fincas.csv << 'FIN'
{FINCAS}
FIN
echo "Archivo creado:"
ls -l fincas.csv
"""),
    ("md", """
## 2. ¿Dónde estoy? Explorar el sistema

| Comando | Qué hace |
|---|---|
| `pwd` | muestra la carpeta actual (*print working directory*) |
| `whoami` | usuario actual |
| `hostname` | nombre de la máquina |
| `uname -a` | información del núcleo (*kernel*) |
| `date` | fecha y hora |
| `ls` / `ls -l` / `ls -la` | lista archivos (formato largo, incluye ocultos) |
"""),
    ("code", """
%%bash
pwd
whoami
hostname
uname -a
date
"""),
    ("code", """
%%bash
# ¿Qué distribución de Linux usa Colab? (en Mac este archivo no existe)
cat /etc/os-release 2>/dev/null | head -n 4 || echo "No es Linux 🙂"
echo "---- la raíz del sistema de archivos ----"
ls /
"""),
    ("code", """
%%bash
# ¿Cuántos procesadores y cuánta memoria y disco tenemos?
nproc 2>/dev/null || sysctl -n hw.ncpu
free -h 2>/dev/null || echo "(free no existe en este sistema)"
df -h . | tail -n 1
"""),
    ("md", """
### ✏️ Ejercicio 1 · Moverse por el árbol

Usa `ls` para listar el contenido de la carpeta **`/etc`** (configuración del sistema) y luego cuenta cuántos elementos tiene usando una tubería con `wc -l`.

<details><summary>💡 Pista</summary>

`ls /etc | wc -l`
</details>
"""),
    ("ej", """
%%bash
# TU CÓDIGO AQUÍ
""", """
%%bash
ls /etc
echo "Número de elementos en /etc:"
ls /etc | wc -l
"""),
    ("md", """
## 3. Crear, copiar, mover y borrar

| Comando | Qué hace |
|---|---|
| `mkdir carpeta` | crea una carpeta (`-p` crea también las intermedias) |
| `touch archivo` | crea un archivo vacío |
| `echo "texto" > archivo` | escribe texto en un archivo (lo **sobrescribe**) |
| `echo "texto" >> archivo` | **agrega** texto al final |
| `cp origen destino` | copia (`cp -r` para carpetas) |
| `mv origen destino` | mueve o renombra |
| `rm archivo` | borra ⚠️ **no hay papelera** |
"""),
    ("code", """
%%bash
mkdir -p prueba
touch prueba/vacio.txt
echo "hola desde la terminal" > prueba/saludo.txt
echo "segunda línea" >> prueba/saludo.txt
cat prueba/saludo.txt
cp prueba/saludo.txt prueba/copia.txt
mv prueba/copia.txt prueba/renombrado.txt
ls -l prueba
rm -r prueba   # limpiamos
"""),
    ("md", """
### ✏️ Ejercicio 2 · La estructura de un proyecto

Con **un solo comando** `mkdir -p`, crea esta estructura de carpetas:

```
proyecto/
├── datos/
├── resultados/
└── scripts/
```

Luego crea el archivo `proyecto/LEEME.txt` con el texto exacto `Proyecto de la escuela` y copia `fincas.csv` dentro de `proyecto/datos/`.

<details><summary>💡 Pista</summary>

`mkdir -p proyecto/datos proyecto/resultados proyecto/scripts` (o `mkdir -p proyecto/{datos,resultados,scripts}`), `echo "..." > proyecto/LEEME.txt`, `cp fincas.csv proyecto/datos/`
</details>
"""),
    ("ej", """
%%bash
# TU CÓDIGO AQUÍ
""", """
%%bash
mkdir -p proyecto/{datos,resultados,scripts}
echo "Proyecto de la escuela" > proyecto/LEEME.txt
cp fincas.csv proyecto/datos/
ls -R proyecto
"""),
    ("code", """
revisar(lambda: all(os.path.isdir(f"proyecto/{c}") for c in ["datos", "resultados", "scripts"]),
        "Las tres carpetas existen.", "Faltan carpetas dentro de proyecto/.")
revisar(lambda: open("proyecto/LEEME.txt").read().strip() == "Proyecto de la escuela",
        "LEEME.txt tiene el texto correcto.", "Revisa el contenido de proyecto/LEEME.txt.")
revisar(lambda: open("proyecto/datos/fincas.csv").read() == open("fincas.csv").read(),
        "fincas.csv está copiado en proyecto/datos/.", "Copia fincas.csv a proyecto/datos/.")
"""),
    ("md", """
## 4. Ver y analizar archivos de texto

| Comando | Qué hace |
|---|---|
| `cat archivo` | muestra todo el archivo |
| `head -n 5 archivo` | primeras 5 líneas |
| `tail -n 5 archivo` | últimas 5 líneas (`tail -n +2` = desde la línea 2) |
| `wc -l archivo` | cuenta líneas (`-w` palabras, `-c` bytes) |
| `grep patrón archivo` | líneas que contienen el patrón (`-i` ignora mayúsculas, `-c` cuenta, `-v` invierte) |
| `cut -d, -f2 archivo` | extrae la columna 2 de un archivo separado por comas |
| `sort` | ordena (`-n` numérico, `-r` al revés, `-t, -k3` por la columna 3) |
| `uniq -c` | cuenta líneas repetidas **consecutivas** (por eso va después de `sort`) |
"""),
    ("code", """
%%bash
cd proyecto/datos
head -n 5 fincas.csv
echo "..."
tail -n 3 fincas.csv
echo
echo "Líneas en total (incluye el encabezado):"
wc -l < fincas.csv
"""),
    ("code", """
%%bash
cd proyecto/datos
echo "Fincas con variedad Geisha:"
grep "Geisha" fincas.csv
echo
echo "Solo los nombres de los departamentos (columna 2), sin el encabezado:"
cut -d, -f2 fincas.csv | tail -n +2 | head -n 8
"""),
    ("md", """
### ✏️ Ejercicio 3 · Contar y filtrar

1. Guarda en `proyecto/resultados/n_fincas.txt` el **número de fincas** (líneas sin contar el encabezado).
2. Guarda en `proyecto/resultados/copan.csv` todas las líneas de fincas del departamento de **Copán**.

<details><summary>💡 Pista</summary>

`tail -n +2 fincas.csv | wc -l > ...` y `grep ",Copán," fincas.csv > ...` (las comas evitan coincidencias con nombres de fincas).
</details>
"""),
    ("ej", """
%%bash
cd proyecto
# TU CÓDIGO AQUÍ
""", """
%%bash
cd proyecto
tail -n +2 datos/fincas.csv | wc -l > resultados/n_fincas.txt
grep ",Copán," datos/fincas.csv > resultados/copan.csv
cat resultados/n_fincas.txt
cat resultados/copan.csv
"""),
    ("code", """
import pandas as pd
fincas = pd.read_csv("fincas.csv")
revisar(lambda: int(open("proyecto/resultados/n_fincas.txt").read().strip()) == len(fincas),
        f"Hay {len(fincas)} fincas.", "El número de fincas no coincide (¿contaste el encabezado?).")
revisar(lambda: len([l for l in open("proyecto/resultados/copan.csv") if l.strip()]) == (fincas.departamento == "Copán").sum(),
        "copan.csv tiene todas las fincas de Copán.", "Revisa el filtro con grep.")
"""),
    ("md", """
## 5. Tuberías y redirección ☕

La **tubería** `|` conecta la salida de un comando con la entrada del siguiente. Es la idea más poderosa de la terminal: programas pequeños que se combinan.

```
cut -d, -f2 fincas.csv | tail -n +2 | sort | uniq -c | sort -nr
   (columna 2)         (sin encabezado) (ordenar) (contar) (de mayor a menor)
```
"""),
    ("code", """
%%bash
cd proyecto/datos
echo "Fincas por variedad:"
cut -d, -f4 fincas.csv | tail -n +2 | sort | uniq -c | sort -nr
"""),
    ("md", """
### ✏️ Ejercicio 4 · ¿Qué departamento tiene más fincas?

Guarda en `proyecto/resultados/por_departamento.txt` el número de fincas **por departamento**, ordenado de **mayor a menor**.

### ✏️ Ejercicio 5 · La finca más alta

Guarda en `proyecto/resultados/mas_alta.txt` **solo el nombre** de la finca con mayor altitud.

<details><summary>💡 Pista para el 5</summary>

Ordena numéricamente por la columna 3 con `sort -t, -k3 -nr`, quédate con la primera línea con `head -n 1` y extrae la columna 1 con `cut -d, -f1`. No olvides quitar el encabezado.
</details>
"""),
    ("ej", """
%%bash
cd proyecto
# TU CÓDIGO AQUÍ (ejercicio 4)

# TU CÓDIGO AQUÍ (ejercicio 5)
""", """
%%bash
cd proyecto
cut -d, -f2 datos/fincas.csv | tail -n +2 | sort | uniq -c | sort -nr > resultados/por_departamento.txt
tail -n +2 datos/fincas.csv | sort -t, -k3 -nr | head -n 1 | cut -d, -f1 > resultados/mas_alta.txt
cat resultados/por_departamento.txt
cat resultados/mas_alta.txt
"""),
    ("code", """
top = fincas.departamento.value_counts().idxmax()
alta = fincas.loc[fincas.altitud_m.idxmax(), "finca"]
revisar(lambda: top in open("proyecto/resultados/por_departamento.txt").readline(),
        f"{top} tiene más fincas.", "La primera línea debería ser el departamento con más fincas.")
revisar(lambda: open("proyecto/resultados/mas_alta.txt").read().strip() == alta,
        f"La finca más alta es {alta}.", "Revisa el sort numérico por la columna 3.")
"""),
    ("md", """
## 6. Bucles y *scripts* de bash

Un **bucle** repite un comando para muchos archivos:

```bash
for i in 1 2 3; do
    echo "vuelta $i"
done
```

Un ***script*** es un archivo de texto con comandos. La primera línea `#!/bin/bash` (*shebang*) indica qué programa lo ejecuta, y `$1`, `$2`… son los argumentos que le pasamos.
"""),
    ("code", """
%%bash
for departamento in Copán Lempira "La Paz"; do
    n=$(grep -c ",$departamento," proyecto/datos/fincas.csv)
    echo "$departamento: $n fincas"
done
"""),
    ("code", """
%%bash
cat > proyecto/scripts/saludo.sh << 'FIN'
#!/bin/bash
# Uso: ./saludo.sh NOMBRE
echo "¡Hola, $1! Hoy es $(date +%A)."
FIN
chmod +x proyecto/scripts/saludo.sh
ls -l proyecto/scripts/saludo.sh
./proyecto/scripts/saludo.sh Honduras
"""),
    ("md", """
### ✏️ Ejercicio 6 · Tu propio *script*

Crea el *script* `proyecto/scripts/resumen.sh` que reciba un **departamento** como argumento e imprima **solo el número** de fincas de ese departamento. Hazlo ejecutable con `chmod +x` y pruébalo con `./proyecto/scripts/resumen.sh Lempira`.

<details><summary>💡 Pista</summary>

Dentro del script: `grep -c ",$1," "$(dirname "$0")/../datos/fincas.csv"`. `dirname "$0"` es la carpeta donde vive el script, así funciona desde cualquier lugar.
</details>
"""),
    ("ej", """
%%bash
cat > proyecto/scripts/resumen.sh << 'FIN'
#!/bin/bash
# TU CÓDIGO AQUÍ
FIN
chmod +x proyecto/scripts/resumen.sh
./proyecto/scripts/resumen.sh Lempira
""", """
%%bash
cat > proyecto/scripts/resumen.sh << 'FIN'
#!/bin/bash
# Uso: ./resumen.sh DEPARTAMENTO   → imprime el número de fincas
grep -c ",$1," "$(dirname "$0")/../datos/fincas.csv"
FIN
chmod +x proyecto/scripts/resumen.sh
./proyecto/scripts/resumen.sh Lempira
"""),
    ("code", """
revisar(lambda: all(bash(f"./proyecto/scripts/resumen.sh '{d}'") == str((fincas.departamento == d).sum())
                    for d in ["Lempira", "Copán", "La Paz", "El Paraíso"]),
        "Tu script cuenta bien las fincas de cualquier departamento.",
        "Prueba tu script con varios departamentos; ¿es ejecutable? ¿usa $1?")
"""),
    ("md", """
### ✏️ Ejercicio 7 · Un archivo por departamento

Con un bucle `for`, crea en `proyecto/resultados/` un archivo **`<Departamento>.csv`** para cada departamento, con las fincas de ese departamento (usa `grep`). Ojo con los nombres con espacios: usa comillas.

<details><summary>💡 Pista</summary>

```bash
cut -d, -f2 proyecto/datos/fincas.csv | tail -n +2 | sort -u | while read -r d; do
    grep ",$d," proyecto/datos/fincas.csv > "proyecto/resultados/$d.csv"
done
```
</details>
"""),
    ("ej", """
%%bash
# TU CÓDIGO AQUÍ
""", """
%%bash
cut -d, -f2 proyecto/datos/fincas.csv | tail -n +2 | sort -u | while read -r d; do
    grep ",$d," proyecto/datos/fincas.csv > "proyecto/resultados/$d.csv"
done
ls proyecto/resultados
"""),
    ("code", """
revisar(lambda: all(os.path.exists(f"proyecto/resultados/{d}.csv") and
                    len(open(f"proyecto/resultados/{d}.csv").readlines()) == n
                    for d, n in fincas.departamento.value_counts().items()),
        "Un archivo por departamento, con el número correcto de fincas.", "Falta algún archivo o tiene filas de más.")
"""),
    ("md", """
## 7. git paso a paso

Vamos a convertir `proyecto/` en un **repositorio de git**. Primero configuramos un nombre y correo **solo para este repositorio** (pueden ser inventados).

| Comando | Qué hace |
|---|---|
| `git init` | crea un repositorio nuevo |
| `git status` | muestra qué cambió |
| `git add archivo` | prepara cambios para el próximo *commit* |
| `git commit -m "mensaje"` | guarda una "foto" del proyecto |
| `git log --oneline` | historia de *commits* |
| `git diff` | cambios que aún no están preparados |
| `git switch -c rama` | crea una rama y se cambia a ella |
| `git merge rama` | trae los cambios de otra rama |
"""),
    ("code", """
%%bash
cd proyecto
git init -b main
git config user.name "Estudiante de la Escuela"
git config user.email "estudiante@ejemplo.com"
echo "resultados/" > .gitignore      # no versionamos resultados generados
git status
"""),
    ("code", """
%%bash
cd proyecto
git add LEEME.txt .gitignore datos/fincas.csv scripts/
git commit -m "Primer commit: datos y scripts"
git log --oneline
"""),
    ("code", """
%%bash
cd proyecto
echo "Autora: (tu nombre aquí)" >> LEEME.txt
git diff          # ¿qué cambió desde el último commit?
"""),
    ("md", """
### ✏️ Ejercicio 8 · Guardar un cambio

Guarda (con `git add` y `git commit`) el cambio que acabamos de hacer en `LEEME.txt`, con el mensaje `Agrega autora`. Después muestra la historia con `git log --oneline`.
"""),
    ("ej", """
%%bash
cd proyecto
# TU CÓDIGO AQUÍ
""", """
%%bash
cd proyecto
git add LEEME.txt
git commit -m "Agrega autora"
git log --oneline
"""),
    ("code", """
revisar(lambda: "Agrega autora" in bash("cd proyecto && git log --format=%s") and bash("cd proyecto && git status --porcelain") == "",
        "Cambio guardado y el repositorio está limpio.", "¿Hiciste git add y git commit con el mensaje correcto?")
"""),
    ("md", """
### ✏️ Ejercicio 9 · Ramas

1. Crea una rama llamada `experimento` y cámbiate a ella.
2. Agrega una finca nueva al final de `datos/fincas.csv` (invéntala, con el mismo formato) y haz *commit* con el mensaje `Agrega finca nueva`.
3. Regresa a `main` y fusiona (`merge`) la rama `experimento`.

<details><summary>💡 Pista</summary>

`git switch -c experimento`, `echo "Mi Finca,Olancho,1400,Catuaí,100" >> datos/fincas.csv`, `git commit -am "Agrega finca nueva"`, `git switch main`, `git merge experimento`
</details>
"""),
    ("ej", """
%%bash
cd proyecto
# TU CÓDIGO AQUÍ
""", """
%%bash
cd proyecto
git switch -c experimento
echo "La Ceibita,Olancho,1400,Catuaí,100" >> datos/fincas.csv
git commit -am "Agrega finca nueva"
git switch main
git merge experimento
git log --oneline --graph --all
"""),
    ("code", """
revisar(lambda: bash("cd proyecto && git branch --show-current") == "main"
                and "Agrega finca nueva" in bash("cd proyecto && git log main --format=%s")
                and len(open("proyecto/datos/fincas.csv").readlines()) == len(fincas) + 2,
        "La rama experimento se fusionó en main.", "¿Estás en main? ¿Hiciste el merge?")
"""),
    ("md", """
> 🌐 **Para subir a GitHub** necesitarías una cuenta y un *token* de acceso; los comandos serían `git remote add origin https://github.com/USUARIO/REPO.git` y `git push -u origin main`. **Nunca** pegues tu *token* en un notebook que vayas a compartir.

También puedes **clonar** cualquier repositorio público. Por ejemplo, el de esta escuela:
"""),
    ("code", """
%%bash
git clone --depth 1 https://github.com/edithvillegas/introduccion-ml-redes-neuronales.git material_escuela 2>&1 | tail -n 2 || true
ls material_escuela 2>/dev/null | head || echo "(cambia la URL por la del repositorio del curso)"
"""),
    ("md", """
## 8. Compilado vs. interpretado: C contra Python

Vamos a calcular $\\pi$ con la serie de Leibniz, $\\pi = 4\\left(1 - \\tfrac{1}{3} + \\tfrac{1}{5} - \\tfrac{1}{7} + \\dots\\right)$, con **100 millones** de términos, en tres versiones.

`%%writefile` guarda el contenido de la celda en un archivo.
"""),
    ("code", """
%%writefile pi.c
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 100000000;
    double suma = 0.0;
    for (long k = 0; k < n; k++) {
        suma += (k % 2 == 0 ? 1.0 : -1.0) / (2.0 * k + 1.0);
    }
    printf("pi ≈ %.10f\\n", 4.0 * suma);
    return 0;
}
"""),
    ("code", """
%%writefile pi.py
import sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
suma = 0.0
for k in range(n):
    suma += (1.0 if k % 2 == 0 else -1.0) / (2.0 * k + 1.0)
print(f"pi ≈ {4 * suma:.10f}")
"""),
    ("code", """
%%writefile pi_numpy.py
import sys
import numpy as np
n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
terminos = 1.0 / (2.0 * np.arange(n) + 1.0)   # 1, 1/3, 1/5, ...
terminos[1::2] *= -1                           # alternar los signos
print(f"pi ≈ {4 * terminos.sum():.10f}")
"""),
    ("code", """
%%bash
gcc -O2 pi.c -o pi          # compilar: traduce TODO el programa a código máquina
ls -l pi                    # ¡un archivo binario ejecutable!
echo "---- C (compilado, -O2) ----"
time ./pi
echo "---- NumPy (Python + núcleo en C) ----"
time python3 pi_numpy.py
"""),
    ("md", "Ahora Python puro. ⏳ Tarda varios segundos: mientras tanto, piensa por qué."),
    ("code", """
%%bash
echo "---- Python puro ----"
time python3 pi.py
"""),
    ("md", """
**¿Por qué tanta diferencia?** En cada vuelta del bucle, Python tiene que averiguar de qué tipo es cada variable, crear objetos nuevos para cada número y revisar errores. El compilador de C sabe de antemano que todo son números `double` y genera instrucciones de máquina directas (y con `-O2` además las optimiza). NumPy hace el bucle en C, sobre arreglos completos.

### ✏️ Ejercicio 10 · Optimizaciones del compilador

Compila `pi.c` **sin optimizaciones** (`-O0`) en un ejecutable llamado `pi_lento`, mide su tiempo con `time` y compáralo con el de `-O2`.
"""),
    ("ej", """
%%bash
# TU CÓDIGO AQUÍ
""", """
%%bash
gcc -O0 pi.c -o pi_lento
echo "---- -O0 ----"; time ./pi_lento
echo "---- -O2 ----"; time ./pi
"""),
    ("code", """
revisar(lambda: os.access("pi_lento", os.X_OK) and "3.14159" in bash("./pi_lento 1000000"),
        "pi_lento existe y funciona. ¿Cuántas veces más rápido es -O2?", "Compila con gcc -O0 pi.c -o pi_lento")
"""),
    ("md", """
## 9. 🏆 Retos finales

### ✏️ Reto 1 · Hola, Honduras en C

Escribe un programa en C, `hola.c`, que imprima exactamente `Hola, Honduras`. Compílalo en un ejecutable llamado `hola` y ejecútalo.

<details><summary>💡 Pista</summary>

Usa `%%writefile hola.c` en una celda y `gcc hola.c -o hola && ./hola` en otra.
</details>
"""),
    ("ej", """
# TU CÓDIGO AQUÍ (puedes usar varias celdas)
""", """
%%writefile hola.c
#include <stdio.h>

int main(void) {
    printf("Hola, Honduras\\n");
    return 0;
}
"""),
    ("ej", """
%%bash
# Compila y ejecuta aquí
""", """
%%bash
gcc hola.c -o hola && ./hola
"""),
    ("code", """
revisar(lambda: bash("./hola") == "Hola, Honduras", "¡Tu primer programa compilado!", "¿Compilaste hola.c en ./hola?")
"""),
    ("md", """
### ✏️ Reto 2 · Producción total por departamento

Con una sola línea de bash (pista: `awk`), calcula la **producción total** (`produccion_qq`, columna 5) de cada departamento y guárdala en `proyecto/resultados/produccion.txt`, ordenada de mayor a menor. Formato de cada línea: `Departamento,total`.

<details><summary>💡 Pista</summary>

```bash
tail -n +2 datos/fincas.csv | awk -F, '{s[$2] += $5} END {for (d in s) print d "," s[d]}' | sort -t, -k2 -nr
```
</details>
"""),
    ("ej", """
%%bash
cd proyecto
# TU CÓDIGO AQUÍ
""", """
%%bash
cd proyecto
tail -n +2 datos/fincas.csv | awk -F, '{s[$2] += $5} END {for (d in s) print d "," s[d]}' | sort -t, -k2 -nr > resultados/produccion.txt
cat resultados/produccion.txt
"""),
    ("code", """
datos_actuales = pd.read_csv("proyecto/datos/fincas.csv")
esperado = datos_actuales.groupby("departamento").produccion_qq.sum().sort_values(ascending=False)
def _ok():
    lineas = [l.strip().rsplit(",", 1) for l in open("proyecto/resultados/produccion.txt") if l.strip()]
    valores = [int(v) for _, v in lineas]
    return ({d: int(v) for d, v in lineas} == esperado.to_dict()          # mismas sumas
            and valores == sorted(valores, reverse=True))                 # de mayor a menor
revisar(_ok, f"{esperado.index[0]} produce más: {esperado.iloc[0]} quintales.", "Revisa la suma o el orden.")
"""),
    ("md", """
---

## Resumen

- La terminal permite **automatizar** y **reproducir** el trabajo: `ls`, `cd`, `mkdir`, `cp`, `mv`, `rm`, `cat`, `head`, `grep`, `cut`, `sort`, `uniq`, `wc`…
- La **tubería** `|` y la redirección `>` / `>>` combinan comandos pequeños en análisis completos.
- **git** guarda la historia del proyecto: `init`, `add`, `commit`, `log`, `diff`, `switch`, `merge`, `push`, `pull`.
- Los lenguajes **compilados** (C) son mucho más rápidos; los **interpretados** (Python) son más cómodos. NumPy y PyTorch combinan lo mejor de ambos.

**Para seguir:** Software Carpentry en español (*La terminal de Unix*, *Control de versiones con Git*) · *Pro Git* en español: https://git-scm.com/book/es
"""),
]

if __name__ == "__main__":
    construir("03_terminal_linux", celdas)
