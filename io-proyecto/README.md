# Solver IO II

Aplicación didáctica en español para modelos de transporte (PuLP) y rutas más cortas (Floyd-Warshall).

## Instalación y ejecución

Requiere Python 3.10 o superior. Desde esta carpeta, ejecuta `python -m venv .venv` para crear un entorno con el Python disponible en `PATH`. Luego ejecuta estos comandos, uno por uno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

La aplicación incluye casos precargados de transporte, comparación de esquina noroeste, costo mínimo, Vogel y óptimo global, análisis de precios sombra, formulación matemática dinámica y un módulo Floyd-Warshall con matrices iterativas y reconstrucción de rutas. PuLP se instala con el extra `cbc` para disponer del solver CBC que resuelve el modelo lineal.

Si aparece un error de CBC al resolver, instala el solver en el mismo entorno que inicia Streamlit y vuelve a iniciar la aplicación:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "pulp[cbc]>=3.0,<4.0"
.\.venv\Scripts\python.exe -m streamlit run app.py
```
