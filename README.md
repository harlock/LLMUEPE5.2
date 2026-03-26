# UEPE Project Structured

Estructura física propuesta para tu proyecto completo, basada en los archivos y comandos que ya ejecutas hoy.

## Qué incluye

- `ingestion/spiders`: discovery y descarga de HTML desde Confluence
- `ingestion/transform`: utilidades offline como `html_to_md_batch.py` y `split_urls.py`
- `core`: motor QA principal (`qa_engine_core.py`)
- `app_server`: API FastAPI para `/health`, `/build` y `/query`
- `frontend/vue-client`: cliente Vue/Tailwind
- `experiments`: scripts alternos, pipelines y versiones históricas
- `data`: descubrimiento, HTML, Markdown e índices

## Crear entorno

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Orden de ejecución recomendado

### 1) Discovery
```bash
bash scripts/run_discovery.sh
```

### 2) Separar internos y externos
```bash
bash scripts/split_urls.sh
```

### 3) Descargar HTML
```bash
bash scripts/download_html.sh
```

### 4) Convertir HTML -> Markdown
```bash
bash scripts/html_to_md.sh
```

### 5) Levantar API
```bash
bash scripts/run_api_nonblocking.sh
```

o, si quieres la versión original:

```bash
bash scripts/run_api.sh
```

## Nota sobre Uvicorn

Usa el módulo con punto, no con slash:

```bash
uvicorn app_server.qa_api_app:app --host 0.0.0.0 --port 8000 --reload
```

## Archivo de referencia
Consulta `docs/PROYECTO_ESTRUCTURA_Y_EJECUCION.md` para la explicación completa.
