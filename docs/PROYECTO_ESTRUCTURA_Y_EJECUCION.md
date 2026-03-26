# Propuesta de estructura y orden de ejecución del proyecto UEPE QA

## 1. Objetivo de la reestructuración

Esta propuesta organiza el proyecto en capas para que el flujo completo quede claro y mantenible:

1. **Discovery de URLs**
2. **Descarga de HTML**
3. **Conversión HTML -> Markdown**
4. **Construcción/consulta del motor QA**
5. **Exposición por API**
6. **Consumo desde cliente Vue**

La idea es separar el código operativo del código experimental. En los archivos actuales hay varias versiones del motor y varios pipelines de indexación que se solapan entre sí. Por eso conviene dejar una ruta principal y mover lo demás a una carpeta de laboratorio o prototipos.

---

## 2. Estructura recomendada

```text
UEPE_QA/
├── app_server/
│   ├── __init__.py
│   └── qa_api_app.py
│
├── frontend/
│   └── vue-client/
│       └── index.html
│
├── src/
│   ├── ingestion/
│   │   ├── spiders/
│   │   │   ├── discover_urls_spider.py
│   │   │   └── confluence_download_html_spider.py
│   │   └── transform/
│   │       ├── html_to_md_batch.py
│   │       └── split_urls.py
│   │
│   ├── qa/
│   │   ├── qa_engine_core.py
│   │   └── markdown_install_qa_dynamic.py
│   │
│   └── pipelines/
│       ├── indexing_pipeline.py
│       ├── pipeline_markdown_install.py
│       ├── pipeline_V2.py
│       ├── pipeline_V3.py
│       └── markdown_install_qa_haystack_ollama.py
│
├── config/
│   ├── config_batch.yaml
│   └── config_discover.yaml
│
├── data/
│   ├── discovery/
│   │   ├── urls_instalation.jsonl
│   │   ├── urls_internal.jsonl
│   │   └── urls_external.jsonl
│   │
│   ├── raw/
│   │   └── out_html/
│   │
│   ├── processed/
│   │   └── out_md/
│   │
│   └── indexes/
│
├── docs/
│   ├── README.md
│   └── README_API.md
│
├── requirements.txt
├── requirements_full_project.txt
└── .venv/
```

---

## 3. Dónde colocar cada archivo actual

### 3.1 Ingesta
- `discover_urls_spider.py` -> `src/ingestion/spiders/discover_urls_spider.py`
- `confluence_download_html_spider.py` -> `src/ingestion/spiders/confluence_download_html_spider.py`
- `html_to_md_batch.py` -> `src/ingestion/transform/html_to_md_batch.py`
- `split_urls.py` -> `src/ingestion/transform/split_urls.py`

### 3.2 Motor QA
- `qa_engine_core.py` -> `src/qa/qa_engine_core.py`
- `markdown_install_qa_dynamic.py` -> `src/qa/markdown_install_qa_dynamic.py`

### 3.3 API
- `qa_api_app.py` -> `app_server/qa_api_app.py`

### 3.4 Cliente web
- `index.html` -> `frontend/vue-client/index.html`

### 3.5 Pipelines experimentales / laboratorio
- `indexing_pipeline.py` -> `src/pipelines/indexing_pipeline.py`
- `pipeline_markdown_install.py` -> `src/pipelines/pipeline_markdown_install.py`
- `pipeline_V2.py` -> `src/pipelines/pipeline_V2.py`
- `pipeline_V3.py` -> `src/pipelines/pipeline_V3.py`
- `markdown_install_qa_haystack_ollama.py` -> `src/pipelines/markdown_install_qa_haystack_ollama.py`

### 3.6 Configuración
- `config_batch.yaml` -> `config/config_batch.yaml`
- `config_discover.yaml` -> `config/config_discover.yaml`

### 3.7 Datos generados
- `urls_instalation.jsonl` -> `data/discovery/urls_instalation.jsonl`
- `out_html/` -> `data/raw/out_html/`
- `out_md/` -> `data/processed/out_md/`

---

## 4. Decisión importante: qué es flujo principal y qué es experimental

## Flujo principal recomendado
Usa como línea principal:

- `discover_urls_spider.py`
- `confluence_download_html_spider.py`
- `html_to_md_batch.py`
- `qa_engine_core.py`
- `qa_api_app.py`
- `frontend/vue-client/index.html`

## Flujo experimental / laboratorio
Deja en `src/pipelines/` los archivos:

- `indexing_pipeline.py`
- `pipeline_markdown_install.py`
- `pipeline_V2.py`
- `pipeline_V3.py`
- `markdown_install_qa_haystack_ollama.py`
- `markdown_install_qa_dynamic.py` (si no lo usas en producción)

Estos archivos son útiles para comparar enfoques, pero no conviene mezclarlos con la ruta operativa del sistema porque representan versiones distintas del retrieval y del grounding.

---

## 5. Orden correcto de ejecución

## Paso 0. Crear entorno
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_full_project.txt
```

En Windows:
```bash
.venv\Scripts\activate
```

---

## Paso 1. Discovery de URLs
```bash
scrapy runspider src/ingestion/spiders/discover_urls_spider.py \
  -a config=config/config_discover.yaml \
  -O data/discovery/urls_instalation.jsonl \
  -s LOG_LEVEL=INFO
```

### Resultado esperado
Se genera el archivo:
- `data/discovery/urls_instalation.jsonl`

---

## Paso 2. Separar internos y externos (opcional)
```bash
python src/ingestion/transform/split_urls.py --input data/discovery/urls_instalation.jsonl
```

### Resultado esperado
Puedes generar:
- `data/discovery/urls_internal.jsonl`
- `data/discovery/urls_external.jsonl`

---

## Paso 3. Descargar HTML a disco
```bash
scrapy runspider src/ingestion/spiders/confluence_download_html_spider.py \
  -a config=config/config_batch.yaml \
  -a urls_file=data/discovery/urls_instalation.jsonl \
  -s LOG_LEVEL=INFO
```

### Resultado esperado
Se llenará:
- `data/raw/out_html/`

Cada carpeta de página debería contener:
- `page.html`
- `export_view.html`
- `url_export_view.html` (si aplica)
- `meta.json`

---

## Paso 4. Convertir HTML guardado a Markdown
```bash
python src/ingestion/transform/html_to_md_batch.py --config config/config_batch.yaml
```

### Resultado esperado
Se generará:
- `data/processed/out_md/`

Cada carpeta debe contener al menos:
- `page.md`
- `meta.json`

---

## Paso 5. Levantar la API
```bash
uvicorn app_server.qa_api_app:app --host 0.0.0.0 --port 8000 --reload
```

> Importante: usa **punto** en lugar de slash.
>
> Correcto:
> `uvicorn app_server.qa_api_app:app`
>
> Incorrecto:
> `uvicorn app_server/qa_api_app:app`

---

## Paso 6. Abrir el cliente Vue
Abre en el navegador:
- `frontend/vue-client/index.html`

La interfaz apunta a la API y usa:
- `/health`
- `/build`
- `/query`

---

## 6. Flujo funcional resumido

```text
discover_urls_spider.py
        ↓
urls_instalation.jsonl
        ↓
confluence_download_html_spider.py
        ↓
out_html/
        ↓
html_to_md_batch.py
        ↓
out_md/
        ↓
qa_engine_core.py
        ↓
qa_api_app.py
        ↓
frontend/vue-client/index.html
```

---

## 7. Relación entre configuraciones y datos

## `config/config_discover.yaml`
Se usa para la etapa de descubrimiento.
Define:
- URL inicial
- dominios permitidos
- profundidad
- regex allow/deny
- manejo de externos
- network throttling

## `config/config_batch.yaml`
Se usa para:
- descarga HTML
- definición de `output_dir`
- definición de `md_output_dir`
- selectores del contenido principal
- fallback `export_view`
- parámetros de red y headers

---

## 8. Observaciones técnicas importantes

### 8.1 `qa_api_app.py`
Este archivo ya está diseñado para exponer:
- `GET /health`
- `POST /build`
- `POST /query`

Por tanto sí debe quedarse en una carpeta de servidor y no mezclado con scraping.

### 8.2 `qa_engine_core.py`
Es el motor principal recomendado para la API porque:
- ya integra búsqueda y respuesta
- ya maneja Haystack opcional
- ya maneja embeddings con Ollama
- ya prepara metadatos por sección

### 8.3 `index.html`
Tu cliente Vue es correcto como frontend estático simple.
No necesita build con npm si lo vas a usar por CDN.

### 8.4 Archivos de pipeline
Los scripts:
- `indexing_pipeline.py`
- `pipeline_markdown_install.py`
- `pipeline_V2.py`
- `pipeline_V3.py`

deben considerarse **prototipos comparativos** o **laboratorio**, no la ruta de producción principal.

---

## 9. Recomendación final de operación

Si quieres estabilidad y orden:

### Mantén como proyecto principal:
- `src/ingestion`
- `src/qa`
- `app_server`
- `frontend`
- `config`
- `data`

### Mantén aparte como laboratorio:
- `src/pipelines`

Esto permite:
- no mezclar experimentos con producción
- no romper la API cuando pruebes otro retriever
- mantener trazable el flujo discovery -> html -> md -> qa -> api

---

## 10. Comando final recomendado de operación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_full_project.txt

scrapy runspider src/ingestion/spiders/discover_urls_spider.py -a config=config/config_discover.yaml -O data/discovery/urls_instalation.jsonl -s LOG_LEVEL=INFO

scrapy runspider src/ingestion/spiders/confluence_download_html_spider.py -a config=config/config_batch.yaml -a urls_file=data/discovery/urls_instalation.jsonl -s LOG_LEVEL=INFO

python src/ingestion/transform/html_to_md_batch.py --config config/config_batch.yaml

uvicorn app_server.qa_api_app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 11. Siguiente mejora recomendada

Como siguiente paso, convendría crear tres archivos adicionales:

- `src/qa/config.py`
- `src/qa/models.py`
- `src/qa/ollama_client.py`

para dividir `qa_engine_core.py` y dejar el motor más mantenible.

