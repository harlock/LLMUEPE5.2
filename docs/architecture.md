# Estructura propuesta

## Capa de ingesta
- `ingestion/spiders/discover_urls_spider.py`
- `ingestion/spiders/confluence_download_html_spider.py`
- `ingestion/transform/html_to_md_batch.py`
- `ingestion/transform/split_urls.py`

## Capa de datos
- `data/discovery`
- `data/raw/confluence_html`
- `data/raw/out_md`
- `data/processed/indexes`

## Capa de QA
- `core/qa_engine_core.py`

## Capa API
- `app_server/qa_api_app.py`
- `app_server/qa_api_app_nonblocking.py`

## Capa frontend
- `frontend/vue-client/index.html`

## Capa experimental
- `experiments/`
