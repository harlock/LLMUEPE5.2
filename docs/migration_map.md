# Mapa de migración

- `discover_urls_spider.py` -> `ingestion/spiders/discover_urls_spider.py`
- `confluence_download_html_spider.py` -> `ingestion/spiders/confluence_download_html_spider.py`
- `html_to_md_batch.py` -> `ingestion/transform/html_to_md_batch.py`
- `split_urls.py` -> `ingestion/transform/split_urls.py`
- `qa_engine_core.py` -> `core/qa_engine_core.py`
- `qa_api_app.py` -> `app_server/qa_api_app.py`
- `qa_api_app_nonblocking.py` -> `app_server/qa_api_app_nonblocking.py`
- `index.html` -> `frontend/vue-client/index.html`
- `config_discover.yaml` -> `config/ingestion/config_discover.yaml`
- `config_batch.yaml` -> `config/ingestion/config_batch.yaml`
- `urls_instalation.jsonl` -> `data/discovery/urls_instalation.jsonl`
- pipelines alternos -> `experiments/`
