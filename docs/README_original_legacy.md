# UEPE5D Confluence Scraper (2 fases: discover -> scrape)

Este paquete hace:
1) Descubrimiento de URLs dentro del Space UEPE5D (Confluence)
2) Scraping de esas URLs guardando HTML + Markdown + imágenes por página

## Archivos
- config.yaml
- discover_urls_spider.py
- scrape_site_spider.py
- requirements.txt

---

## 1) Preparar entorno


```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt


## 1) Ejecuta el discovery (fase 1)

scrapy runspider discover_urls_spider.py -a config=config_discover.yaml -O urls_instalation.jsonl -s LOG_LEVEL=INFO

## 2) Separa internos y externos (opcional)
python split_urls.py --input urls.jsonl

## 3) Descarga de htmls a disco

scrapy runspider confluence_download_html_spider.py \
       -a config=config_batch.yaml \
       -a urls_file=urls_instalation.jsonl \
       -s LOG_LEVEL=INFO


#
# Etapa 4 (offline): Convertir HTML guardado -> Markdown.
# Lee carpetas generadas por el spider (meta.json + page.html/export_view.html),
# elige la mejor fuente (export_view.html > url_export_view.html > page.html),
# extrae SOLO el contenido principal (AkMainContent / content-body),
# y genera page.md con:
# - Título garantizado (# <title>)
# - Encabezados/subtítulos
# - Párrafos, listas, quotes, panels/notes
# - Tablas (HTML) + extracción de code dentro de celdas sin romper la tabla
# - Code blocks (con language-xxx o inferencia: json/yaml/shell/hcl/python/sql/xml)
#
# Run:
  
  python html_to_md_batch.py --config config_batch.yaml
  
  uvicorn app_server/qa_api_app:app --host 0.0.0.0 --port 8000 --reload
  
  



```
## instalar ollama

https://ollama.com/download

## continar aqui con la docuntacion

<!--
## 3) Scrapea SOLO internos (recomendado)
scrapy runspider scrape_site_spider.py -a config=config.yaml -a urls_file=internal_urls.jsonl -O scraped_pages.jsonl


scrapy runspider scrape_site_spider.py \
  -a config=config.yaml \
  -a urls_file=urls_instalation.jsonl \
  -O scraped_pages.jsonl \
  -s LOG_FILE=scrape.log \
  -s LOG_LEVEL=INFO

export MILVUS_URI="https://TU_ENDPOINT"
export MILVUS_TOKEN="TU_TOKEN"

python up_milvus_blocks.py --out_dir ./out_UEPE5D --collection knowledge_base --drop

scrapy runspider scrape_site_spider.py \
  -a config=config_v2.yaml \
  -a urls_file=urls_instalation.jsonl \
  -O scraped_pages.jsonl \
  -s LOG_FILE=scrape.log \
  -s LOG_LEVEL=INFO
-->

```


```