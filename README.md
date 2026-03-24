# UEPE5D Confluence Scraper (2 fases: discover -> scrape)

Este repositorio contiene el ecosistema completo para el Copiloto de Soporte Técnico de UEPE 5.2. El sistema es una arquitectura RAG (Retrieval-Augmented Generation) Full-Stack que extrae documentación oficial de Confluence, la indexa en una base de datos vectorial (Qdrant) y sirve una interfaz de chat moderna (React) impulsada por un LLM local.

El proyecto se compone de 3 capas principales:

1. Pipeline de Ingesta (Scraping & OCR)
2. Backend RAG (FastAPI + Haystack 2.0)
3. Frontend de Usuario (React + Vite)

Este paquete hace:
1) Descubrimiento de URLs dentro del Space UEPE5D (Confluence)
2) Scraping de esas URLs guardando HTML + Markdown + imágenes por página

## Archivos
- config.yaml
- discover_urls_spider.py
- scrape_site_spider.py
- requirements.txt

---

## 🚀 Fase 1: Entorno y Extracción de Datos (Scraping)

Esta fase se encarga de descubrir las URLs del espacio de Confluence, descargar los archivos HTML y convertirlos a Markdown listo para vectorizar.

1. Preparar el entorno de Python

Abre una terminal en la raíz del proyecto y ejecuta:

En Windows:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Ejecuta el discovery (fase 1)

```bash
scrapy runspider discover_urls_spider.py -a config=config_discover.yaml -O urls_instalation.jsonl -s LOG_LEVEL=INFO
```

## 2) Separa internos y externos (opcional)

```bash
python split_urls.py --input urls.jsonl
```

## 3) Descarga de HTMLs a disco (Batch)

```bash
scrapy runspider confluence_download_html_spider.py -a config=config_batch.yaml -a urls_file=urls_instalation.jsonl -s LOG_LEVEL=INFO
```

# Etapa 4 (offline): Convertir HTML guardado -> Markdown.

- Lee carpetas generadas por el spider (meta.json + page.html/export_view.html),
- elige la mejor fuente (export_view.html > url_export_view.html > page.html),
- extrae SOLO el contenido principal (AkMainContent / content-body),
- y genera page.md con:
- Título garantizado (# <title>)
- Encabezados/subtítulos
- Párrafos, listas, quotes, panels/notes
- Tablas (HTML) + extracción de code dentro de celdas sin romper la tabla
- Code blocks (con language-xxx o inferencia: json/yaml/shell/hcl/python/sql/xml)

Este script procesa las carpetas generadas, limpia el HTML y genera archivos `page.md` con títulos, código formateado y tablas.

```bash
python html_to_md_batch.py --config config_batch.yaml
```

## Preparar el Modelo de IA Local

Abre una terminal y descarga el modelo Llama 3.2 que utiliza el backend:

```bash
ollama run llama3.2
```

# 🧠 Fase 2: Indexación y Backend RAG

Una vez que tienes los archivos Markdown generados en la carpeta `out_md`, es momento de crear la base de datos vectorial y levantar la API.

## 1. Indexar los documentos en Qdrant

Este paso lee los archivos `.md`, los divide en fragmentos (chunks) y los guarda físicamente en la carpeta local `./qdrant_db`.

```bash
python indexing_pipeline.py
```

Nota: Solo necesitas ejecutar esto una vez, o cuando agregues nueva documentación a `out_md`.

## 2. Levantar la API de Soporte (FastAPI)

```bash
python app.py
```

El servidor quedará ejecutándose en `http://0.0.0.0:8000`. Puedes probar los endpoints directamente en `http://127.0.0.1:8000/docs`.

# 💻 Fase 3: Interfaz de Usuario (Frontend)

Para interactuar con el Copiloto de forma amigable, utilizaremos la interfaz construida en React y Tailwind CSS.

## 1. Instalar dependencias

Abre una nueva pestaña/ventana en tu terminal (deja el backend corriendo en la otra), navega a la carpeta del frontend e instala los paquetes:

```bash
cd frontend-uepe
npm install
```

## 2. Ejecutar la aplicación web

```bash
npm run dev
```

¡Listo! Abre tu navegador en la URL que te indique la terminal (usualmente `http://localhost:5173`) para comenzar a chatear con el Asistente Técnico de UEPE 5.2.

# 🛠️ Estructura del Proyecto

```bash
LLMUEPE5.2/
├── .venv/                      # Entorno virtual de Python
├── frontend-uepe/              # 🎨 Interfaz visual en React/Vite
├── out_html/                   # Descargas crudas del scraper
├── out_md/                     # Archivos Markdown procesados y listos
├── qdrant_db/                  # 💾 Base de datos vectorial persistente
├── app.py                      # ⚙️ API FastAPI (Motor de Búsqueda RAG)
├── indexing_pipeline.py        # Script para vectorizar los Markdown
├── requirements.txt            # Dependencias de Python
└── ... (Scripts de Scraping)
```

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
