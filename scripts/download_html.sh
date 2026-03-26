#!/usr/bin/env bash
set -euo pipefail
scrapy runspider ingestion/spiders/confluence_download_html_spider.py   -a config=config/ingestion/config_batch.yaml   -a urls_file=data/discovery/urls_instalation.jsonl   -s LOG_LEVEL=INFO
