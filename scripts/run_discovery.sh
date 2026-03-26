#!/usr/bin/env bash
set -euo pipefail
scrapy runspider ingestion/spiders/discover_urls_spider.py   -a config=config/ingestion/config_discover.yaml   -O data/discovery/urls_instalation.jsonl   -s LOG_LEVEL=INFO
