#!/usr/bin/env bash
set -euo pipefail
python ingestion/transform/split_urls.py   --input data/discovery/urls_instalation.jsonl   --output-dir data/discovery
