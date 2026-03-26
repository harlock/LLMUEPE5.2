#!/usr/bin/env bash
set -euo pipefail
python ingestion/transform/html_to_md_batch.py --config config/ingestion/config_batch.yaml
