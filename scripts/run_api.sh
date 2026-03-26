#!/usr/bin/env bash
set -euo pipefail
uvicorn app_server.qa_api_app:app --host 0.0.0.0 --port 8020 --reload
