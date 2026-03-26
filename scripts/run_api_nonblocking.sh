#!/usr/bin/env bash
set -euo pipefail
AUTO_BUILD_ON_STARTUP=false uvicorn app_server.qa_api_app_nonblocking:app --host 0.0.0.0 --port 8010 --reload
