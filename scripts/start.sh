#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m uvicorn chest_agent.app:app --host "${CHEST_HOST:-127.0.0.1}" --port "${CHEST_PORT:-7860}"
