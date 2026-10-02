#!/usr/bin/env bash
# Eenmalig installeren en starten op macOS (Apple Silicon) of Linux.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Eerste keer: virtuele omgeving maken en pakketten installeren (duurt een paar minuten)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
exec .venv/bin/python -m app "$@"
