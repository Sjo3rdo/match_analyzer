#!/usr/bin/env bash
# Installeren en starten op macOS (Apple Silicon) of Linux.
# De eerste keer wordt alles geïnstalleerd; na een update (git pull) alleen wat er nieuw is.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Eerste keer: virtuele omgeving maken en pakketten installeren (duurt een paar minuten)..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
fi
want=$(.venv/bin/python -c 'import hashlib; print(hashlib.sha1(open("requirements.txt", "rb").read()).hexdigest())')
have=$(cat .venv/.requirements.sha 2>/dev/null || true)
if [ "$want" != "$have" ]; then
  echo "Pakketten installeren of bijwerken..."
  .venv/bin/pip install -r requirements.txt
  echo "$want" > .venv/.requirements.sha
fi
exec .venv/bin/python -m app "$@"
