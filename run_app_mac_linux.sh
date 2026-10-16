#!/usr/bin/env bash
# One-command launcher for the web dashboard (macOS / Linux):  bash run_app_mac_linux.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
python app.py
