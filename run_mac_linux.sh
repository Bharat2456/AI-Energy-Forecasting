#!/usr/bin/env bash
# One-command setup + run for macOS / Linux:  bash run_mac_linux.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
