"""Launch the local web dashboard:   python app.py

Opens http://127.0.0.1:5000 in your browser. Everything stays on your computer
(no login, no internet, no database). `python main.py` still works exactly as before.
"""
from __future__ import annotations
import argparse
import threading
import webbrowser

from webapp.server import app


def main():
    p = argparse.ArgumentParser(description="AI Energy Forecasting - local web dashboard")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    a = p.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"\n  ⚡ Energy Forecasting dashboard running at {url}\n  Press Ctrl+C to stop.\n")
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
