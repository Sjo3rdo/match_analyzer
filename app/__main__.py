"""Start de app: python -m app [--port 8000] [--no-browser]"""
import argparse
import threading
import time
import urllib.request
import webbrowser

import uvicorn


def _open_when_ready(url: str, timeout: float = 60.0) -> None:
    """Wacht tot de server antwoordt en open dan pas de browser (anders zie je een lege pagina)."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(url + "/api/pitch", timeout=2):
                break
        except Exception:
            time.sleep(0.3)
    webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description="Match Analyzer")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    print(f"Match Analyzer draait op {url}  (stoppen: Ctrl+C)")
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
