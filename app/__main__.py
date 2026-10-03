"""Start de app: python -m app [--port 8000] [--no-browser]"""
import argparse
import json
import socket
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


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _is_match_analyzer(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/version", timeout=2) as r:
            return "version" in json.load(r)
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Match Analyzer")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    port = args.port
    if not _port_free(port):
        url = f"http://127.0.0.1:{port}"
        if _is_match_analyzer(url):  # de app draait al (bijv. in een ander Terminal-venster)
            print(f"Match Analyzer draait al op {url}; ik open de browser.")
            if not args.no_browser:
                webbrowser.open(url)
            return
        port = next((p for p in range(port + 1, port + 20) if _port_free(p)), port)
        print(f"Poort {args.port} is bezet door een ander programma; ik gebruik poort {port}.")
    url = f"http://127.0.0.1:{port}"
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url,), daemon=True).start()
    print(f"Match Analyzer draait op {url}  (stoppen: Ctrl+C)")
    uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
