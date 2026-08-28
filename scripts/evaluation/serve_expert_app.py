#!/usr/bin/env python3
"""
scripts/evaluation/serve_expert_app.py

Startet einen schlanken lokalen Web-Server für die Experten-Evaluation:
- Lädt automatisch die verblindeten Items aus data/expert_eval/blinded_items.json
- Speichert Ratings in Echtzeit in results/expert_eval/expert_eval_ratings.json
- Öffnet den Browser unter http://localhost:8080
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


PORT = 8080
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "expert_eval")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "expert_eval")
UI_DIR = os.path.join(os.path.dirname(__file__), "web_ui")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


class ExpertEvalHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve blinded_items.json directly
        if path == "/data/expert_eval/blinded_items.json":
            return os.path.join(DATA_DIR, "blinded_items.json")
        # Serve web_ui files
        if path == "/" or path == "/index.html":
            return os.path.join(UI_DIR, "index.html")
        return super().translate_path(path)

    def do_POST(self):
        if self.path == "/api/save_rating":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                ratings_json = json.loads(post_data.decode("utf-8"))
                save_path = os.path.join(RESULTS_DIR, "expert_eval_ratings.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(ratings_json, f, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "saved"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    os.chdir(ROOT_DIR)
    blinded_file = os.path.join(DATA_DIR, "blinded_items.json")
    if not os.path.exists(blinded_file):
        print(f"[HINWEIS] {blinded_file} nicht gefunden. Starte build_expert_evaluation_set.py...")
        os.system(f"{sys.executable} scripts/evaluation/build_expert_evaluation_set.py")

    port = 8085
    httpd = None
    while port < 8100:
        try:
            server_address = ("", port)
            httpd = HTTPServer(server_address, ExpertEvalHandler)
            break
        except OSError:
            port += 1

    if not httpd:
        print("Kein freier Port gefunden.")
        return

    url = f"http://localhost:{port}"
    print(f"\n========================================================")
    print(f" Experten-Evaluations-Server läuft auf: {url}")
    print(f" Ergebnisse werden gespeichert in: results/expert_eval/expert_eval_ratings.json")
    print(f" Beenden mit Strg+C")
    print(f"========================================================\n")

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer beendet.")


if __name__ == "__main__":
    main()
