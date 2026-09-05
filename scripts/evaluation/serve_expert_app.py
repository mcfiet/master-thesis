#!/usr/bin/env python3
"""
scripts/evaluation/serve_expert_app.py

Startet den Web-Server für die Experten-Evaluation:
1. Experten-Interface: http://<host>:8085/ (verblindet, keine Modellnamen)
2. Admin-Dashboard:    http://<host>:8085/admin (Passwort-geschützt: test123#)
   - Zeigt Modellherkunft, R_style, R_sem, Flesch, LIX und Live-Bewertungen
   - Live-Statistiken und Side-by-Side Textvergleich
3. Speichert Ratings in Echtzeit in results/expert_eval/expert_eval_ratings.json
"""

import os
import sys
import json
import argparse
import urllib.parse
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler


ADMIN_PASSWORD = "test123#"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
UI_DIR = os.path.join(os.path.dirname(__file__), "web_ui")


def get_data_dir() -> str:
    for candidate in [
        os.path.join(ROOT_DIR, "data", "expert_eval"),
        os.path.join(ROOT_DIR, "data2", "expert_eval"),
    ]:
        if os.path.exists(os.path.join(candidate, "blinded_items.json")):
            return candidate
    return os.path.join(ROOT_DIR, "data", "expert_eval")


def get_results_dir() -> str:
    for candidate in [
        os.path.join(ROOT_DIR, "results", "expert_eval"),
        os.path.join(ROOT_DIR, "results2", "expert_eval"),
    ]:
        if os.path.exists(candidate):
            return candidate
    res = os.path.join(ROOT_DIR, "results", "expert_eval")
    os.makedirs(res, exist_ok=True)
    return res


class ExpertEvalHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _is_admin_authorized(self, parsed_url) -> bool:
        query = urllib.parse.parse_qs(parsed_url.query)
        if "token" in query and query["token"][0] == ADMIN_PASSWORD:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and auth_header.split("Bearer ")[1].strip() == ADMIN_PASSWORD:
            return True
        return False

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path

        # 0. HTML Pages
        if clean_path in ["/", "/index.html"]:
            index_path = os.path.join(UI_DIR, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(index_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        elif clean_path in ["/admin", "/admin.html"]:
            admin_path = os.path.join(UI_DIR, "admin.html")
            if os.path.exists(admin_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with open(admin_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 1. API: Blinded Items Data
        elif clean_path in ["/api/items", "/blinded_items.json", "/data/expert_eval/blinded_items.json"]:
            blinded_path = os.path.join(get_data_dir(), "blinded_items.json")
            if os.path.exists(blinded_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                with open(blinded_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "blinded_items.json nicht gefunden"}')
            return

        # 2. API: Server Status
        elif clean_path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            ratings_path = os.path.join(get_results_dir(), "expert_eval_ratings.json")
            ratings_count = 0
            if os.path.exists(ratings_path):
                try:
                    with open(ratings_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                        ratings_count = len(d.get("ratings", d))
                except Exception:
                    pass
            self.wfile.write(json.dumps({
                "status": "online",
                "ratings_count": ratings_count,
                "total_items": 50
            }).encode("utf-8"))
            return

        # 3. API: Live Ratings
        elif clean_path == "/api/ratings":
            ratings_path = os.path.join(get_results_dir(), "expert_eval_ratings.json")
            if os.path.exists(ratings_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with open(ratings_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")
            return

        # 4. API: Admin Data (Passwort-geschützt)
        elif clean_path == "/api/admin_data":
            if not self._is_admin_authorized(parsed):
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Unauthorized"}).encode("utf-8"))
                return

            secret_key_path = os.path.join(get_data_dir(), "secret_key_mapping.json")
            ratings_path = os.path.join(get_results_dir(), "expert_eval_ratings.json")

            items_list = []
            if os.path.exists(secret_key_path):
                with open(secret_key_path, "r", encoding="utf-8") as f:
                    secret_dict = json.load(f)
                    items_list = list(secret_dict.values())

            live_ratings = {}
            if os.path.exists(ratings_path):
                try:
                    with open(ratings_path, "r", encoding="utf-8") as f:
                        live_ratings = json.load(f)
                except Exception:
                    pass

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "items": items_list,
                "ratings": live_ratings
            }, ensure_ascii=False).encode("utf-8"))
            return

        # 5. API: Export Combined Master CSV (Passwort-geschützt)
        elif clean_path == "/api/export_master_csv":
            if not self._is_admin_authorized(parsed):
                self.send_response(401)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Unauthorized")
                return

            master_csv_path = os.path.join(get_data_dir(), "expert_study_master_table.csv")
            if os.path.exists(master_csv_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="expert_study_master_table.csv"')
                self.end_headers()
                with open(master_csv_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
            return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/save_rating":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                ratings_json = json.loads(post_data.decode("utf-8"))
                save_path = os.path.join(get_results_dir(), "expert_eval_ratings.json")
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

        elif self.path.startswith("/api/admin/reset_ratings"):
            parsed = urllib.parse.urlparse(self.path)
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

            is_auth = self._is_admin_authorized(parsed)
            if not is_auth:
                try:
                    body = json.loads(post_data.decode("utf-8"))
                    if body.get("password") == ADMIN_PASSWORD:
                        is_auth = True
                except Exception:
                    pass

            if not is_auth:
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Falsches Admin-Passwort"}')
                return

            # Erstelle automatisches Backup und leere die Datei
            save_path = os.path.join(get_results_dir(), "expert_eval_ratings.json")
            if os.path.exists(save_path):
                import time, shutil
                backup_path = os.path.join(get_results_dir(), f"expert_eval_ratings_backup_{int(time.time())}.json")
                try:
                    shutil.copy2(save_path, backup_path)
                except Exception:
                    pass

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({}, f)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "reset_success"}')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Startet den Server für die Experten-Evaluation.")
    parser.add_argument("--host", default="0.0.0.0", help="Host-Adresse (z.B. 0.0.0.0 oder 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8085, help="Port (Standard: 8085)")
    parser.add_argument("--no-browser", action="store_true", help="Browser nicht automatisch öffnen")
    args = parser.parse_args()

    os.chdir(ROOT_DIR)
    blinded_file = os.path.join(get_data_dir(), "blinded_items.json")
    if not os.path.exists(blinded_file):
        print(f"[HINWEIS] {blinded_file} nicht gefunden. Bitte zuerst build_expert_evaluation_set.py ausführen!")

    server_port = args.port
    max_retries = 10
    httpd = None

    for attempt in range(max_retries):
        try:
            server_address = (args.host, server_port)
            HTTPServer.allow_reuse_address = True
            httpd = HTTPServer(server_address, ExpertEvalHandler)
            break
        except OSError as e:
            if e.errno == 48 or "Address already in use" in str(e):
                print(f"[Port {server_port} belegt, probiere Port {server_port + 1}...]")
                server_port += 1
            else:
                raise e

    if httpd is None:
        print(f"[FEHLER] Kein freier Port im Bereich {args.port}-{args.port + max_retries} gefunden.")
        return

    display_host = "localhost" if args.host in ["0.0.0.0", "127.0.0.1", ""] else args.host
    url = f"http://{display_host}:{server_port}"
    admin_url = f"http://{display_host}:{server_port}/admin"

    print(f"\n========================================================")
    print(f" Experten-App (Verblindet):  {url}")
    print(f" Admin-Dashboard:            {admin_url} (Passwort: {ADMIN_PASSWORD})")
    print(f" Listening on: {args.host}:{args.port}")
    print(f" Live-Speicherung in: results/expert_eval/expert_eval_ratings.json")
    print(f" Beenden mit Strg+C")
    print(f"========================================================\n")

    if not args.no_browser and args.host in ["127.0.0.1", "localhost"]:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer ordnungsgemäß beendet.")


if __name__ == "__main__":
    main()
