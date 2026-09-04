# Minimales, leichtgewichtiges Dockerfile für die Experten-Evaluation Web-App
# Benötigt nur Python 3 Standardbibliothek (Image-Größe < 50MB)
FROM python:3.11-alpine

WORKDIR /app

# Kopiere die Web-App, Daten und Server-Skript
COPY scripts/evaluation/serve_expert_app.py ./scripts/evaluation/serve_expert_app.py
COPY scripts/evaluation/web_ui/ ./scripts/evaluation/web_ui/
COPY data/expert_eval/blinded_items.json ./data/expert_eval/blinded_items.json

# Verzeichnis für die Ergebnisse erstellen
RUN mkdir -p results/expert_eval

EXPOSE 8085

# Server starten
CMD ["python", "scripts/evaluation/serve_expert_app.py", "--host", "0.0.0.0", "--port", "8085"]
