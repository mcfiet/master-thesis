# Jupyter Server auf Windows starten (Anleitung)

Diese Anleitung beschreibt, wie du den Jupyter-Server auf deinem Windows-PC startest und dich von deinem Mac aus per VS Code damit verbindest.

---

## Schritt 1: SSH-Tunnel auf dem Mac starten

Öffne ein Terminal auf deinem **Mac** und starte den SSH-Tunnel. Dieser leitet Port `8888` vom Windows-PC direkt an deinen Mac weiter:

```bash
ssh -L 8888:localhost:8888 fiete@192.168.0.94
```
*(Lass dieses Terminal-Fenster im Hintergrund geöffnet).*

---

## Schritt 2: Jupyter-Server auf Windows starten

Sobald du über das SSH-Terminal auf dem Windows-PC eingeloggt bist (oder in einem separaten Terminal auf dem Windows-PC direkt), führe folgende Befehle aus:

```cmd
# 1. In das Projektverzeichnis wechseln
cd C:\Users\fiete\git\master

# 2. Die virtuelle Umgebung aktivieren
.venv\Scripts\activate

# 3. Jupyter Server starten
jupyter lab --no-browser --port=8888 --ServerApp.allow_origin="*"
```

---

## Schritt 3: In VS Code auf dem Mac verbinden

1. Kopiere die URL mit dem **neuesten Token** aus der Terminalausgabe auf Windows, z. B.:
   `http://127.0.0.1:8888/lab?token=bf74fa3c4e42...`
2. Öffne dein Notebook (`.ipynb`) in VS Code auf dem Mac.
3. Klicke oben rechts auf **Kernel auswählen** (Select Kernel) -> **Existierender Jupyter-Server** (Existing Jupyter Server) und füge die kopierte URL ein.
4. Wähle anschließend als Kernel **"Python (Master Thesis)"** aus.

---

## Nützliche Befehle zur Fehlerbehebung

### Alle alten Jupyter/Python-Prozesse auf Windows killen:
Sollte sich ein Server aufhängen oder Port `8888` blockiert sein, führe dies auf Windows aus:
```cmd
taskkill /f /im jupyter-lab.exe /im jupyter.exe /im python.exe
```

### Projektdateien vom Mac auf Windows synchronisieren:
Falls du neue Daten oder Skripte auf dem Mac hinzugefügt hast, kopiere sie ohne die schwere `.venv` rüber:
```bash
scp -r "/Users/fietescheel/Documents/Master Thesis/data" fiete@192.168.0.94:C:/Users/fiete/git/master/
scp -r "/Users/fietescheel/Documents/Master Thesis/results" fiete@192.168.0.94:C:/Users/fiete/git/master/
```
