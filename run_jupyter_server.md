# Jupyter Server auf Linux oder Windows starten (Anleitung)

Diese Anleitung beschreibt, wie du den Jupyter-Server auf einem Remote-Server oder auf einem lokalen PC startest und dich per VS Code von deinem Mac aus damit verbindest.

---

## 1. Verbindung mit Linux Remote GPU-Server

### Schritt 1: SSH-Tunnel auf dem Mac starten

Öffne ein Terminal auf deinem **Mac** und tunnel den Port `8888` von Linux:

```bash
ssh -L 8888:localhost:8888 fiete@linux
```

_(Lass dieses Terminal-Fenster im Hintergrund geöffnet)._

### Schritt 2: Jupyter-Server auf Linux starten

Stelle sicher, dass du auf dem richtigen GPU-Node bist (falls Slurm o.ä. verwendet wird), aktiviere deine virtuelle Umgebung und starte den Server:

```bash
# 1. In das Projektverzeichnis wechseln
cd /home/fiete/master-thesis

# 2. Die virtuelle Umgebung aktivieren
source .venv/bin/activate

# 3. Jupyter Server starten
jupyter lab --no-browser --port=8888 --ip=127.0.0.1
```

---

## 2. Verbindung mit lokalem Windows-PC

### Schritt 1: SSH-Tunnel auf dem Mac starten

Öffne ein Terminal auf deinem **Mac**:

```bash
ssh -L 8888:localhost:8888 fiete@192.168.0.94
```

### Schritt 2: Jupyter-Server auf Windows starten

Führe auf dem Windows-PC (oder über das SSH-Terminal) folgende Befehle aus:

```cmd
# 1. In das Projektverzeichnis wechseln
cd C:\Users\fiete\git\master

# 2. Die virtuelle Umgebung aktivieren
.venv\Scripts\activate

# 3. Jupyter Server starten
jupyter lab --no-browser --port=8888 --ip=127.0.0.1
```

---

## Schritt 3: In VS Code auf dem Mac verbinden

1. Kopiere die URL mit dem **neuesten Token** aus der Terminalausgabe des Servers (Linux oder Windows), z. B.:
   `http://127.0.0.1:8888/lab?token=bf74fa3c4e42...`
2. Öffne dein Notebook (`.ipynb`) in VS Code auf dem Mac.
3. Klicke oben rechts auf **Kernel auswählen** (Select Kernel) -> **Existierender Jupyter-Server** (Existing Jupyter Server) und füge die kopierte URL ein.
4. Wähle anschließend als Kernel **"Python (Master Thesis)"** aus.

---

## Fehlerbehebung & GPU / CUDA Support

### PyTorch meldet `CUDA verfügbar: False` oder `No GPU`

#### 1. NVIDIA-Treiber-Kompatibilität prüfen

Führe im Terminal `nvidia-smi` aus, um die unterstützte CUDA-Version des Treibers zu ermitteln (oben rechts im Output, z. B. `CUDA Version: 12.4`).
Wenn deine PyTorch-Version mit einer höheren CUDA-Version (z. B. `+cu130`) kompiliert wurde, schlägt die Initialisierung der GPU fehl.

#### 2. PyTorch >= 2.6 mit passendem CUDA-Support (z. B. 12.4) installieren

Aufgrund von Sicherheitsänderungen in Hugging Face `transformers` (CVE-2025-32434) wird mindestens **PyTorch v2.6.0** benötigt, um Gewichte im klassischen PyTorch-Format zu laden.

Führe bei CUDA- oder Version-Mismatches Folgendes in der virtuellen Umgebung aus:

```bash
# Virtuelle Umgebung aktivieren
source .venv/bin/activate

# PyTorch mit passendem CUDA-Support (z.B. cu124 oder cu121) installieren
pip install --upgrade "torch>=2.6" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

_Danach den Jupyter-Kernel neu starten._

---

## Nützliche Befehle

### Alle alten Jupyter/Python-Prozesse auf Windows killen:

```cmd
taskkill /f /im jupyter-lab.exe /im jupyter.exe /im python.exe
```

### Projektdateien synchronisieren:

Falls du neue Daten oder Skripte auf dem Mac hinzugefügt hast, kopiere sie ohne die schwere `.venv` auf die jeweiligen Systeme:

#### Nach (Linux):

```bash
rsync -avz --exclude='.venv' --exclude='.git' "/Users/fietescheel/Documents/Master Thesis/" fiete@linux:/home/fiete/master-thesis/
```

#### Nach Windows:

```bash
scp -r "/Users/fietescheel/Documents/Master Thesis/data" fiete@192.168.0.94:C:/Users/fiete/git/master/
scp -r "/Users/fietescheel/Documents/Master Thesis/results" fiete@192.168.0.94:C:/Users/fiete/git/master/
```
