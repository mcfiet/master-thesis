import json
from pathlib import Path

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Vergleich: Normales vs. Neues MixUp-Verfahren\n",
                    "\n",
                    "Dieses Notebook vergleicht das normale (standardmäßige) MixUp-Verfahren mit unserer neuen (gleichverteilten) Methode.\n",
                    "\n",
                    "Beide Methoden schneiden **zusammenhängende Blöcke** aus den Texten aus und mischen sie. Der einzige Unterschied liegt darin, wie die Blockgrößen (und damit das Target) bestimmt werden:\n",
                    "\n",
                    "1. **Normal (Marc's Original):**\n",
                    "   * Schneidet zufällige Start/End-Bereiche aus beiden Texten völlig unabhängig aus.\n",
                    "   * Das Target wird über das Verhältnis der Zeichenanzahl berechnet.\n",
                    "2. **Neu (Gleichverteilt):**\n",
                    "   * Zieht zuerst ein gleichverteiltes Target $\\lambda \\sim U(0.0, 1.0)$ vorab.\n",
                    "   * Berechnet die exakte Anzahl benötigter Sätze ($num\\_ls$ und $num\\_as$) für eine dynamische Gesamtgröße $N$ (zwischen 8 und 15 Sätzen).\n",
                    "   * Schneidet zusammenhängende Blöcke dieser Längen aus den Artikeln aus."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import random\n",
                    "import numpy as np\n",
                    "import torch\n",
                    "from torch.utils.data import Dataset, DataLoader\n",
                    "import spacy\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from tqdm.notebook import tqdm\n",
                    "import os\n",
                    "\n",
                    "# Configuration\n",
                    "CSV_PATH = \"../results/information_loss_analysis_cleaned.csv\"\n",
                    "MIN_SIM = 0.8\n",
                    "MAX_SIM = 0.98\n",
                    "\n",
                    "print(\"Libraries imported successfully.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Daten laden und Spacy-Sentencizer vorbereiten"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_csv(CSV_PATH)\n",
                    "mask = (df[\"semantic_similarity_8192\"] >= MIN_SIM) & (df[\"semantic_similarity_8192\"] <= MAX_SIM)\n",
                    "df_filtered = df[mask].dropna(subset=[\"ls_text\", \"as_text\"])\n",
                    "print(f\"Gefundene Artikelpaare: {len(df_filtered)}\")\n",
                    "\n",
                    "# Spacy für schnelles Satz-Splitting (Sentencizer) vorbereiten\n",
                    "nlp = spacy.blank(\"de\")\n",
                    "nlp.add_pipe(\"sentencizer\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Hilfsfunktion für zusammenhängende Slices"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def get_contiguous_slice(sentences, k):\n",
                    "    \"\"\"Schneidet einen zusammenhängenden Block von k Sätzen aus.\n",
                    "    Falls der Text zu kurz ist, wird er ganz zurückgegeben.\"\"\"\n",
                    "    num_sents = len(sentences)\n",
                    "    if num_sents == 0 or k <= 0:\n",
                    "        return []\n",
                    "    if num_sents <= k:\n",
                    "        return list(sentences)\n",
                    "    else:\n",
                    "        start = random.randint(0, num_sents - k)\n",
                    "        return sentences[start : start + k]"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Dataset Klassen definieren"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class NormalMixupDataset(Dataset):\n",
                    "    def __init__(self, df, nlp_sentencizer):\n",
                    "        self.ls_data = []\n",
                    "        self.as_data = []\n",
                    "        for _, row in tqdm(df.iterrows(), total=len(df), desc=\"Segmentiere (Normal)\"):\n",
                    "            ls_sents = [s.text.strip() for s in nlp_sentencizer(str(row[\"ls_text\"])).sents if s.text.strip()]\n",
                    "            as_sents = [s.text.strip() for s in nlp_sentencizer(str(row[\"as_text\"])).sents if s.text.strip()]\n",
                    "            self.ls_data.append(ls_sents)\n",
                    "            self.as_data.append(as_sents)\n",
                    "            \n",
                    "    def __len__(self):\n",
                    "        return len(self.ls_data)\n",
                    "        \n",
                    "    def __getitem__(self, idx):\n",
                    "        leichte_saetze = self.ls_data[idx]\n",
                    "        alltags_saetze = self.as_data[idx]\n",
                    "        \n",
                    "        num_leicht = len(leichte_saetze)\n",
                    "        num_alltag = len(alltags_saetze)\n",
                    "        \n",
                    "        if num_leicht == 0 or num_alltag == 0:\n",
                    "            return \"\", 0.5\n",
                    "            \n",
                    "        # Marc's Originale Logik (Zufällige Slices)\n",
                    "        start_leichte_saetze, ende_leichte_saetze = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])\n",
                    "        sample_leicht = leichte_saetze[start_leichte_saetze:ende_leichte_saetze]\n",
                    "        \n",
                    "        start_alltags_saetze, ende_alltags_saetze = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])\n",
                    "        sample_alltag = alltags_saetze[start_alltags_saetze:ende_alltags_saetze]\n",
                    "        \n",
                    "        kompletter_absatz = sample_leicht + sample_alltag\n",
                    "        random.shuffle(kompletter_absatz)\n",
                    "        \n",
                    "        str_sample_leicht = ''.join(sample_leicht)\n",
                    "        str_sample_alltag = ''.join(sample_alltag)\n",
                    "        len_sample_leicht = len(str_sample_leicht)\n",
                    "        len_sample_alltag = len(str_sample_alltag)\n",
                    "        \n",
                    "        total_len = len_sample_leicht + len_sample_alltag\n",
                    "        regression_target = len_sample_leicht / total_len if total_len > 0 else 0.5\n",
                    "        \n",
                    "        return ' '.join(kompletter_absatz), regression_target\n",
                    "\n",
                    "\n",
                    "class NewMixupDataset(Dataset):\n",
                    "    def __init__(self, df, nlp_sentencizer):\n",
                    "        self.ls_data = []\n",
                    "        self.as_data = []\n",
                    "        for _, row in tqdm(df.iterrows(), total=len(df), desc=\"Segmentiere (Neu/Uniform)\"):\n",
                    "            ls_sents = [s.text.strip() for s in nlp_sentencizer(str(row[\"ls_text\"])).sents if s.text.strip()]\n",
                    "            as_sents = [s.text.strip() for s in nlp_sentencizer(str(row[\"as_text\"])).sents if s.text.strip()]\n",
                    "            self.ls_data.append(ls_sents)\n",
                    "            self.as_data.append(as_sents)\n",
                    "            \n",
                    "    def __len__(self):\n",
                    "        return len(self.ls_data)\n",
                    "        \n",
                    "    def __getitem__(self, idx):\n",
                    "        leichte_saetze = self.ls_data[idx]\n",
                    "        alltags_saetze = self.as_data[idx]\n",
                    "        \n",
                    "        num_leicht = len(leichte_saetze)\n",
                    "        num_alltag = len(alltags_saetze)\n",
                    "        \n",
                    "        if num_leicht == 0 or num_alltag == 0:\n",
                    "            return \"\", 0.5\n",
                    "            \n",
                    "        # 1. Ziehe ein stufenloses Lambda aus einer Gleichverteilung\n",
                    "        lam = random.uniform(0.0, 1.0)\n",
                    "        \n",
                    "        # 2. Definiere Ziel-Satzanzahl (z. B. N zufällig zwischen 8 und 15 Sätzen)\n",
                    "        N = random.randint(8, 15)\n",
                    "        num_ls = int(round(lam * N))\n",
                    "        num_as = N - num_ls\n",
                    "        \n",
                    "        # 3. Zusammenhängende Slices mit berechneter Länge ziehen\n",
                    "        sample_leicht = get_contiguous_slice(leichte_saetze, num_ls)\n",
                    "        sample_alltag = get_contiguous_slice(alltags_saetze, num_as)\n",
                    "        \n",
                    "        # 4. Zusammenfügen und mischen\n",
                    "        kompletter_absatz = sample_leicht + sample_alltag\n",
                    "        random.shuffle(kompletter_absatz)\n",
                    "        \n",
                    "        # 5. Target als tatsächliches Satzverhältnis berechnen\n",
                    "        total_sents = len(sample_leicht) + len(sample_alltag)\n",
                    "        regression_target = len(sample_leicht) / total_sents if total_sents > 0 else 0.5\n",
                    "        \n",
                    "        return ' '.join(kompletter_absatz), regression_target"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Simulation & Target-Erfassung"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "normal_ds = NormalMixupDataset(df_filtered, nlp)\n",
                    "new_ds = NewMixupDataset(df_filtered, nlp)\n",
                    "\n",
                    "normal_targets = []\n",
                    "new_targets = []\n",
                    "\n",
                    "EPOCHS = 10\n",
                    "for epoch in range(EPOCHS):\n",
                    "    for i in range(len(df_filtered)):\n",
                    "        _, t_norm = normal_ds[i]\n",
                    "        _, t_new = new_ds[i]\n",
                    "        \n",
                    "        normal_targets.append(t_norm)\n",
                    "        new_targets.append(t_new)\n",
                    "\n",
                    "print(f\"Simulation beendet. {len(normal_targets)} Samples generiert.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Visualisierung & Vergleich"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(14, 6))\n",
                    "\n",
                    "# 1. Normal (Marc's Code)\n",
                    "plt.subplot(1, 2, 1)\n",
                    "sns.histplot(normal_targets, bins=25, kde=True, color=\"salmon\", stat=\"probability\")\n",
                    "plt.title(\"1. Normal (Marc's Code)\")\n",
                    "plt.xlabel(\"Regression Target (Anteil Leichte Sprache)\")\n",
                    "plt.ylabel(\"Wahrscheinlichkeit\")\n",
                    "plt.xlim(-0.05, 1.05)\n",
                    "plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
                    "\n",
                    "# 2. Neu (Gleichverteilt)\n",
                    "plt.subplot(1, 2, 2)\n",
                    "sns.histplot(new_targets, bins=25, kde=True, color=\"skyblue\", stat=\"probability\")\n",
                    "plt.title(\"2. Neu (Gleichverteilt)\")\n",
                    "plt.xlabel(\"Regression Target (Anteil Leichte Sprache)\")\n",
                    "plt.ylabel(\"Wahrscheinlichkeit\")\n",
                    "plt.xlim(-0.05, 1.05)\n",
                    "plt.grid(True, linestyle=\"--\", alpha=0.5)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig(\"../results/mixup_comparison_two_variants.png\", dpi=300)\n",
                    "plt.show()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    output_path = Path("/Users/fietescheel/Documents/Master Thesis/notebooks/3_mixup_dataloader_test.ipynb")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Notebook successfully created at {output_path}")

if __name__ == '__main__':
    create_notebook()
