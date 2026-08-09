# GBERT-Integration für Komplexitätsbewertung & DPO-Reward (Woche 20)

Dieses Dokument dokumentiert die Integration des deutschen Sprachmodells GBERT (`deepset/gbert-base`) zur kontinuierlichen Komplexitätsbewertung (Style-Score) sowie als Reward-Instanz für das DPO-Tuning.

---

## 1. Warum GBERT?
Bisherige Ansätze nutzten primär rekurrente Netze (BiLSTM) auf vortrainierten Word2Vec- oder FastText-Embeddings zur Komplexitätsbestimmung. GBERT bietet durch seinen Transformer-basierten Aufmerksamkeitsmechanismus (Attention) und das Training auf großen deutschen Textkorpora eine deutlich höhere syntaktische und semantische Repräsentationstiefe. Dies ermöglicht eine robustere Identifikation von typischen Strukturen einfacher oder komplexer Sprache (z. B. Genitiv-Konstruktionen, Passivsätze und Nebensatzverschachtelungen).

---

## 2. GBERT MixUp Regression
Im Jupyter-Notebook [`3c_mixup_gbert_regression.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/metric/mixup/3c_mixup_gbert_regression.ipynb) wurde ein GBERT-Regressor mittels des dynamischen Hybrid-MixUp-Verfahrens trainiert:

* **Methode:** Ein linearer Regressor auf dem CLS-Token des GBERT-Modells wird darauf trainiert, die kontinuierliche Komplexität $\lambda \in [0, 1]$ gemischter Embeddings vorherzusagen.
* **Lerntempo:** Zyklische Lernrate (Cyclic Learning Rate) zur Stabilisierung des Trainings.
* **Ergebnisse der Out-of-Domain-Evaluierung (Lebenshilfe-Datensatz):**
  * **Datengrundlage:** 361 LS-Chunks und 245 AS-Chunks.
  * **Ø Lambda (LS):** `0.8598` (Erwartungswert nahe 1.0)
  * **Ø Lambda (AS):** `0.0564` (Erwartungswert nahe 0.0)
  * **Accuracy (Schwelle 0.5):** **95.38%**
  * **Balanced Accuracy:** **95.60%**
  * **MAE (Mean Absolute Error):** `0.1063`

Das GBERT-Modell separiert Alltagssprache und Leichte Sprache hochgradig präzise mit einem minimalen Fehler auf ungesehenen Daten.

---

## 3. GBERT Synthetische Regression
Im Notebook [`2_synthetic_gbert_regression.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/metric/synthetic/2_synthetic_gbert_regression.ipynb) wurde die GBERT-Architektur auf kontinuierlichen Komplexitätsscores trainiert, die von einem LLM synthetisch erzeugt wurden:

* **Ergebnisse der Out-of-Domain-Evaluierung (Lebenshilfe-Datensatz):**
  * **Anzahl Samples:** 245
  * **MSE (Mean Squared Error):** `0.0323`
  * **MAE:** `0.0931`
  * **Pearson-Korrelation:** **0.8715**
  * **Spearman-Korrelation:** **0.8640**

Die extrem hohe Korrelation von $> 0.86$ zeigt, dass die GBERT-Regression die menschlichen/synthetischen Komplexitätsabstufungen extrem trennscharf abbilden kann.

---

## 4. DPO-Training mit GBERT-Reward
Der feinjustierte GBERT-Regressor dient als Reward-Modell für das anschließende DPO-Tuning des Übersetzungsmodells:

1. **Custom PyTorch-Loop:** [`4_dpo_w05_w05_gbert_reward_trainer.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/translation/4_dpo_w05_w05_gbert_reward_trainer.ipynb) nutzt den GBERT-Simplicity-Score als Teil der Verbund-Reward-Funktion (Style-Reward und SBERT-Semantik-Kosinus-Ähnlichkeit) zur Bestimmung von Gewinner- ($y_w$) und Verlierer-Kandidaten ($y_l$).
2. **Hugging Face Trainer:** [`5_dpo_with_huggingface_trainer.ipynb`](file:///Users/fietescheel/Documents/Master%20Thesis/notebooks/translation/5_dpo_with_huggingface_trainer.ipynb) integriert GBERT-basierte Präferenzen in das DPO-Training über die offizielle Hugging Face `trl` Bibliothek (`DPOTrainer`) mit PEFT/LoRA für ressourceneffizientes Feintuning.
