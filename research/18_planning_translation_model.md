# Planung Übersetzungsmodell (Step 3)

## 1. Modell-Architekturen & Ansätze

### A. Sequence-to-Sequence (Seq2Seq / Encoder-Decoder)
* **Modelle:** `mBART-50` (`facebook/mbart-large-50`), `mt5` (`google/mt5-base` / `mt5-large`).
* **Funktionsweise:** Klassisches maschinelles Übersetzen (NMT). Der Encoder verarbeitet Alltagssprache (AS), der Decoder generiert Leichte Sprache (LS).
* **Vorteile:**
  * Strikte Zielorientierung (kein Generieren von Chat-Smalltalk/Boilerplate).
  * Deterministisch & ressourcenschonend (Fine-Tuning auf Standard-GPUs problemlos).
* **Nachteile:**
  * Tut sich schwer bei starker Umstrukturierung (wenn 1 langer AS-Satz zu 3–4 LS-Sätzen wird).
  * Neigt bei langen Absätzen zu Wiederholungen oder frühzeitigem Abbruch.

### B. Causal LLMs (Decoder-Only via LoRA / QLoRA)
* **Modelle:** `LLaMA-3-8B-Instruct`, `Mistral-7B-v0.3`, `Qwen2.5-7B`.
* **Funktionsweise:** Fine-Tuning mit System-Prompt (z. B. *"Übersetze in Leichte Sprache..."*) über Parameter-Efficient Fine-Tuning (LoRA).
* **Vorteile:**
  * Sehr starke deutsche Sprachkompetenz und Flexibilität.
  * Beherrscht Absatz- & Block-Transformationen mühelos (löst das 1:n-Satzaufspaltungsproblem).
  * Kann komplexe Regeln (kurze Sätze, Aktivformen, Erklärungen) durch Prompting + Fine-Tuning verbinden.
* **Nachteile:**
  * Risiko von Halluzinationen (Erfinden von Inhalten) oder Über-Vereinfachung (Verlust wichtiger Fakten).
  * Benötigt Quantisierung (QLoRA 4-bit/8-bit) und mindestens 16–24 GB VRAM im Training.

### C. Zero-Shot / Few-Shot Baseline (Prompt-Only)
* **Modelle:** Un-finetuned `LLaMA-3-8B-Instruct`, `GPT-4o`.
* **Zweck:** Dient als **Baseline 0 / Referenzpunkt**, um zu sehen, was ein fertiges LLM ohne Korpus-Fine-Tuning leistet.

---

## 2. Integration unserer trainierten Metriken & Modelle

Wir haben in Step 1 & 2 drei eigene/spezifische Metrik-Werkzeuge aufgebaut:
1. **BiLSTM MixUp Regressor (Variante D):** Sagt kontinuierlichen Komplexitätswert $\lambda \in [0.0, 1.0]$ vorher ($0.0 = \text{AS}, 1.0 = \text{LS}$).
2. **SBERT / Jina Embeddings:** Misst die semantische Kosinus-Ähnlichkeit ($\text{Sim}$) zwischen AS-Quelle und LS-Ziel.
3. **Bi-direktionale NER (spaCy):** Misst den Erhalt von Fakten (Namen, Zahlen, Orte, Daten).

### Einsatzmöglichkeit 1: DPO Preference Data & Reward-Generierung
* **Ziel:** Das LLM nicht nur mit SFT trainieren, sondern über Direct Preference Optimization (DPO) gezielt auf hohe LS-Qualität UND Inhaltstreue ausrichten.
* **Ablauf:**
  1. Das SFT-Modell generiert für einen AS-Text verschiedene Übersetzungsvarianten ($y_1, y_2$).
  2. **Scoring der Varianten:**
     * **Stil-Score:** $\lambda = \text{MixUpRegressor}(y_i)$ (Ziel: $\lambda \to 1.0$).
     * **Semantik-Score:** $\text{Sim} = \text{SBERT}(x_{AS}, y_i)$ (Ziel: $\text{Sim} \ge 0.80$).
     * **Gesamt-Reward:** $R = \lambda \times \text{Sim}$.
  3. Die Variante mit dem höchsten Score wird `Chosen`, die mit geringem Score oder Halluzination wird `Rejected`.
  4. DPO trainiert das Modell zielgerichtet darauf, präferierte Übersetzungen zu bevorzugen.

### Einsatzmöglichkeit 2: Automatische Evaluierung & Modellvergleich
* Die trainierten Metriken bilden zusammen mit Standard-Benchmarks (SARI) die automatische Evaluierungs-Pipeline für alle Modelle (mBART vs. LLaMA-3 SFT vs. LLaMA-3 DPO vs. Zero-Shot):
  * **Formale LS-Güte:** Ø MixUp-Score $\lambda$ (Variante D).
  * **Inhaltlicher Erhalt:** Ø SBERT-Similarity + Ø NER-Recall.
  * **Regel-Check:** Automatische Satzlängen- und Passiv-Analyse via spaCy.

---

## 3. Daten-Alignment & Vorbereitung

* **Block-/Absatz-Alignment:** Wir matchen Texte auf **Absatz-Ebene (ca. 100–300 Tokens)** statt auf Satz-Ebene.
  * *Warum:* Ein AS-Satz verteilt sich in LS oft auf mehrere Sätze. Auf Absatz-Ebene bleibt der Kontext erhalten und die Modelle lernen flüssige Aufspaltungen.
* **Splits:** 80% Train, 10% Validation, 10% Test (strikt nach Dokumenten getrennt).
* **Out-of-Domain Testset:** Das proprietäre Lebenshilfe-Set (49 Paare) zur Messung der echten Generalisierung.

---

## 4. Nächste konkrete Schritte

1. `scripts/prepare_translation_dataset.py`: Absatz-Paare aus `corpus_final` extrahieren und in Train/Val/Test speichern.
2. `scripts/train_translation_sft.py`: Baseline-Training für `mBART-50` und `LLaMA-3-8B` (LoRA).
3. `scripts/train_translation_dpo.py`: DPO-Training mit MixUp-Score & SBERT als Reward.
4. `scripts/evaluate_translation_models.py`: Vergleichende Auswertung über alle Modelle.
