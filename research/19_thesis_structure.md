# Thesis-Struktur: Automatische Übersetzung in Leichte Sprache

## 1. Einleitung
- **1.1 Problemstellung & Motivation** (Bedeutung von Leichter Sprache, Barrierefreiheit, Automatisierungsbedarf)
- **1.2 Zielsetzung & Forschungsfragen** (Vorstellung der zentralen Leitfrage sowie der Teilfragen)
- **1.3 Aufbau der Arbeit** (Überblick über die Drei-Stufen-Pipeline und Kapitelstruktur)

## 2. Theoretical Background & Stand der Forschung
- **2.1 Leichte Sprache** (Regelwerke, Richtlinien, Zielgruppen, Abgrenzung zur Einfachen Sprache)
- **2.2 Maschinelle Übersetzung & Textvereinfachung** (Seq2Seq-Modelle, Large Language Models, LoRA/QLoRA Fine-Tuning)
- **2.3 Evaluierungsansätze** (Klassische Metriken, automatische Lesbarkeitsindizes, Reward-Modelle, RLHF/DPO)

## 3. Themenblock 1: Datenbasis & Korpus-Erstellung
- **3.1 Multi-Quellen-Crawling & Datenakquise** (Aufbau des 11-Quellen-Korpus: Hannover.de, MDR, Apotheken Umschau etc.)
- **3.2 Alignment & Das 1:n-Problem** (Gegenüberstellung von Satz- vs. Absatz-/Block-Alignment zur Wahrung semantischer Einheiten)
- **3.3 Quality Assurance & Similarity Sweet-Spot** (Filtering via Long-Context Embeddings im Bereich 0.80 ≤ Similarity ≤ 0.98)
- **3.4 Datenbereinigung & Normalisierung** (Mediopunkt-Behandlung, Metadaten-Removal, Automatisierte Post-Cleaning-Pipeline)
- **3.5 Korpus-Statistiken & Lexikalische Diversität** (Token-/Satzanzahlen, Wortlängen, Type-Token-Ratio)

## 4. Themenblock 2: Metrik & Bewerten von Sprachkomplexität
- **4.1 Klassifikationsmodelle** (BiLSTM vs. SBERT; Satzebene vs. Dokumentenebene mit Majority Voting)
- **4.2 Out-of-Domain-Generalisierung & Empirische Bias-Kontrolle** (Ausschluss von Length-, Layout- und Typografie-Shortcuts)
- **4.3 Continuous MixUp Regression** (Kontinuierliche Komplexitätsbewertung; Vergleich der Dataloader-Varianten A–D; Hybrid + Cyclic LR)
- **4.4 Evaluierung synthetischer Sprachstufen** (LLM-generierte Zwischenstufen & Monotonie-Überprüfung des Regressors)

## 5. Themenblock 3: Modellierung der Übersetzung & Reward-Guided Fine-Tuning
- **5.1 Datenvorbereitung & Trainings-Splits** (Gefilterter Sweet-Spot-Datensatz, Block-basiertes Alignment)
- **5.2 Supervised Fine-Tuning (SFT)** (Baselines mit Encoder-Decoder-Modellen [mt5] und Decoder-Only-LLMs [Mistral, LLaMA])
- **5.3 Reward-Guided Optimization (RLHF / DPO)** (Integration des MixUp-Regressors als Reward-Funktion zur Regel- & Stilerzwingung)
- **5.4 Mehrdimensionale Evaluierung** (Formale Regelkonformität vs. Faktentreue und Informationsverlust mittels NLI/NER)

## 6. Diskussion & Gesamtevaluation
- **6.1 Inhaltssynthese der drei Themenblöcke** (Zusammenwirken von Daten, Metrik und Übersetzung)
- **6.2 Stärken, Grenzen & Systemrestriktionen** (Kritische Würdigung der Methodik und Ergebnisse)
- **6.3 Beantwortung der Forschungsfragen** (Systematische Auflösung von FF 2.1–5.4)

## 7. Fazit & Ausblick
- **7.1 Zusammenfassung der Kernergebnisse**
- **7.2 Ausblick & Zukünftige Forschungsarbeiten** (Zielgruppen-Validierung, Live-Anwendungen)
