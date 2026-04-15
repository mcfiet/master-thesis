# Statusbericht Master Thesis: Automatische Übersetzung in Leichte Sprache
## Status Quo und Planung

<div style="font-family: sans-serif; max-width: 1400px; margin: auto;">

  <!-- Main Grid -->
  <div style="display: flex; justify-content: space-between; align-items: stretch; margin-bottom: 25px; gap: 15px;">
    
    <!-- Box 1: Dataset -->
    <div style="flex: 1; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-color: #fff; display: flex; flex-direction: column;">
      <h3 style="color: #1e293b; text-align: center; margin-top: 0; margin-bottom: 15px; font-size: 1.1em;">Dataset / Corpus</h3>
      
      <!-- Status Quo -->
      <div style="margin-bottom: 12px; padding: 0 5px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">Status Quo</p>
        <p style="font-size: 0.75em; color: #475569; margin: 0; line-height: 1.3;">
          Web-Crawl & Synthetic Data Generation (EL-Websites ➔ Normal sentences generated). SBERT training for EL discrimination.
        </p>
      </div>

      <!-- Limitations Box -->
      <div style="margin-bottom: 12px; padding: 10px; background: #fff1f2; border-left: 4px solid #f43f5e; border-radius: 4px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #be123c; text-transform: uppercase; margin-bottom: 3px;">Limitations</p>
        <p style="font-size: 0.75em; color: #9f1239; margin: 0; line-height: 1.3;">
          <b>1:n Problem:</b> One standard sentence results in many isolated EL sentences. Leads to massive information loss & contradictory translations.
        </p>
      </div>
<div style="display: flex; gap: 1rem;">
      <!-- Minimum Goal Box -->
      <div style="margin-bottom: 8px; padding: 10px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #1e40af; text-transform: uppercase; margin-bottom: 5px;">Minimum Goal</p>
        <ul style="font-size: 0.75em; color: #1e3a8a; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>Cleaning & resolving 1:n</li>
        </ul>
      </div>

      <!-- Optimal Goal Box -->
      <div style="padding: 10px; background: #faf5ff; border-left: 4px solid #a855f7; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #6b21a8; text-transform: uppercase; margin-bottom: 5px;">Optimal Goal</p>
        <ul style="font-size: 0.75em; color: #581c87; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>Expansion of metric dataset</li>
        </ul>
      </div>
    </div>
</div>
    <div style="display: flex; align-items: center; color: #cbd5e1; font-size: 20px;">➔</div>

    <!-- Box 2: Metric -->
    <div style="flex: 1; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-color: #fff; display: flex; flex-direction: column;">
      <h3 style="color: #1e293b; text-align: center; margin-top: 0; margin-bottom: 15px; font-size: 1.1em;">Metric Model</h3>
      
      <!-- Status Quo -->
      <div style="margin-bottom: 12px; padding: 0 5px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">Status Quo</p>
        <p style="font-size: 0.75em; color: #475569; margin: 0; line-height: 1.3;">
          SBERT training for classification (EL vs. Normal). Focus so far on formal compliance with EL rules.
        </p>
      </div>

      <!-- Limitations Box -->
      <div style="margin-bottom: 12px; padding: 10px; background: #fff1f2; border-left: 4px solid #f43f5e; border-radius: 4px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #be123c; text-transform: uppercase; margin-bottom: 3px;">Limitations</p>
        <p style="font-size: 0.75em; color: #9f1239; margin: 0; line-height: 1.3;">
          <b>Real-world Data Test:</b> Only ~74% Acc. (previously only tested on synthetic data). Evaluates only form, no content correctness/semantics.
        </p>
      </div>
<div style="display: flex; gap: 1rem;">
      <!-- Minimum Goal Box -->
      <div style="margin-bottom: 8px; padding: 10px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #1e40af; text-transform: uppercase; margin-bottom: 5px;">Minimum Goal</p>
        <ul style="font-size: 0.75em; color: #1e3a8a; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>Robust generalization</li>
        </ul>
      </div>

      <!-- Optimal Goal Box -->
      <div style="padding: 10px; background: #faf5ff; border-left: 4px solid #a855f7; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #6b21a8; text-transform: uppercase; margin-bottom: 5px;">Optimal Goal</p>
        <ul style="font-size: 0.75em; color: #581c87; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>Semantics integration</li>
          <li>Information loss check</li>
        </ul>
      </div>
    </div>
</div>
    <div style="display: flex; align-items: center; color: #cbd5e1; font-size: 20px;">➔</div>

    <!-- Box 3: Translation -->
    <div style="flex: 1; border-radius: 15px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); background-color: #fff; display: flex; flex-direction: column;">
      <h3 style="color: #1e293b; text-align: center; margin-top: 0; margin-bottom: 15px; font-size: 1.1em;">Translation Model</h3>
      
      <!-- Status Quo -->
      <div style="margin-bottom: 12px; padding: 0 5px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #64748b; text-transform: uppercase; margin-bottom: 2px;">Status Quo</p>
        <p style="font-size: 0.75em; color: #475569; margin: 0; line-height: 1.3;">
          mt5 & Mistral (LoRa) fine-tuned. <b>mt5</b> structurally closer to target; <b>Mistral</b> linguistically cleaner. Classical metrics insufficient.
        </p>
      </div>

      <!-- Limitations Box -->
      <div style="margin-bottom: 12px; padding: 10px; background: #fff1f2; border-left: 4px solid #f43f5e; border-radius: 4px;">
        <p style="font-size: 0.7em; font-weight: bold; color: #be123c; text-transform: uppercase; margin-bottom: 3px;">Limitations</p>
        <p style="font-size: 0.75em; color: #9f1239; margin: 0; line-height: 1.3;">
          <b>Evaluation:</b> BLEU etc. unusable for EL. Model suffers massively from faulty alignment of training data.
        </p>
      </div>
<div style="display: flex; gap: 1rem;">
      <!-- Minimum Goal Box -->
      <div style="margin-bottom: 8px; padding: 10px; background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #1e40af; text-transform: uppercase; margin-bottom: 5px;">Minimum Goal</p>
        <ul style="font-size: 0.75em; color: #1e3a8a; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>Metric as Reward</li>
          <li>Feedback Lebenshilfe</li>
        </ul>
      </div>

      <!-- Optimal Goal Box -->
      <div style="padding: 10px; background: #faf5ff; border-left: 4px solid #a855f7; border-radius: 4px; flex: 1;">
        <p style="font-size: 0.7em; font-weight: bold; color: #6b21a8; text-transform: uppercase; margin-bottom: 5px;">Optimal Goal</p>
        <ul style="font-size: 0.75em; color: #581c87; margin: 0; padding-left: 12px; list-style-type: disc;">
          <li>User study</li>
          <li>Proof of reading comprehension</li>
        </ul>
      </div>
      </div>
    </div>
  </div>

  <!-- Pipeline Process Section -->
  <div style="background-color: #f8fafc; border-radius: 15px; padding: 25px; margin-bottom: 30px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
    <h3 style="text-align: center; margin-top: 0; color: #1e293b; margin-bottom: 20px; font-size: 1.2em;">Pipeline Process</h3>
    <div style="display: flex; justify-content: space-between; gap: 30px;">
      <div style="flex: 1; text-align: center;">
        <p style="color: #3b82f6; font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">Step 1:</p>
        <p style="font-size: 0.8em; color: #475569; line-height: 1.4;">Collecting and preparing parallel corpus data<br>(Easy Language ↔ Standard German)</p>
      </div>
      <div style="flex: 1; text-align: center;">
        <p style="color: #a855f7; font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">Step 2:</p>
        <p style="font-size: 0.8em; color: #475569; line-height: 1.4;">Training the metric model (SBERT) to<br>evaluate Easy Language quality</p>
      </div>
      <div style="flex: 1; text-align: center;">
        <p style="color: #22c55e; font-weight: bold; margin-bottom: 8px; font-size: 0.9em;">Step 3:</p>
        <p style="font-size: 0.8em; color: #475569; line-height: 1.4;">Training the translation model using<br>the metric as a reward function</p>
      </div>
    </div>
  </div>

</div>

---

## Titel (Vorschläge)

1.  **Automatische Übersetzung in Leichte Sprache:** Optimierung neuronaler Sprachmodelle durch SBERT-basierte Reward-Metriken.
2.  **Maschinelle Übersetzung in Leichte Sprache:** Entwicklung generativer Sprachmodelle durch automatisierte Bewertungsverfahren.
3.  **Von der Datenbasis zur Generierung:** Entwicklung eines integrierten Ansatzes für die automatische Übersetzung in Leichte Sprache.

---
## Nächste Schritte

Der Fokus der nächsten Arbeitsphase liegt primär auf der **Optimierung der Datenbasis**, da die Qualität des Korpus der entscheidende Flaschenhals für die nachfolgenden Modelle ist.

1.  **Quellenanalyse & Selektion:**
    *   Systematische Identifikation und Evaluierung von Quellen, die sich für das Zusammenführen von Standarddeutsch und Leichter Sprache eignen.
    *   Unterscheidung zwischen hochwertigen Quellen (professionelle Übersetzungen) und weniger geeigneten Datenbeständen.
2.  **Block-basiertes Sampling (Sample-Einheiten):**
    *   Umstellung der Datenstruktur von einer rein satzweisen Betrachtung auf **ganze Textblöcke**.
    *   **Vorteil:** Ein Standard-Satz teilt sich oft in mehrere Sätze der Leichten Sprache auf (1:n-Problem). Durch das Sampling ganzer Blöcke wird das Alignment erheblich vereinfacht und konsistenter, da semantische Einheiten zusammenbleiben.
3.  **Verbesserung des Alignments:**
    *   Entwicklung eines Prozesses, um diese Blöcke effizient gegenüberzustellen, ohne die inhaltliche Kohärenz zu verlieren.