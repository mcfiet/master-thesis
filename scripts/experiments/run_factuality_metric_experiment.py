#!/usr/bin/env python3
"""
Evaluierung von Metriken zur Faktenkonsistenz und Halluzinationserkennung
(SBERT vs. Cross-Encoder vs. NLI vs. NER vs. Regex Number Check)
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# Determinismus
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cpu"
print(f"Nutze Device: {DEVICE}")

# 1. Lade SpaCy für NER
print("Lade SpaCy...")
nlp = spacy.load("de_core_news_sm")

# 2. Lade SBERT
print("Lade SBERT...")
sbert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=DEVICE)

# 3. Lade NLI Cross-Encoder
print("Lade NLI Cross-Encoder...")
nli_model = CrossEncoder("cross-encoder/nli-distilroberta-base", device=DEVICE)

# Helper: NER Overlap
def compute_ner_metrics(as_text: str, ls_text: str):
    doc_as = nlp(str(as_text or ""))
    doc_ls = nlp(str(ls_text or ""))
    
    ents_as = set([e.text.strip().lower() for e in doc_as.ents if len(e.text.strip()) > 1])
    ents_ls = set([e.text.strip().lower() for e in doc_ls.ents if len(e.text.strip()) > 1])
    
    if len(ents_ls) == 0 and len(ents_as) == 0:
        jaccard = 1.0
        recall_ls_in_as = 1.0
    elif len(ents_ls) == 0:
        jaccard = 0.0
        recall_ls_in_as = 1.0
    elif len(ents_as) == 0:
        jaccard = 0.0
        recall_ls_in_as = 0.0
    else:
        intersection = ents_as.intersection(ents_ls)
        jaccard = len(intersection) / len(ents_as.union(ents_ls))
        recall_ls_in_as = len(intersection) / len(ents_ls)
        
    return jaccard, recall_ls_in_as, len(ents_as), len(ents_ls)

# Helper: Number Regex Consistency
def compute_number_consistency(as_text: str, ls_text: str):
    pattern = r'\b\d+(?:[.,]\d+)?\b'
    nums_as = set(re.findall(pattern, str(as_text or "")))
    nums_ls = set(re.findall(pattern, str(ls_text or "")))
    
    if len(nums_ls) == 0:
        return 1.0, 0, 0
    
    valid_nums = nums_ls.intersection(nums_as)
    hallucinated_nums = nums_ls - nums_as
    
    consistency_score = len(valid_nums) / len(nums_ls)
    return consistency_score, len(nums_as), len(hallucinated_nums)

# -----------------------------------------------------------------------------
# Datensatz-Erstellung (4 Klassen)
# -----------------------------------------------------------------------------
print("Erstelle 4-Klassen Benchmark-Datensatz...")

samples = []

# KLASSE 1: Gold Truth Positives
# 1.1 Lebenshilfe
lh_path = "data/lebenshilfe/lebenshilfe_dataset_clean.json"
if os.path.exists(lh_path):
    with open(lh_path, "r", encoding="utf-8") as f:
        lh_data = json.load(f)
    for row in lh_data:
        as_t = str(row.get("as_text") or "").strip()
        ls_t = str(row.get("ls_text") or "").strip()
        if len(as_t) > 20 and len(ls_t) > 20:
            samples.append({
                "category": "1_Gold_Positives",
                "subtype": "Lebenshilfe",
                "as_text": as_t,
                "ls_text": ls_t,
                "is_factually_correct": 1
            })

# 1.2 Corpus Master
cm_path = "data/analysis/corpus_master.csv"
if os.path.exists(cm_path):
    df_cm = pd.read_csv(cm_path)
    df_cm_clean = df_cm.dropna(subset=["as_text", "ls_text"]).sample(n=min(40, len(df_cm)), random_state=42)
    for _, row in df_cm_clean.iterrows():
        samples.append({
            "category": "1_Gold_Positives",
            "subtype": f"Corpus_{row.get('source', 'master')}",
            "as_text": str(row["as_text"]).strip(),
            "ls_text": str(row["ls_text"]).strip(),
            "is_factually_correct": 1
        })

# KLASSE 2: Reale Modell-Halluzinationen (aus dpo_pairs_w05_w05.jsonl)
dpo_path = "data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl"
if os.path.exists(dpo_path):
    dpo_rows = []
    with open(dpo_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 500: break
            dpo_rows.append(json.loads(line))
            
    for idx in range(min(40, len(dpo_rows))):
        item = dpo_rows[idx]
        as_t = str(item.get("prompt") or item.get("as_text") or "").strip()
        ls_t = str(item.get("rejected") or "").strip()
        if len(as_t) > 20 and len(ls_t) > 20:
            samples.append({
                "category": "2_Real_Model_Hallucinations",
                "subtype": f"mBART_SFT_{item.get('source', 'generation')}",
                "as_text": as_t,
                "ls_text": ls_t,
                "is_factually_correct": 0
            })

# KLASSE 3: Random Shuffle Negatives (Themenwechsel)
gold_samples = [s for s in samples if s["category"] == "1_Gold_Positives"]
if len(gold_samples) >= 30:
    sampled_gold = random.sample(gold_samples, 30)
    for i in range(len(sampled_gold)):
        as_t = sampled_gold[i]["as_text"]
        target_idx = (i + 7) % len(sampled_gold)
        ls_t = sampled_gold[target_idx]["ls_text"]
        samples.append({
            "category": "3_Random_Shuffle_Negatives",
            "subtype": "Thematic_Mismatch",
            "as_text": as_t,
            "ls_text": ls_t,
            "is_factually_correct": 0
        })

# KLASSE 4: Kontrollierte Minimal-Perturbationen (Adversarial Slices)
perturbation_templates = [
    ("Kapitän Francesco Schettino wurde zu 16 Jahren Haft verurteilt.", "Der Kapitän war 16 Jahre alt und muss nicht ins Gefängnis.", "Number_Slot_Shift"),
    ("Die NZZ machte im letzten Jahr einen Gewinn von 20,47 Millionen Euro.", "Im Jahr 2047 hat die NZZ einen Gewinn von 20 Euro gemacht.", "Number_To_Year_Shift"),
    ("Der Beirat für Menschen mit Behinderungen besteht aus 15 Mitgliedern.", "Der Beirat für Menschen mit Behinderungen besteht aus 500 Mitgliedern.", "Number_Exaggeration"),
    ("Ferrari verkaufte 7900 Autos im Jahr 2015.", "Ferrari verkaufte 20 Autos im Jahr 2015.", "Number_Reduction"),
    ("Das Teleskop hat einen Durchmesser von 39 Metern.", "Das Teleskop hat einen Durchmesser von 3900 Kilometern.", "Number_Scale_Shift"),
    ("Die Beratung findet montags von 8 bis 12 Uhr statt.", "Die Beratung findet sonntags um Mitternacht statt.", "Date_Time_Shift"),
    ("Mehr als 80 Prozent der Befragten haben Sorgen.", "Nur 2 Prozent der Befragten haben Sorgen.", "Percentage_Inversion"),
    ("Der Bau kostet rund 1,5 Millionen Euro.", "Der Bau kostet rund 150 Milliarden Euro.", "Financial_Scale_Shift"),
    ("Die Beratung in den Pflegestützpunkten ist für alle Bürger kostenlos.", "Die Beratung in den Pflegestützpunkten ist nicht kostenlos und kostet viel Geld.", "Negation_Inversion"),
    ("Alle Menschen haben ein Recht auf barrierefreies Wohnen.", "Kein Mensch hat ein Recht auf barrierefreies Wohnen.", "Negation_Inversion"),
    ("Die Justiz soll die Bürger vor Kriminellen schützen.", "Die Justiz soll die Bürger nicht vor Kriminellen schützen.", "Negation_Inversion"),
    ("Sie müssen für diesen Antrag kein Geld bezahlen.", "Sie müssen für diesen Antrag viel Geld bezahlen.", "Polarity_Flip"),
    ("Die Mitarbeiter dürfen Ihre privaten Daten nicht an fremde Personen weitergeben.", "Die Mitarbeiter dürfen Ihre privaten Daten an alle fremden Personen weitergeben.", "Negation_Removal"),
    ("Das neue Gesetz tritt am ersten Januar in Kraft.", "Das neue Gesetz tritt niemals in Kraft.", "Negation_Temporal"),
    ("Kinder mit Behinderungen werden in der Schule gefördert.", "Kinder mit Behinderungen werden in der Schule nicht gefördert.", "Negation_Inversion"),
    ("Das Gericht hat entschieden, dass diese Gebühr unzulässig ist.", "Das Gericht hat entschieden, dass diese Gebühr vollkommen erlaubt ist.", "Polarity_Flip"),
    ("Der ehemalige Sportwagenbauer Ferrari befindet sich auf Talfahrt an der Börse.", "Der Sportwagen-Fahrer Herr Ferrari hat ein schnelles Rennen gewonnen.", "Entity_Role_Shift"),
    ("Lewis Hamilton spielt in seiner Freizeit Klavier zu Liedern von Adele.", "Lewis Hamilton ist ein deutscher Schreiner und baut Klaviere für Adele.", "Entity_Hallucination"),
    ("Die Polizei überwacht gefährliche Straftäter nach dem Justizgesetz.", "Die gefährlichen Straftäter überwachen die Polizei in ganz Hamburg.", "Subject_Object_Inversion"),
    ("Das Schiff Costa Concordia rammte vor der italienischen Küste einen Felsen.", "In Sachsen-Anhalt gab es ein schweres Erdbeben bei einem Schiffsunglück.", "Geographic_Hallucination"),
    ("Die Bischofskonferenz im Vatikan berät über Reformen der katholischen Kirche.", "Sachsen-Anhalt Die Bischwürden wollen keine Früchte mehr in Rom essen.", "Neologism_Hallucination"),
    ("Die Verbraucherzentrale hilft Mietern bei Streitigkeiten mit Vermietern.", "Die Vermieter helfen der Verbraucherzentrale gegen alle Mieter.", "Subject_Object_Inversion"),
    ("Der Senat ist die Landesregierung der Freien und Hansestadt Hamburg.", "Der Senat ist ein Fußballverein aus Sachsen-Anhalt.", "Entity_Category_Shift"),
    ("Die Techniker-Krankenkasse erstattet die Kosten für medizinische Hilfsmittel.", "Die Patienten müssen die Techniker-Krankenkasse monatlich bar im Krankenhaus bezahlen.", "Role_Payment_Shift"),
    ("Österreich ist Mitglied im europäischen Teleskop-Konsortium MOSAIC.", "Saudi-Arabien baut das europäische Teleskop mitten in Wien.", "Country_Substitution"),
    ("Verbraucherschützer haben gegen das Flugreise-Portal vor Gericht geklagt.", "Das Flugreise-Portal hat alle Verbraucher ins Gefängnis geklagt.", "Subject_Object_Inversion"),
    ("In Pinneberg wurde ein neuer Rat für Menschen mit Behinderung gewählt.", "In Pinneberg wurden alle Räte für behinderte Menschen verboten.", "Polarity_Abolish"),
    ("Die Hamburger Wasserwerke kontrollieren täglich die Trinkwasserqualität.", "Das Trinkwasser in Hamburg wird aus der Atacama-Wüste importiert.", "Geographic_Nonsense"),
    ("Das Landgericht Frankfurt verbietet unerlaubte Zusatzgebühren beim Online-Kauf.", "Das Landgericht Frankfurt zwingt alle Käufer zu doppelten Bankgebühren.", "Polarity_Verdict_Flip"),
    ("Elternvertreter fordern verlässliche Betreuungszeiten in Kindertagesstätten.", "Die Kindertagesstätten fordern die Abschaffung aller Elternvertreter.", "Role_Reversal")
]

for as_t, ls_t, subtype in perturbation_templates:
    samples.append({
        "category": "4_Targeted_Minimal_Perturbations",
        "subtype": subtype,
        "as_text": as_t,
        "ls_text": ls_t,
        "is_factually_correct": 0
    })

df_benchmark = pd.DataFrame(samples)
print(f"Benchmark-Datensatz erstellt mit {len(df_benchmark)} Stichproben:")
print(df_benchmark["category"].value_counts())

os.makedirs("data/analysis", exist_ok=True)
df_benchmark.to_json("data/analysis/factual_consistency_benchmark_dataset.json", orient="records", force_ascii=False, indent=2)
print("Gespeichert in: data/analysis/factual_consistency_benchmark_dataset.json")

# -----------------------------------------------------------------------------
# Metrik-Berechnung
# -----------------------------------------------------------------------------
print("\nStarte Berechnung aller Metriken...")

as_list = df_benchmark["as_text"].tolist()
ls_list = df_benchmark["ls_text"].tolist()

# 1. SBERT Similarity
print("1. Berechne SBERT Embeddings...")
emb_as = sbert_model.encode(as_list, batch_size=16, convert_to_tensor=True, show_progress_bar=True)
emb_ls = sbert_model.encode(ls_list, batch_size=16, convert_to_tensor=True, show_progress_bar=True)
sbert_cos = util.cos_sim(emb_as, emb_ls).diagonal().cpu().numpy()
sbert_norm = np.clip((sbert_cos + 1.0) / 2.0, 0.0, 1.0)

# 2. NLI Cross-Encoder
print("2. Berechne NLI Entailment & Contradiction Scores...")
pairs = list(zip(as_list, ls_list))
nli_logits = nli_model.predict(pairs, batch_size=16, show_progress_bar=True)
nli_probs = F.softmax(torch.tensor(nli_logits), dim=-1).numpy()
p_contradiction = nli_probs[:, 0]
p_neutral = nli_probs[:, 1]
p_entailment = nli_probs[:, 2]

nli_factuality_score = np.clip((p_entailment - p_contradiction + 1.0) / 2.0, 0.0, 1.0)

# 3. NER Overlap & Recall
print("3. Berechne SpaCy NER Metriken...")
ner_jaccards = []
ner_recalls = []
for a, l in tqdm(zip(as_list, ls_list), total=len(as_list)):
    jac, rec, _, _ = compute_ner_metrics(a, l)
    ner_jaccards.append(jac)
    ner_recalls.append(rec)

# 4. Number Consistency
print("4. Berechne Numerische Konsistenz...")
num_consistencies = []
for a, l in tqdm(zip(as_list, ls_list), total=len(as_list)):
    c_score, _, _ = compute_number_consistency(a, l)
    num_consistencies.append(c_score)

df_benchmark["sbert_raw"] = sbert_cos
df_benchmark["sbert_score"] = sbert_norm
df_benchmark["nli_p_entail"] = p_entailment
df_benchmark["nli_p_neutral"] = p_neutral
df_benchmark["nli_p_contra"] = p_contradiction
df_benchmark["nli_factuality"] = nli_factuality_score
df_benchmark["ner_jaccard"] = ner_jaccards
df_benchmark["ner_recall"] = ner_recalls
df_benchmark["number_consistency"] = num_consistencies
df_benchmark["composite_factuality"] = df_benchmark["sbert_score"] * df_benchmark["nli_factuality"] * df_benchmark["number_consistency"]

res_path = "results/evaluation/factual_consistency_metric_results.csv"
os.makedirs("results/evaluation", exist_ok=True)
df_benchmark.to_csv(res_path, index=False)
print(f"Ergebnisse gespeichert in: {res_path}")

print("\n" + "="*80)
print("DURCHSCHNITTSWERTE PRO TESTKLASSE")
print("="*80)

metrics_to_show = ["sbert_score", "nli_factuality", "nli_p_contra", "ner_jaccard", "number_consistency", "composite_factuality"]
df_summary = df_benchmark.groupby("category")[metrics_to_show].mean().round(4)
print(df_summary.to_string())

print("\n" + "="*80)
print("TRENNKRAFT & ROC-AUC (Unterscheidung: Gold Positives vs. Alle Negatives)")
print("="*80)

y_true = df_benchmark["is_factually_correct"].values

for m in metrics_to_show:
    if m == "nli_p_contra":
        auc = roc_auc_score(y_true, 1.0 - df_benchmark[m].values)
    else:
        auc = roc_auc_score(y_true, df_benchmark[m].values)
    
    gold_mean = df_benchmark[df_benchmark["category"] == "1_Gold_Positives"][m].mean()
    hallu_mean = df_benchmark[df_benchmark["category"] == "2_Real_Model_Hallucinations"][m].mean()
    pert_mean = df_benchmark[df_benchmark["category"] == "4_Targeted_Minimal_Perturbations"][m].mean()
    
    delta_hallu = gold_mean - hallu_mean
    delta_pert = gold_mean - pert_mean
    
    print(f"Metrik: {m:<22} | ROC-AUC: {auc:.4f} | Δ(Gold - Hallu): {delta_hallu:+.4f} | Δ(Gold - Perturb): {delta_pert:+.4f}")

print("\nEvaluierung erfolgreich abgeschlossen.")
