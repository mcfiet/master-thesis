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

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Nutze Device: {DEVICE}")

# 1. Lade SpaCy für NER
print("Lade SpaCy...")
nlp = spacy.load("de_core_news_sm")

# 2. Lade SBERT
print("Lade SBERT...")
sbert_model = SentenceTransformer("jinaai/jina-embeddings-v2-base-de", device=DEVICE, trust_remote_code=True)
if hasattr(sbert_model, "max_seq_length"):
    sbert_model.max_seq_length = 8192

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
# Datensatz-Laden (Festes 4-Klassen Benchmark Testset)
# -----------------------------------------------------------------------------
benchmark_file = "data/evaluation_sets/benchmark_factuality_testset.json"
if os.path.exists(benchmark_file):
    print(f"Lade festes Benchmark-Testset aus {benchmark_file}...")
    df_benchmark = pd.read_json(benchmark_file)
else:
    print(f"Warnung: {benchmark_file} nicht gefunden, lade data/analysis/factual_consistency_benchmark_dataset.json...")
    df_benchmark = pd.read_json("data/analysis/factual_consistency_benchmark_dataset.json")

print(f"Benchmark-Datensatz geladen mit {len(df_benchmark)} Stichproben:")
print(df_benchmark["category"].value_counts())

# -----------------------------------------------------------------------------
# Metrik-Berechnung
# -----------------------------------------------------------------------------
print("\nStarte Berechnung aller Metriken...")

as_list = df_benchmark["as_text"].tolist()
ls_list = df_benchmark["ls_text"].tolist()

# 1. SBERT Similarity
print("1. Berechne SBERT Embeddings...")
with torch.inference_mode():
    emb_as = sbert_model.encode(as_list, batch_size=4, convert_to_tensor=True, show_progress_bar=True)
    emb_ls = sbert_model.encode(ls_list, batch_size=4, convert_to_tensor=True, show_progress_bar=True)
    sbert_cos = util.cos_sim(emb_as, emb_ls).diagonal().cpu().numpy()
    sbert_norm = np.clip((sbert_cos + 1.0) / 2.0, 0.0, 1.0)
    del emb_as, emb_ls
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 2. NLI Cross-Encoder
print("2. Berechne NLI Entailment & Contradiction Scores...")
pairs = list(zip(as_list, ls_list))
with torch.inference_mode():
    nli_logits = nli_model.predict(pairs, batch_size=8, show_progress_bar=True)
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

summary_cat_path = "results/evaluation/factual_consistency_summary_by_category.csv"
df_summary.to_csv(summary_cat_path)
print(f"Kategorie-Zusammenfassung gespeichert in: {summary_cat_path}")

print("\n" + "="*80)
print("TRENNKRAFT & ROC-AUC (Unterscheidung: Gold Positives vs. Alle Negatives)")
print("="*80)

y_true = df_benchmark["is_factually_correct"].values
roc_auc_records = []

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
    
    roc_auc_records.append({
        "metric": m,
        "roc_auc": round(float(auc), 4),
        "gold_mean": round(float(gold_mean), 4),
        "hallu_mean": round(float(hallu_mean), 4),
        "pert_mean": round(float(pert_mean), 4),
        "delta_gold_hallu": round(float(delta_hallu), 4),
        "delta_gold_pert": round(float(delta_pert), 4)
    })
    print(f"Metrik: {m:<22} | ROC-AUC: {auc:.4f} | Δ(Gold - Hallu): {delta_hallu:+.4f} | Δ(Gold - Perturb): {delta_pert:+.4f}")

df_roc = pd.DataFrame(roc_auc_records)
roc_table_path = "results/evaluation/factual_consistency_roc_auc_table.csv"
df_roc.to_csv(roc_table_path, index=False)
print(f"ROC-AUC Tabelle gespeichert in: {roc_table_path}")

print("\nEvaluierung erfolgreich abgeschlossen.")

