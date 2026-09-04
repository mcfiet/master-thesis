#!/usr/bin/env python3
"""
scripts/evaluation/build_expert_evaluation_set.py

Erstellt den 50-Item Evaluationspool für die Expertenevaluation (Disjoint Design):
1. Lädt corpus_master.csv und lebenshilfe_dataset_clean.json.
2. Führt dynamischen Train/Test-Split mit seed=42 durch (81 ungesehene Web-Artikel + 37 Lebenshilfe-Artikel = 118 Test-Artikel).
3. Wählt 48 vollständig disjunkte Quellartikel aus (kein Thema doppelt):
   - 12 Artikel für Bedingung 1 (AS Original)
   - 12 Artikel für Bedingung 2 (Gold LS Mensch)
   - 12 Artikel für Bedingung 3 (mBART SFT len1024 Übersetzung)
   - 12 Artikel für Bedingung 4 (mBART DPO len1024 Übersetzung)
   -  2 Kontroll-Artikel für Bedingung 5 (1x extrem einfache LS, 1x extremer Schachtelsatz)
4. Lädt die echten Modellgewichte (mBART SFT & DPO) und generiert vollständige Übersetzungen.
5. Berechnet für alle 50 Texte die automatisierten Metriken exakt wie in der Pipeline:
   - R_style (BiLSTM MixUp Regressor mit dynamischer Länge ohne Zero-Padding)
   - R_sem_as (SBERT Quelltreue zum AS-Text)
   - sim_ref (SBERT Ähnlichkeit zur Gold-Referenz)
   - Composite Reward (0.5 * R_style + 0.5 * R_sem)
   - Flesch Reading Ease (DE) & LIX Lesbarkeitsindex
   - Wort-, Satz- und Längenstatistiken
6. Verblindet und shuffelt alle 50 Textkarten (ITEM_01 bis ITEM_50).
7. Speichert:
   - data/expert_eval/study_source_articles.csv
   - data/expert_eval/blinded_items.json (für die Web-App des Experten)
   - data/expert_eval/secret_key_mapping.json (Ground Truth für die statistische Auswertung)
   - data/expert_eval/expert_study_master_table.csv (vollständige Master-Tabelle)
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm


# ==============================================================================
# REWARD MODEL ARCHITECTURE (BILSTM MIXUP REGRESSOR)
# ==============================================================================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out)


# ==============================================================================
# LINGUISTIC METRICS (LIX & FLESCH DE)
# ==============================================================================
def calculate_lix(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    long_words = sum(1 for w in words if len(w) > 6)
    return round((num_words / num_sentences) + (long_words * 100 / num_words), 2)


def calculate_flesch_de(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    num_syllables = 0
    for word in words:
        count = sum(1 for char in word if char in vowels)
        num_syllables += max(1, count)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    return round(180.0 - asl - (58.5 * asw), 2)


def get_text_statistics(text: str) -> Dict[str, Any]:
    words = [w for w in text.split() if w.strip()]
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    avg_word_len = round(sum(len(w) for w in words) / max(1, num_words), 2)
    avg_sent_len = round(num_words / num_sentences, 2)
    return {
        "word_count": num_words,
        "sentence_count": num_sentences,
        "avg_word_length": avg_word_len,
        "avg_sentence_length": avg_sent_len
    }


# ==============================================================================
# MODEL LOADER & TRANSLATION GENERATOR
# ==============================================================================
def load_seq2seq_model(model_path: str, base_model_name: str, device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel

    if not os.path.exists(model_path):
        print(f"[WARNUNG] Modellpfad nicht gefunden: {model_path}")
        return None, None

    print(f"Lade Modell aus: {model_path} (Basis: {base_model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    if "mbart" in model_path.lower() or "mbart" in base_model_name.lower():
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

    try:
        # Versuch 1: Standalone Modell direkt laden
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        print(f"  -> Erfolgreich als eigenständiges Modell geladen.")
    except Exception:
        # Versuch 2: PEFT LoRA Adapter auf Basismodell laden
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(device)
        model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload().to(device)
        print(f"  -> Erfolgreich als LoRA-Adapter gemerged.")

    model.eval()
    return model, tokenizer


def generate_translations_batch(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    max_source_len: int = 1024,
    max_target_len: int = 1024,
    batch_size: int = 2
) -> List[str]:
    if model is None or tokenizer is None:
        return texts

    gen_texts = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for b in range(num_batches):
        batch_src = texts[b * batch_size : (b + 1) * batch_size]
        inputs = tokenizer(
            batch_src,
            padding=True,
            truncation=True,
            max_length=max_source_len,
            return_tensors="pt"
        ).to(device)

        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_length": max_target_len,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
            "length_penalty": 1.0,
        }

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gen_texts.extend([d.strip() for d in decoded])

    return gen_texts


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Erstellt den 50-Item Disjoint Evaluationspool für Experten.")
    parser.add_argument("--corpus_csv", default="data/analysis/corpus_master.csv")
    parser.add_argument("--lh_json", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--sft_model_path", default="results/models/token_length_exp/sft_len1024")
    parser.add_argument("--dpo_model_path", default="results/models/token_length_exp/dpo_len1024")
    parser.add_argument("--base_model_name", default="facebook/mbart-large-50")
    parser.add_argument("--reward_model_path", default="results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt")
    parser.add_argument("--reward_vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--output_dir", default="data/expert_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    if not os.path.exists(args.sft_model_path) and os.path.exists("results/models/token_length_exp/sft_len1024"):
        args.sft_model_path = "results/models/token_length_exp/sft_len1024"
    if not os.path.exists(args.dpo_model_path) and os.path.exists("results/models/token_length_exp/dpo_len1024"):
        args.dpo_model_path = "results/models/token_length_exp/dpo_len1024"
    if not os.path.exists(args.reward_model_path):
        if os.path.exists("results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt"):
            args.reward_model_path = "results/models/regressor_length_exp/bilstm_mixup_regression_1024.pt"
        elif os.path.exists("results/models/bilstm_mixup_regression.pt"):
            args.reward_model_path = "results/models/bilstm_mixup_regression.pt"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/expert_eval", exist_ok=True)
    print(f"=== Erstelle 50-Item Disjoint Evaluationspool (Seed: {args.seed}, Device: {args.device}) ===")

    # 1. SpaCy Parser laden
    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        nlp = spacy.blank("de")
        nlp.add_pipe("sentencizer")

    # 2. Datenbasis laden und dynamischen Test-Split anwenden
    from sklearn.model_selection import train_test_split

    if not os.path.exists(args.corpus_csv):
        raise FileNotFoundError(f"Corpus Master nicht gefunden: {args.corpus_csv}")
    
    df_cm = pd.read_csv(args.corpus_csv)
    mask = (df_cm["semantic_similarity_8192"] >= 0.70) & (df_cm["semantic_similarity_8192"] <= 0.98)
    df_filtered = df_cm[mask].dropna(subset=["ls_text", "as_text"]).reset_index(drop=True)

    train_val_df, test_df = train_test_split(df_filtered, test_size=0.1, random_state=args.seed)
    print(f"Web-Korpus geladen: {len(df_filtered)} Paare | Held-Out Testset: {len(test_df)} Artikel")

    lh_data = []
    if os.path.exists(args.lh_json):
        with open(args.lh_json, "r", encoding="utf-8") as f:
            lh_data = json.load(f)
    print(f"Lebenshilfe Gold geladen: {len(lh_data)} Artikel")

    # 3. 48 disjunkte Quellartikel auswählen (kein Thema doppelt!)
    web_articles = test_df.to_dict("records")
    random.shuffle(web_articles)
    lh_articles = list(lh_data)
    random.shuffle(lh_articles)

    as_items_raw = web_articles[0:7] + [{"as_text": x["as_text"], "ls_text": x["ls_text"], "source": "lebenshilfe"} for x in lh_articles[0:5]]
    gold_items_raw = web_articles[7:14] + [{"as_text": x["as_text"], "ls_text": x["ls_text"], "source": "lebenshilfe"} for x in lh_articles[5:10]]
    sft_items_raw = web_articles[14:22] + [{"as_text": x["as_text"], "ls_text": x["ls_text"], "source": "lebenshilfe"} for x in lh_articles[10:14]]
    dpo_items_raw = web_articles[22:30] + [{"as_text": x["as_text"], "ls_text": x["ls_text"], "source": "lebenshilfe"} for x in lh_articles[14:18]]

    print(f"\nZuweisung der 48 Studien-Artikel:")
    print(f"  Bedingung 1 (AS Original): {len(as_items_raw)} disjunkte Artikel")
    print(f"  Bedingung 2 (Gold LS):     {len(gold_items_raw)} disjunkte Artikel")
    print(f"  Bedingung 3 (mBART SFT):   {len(sft_items_raw)} disjunkte Artikel")
    print(f"  Bedingung 4 (mBART DPO):   {len(dpo_items_raw)} disjunkte Artikel")

    # 4. SFT und DPO Inferenz
    print("\n--- Starte SFT Übersetzung (12 Texte) ---")
    sft_model, sft_tok = load_seq2seq_model(args.sft_model_path, args.base_model_name, args.device)
    sft_sources = [str(x["as_text"]).strip() for x in sft_items_raw]
    sft_generated_texts = generate_translations_batch(sft_model, sft_tok, sft_sources, args.device)
    del sft_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Starte DPO Übersetzung (12 Texte) ---")
    dpo_model, dpo_tok = load_seq2seq_model(args.dpo_model_path, args.base_model_name, args.device)
    dpo_sources = [str(x["as_text"]).strip() for x in dpo_items_raw]
    dpo_generated_texts = generate_translations_batch(dpo_model, dpo_tok, dpo_sources, args.device)
    del dpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Reward-Modell (BiLSTM MixUp) und SBERT initialisieren
    print("\n--- Initialisiere Metrik-Pipeline (BiLSTM Regressor & SBERT) ---")
    with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        stoi = vocab_raw.get("stoi", vocab_raw)
    unk_idx = stoi.get("<unk>") or stoi.get("<UNK>") or 1

    bilstm = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(args.device)
    state = torch.load(args.reward_model_path, map_location=args.device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    bilstm.load_state_dict(state)
    bilstm.eval()
    print(f"BiLSTM MixUp Regressor geladen aus: {args.reward_model_path}")

    from sentence_transformers import SentenceTransformer, util
    sbert = SentenceTransformer(args.sbert_model_name, device=args.device)
    print(f"SentenceTransformer geladen: {args.sbert_model_name}")

    def score_simplicity(text_str: str) -> float:
        doc = nlp(str(text_str or ""))
        tokens = [t.text.lower() for t in doc if not t.is_space]
        indices = [stoi.get(t, unk_idx) for t in tokens[:1024]]
        if len(indices) == 0:
            indices = [0]
        inp_tensor = torch.tensor([indices], dtype=torch.long, device=args.device)
        with torch.no_grad():
            score = bilstm(inp_tensor).item()
        return round(float(score), 4)

    # 6. Zusammenstellung der 50 Einzeltexte
    candidate_items = []

    # Bedingung 1: AS Original (12x)
    for idx, raw in enumerate(as_items_raw):
        candidate_items.append({
            "source_article_id": f"SRC_AS_{idx+1:02d}",
            "condition": "AS_Original",
            "source_domain": raw.get("source", "web"),
            "domain_type": "Out-of-Domain" if raw.get("source") == "lebenshilfe" else "In-Domain Held-Out",
            "source_text_as": str(raw["as_text"]).strip(),
            "target_text_ls_gold": str(raw["ls_text"]).strip(),
            "evaluation_text": str(raw["as_text"]).strip()
        })

    # Bedingung 2: Gold LS (12x)
    for idx, raw in enumerate(gold_items_raw):
        candidate_items.append({
            "source_article_id": f"SRC_GOLD_{idx+1:02d}",
            "condition": "Gold_Standard_LS",
            "source_domain": raw.get("source", "web"),
            "domain_type": "Out-of-Domain" if raw.get("source") == "lebenshilfe" else "In-Domain Held-Out",
            "source_text_as": str(raw["as_text"]).strip(),
            "target_text_ls_gold": str(raw["ls_text"]).strip(),
            "evaluation_text": str(raw["ls_text"]).strip()
        })

    # Bedingung 3: mBART SFT (12x)
    for idx, (raw, gen_t) in enumerate(zip(sft_items_raw, sft_generated_texts)):
        candidate_items.append({
            "source_article_id": f"SRC_SFT_{idx+1:02d}",
            "condition": "mBART_SFT",
            "source_domain": raw.get("source", "web"),
            "domain_type": "Out-of-Domain" if raw.get("source") == "lebenshilfe" else "In-Domain Held-Out",
            "source_text_as": str(raw["as_text"]).strip(),
            "target_text_ls_gold": str(raw["ls_text"]).strip(),
            "evaluation_text": gen_t
        })

    # Bedingung 4: mBART DPO (12x)
    for idx, (raw, gen_t) in enumerate(zip(dpo_items_raw, dpo_generated_texts)):
        candidate_items.append({
            "source_article_id": f"SRC_DPO_{idx+1:02d}",
            "condition": "mBART_DPO",
            "source_domain": raw.get("source", "web"),
            "domain_type": "Out-of-Domain" if raw.get("source") == "lebenshilfe" else "In-Domain Held-Out",
            "source_text_as": str(raw["as_text"]).strip(),
            "target_text_ls_gold": str(raw["ls_text"]).strip(),
            "evaluation_text": gen_t
        })

    # Bedingung 5: 2 Kontroll-Texte
    candidate_items.append({
        "source_article_id": "SRC_CTRL_01",
        "condition": "Control_Easy_LS",
        "source_domain": "calibration",
        "domain_type": "Control",
        "source_text_as": "Das ist ein Kontrolltext in sehr einfacher Leichter Sprache.",
        "target_text_ls_gold": "Das ist ein Kontrolltext in sehr einfacher Leichter Sprache.",
        "evaluation_text": "Das ist ein Haus.\nIn dem Haus wohnen Menschen.\nDie Menschen haben einen Garten.\nDie Sonne scheint."
    })
    candidate_items.append({
        "source_article_id": "SRC_CTRL_02",
        "condition": "Control_Hard_AS",
        "source_domain": "calibration",
        "domain_type": "Control",
        "source_text_as": "Das ist ein extremer Schachtelsatz aus dem Verwaltungsrecht.",
        "target_text_ls_gold": "Das ist ein extremer Schachtelsatz.",
        "evaluation_text": "Die Zuständigkeit der nach Landesrecht zur Ausführung dieses Gesetzes bestimmten Behörden erstreckt sich, unbeschadet abweichender bundesgesetzlicher Vorschriften, auf die Überwachung der Einhaltung aller in den jeweiligen Rechtsverordnungen statuierten Pflichten durch die betroffenen Rechtssubjekte."
    })

    print(f"\nGesamtzahl zusammengestellter Einzeltexte: {len(candidate_items)}")

    # 7. Automatisierte Metriken berechnen
    print("\nBerechne automatisierte Metriken (R_style, R_sem, Flesch, LIX) für alle 50 Texte...")
    for item in tqdm(candidate_items, desc="Berechne Metriken"):
        eval_txt = item["evaluation_text"]
        as_txt = item["source_text_as"]
        gold_txt = item["target_text_ls_gold"]

        item["r_style"] = score_simplicity(eval_txt)

        with torch.no_grad():
            emb_eval = sbert.encode(eval_txt, convert_to_tensor=True, show_progress_bar=False)
            emb_as = sbert.encode(as_txt, convert_to_tensor=True, show_progress_bar=False)
            emb_gold = sbert.encode(gold_txt, convert_to_tensor=True, show_progress_bar=False)
            item["r_sem_as"] = round(float(util.cos_sim(emb_eval, emb_as)[0][0]), 4)
            item["sim_gold"] = round(float(util.cos_sim(emb_eval, emb_gold)[0][0]), 4)

        item["composite_reward"] = round(0.5 * item["r_style"] + 0.5 * item["r_sem_as"], 4)
        item["lix"] = calculate_lix(eval_txt)
        item["flesch_de"] = calculate_flesch_de(eval_txt)

        stats = get_text_statistics(eval_txt)
        item.update(stats)

    # 8. Verblinden und Shuffeln
    print("\nVerblinde und shuffele alle 50 Items...")
    random.shuffle(candidate_items)

    blinded_items_list = []
    secret_key_mapping = {}
    master_rows = []

    for idx, item in enumerate(candidate_items):
        item_id = f"ITEM_{idx+1:02d}"
        
        blinded_items_list.append({
            "item_id": item_id,
            "text": item["evaluation_text"],
            "word_count": item["word_count"],
            "sentence_count": item["sentence_count"]
        })

        secret_key_mapping[item_id] = {
            "blinded_item_id": item_id,
            "source_article_id": item["source_article_id"],
            "condition": item["condition"],
            "source_domain": item["source_domain"],
            "domain_type": item["domain_type"],
            "evaluation_text": item["evaluation_text"],
            "source_text_as": item["source_text_as"],
            "target_text_ls_gold": item["target_text_ls_gold"],
            "r_style": item["r_style"],
            "r_sem_as": item["r_sem_as"],
            "sim_gold": item["sim_gold"],
            "composite_reward": item["composite_reward"],
            "lix": item["lix"],
            "flesch_de": item["flesch_de"],
            "word_count": item["word_count"],
            "sentence_count": item["sentence_count"],
            "avg_word_length": item["avg_word_length"],
            "avg_sentence_length": item["avg_sentence_length"]
        }
        master_rows.append(secret_key_mapping[item_id])

    # 9. Speichern
    blinded_path = os.path.join(args.output_dir, "blinded_items.json")
    secret_path = os.path.join(args.output_dir, "secret_key_mapping.json")
    master_csv_path = os.path.join(args.output_dir, "expert_study_master_table.csv")

    with open(blinded_path, "w", encoding="utf-8") as f:
        json.dump(blinded_items_list, f, ensure_ascii=False, indent=2)

    with open(secret_path, "w", encoding="utf-8") as f:
        json.dump(secret_key_mapping, f, ensure_ascii=False, indent=2)

    df_master = pd.DataFrame(master_rows)
    df_master.to_csv(master_csv_path, index=False, encoding="utf-8")

    print(f"\n[FERTIG] Experten-Evaluationspool erfolgreich erstellt!")
    print(f"  -> Verblindete Items (Web-App): {blinded_path} ({len(blinded_items_list)} Items)")
    print(f"  -> Geheimer Schlüssel:         {secret_path}")
    print(f"  -> Master-Tabelle CSV:         {master_csv_path}")

    print("\n--- Deskriptive Übersicht nach Bedingung ---")
    summary = df_master.groupby("condition")[["r_style", "r_sem_as", "flesch_de", "lix", "word_count"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
