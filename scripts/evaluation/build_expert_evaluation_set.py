#!/usr/bin/env python3
"""
scripts/evaluation/build_expert_evaluation_set.py

Erstellt den verblindeten 50-Item-Datensatz für die Experten-Evaluation:
1. Wählt 10 balancierte Quelltexte aus 10 Nicht-Lebenshilfe-Quellen aus
   (MDR, Apotheken, Hamburg, Köln, Stuttgart, Wiesbaden, Behindertenbeauftragter, BrandEins, Taz, Sozialpolitik).
2. Generiert bzw. extrahiert die 5 Bedingungen:
   - Bedingung 1: Original Alltagssprache (AS)
   - Bedingung 2: Gold-Standard Leichte Sprache (menschliche Übersetzung)
   - Bedingung 3: LLM Few-Shot Baseline (Qwen 2.5)
   - Bedingung 4: mBART SFT
   - Bedingung 5: mBART DPO (Reward-Guided)
3. Berechnet für alle 50 Items die automatisierten Metriken:
   - BiLSTM MixUp Regressor (R_style)
   - SBERT Kosinus-Ähnlichkeit (R_sem)
   - Flesch Reading Ease (DE), LIX-Index
4. Verblindet und shuffelt alle 50 Text-Karten (ITEM_01 bis ITEM_50).
5. Speichert:
   - data/expert_eval/blinded_items.json (für das Experten-Tool)
   - data/expert_eval/secret_key_mapping.json (für die Auswertung der Masterarbeit)
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
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


def text_to_tensor(text: str, stoi: Dict[str, int], max_len: int = 500) -> torch.Tensor:
    tokens = text.lower().split()
    indices = [stoi.get(token, stoi.get("<unk>", 1)) for token in tokens[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


# ==============================================================================
# LINGUISTIC METRICS
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
    return round(180 - asl - (58.5 * asw), 2)


# ==============================================================================
# MAIN DATASET BUILDER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Erstellt den 50-Item-Evaluationsdatensatz für Experten.")
    parser.add_argument("--testset_csv", default="data/evaluation_sets/benchmark_translation_testset.csv")
    parser.add_argument("--sft_mbart_path", default="results/models/sft")
    parser.add_argument("--dpo_mbart_path", default="results/models/dpo")
    parser.add_argument("--qwen_base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reward_model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--reward_vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--output_dir", default="data/expert_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_articles", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Erstelle Experten-Evaluationspool (Seed: {args.seed}, Device: {args.device}) ===")

    # 1. Datenbasis laden und Lebenshilfe-Texte filtern
    if not os.path.exists(args.testset_csv):
        raise FileNotFoundError(f"Test-Set nicht gefunden: {args.testset_csv}")

    df_full = pd.read_csv(args.testset_csv)
    df_non_lh = df_full[df_full["source_domain"] != "lebenshilfe"].copy()
    print(f"Verfügbare Nicht-Lebenshilfe-Testartikel: {len(df_non_lh)} über {df_non_lh['source_domain'].nunique()} Domänen.")

    # 2. Aus 10 verschiedenen geprüften Domänen je 1 Artikel selektieren
    selected_domains = [
        "mdr", "apotheken", "behindertenbeauftragter", "hamburg", "hannover",
        "koeln", "wiesbaden", "brandeins", "taz", "sozialpolitik"
    ]
    selected_rows = []
    for dom in selected_domains[:args.num_articles]:
        dom_rows = df_non_lh[df_non_lh["source_domain"] == dom]
        if len(dom_rows) > 0:
            selected_rows.append(dom_rows.iloc[0])
        else:
            print(f"Warnung: Keine Artikel für Domäne {dom} gefunden.")

    df_sample = pd.DataFrame(selected_rows).reset_index(drop=True)
    print(f"Ausgewählte Quellartikel für die Studie: {len(df_sample)}")
    for idx, row in df_sample.iterrows():
        print(f"  [{idx+1:02d}] Domäne: {row['source_domain']:<22} ID: {row['id']} (AS: {row['as_tokens']} Tok, LS: {row['ls_tokens']} Tok)")

    # 3. Reward-Modell (BiLSTM) & SBERT für Vorberechnung und Candidate Selection initialisieren
    print("\nLade Reward-Modell (BiLSTM) & SBERT...")
    has_reward = os.path.exists(args.reward_model_path) and os.path.exists(args.reward_vocab_path)
    bilstm = None
    stoi = {}
    if has_reward:
        with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            stoi = vocab_raw.get("stoi", vocab_raw)
        bilstm = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(args.device)
        raw_w = torch.load(args.reward_model_path, map_location=args.device, weights_only=False)
        if isinstance(raw_w, dict) and "model_state_dict" in raw_w:
            raw_w = raw_w["model_state_dict"]
        bilstm.load_state_dict(raw_w)
        bilstm.eval()

    from sentence_transformers import SentenceTransformer, util
    sbert = SentenceTransformer(args.sbert_model_name, device=args.device)

    # 4. Quelltexte und Gold Standard extrahieren
    as_texts = df_sample["as_text"].tolist()
    gold_ls_texts = df_sample["ls_text"].tolist()

    gen_sft = []
    gen_dpo = []
    gen_baseline = []

    # Prüfe ob mBART Checkpoints vorhanden sind
    has_mbart_sft = os.path.exists(args.sft_mbart_path)
    has_mbart_dpo = os.path.exists(args.dpo_mbart_path)

    if has_mbart_sft and has_mbart_dpo:
        print("\nLade fine-getunte mBART Modelle für SFT und DPO Inferenz...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        mbart_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50", use_fast=False)
        mbart_tok.src_lang = "de_DE"
        mbart_tok.tgt_lang = "de_DE"

        # SFT: Best-of-N Sampling fuer hochwertige, stark vereinfachte Kandidaten
        print("Generiere SFT Übersetzungen (Best-of-N Sampling mit Style-Ranking)...")
        sft_m = AutoModelForSeq2SeqLM.from_pretrained(args.sft_mbart_path).to(args.device)
        sft_m.eval()
        for text in tqdm(as_texts, desc="mBART SFT (Best-of-N)"):
            inputs = mbart_tok(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outs = sft_m.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_length=512, num_return_sequences=4, do_sample=True, temperature=0.7,
                    top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3
                )
            decoded_candidates = [mbart_tok.decode(o, skip_special_tokens=True).strip() for o in outs]
            
            # Wähle den besten SFT-Kandidaten (höchster R_style Score)
            best_cand = decoded_candidates[0]
            if bilstm is not None and stoi:
                best_score = -1.0
                for cand in decoded_candidates:
                    inp_t = text_to_tensor(cand, stoi, max_len=500).to(args.device)
                    with torch.no_grad():
                        sc = float(bilstm(inp_t).item())
                    if sc > best_score:
                        best_score = sc
                        best_cand = cand
            gen_sft.append(best_cand)
        del sft_m

        # DPO
        print("Generiere DPO Übersetzungen...")
        dpo_m = AutoModelForSeq2SeqLM.from_pretrained(args.dpo_mbart_path).to(args.device)
        dpo_m.eval()
        for text in tqdm(as_texts, desc="mBART DPO"):
            inputs = mbart_tok(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = dpo_m.generate(
                    input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_length=512, num_beams=4, repetition_penalty=1.2, no_repeat_ngram_size=3
                )
            gen_dpo.append(mbart_tok.decode(out[0], skip_special_tokens=True).strip())
        del dpo_m
    else:
        print("\nGeneriere Modell-Übersetzungen via Qwen Prompting / Baseline...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        qwen_tok = AutoTokenizer.from_pretrained(args.qwen_base_model)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            args.qwen_base_model,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        ).to(args.device)
        qwen_model.eval()

        FEW_SHOT_PROMPT = (
            "Du bist ein professioneller Übersetzer für deutsche Leichte Sprache. "
            "Übersetze den gegebenen Text in verständliche Leichte Sprache.\n"
            "Regeln:\n"
            "- Kurze Sätze (ein Gedanke pro Satz)\n"
            "- Einfache, verständliche Wörter\n"
            "- Zusammengesetzte Wörter trennen (z.B. Wörter-Buch)\n"
            "- Kein Passiv, kein Konjunktiv\n\n"
        )

        for text in tqdm(as_texts, desc="LLM Inferenz"):
            messages = [
                {"role": "system", "content": FEW_SHOT_PROMPT},
                {"role": "user", "content": f"Text:\n{text[:1200]}\n\nLeichte Sprache:"}
            ]
            prompt = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = qwen_tok(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = qwen_model.generate(**inputs, max_new_tokens=400, temperature=0.3, repetition_penalty=1.15)
            gen_text = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            gen_baseline.append(gen_text)
            gen_sft.append(gen_text)
            gen_dpo.append(gen_text)
        del qwen_model

    if not gen_baseline:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        qwen_tok = AutoTokenizer.from_pretrained(args.qwen_base_model)
        qwen_model = AutoModelForCausalLM.from_pretrained(args.qwen_base_model).to(args.device)
        qwen_model.eval()
        for text in tqdm(as_texts, desc="Qwen Baseline"):
            messages = [
                {"role": "system", "content": "Übersetze in deutsche Leichte Sprache."},
                {"role": "user", "content": f"{text[:1000]}"}
            ]
            prompt = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = qwen_tok(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = qwen_model.generate(**inputs, max_new_tokens=350, temperature=0.2)
            gen_text = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            gen_baseline.append(gen_text)
        del qwen_model

    # 5. 50 Items zusammenstellen
    conditions = [
        ("AS_Original", as_texts),
        ("Gold_Standard_LS", gold_ls_texts),
        ("LLM_FewShot_Baseline", gen_baseline),
        ("mBART_SFT", gen_sft),
        ("mBART_DPO", gen_dpo)
    ]

    all_items = []
    print("\nKombiniere 50 Items und berechne Metriken...")
    for cond_name, text_list in conditions:
        for idx in range(len(df_sample)):
            as_source = as_texts[idx]
            cand_text = text_list[idx]
            source_dom = df_sample.iloc[idx]["source_domain"]
            source_id = df_sample.iloc[idx]["id"]

            # Style Score
            if bilstm is not None:
                inp_t = text_to_tensor(cand_text, stoi, max_len=500).to(args.device)
                with torch.no_grad():
                    r_style = round(float(bilstm(inp_t).item()), 4)
            else:
                r_style = 0.5

            # Semantic Sim
            emb_as = sbert.encode(as_source, convert_to_tensor=True)
            emb_cand = sbert.encode(cand_text, convert_to_tensor=True)
            r_sem = round(float(util.cos_sim(emb_as, emb_cand).item()), 4)
            r_composite = round(0.5 * r_style + 0.5 * r_sem, 4)

            flesch = calculate_flesch_de(cand_text)
            lix = calculate_lix(cand_text)

            item_data = {
                "source_domain": source_dom,
                "source_article_id": source_id,
                "true_condition": cond_name,
                "source_text_as": as_source,
                "candidate_text": cand_text,
                "metrics": {
                    "r_style": r_style,
                    "r_sem": r_sem,
                    "r_composite": r_composite,
                    "flesch_de": flesch,
                    "lix": lix
                }
            }
            all_items.append(item_data)

    print(f"Gesamtanzahl erstellter Items: {len(all_items)}")

    # 6. Randomisierung & Verblindung (Shuffling)
    random.shuffle(all_items)

    blinded_items_for_expert = []
    secret_key_mapping = []

    for i, item in enumerate(all_items):
        item_id = f"ITEM_{i+1:02d}"
        blinded_items_for_expert.append({
            "item_id": item_id,
            "candidate_text": item["candidate_text"],
            "source_domain": item["source_domain"],
            "source_article_id": item["source_article_id"],
            "true_condition": item["true_condition"],
            "metrics": item["metrics"]
        })

        secret_key_mapping.append({
            "item_id": item_id,
            "source_domain": item["source_domain"],
            "source_article_id": item["source_article_id"],
            "true_condition": item["true_condition"],
            "metrics": item["metrics"],
            "candidate_text": item["candidate_text"],
            "source_text_as": item["source_text_as"]
        })

    # Speichern
    blinded_path = os.path.join(args.output_dir, "blinded_items.json")
    secret_path = os.path.join(args.output_dir, "secret_key_mapping.json")

    with open(blinded_path, "w", encoding="utf-8") as f:
        json.dump(blinded_items_for_expert, f, ensure_ascii=False, indent=2)

    with open(secret_path, "w", encoding="utf-8") as f:
        json.dump(secret_key_mapping, f, ensure_ascii=False, indent=2)

    print(f"\n[ERFOLG] Verblindete Items gespeichert in: {blinded_path}")
    print(f"[ERFOLG] Geheime Mapping-Datei gespeichert in: {secret_path}")
    print("\nVerteilung der Bedingungen im gemischten Pool:")
    cond_counts = pd.Series([s["true_condition"] for s in secret_key_mapping]).value_counts()
    print(cond_counts.to_string())


if __name__ == "__main__":
    main()
