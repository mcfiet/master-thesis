#!/usr/bin/env python3
import os
import sys
import json
import argparse
import math
import re
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def count_vowels(word: str) -> int:
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    return max(1, sum(1 for c in word if c in vowels))


def compute_metrics(text: str) -> Dict[str, float]:
    tokens = [w for w in re.findall(r"\b[A-Za-zÄÖÜäöüß0-9\-_]+\b", text) if w.strip()]
    if not tokens:
        return {
            "flesch_de": 0.0,
            "lix": 0.0,
            "wstf_1": 0.0,
            "wstf_4": 0.0,
            "n_words": 0,
            "n_sents": 0,
            "avg_sent_len": 0.0,
            "avg_word_syllables": 0.0,
        }

    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_sents = max(1, len(sents))
    n_words = len(tokens)

    syllables = [count_vowels(w) for w in tokens]
    total_syllables = sum(syllables)
    long_words_lix = sum(1 for w in tokens if len(w) > 6)
    poly_words_3syl = sum(1 for s in syllables if s >= 3)
    mono_words = sum(1 for s in syllables if s == 1)

    asl = n_words / n_sents
    asw = total_syllables / n_words
    long_ratio = (long_words_lix / n_words) * 100.0
    poly_ratio = (poly_words_3syl / n_words) * 100.0
    mono_ratio = (mono_words / n_words) * 100.0

    # 1. Flesch Reading Ease (Deutsche Fassung nach Amstad)
    flesch_de = 180.0 - asl - (58.5 * asw)

    # 2. LIX (Lesbarkeitsindex)
    lix = asl + long_ratio

    # 3. Wiener Sachtextformel (WSTF 1 & 4)
    wstf_1 = (0.1935 * poly_ratio) + (0.1672 * asl) + (0.1297 * long_ratio) - (0.0327 * mono_ratio) - 0.8749
    wstf_4 = (0.2656 * asl) + (0.2744 * poly_ratio) - 1.693

    return {
        "flesch_de": flesch_de,
        "lix": lix,
        "wstf_1": wstf_1,
        "wstf_4": wstf_4,
        "n_words": n_words,
        "n_sents": n_sents,
        "avg_sent_len": asl,
        "avg_word_syllables": asw,
    }


if HAS_TORCH:
    class BiLSTMRegressor(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
            super().__init__()
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
            return self.sigmoid(out).squeeze(-1)


def load_vocab(path: str) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        v = json.load(f)
    return v.get("stoi", v)


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 256) -> List[int]:
    tokens = [t.lower() for t in re.findall(r"\b\w+\b", text)[:max_len]]
    enc = [vocab.get(t, 1) for t in tokens] or [0]
    return enc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.json")
    parser.add_argument("--output_csv", default="results/evaluation/synthetic_rule_benchmark_256_eval.csv")
    parser.add_argument("--summary_json", default="results/evaluation/synthetic_rule_benchmark_256_summary.json")
    parser.add_argument("--model_path", default="results/models/regressor_length_exp/bilstm_mixup_regression_256.pt")
    parser.add_argument("--vocab_path", default="data/regressor_length_exp/mixup_vocab_256.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    device = torch.device("cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu") if HAS_TORCH else None
    neural_model = None
    vocab = {}
    if HAS_TORCH and os.path.exists(args.model_path) and os.path.exists(args.vocab_path):
        try:
            vocab = load_vocab(args.vocab_path)
            st = torch.load(args.model_path, map_location=device)
            if "model_state_dict" in st:
                st = st["model_state_dict"]
            emb_w = st.get("embedding.weight", None)
            v_size = emb_w.shape[0] if emb_w is not None else len(vocab)
            neural_model = BiLSTMRegressor(v_size).to(device)
            neural_model.load_state_dict(st)
            neural_model.eval()
            print("256-Token BiLSTM Regressor geladen!")
        except Exception as e:
            print(f"Warning: Konnte BiLSTM-Modell nicht laden: {e}")

    # Berechne Metriken für alle Textvarianten
    rows = []
    base_lookup = {}

    for it in items:
        t_id = it["text_id"]
        v_name = it["variant"]
        txt = it["text"]

        m = compute_metrics(txt)
        row = {
            "text_id": t_id,
            "title": it["title"],
            "domain": it["domain"],
            "variant": v_name,
            "token_count": it["token_count"],
            "syllable_count": it["syllable_count"],
            "flesch_de": m["flesch_de"],
            "lix": m["lix"],
            "wstf_1": m["wstf_1"],
            "wstf_4": m["wstf_4"],
            "avg_sent_len": m["avg_sent_len"],
            "avg_word_syllables": m["avg_word_syllables"],
        }

        if neural_model is not None:
            with torch.no_grad():
                enc = torch.tensor([encode_text(txt, vocab, 256)], dtype=torch.long).to(device)
                score = float(neural_model(enc).cpu().item())
                row["neural_score_256"] = score
        else:
            row["neural_score_256"] = 0.0

        if v_name == "base_ls":
            base_lookup[t_id] = row

        rows.append(row)

    # Berechne Deltas vs base_ls
    for row in rows:
        t_id = row["text_id"]
        b = base_lookup.get(t_id, row)
        row["delta_flesch_de"] = row["flesch_de"] - b["flesch_de"]
        row["delta_lix"] = row["lix"] - b["lix"]
        row["delta_wstf_1"] = row["wstf_1"] - b["wstf_1"]
        row["delta_wstf_4"] = row["wstf_4"] - b["wstf_4"]
        row["delta_neural_score"] = row["neural_score_256"] - b["neural_score_256"]
        row["token_delta"] = row["token_count"] - b["token_count"]
        row["syllable_delta"] = row["syllable_count"] - b["syllable_count"]

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"Ergebnisse gespeichert: {args.output_csv}")

    # Aggregierte Zusammenfassung nach Varianten
    summary_by_variant = []
    for var_name in df["variant"].unique():
        df_v = df[df["variant"] == var_name]
        summary_by_variant.append({
            "variant": var_name,
            "mean_flesch_de": round(float(df_v["flesch_de"].mean()), 3),
            "mean_delta_flesch": round(float(df_v["delta_flesch_de"].mean()), 4),
            "mean_lix": round(float(df_v["lix"].mean()), 3),
            "mean_delta_lix": round(float(df_v["delta_lix"].mean()), 4),
            "mean_wstf_4": round(float(df_v["wstf_4"].mean()), 3),
            "mean_delta_wstf_4": round(float(df_v["delta_wstf_4"].mean()), 4),
            "mean_neural_score_256": round(float(df_v["neural_score_256"].mean()), 4),
            "mean_delta_neural_score": round(float(df_v["delta_neural_score"].mean()), 4),
            "mean_token_delta": round(float(df_v["token_delta"].mean()), 2),
            "mean_syllable_delta": round(float(df_v["syllable_delta"].mean()), 2),
        })

    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_by_variant, f, indent=2, ensure_ascii=False)
    print(f"Zusammenfassung gespeichert: {args.summary_json}")


if __name__ == "__main__":
    main()
