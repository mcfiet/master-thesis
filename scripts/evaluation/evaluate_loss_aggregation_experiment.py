#!/usr/bin/env python3
"""
Evaluierungsskript für das Loss-Aggregations-Experiment (Sum vs. Mean DPO)
Masterarbeit: Automatische Übersetzung von Alltagssprache (AS) in Leichte Sprache (LS)

Vergleicht:
1. SFT Baseline (Referenz)
2. DPO Sum (Klassische Summierung der Log-Wahrscheinlichkeiten)
3. DPO Mean (Längen-normalisierter / Per-Token DPO)
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


# ==============================================================================
# BILSTM REGRESSOR DEFINITION (REWARD MODEL)
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
# LEXIKALISCHE METRIKEN (BLEU & ROUGE-L)
# ==============================================================================
def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))

def compute_sentence_bleu(ref_toks: List[str], cand_toks: List[str], max_n: int = 4) -> float:
    if len(cand_toks) == 0 or len(ref_toks) == 0:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng = compute_ngram_counts(cand_toks, n)
        ref_ng = compute_ngram_counts(ref_toks, n)
        if sum(cand_ng.values()) == 0:
            precisions.append(0.0)
            continue
        matches = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
        precisions.append(matches / max(1, sum(cand_ng.values())))
    if min(precisions) <= 1e-9:
        geom_mean = 0.0
    else:
        geom_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    c, r = len(cand_toks), len(ref_toks)
    bp = 1.0 if c > r else np.exp(1.0 - (r / max(1, c)))
    return float(bp * geom_mean)

def compute_rouge_l(ref_toks: List[str], cand_toks: List[str]) -> Dict[str, float]:
    if len(ref_toks) == 0 or len(cand_toks) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    m, n = len(ref_toks), len(cand_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_toks[i] == cand_toks[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    p = lcs / max(1, n)
    r = lcs / max(1, m)
    f1 = 2 * p * r / max(1e-9, (p + r))
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


# ==============================================================================
# EVALUATOR ENGINE
# ==============================================================================
class LossAggregationEvaluator:
    def __init__(
        self,
        reward_model_path: str = "results/models/bilstm_mixup_regression.pt",
        reward_vocab_path: str = "data/vocabs/mixup_vocab.json",
        sbert_model_name: str = "jinaai/jina-embeddings-v2-base-de",
        sbert_max_seq_len: int = 8192,
        max_seq_len: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self.sbert_max_seq_len = sbert_max_seq_len

        print("Initialisiere SpaCy Tokenizer...")
        self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])

        # 1. Vocab
        print(f"Lade Vokabular aus {reward_vocab_path}...")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # 2. BiLSTM Regressor
        print(f"Lade BiLSTM Regressor aus {reward_model_path}...")
        self.regressor = BiLSTMRegressor(vocab_size=len(self.vocab)).to(self.device)
        try:
            state_dict = torch.load(reward_model_path, map_location=self.device, weights_only=False)
        except Exception:
            state_dict = torch.load(reward_model_path, map_location=self.device)
        if "model_state_dict" in state_dict:
            self.regressor.load_state_dict(state_dict["model_state_dict"])
        else:
            self.regressor.load_state_dict(state_dict)
        self.regressor.eval()

        # 3. SBERT
        print(f"Lade SBERT Modell ({sbert_model_name}, max_seq_len={sbert_max_seq_len})...")
        self.sbert = SentenceTransformer(sbert_model_name, device=self.device, trust_remote_code=True)
        if sbert_max_seq_len and hasattr(self.sbert, "max_seq_length"):
            self.sbert.max_seq_length = sbert_max_seq_len

    @torch.no_grad()
    def predict_simplicity(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_preds = []
        for text in texts:
            doc = self.nlp(text)
            ids = [self.vocab.get(token.text.lower(), 1) for token in doc if not token.is_space][:self.max_seq_len]
            if not ids:
                ids = [0]
            tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
            p = self.regressor(tensor).squeeze().item()
            all_preds.append(float(p))
        return np.array(all_preds)

    @torch.no_grad()
    def predict_semantic_sim(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        effective_len = getattr(self.sbert, "max_seq_length", self.sbert_max_seq_len)
        if effective_len > 4096:
            sbert_batch_size = 2
        elif effective_len > 1024:
            sbert_batch_size = 4
        elif effective_len > 512:
            sbert_batch_size = 8
        else:
            sbert_batch_size = 16

        all_sims = []
        for i in range(0, len(texts1), sbert_batch_size):
            b1 = texts1[i : i + sbert_batch_size]
            b2 = texts2[i : i + sbert_batch_size]
            emb1 = self.sbert.encode(b1, convert_to_tensor=True, batch_size=sbert_batch_size, show_progress_bar=False)
            emb2 = self.sbert.encode(b2, convert_to_tensor=True, batch_size=sbert_batch_size, show_progress_bar=False)
            sims = util.cos_sim(emb1, emb2).diag().cpu().numpy()
            all_sims.extend(sims.tolist() if isinstance(sims, np.ndarray) and sims.ndim > 0 else [float(sims)])
        return np.array(all_sims)

    def evaluate_model(
        self,
        model_name_or_path: str,
        display_name: str,
        base_model_name: str,
        as_texts: List[str],
        ls_ref_texts: List[str],
        max_source_len: int = 256,
        max_target_len: int = 256,
        batch_size: int = 8,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        print(f"\n{'='*70}")
        print(f"Evaluierung: {display_name} ({model_name_or_path})")
        print(f"{'='*70}")

        # Tokenizer
        tok_path = model_name_or_path if os.path.exists(os.path.join(model_name_or_path, "tokenizer_config.json")) else base_model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

        de_id = None
        if hasattr(tokenizer, "lang_code_to_id") and tokenizer.lang_code_to_id and "de_DE" in tokenizer.lang_code_to_id:
            de_id = tokenizer.lang_code_to_id["de_DE"]
        elif hasattr(tokenizer, "convert_tokens_to_ids"):
            de_id = tokenizer.convert_tokens_to_ids("de_DE")
        if de_id is None or de_id == getattr(tokenizer, "unk_token_id", None):
            de_id = 250003
        print(f"  -> Konfigurierte Ziel-Sprach-ID (de_DE): {de_id}")

        # Model Loading
        config = AutoConfig.from_pretrained(base_model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"[MODELL-LADEN] {display_name}")
        print(f"  -> Pfad: {model_name_or_path}")

        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path, torch_dtype=dtype
            ).to(self.device)
        except Exception:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name, torch_dtype=dtype
            )
            model = PeftModel.from_pretrained(model, model_name_or_path).merge_and_unload().to(self.device)

        print(f"  -> Instanziierte Modell-Klasse: {model.__class__.__name__} ({dtype})\n")
        model.eval()

        # Inferenz
        gen_texts = []
        for i in tqdm(range(0, len(as_texts), batch_size), desc=f"Inferenz {display_name}"):
            batch_src = as_texts[i : i + batch_size]
            inputs = tokenizer(batch_src, max_length=max_source_len, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_length": max_target_len,
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
            }

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_texts.extend(decoded)

        del model
        torch.cuda.empty_cache()

        # Metrik-Berechnung
        r_style = self.predict_simplicity(gen_texts)
        r_sem_raw = self.predict_semantic_sim(as_texts, gen_texts)
        r_sem_as = np.clip((r_sem_raw + 1.0) / 2.0, 0.0, 1.0)
        sim_ref_raw = self.predict_semantic_sim(ls_ref_texts, gen_texts)
        sim_ref = np.clip((sim_ref_raw + 1.0) / 2.0, 0.0, 1.0)

        composite_reward = 0.5 * r_style + 0.5 * r_sem_as

        bleu_list, rouge_l_list = [], []
        src_tokens_list, gen_tokens_list = [], []
        truncation_list = []
        valid_sentence_ends = {".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'"}

        for src_s, ref_s, gen_s in zip(as_texts, ls_ref_texts, gen_texts):
            d_src = self.nlp(src_s)
            d_ref = self.nlp(ref_s)
            d_gen = self.nlp(gen_s)
            t_src = [t.text.lower() for t in d_src if not t.is_space]
            t_ref = [t.text.lower() for t in d_ref if not t.is_space]
            t_gen = [t.text.lower() for t in d_gen if not t.is_space]

            src_tokens_list.append(len(t_src))
            gen_tokens_list.append(len(t_gen))
            bleu_list.append(compute_sentence_bleu(t_ref, t_gen))
            rouge_l_list.append(compute_rouge_l(t_ref, t_gen)["f1"])

            trimmed = gen_s.strip()
            ends_clean = any(trimmed.endswith(end) for end in valid_sentence_ends) if len(trimmed) > 0 else False
            truncation_list.append(not ends_clean)

        comp_ratios = np.array(gen_tokens_list) / np.maximum(1, np.array(src_tokens_list))

        summary = {
            "Modell": display_name,
            "model_key": display_name,
            "model_path": model_name_or_path,
            "r_style_mean": float(np.mean(r_style)),
            "r_sem_as_mean": float(np.mean(r_sem_as)),
            "sim_ref_mean": float(np.mean(sim_ref)),
            "composite_reward_mean": float(np.mean(composite_reward)),
            "bleu_mean": float(np.mean(bleu_list)),
            "rouge_l_mean": float(np.mean(rouge_l_list)),
            "avg_gen_tokens": float(np.mean(gen_tokens_list)),
            "compression_ratio_mean": float(np.mean(comp_ratios)),
            "truncation_rate_pct": float(np.mean(truncation_list) * 100),
        }

        df_details = pd.DataFrame({
            "model_name": display_name,
            "as_text": as_texts,
            "ls_ref_text": ls_ref_texts,
            "generated_text": gen_texts,
            "r_style": r_style,
            "r_sem_as": r_sem_as,
            "sim_ref": sim_ref,
            "composite_reward": composite_reward,
            "bleu": bleu_list,
            "rouge_l": rouge_l_list,
            "gen_tokens": gen_tokens_list,
            "compression_ratio": comp_ratios,
            "is_truncated": truncation_list,
        })

        return summary, df_details


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO Loss Aggregation Experiment (Sum vs. Mean).")
    parser.add_argument("--test_file", type=str, default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--base_model_name", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--sbert_model_name", type=str, default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--sbert_max_seq_len", type=int, default=8192, help="Max sequence length for SBERT (default: 8192)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--plot_dir", type=str, default="results/plots")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print("Lade Testdaten aus:", args.test_file)
    with open(args.test_file, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    if args.max_samples is not None:
        lh_data = lh_data[:args.max_samples]

    as_texts = [item.get("source_text", item.get("as_text", item.get("source", ""))) for item in lh_data]
    ls_ref_texts = [item.get("target_text", item.get("ls_text", item.get("target", ""))) for item in lh_data]

    evaluator = LossAggregationEvaluator(
        sbert_model_name=args.sbert_model_name,
        sbert_max_seq_len=args.sbert_max_seq_len,
    )

    models_to_evaluate = [
        ("results/models/sft", "SFT Baseline"),
        ("results/models/loss_aggregation_exp/dpo_sum", "DPO Sum (Classic)"),
        ("results/models/loss_aggregation_exp/dpo_mean", "DPO Mean (Length-Normalized)"),
    ]

    summaries = []
    all_details = []

    for path, display_name in models_to_evaluate:
        if not os.path.exists(path):
            print(f"[UEBERSPRUNGEN] Modellpfad existiert noch nicht: {path}")
            continue

        summary, df_det = evaluator.evaluate_model(
            model_name_or_path=path,
            display_name=display_name,
            base_model_name=args.base_model_name,
            as_texts=as_texts,
            ls_ref_texts=ls_ref_texts,
        )
        summaries.append(summary)
        all_details.append(df_det)

    if not summaries:
        print("[FEHLER] Keine Modelle zur Auswertung gefunden.")
        return

    df_summary = pd.DataFrame(summaries)
    df_all_details = pd.concat(all_details, ignore_index=True)

    summary_csv = os.path.join(args.output_dir, "loss_aggregation_comparison_summary.csv")
    details_csv = os.path.join(args.output_dir, "loss_aggregation_comparison_details.csv")
    df_summary.to_csv(summary_csv, index=False)
    df_all_details.to_csv(details_csv, index=False)

    print(f"\n[ERFOLG] Zusammenfassung gespeichert: {summary_csv}")
    print(f"[ERFOLG] Details gespeichert: {details_csv}")

    # Visualisierungen erstellen
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "sans-serif"

    # 1. Pareto Frontier
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"SFT Baseline": "#7f7f7f", "DPO Sum (Classic)": "#d62728", "DPO Mean (Length-Normalized)": "#2ca02c"}
    ax.plot(df_summary["r_style_mean"], df_summary["r_sem_as_mean"], marker="o", markersize=10, linewidth=2, color="#1f77b4", linestyle="--")
    for _, row in df_summary.iterrows():
        c = colors.get(row["Modell"], "#333333")
        ax.scatter([row["r_style_mean"]], [row["r_sem_as_mean"]], color=c, s=150, zorder=5)
        ax.annotate(
            f"{row['Modell']}\n(Simp: {row['r_style_mean']:.3f}, Sem: {row['r_sem_as_mean']:.3f})",
            (row["r_style_mean"], row["r_sem_as_mean"]),
            textcoords="offset points",
            xytext=(10, 10),
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=c, alpha=0.8)
        )
    ax.set_xlabel("Ø Simplicity Score ($R_{style}$)", fontsize=12)
    ax.set_ylabel("Ø Semantik zu AS ($R_{sem, AS}$)", fontsize=12)
    ax.set_title("DPO Loss-Aggregation Trade-Off: Sum vs. Mean", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(args.plot_dir, "loss_aggregation_pareto_frontier.png"), dpi=300)
    plt.close()

    print("[ERFOLG] Alle Plots erfolgreich generiert!")

if __name__ == "__main__":
    main()
