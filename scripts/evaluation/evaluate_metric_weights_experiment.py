#!/usr/bin/env python3
"""
=============================================================================
Comparative Evaluation Script: Metric Weighting Experiment (0.5/0.5 vs. 0.7/0.3 vs. 1.0/0.0)
=============================================================================
Evaluates and benchmarks the impact of DPO reward metric weighting on the
Seq2Seq Encoder-Decoder model (mBART-50):
  - Baseline: SFT model (results/models/sft)
  - DPO w05_w05: 50% Style / 50% Semantics
  - DPO w07_w03: 70% Style / 30% Semantics
  - DPO w10_w00: 100% Style / 0% Semantics

Metrics Computed:
  - Neural Simplicity / Style Score (BiLSTM Regressor)
  - Semantic Preservation to Source AS (SBERT Cosine Similarity)
  - Semantic Similarity to Reference LS (SBERT Cosine Similarity)
  - Composite Rewards across all weight regimes (0.5/0.5, 0.7/0.3, 1.0/0.0)
  - Lexical Overlap (BLEU, ROUGE-1, ROUGE-2, ROUGE-L F1)
  - Text Length & Compression Ratio (Output Tokens / Input Tokens)
  - Sentence Truncation Rate (Percentage of outputs with incomplete endings)
=============================================================================
"""

import argparse
import datetime
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
os.makedirs("results/evaluation", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# ==============================================================================
# BILSTM REGRESSOR ARCHITECTURE
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
# LEXICAL EVALUATION METRICS (BLEU & ROUGE)
# ==============================================================================
def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_sentence_bleu(reference_tokens: List[str], candidate_tokens: List[str], max_n: int = 4) -> float:
    if len(candidate_tokens) == 0 or len(reference_tokens) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = compute_ngram_counts(candidate_tokens, n)
        ref_ngrams = compute_ngram_counts(reference_tokens, n)
        if sum(cand_ngrams.values()) == 0:
            precisions.append(0.0)
            continue
        clipped_matches = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
        total_cand = max(1, sum(cand_ngrams.values()))
        precisions.append(clipped_matches / total_cand)

    if min(precisions) <= 1e-9:
        geom_mean = 0.0
    else:
        geom_mean = np.exp(np.mean([np.log(p) for p in precisions]))

    # Brevity penalty
    c = len(candidate_tokens)
    r = len(reference_tokens)
    bp = 1.0 if c > r else np.exp(1.0 - (r / max(1, c)))
    return float(bp * geom_mean)


def compute_rouge_n(ref_tokens: List[str], cand_tokens: List[str], n: int = 1) -> Dict[str, float]:
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    ref_ngrams = compute_ngram_counts(ref_tokens, n)
    cand_ngrams = compute_ngram_counts(cand_tokens, n)

    overlap = sum(min(cand_ngrams[ng], ref_ngrams[ng]) for ng in cand_ngrams)
    total_ref = sum(ref_ngrams.values())
    total_cand = sum(cand_ngrams.values())

    p = overlap / max(1, total_cand)
    r = overlap / max(1, total_ref)
    f1 = 2 * p * r / max(1e-9, (p + r))
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def compute_rouge_l(ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == cand_tokens[j]:
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
class MetricWeightsEvaluator:
    def __init__(
        self,
        reward_model_path: str,
        reward_vocab_path: str,
        sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        max_seq_len: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len

        print("Initializing spacy German tokenizer...")
        self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])

        # 1. Load Vocab
        print(f"Loading vocabulary from {reward_vocab_path}...")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        # 2. Load BiLSTM Simplicity Model
        print(f"Loading BiLSTM Regressor from {reward_model_path}...")
        self.regressor = BiLSTMRegressor(vocab_size=len(self.vocab)).to(self.device)
        state_dict = torch.load(reward_model_path, map_location=self.device)
        if "model_state_dict" in state_dict:
            self.regressor.load_state_dict(state_dict["model_state_dict"])
        else:
            self.regressor.load_state_dict(state_dict)
        self.regressor.eval()

        # 3. Load SBERT Model
        print(f"Loading SBERT model ({sbert_model_name})...")
        self.sbert = SentenceTransformer(sbert_model_name, device=self.device)

    def text_to_tensor(self, texts: List[str]) -> torch.Tensor:
        batch_ids = []
        for text in texts:
            doc = self.nlp(text)
            ids = [self.vocab.get(token.text.lower(), 1) for token in doc if not token.is_space]
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            else:
                ids = ids + [0] * (self.max_seq_len - len(ids))
            batch_ids.append(ids)
        return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict_simplicity(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        all_preds = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tensor = self.text_to_tensor(batch_texts)
            preds = self.regressor(tensor).squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist() if isinstance(preds, np.ndarray) and preds.ndim > 0 else [float(preds)])
        return np.array(all_preds)

    @torch.no_grad()
    def predict_semantic_sim(self, texts1: List[str], texts2: List[str], batch_size: int = 64) -> np.ndarray:
        all_sims = []
        for i in range(0, len(texts1), batch_size):
            b1 = texts1[i : i + batch_size]
            b2 = texts2[i : i + batch_size]
            emb1 = self.sbert.encode(b1, convert_to_tensor=True, show_progress_bar=False)
            emb2 = self.sbert.encode(b2, convert_to_tensor=True, show_progress_bar=False)
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
        prompt_prefix: str = "",
        max_source_len: int = 256,
        max_target_len: int = 256,
        batch_size: int = 8,
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        print(f"\n==================================================================")
        print(f"Evaluating Model: {display_name} ({model_name_or_path})")
        print(f"==================================================================")

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path if os.path.exists(os.path.join(model_name_or_path, "tokenizer_config.json")) else base_model_name,
            use_fast=False
        )
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load Model
        config = AutoConfig.from_pretrained(base_model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        def _load_base():
            for kwargs in [{"use_safetensors": True}, {"use_safetensors": False, "weights_only": False}, {}]:
                try:
                    return AutoModelForSeq2SeqLM.from_pretrained(base_model_name, config=config, torch_dtype=dtype, **kwargs)
                except Exception:
                    continue
            return AutoModelForSeq2SeqLM.from_pretrained(base_model_name, config=config, torch_dtype=dtype)

        has_weights = (
            os.path.exists(os.path.join(model_name_or_path, "model.safetensors")) or
            os.path.exists(os.path.join(model_name_or_path, "pytorch_model.bin"))
        )
        has_adapter = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))

        print(f"[MODELL-LADEN] {display_name}")
        print(f"  -> Pfad: {model_name_or_path}")

        if has_weights:
            weight_file = "model.safetensors" if os.path.exists(os.path.join(model_name_or_path, "model.safetensors")) else "pytorch_model.bin"
            size_mb = os.path.getsize(os.path.join(model_name_or_path, weight_file)) / (1024 * 1024)
            print(f"  -> Lade-Modus: Vollstaendig fusioniertes Standalone-Modell ({weight_file}, {size_mb:.1f} MB)")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path, torch_dtype=dtype, use_safetensors=True
                ).to(self.device)
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path, torch_dtype=dtype, weights_only=False
                ).to(self.device)
        elif has_adapter:
            print(f"  -> Lade-Modus: LoRA-Adapter auf Basismodell '{base_model_name}' (adapter_config.json)")
            base_model = _load_base()
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            model = model.to(self.device)
        else:
            print(f"  -> Lade-Modus: Standard Pretrained Seq2Seq Modell")
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path, config=config, torch_dtype=dtype, use_safetensors=True
                ).to(self.device)
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name_or_path, config=config, torch_dtype=dtype, weights_only=False
                ).to(self.device)

        print(f"  -> Instanziierte Modell-Klasse: {model.__class__.__name__} ({dtype})\n")

        model.eval()

        # Batch Generation
        gen_texts = []
        for i in tqdm(range(0, len(as_texts), batch_size), desc=f"Generating {display_name}"):
            batch_src = [prompt_prefix + txt for txt in as_texts[i : i + batch_size]]
            inputs = tokenizer(
                batch_src,
                max_length=max_source_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_target_len,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_texts.extend(decoded)

        # Clean GPU VRAM
        del model
        if "base_model" in locals():
            del base_model
        torch.cuda.empty_cache()

        # Compute Core Neural Metrics
        r_style = self.predict_simplicity(gen_texts)
        r_sem_as = self.predict_semantic_sim(as_texts, gen_texts)
        r_sem_as_norm = np.clip((r_sem_as + 1.0) / 2.0, 0.0, 1.0)
        sim_ref = self.predict_semantic_sim(ls_ref_texts, gen_texts)
        sim_ref_norm = np.clip((sim_ref + 1.0) / 2.0, 0.0, 1.0)

        # Compute Composite Rewards for different weighting regimes
        reward_05_05 = 0.5 * r_style + 0.5 * r_sem_as_norm
        reward_07_03 = 0.7 * r_style + 0.3 * r_sem_as_norm
        reward_10_00 = 1.0 * r_style + 0.0 * r_sem_as_norm

        # Lexical, Length & Truncation Analysis
        bleu_scores = []
        r1_scores, r2_scores, rl_scores = [], [], []
        src_token_counts, gen_token_counts, ref_token_counts = [], [], []
        compression_ratios = []
        is_truncated = []

        valid_sentence_ends = {".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'"}

        for src, ref, gen in zip(as_texts, ls_ref_texts, gen_texts):
            doc_src = self.nlp(src)
            doc_ref = self.nlp(ref)
            doc_gen = self.nlp(gen)

            toks_src = [t.text.lower() for t in doc_src if not t.is_space]
            toks_ref = [t.text.lower() for t in doc_ref if not t.is_space]
            toks_gen = [t.text.lower() for t in doc_gen if not t.is_space]

            src_token_counts.append(len(toks_src))
            gen_token_counts.append(len(toks_gen))
            ref_token_counts.append(len(toks_ref))

            comp_ratio = len(toks_gen) / max(1, len(toks_src))
            compression_ratios.append(comp_ratio)

            # Truncation check
            trimmed_gen = gen.strip()
            ends_well = any(trimmed_gen.endswith(end) for end in valid_sentence_ends) if len(trimmed_gen) > 0 else False
            is_truncated.append(not ends_well)

            # Lexical metrics
            bleu = compute_sentence_bleu(toks_ref, toks_gen)
            r1 = compute_rouge_n(toks_ref, toks_gen, n=1)["f1"]
            r2 = compute_rouge_n(toks_ref, toks_gen, n=2)["f1"]
            rl = compute_rouge_l(toks_ref, toks_gen)["f1"]

            bleu_scores.append(bleu)
            r1_scores.append(r1)
            r2_scores.append(r2)
            rl_scores.append(rl)

        summary = {
            "model_name": display_name,
            "model_path": model_name_or_path,
            "r_style_mean": float(np.mean(r_style)),
            "r_sem_as_mean": float(np.mean(r_sem_as_norm)),
            "sim_ref_mean": float(np.mean(sim_ref_norm)),
            "reward_05_05_mean": float(np.mean(reward_05_05)),
            "reward_07_03_mean": float(np.mean(reward_07_03)),
            "reward_10_00_mean": float(np.mean(reward_10_00)),
            "bleu_mean": float(np.mean(bleu_scores)),
            "rouge1_f1_mean": float(np.mean(r1_scores)),
            "rouge2_f1_mean": float(np.mean(r2_scores)),
            "rougeL_f1_mean": float(np.mean(rl_scores)),
            "avg_src_tokens": float(np.mean(src_token_counts)),
            "avg_gen_tokens": float(np.mean(gen_token_counts)),
            "avg_ref_tokens": float(np.mean(ref_token_counts)),
            "compression_ratio_mean": float(np.mean(compression_ratios)),
            "truncation_rate": float(np.mean(is_truncated)),
        }

        details = pd.DataFrame({
            "model_name": display_name,
            "as_text": as_texts,
            "ls_ref_text": ls_ref_texts,
            "generated_text": gen_texts,
            "r_style": r_style,
            "r_sem_as": r_sem_as_norm,
            "sim_ref": sim_ref_norm,
            "reward_05_05": reward_05_05,
            "reward_07_03": reward_07_03,
            "reward_10_00": reward_10_00,
            "bleu": bleu_scores,
            "rouge1_f1": r1_scores,
            "rouge2_f1": r2_scores,
            "rougeL_f1": rl_scores,
            "src_tokens": src_token_counts,
            "gen_tokens": gen_token_counts,
            "ref_tokens": ref_token_counts,
            "compression_ratio": compression_ratios,
            "is_truncated": is_truncated,
        })

        return summary, details


# ==============================================================================
# PLOTTING UTILITY
# ==============================================================================
def generate_comparison_plots(summary_df: pd.DataFrame, output_plot_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Simplicity vs Semantic Preservation
    ax1 = axes[0]
    ax1.plot(summary_df["r_style_mean"], summary_df["r_sem_as_mean"], marker="o", linestyle="-", color="#1f77b4", markersize=8)
    for _, row in summary_df.iterrows():
        ax1.annotate(
            row["model_name"],
            (row["r_style_mean"], row["r_sem_as_mean"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )
    ax1.set_xlabel("Ø Simplicity Score ($R_{style}$)", fontsize=11)
    ax1.set_ylabel("Ø Semantik-Erhalt zur Quelle ($R_{sem, AS}$)", fontsize=11)
    ax1.set_title("Simplicity vs. Semantik-Erhalt (Trade-off)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Composite Rewards across Weightings
    ax2 = axes[1]
    x = np.arange(len(summary_df))
    width = 0.25
    ax2.bar(x - width, summary_df["reward_05_05_mean"], width, label="0.5 Style / 0.5 Sem", color="#aec7e8")
    ax2.bar(x, summary_df["reward_07_03_mean"], width, label="0.7 Style / 0.3 Sem", color="#1f77b4")
    ax2.bar(x + width, summary_df["reward_10_00_mean"], width, label="1.0 Style / 0.0 Sem", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(summary_df["model_name"], rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Composite Reward Score", fontsize=11)
    ax2.set_title("Composite Rewards je Gewichtungsschema", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")

    # 3. Output Length & BLEU vs Simplicity
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    p1 = ax3.bar(x - 0.15, summary_df["avg_gen_tokens"], 0.3, label="Ø Gen. Tokens", color="#2ca02c", alpha=0.8)
    p2 = ax3_twin.plot(x + 0.15, summary_df["bleu_mean"], color="#d62728", marker="s", linewidth=2, label="BLEU Score")
    ax3.set_xticks(x)
    ax3.set_xticklabels(summary_df["model_name"], rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("Ø Generierte Tokens", color="#2ca02c", fontsize=11)
    ax3_twin.set_ylabel("BLEU Score", color="#d62728", fontsize=11)
    ax3.set_title("Länge & BLEU vs. Modellvariante", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6, axis="y")

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {output_plot_path}")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Metric Weights Experiment (0.5/0.5 vs 0.7/0.3 vs 1.0/0.0).")
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Path to evaluation test set")
    parser.add_argument("--base_model_name", default="facebook/mbart-large-50", help="Base model architecture")
    parser.add_argument("--sft_model_path", default="results/models/sft", help="Path to SFT baseline model")
    parser.add_argument("--dpo_w05_w05_path", default="results/models/metric_weights_exp/dpo_w05_w05", help="Path to DPO 0.5/0.5 model")
    parser.add_argument("--dpo_w07_w03_path", default="results/models/metric_weights_exp/dpo_w07_w03", help="Path to DPO 0.7/0.3 model")
    parser.add_argument("--dpo_w10_w00_path", default="results/models/metric_weights_exp/dpo_w10_w00", help="Path to DPO 1.0/0.0 model")
    parser.add_argument("--reward_model_path", default="results/models/bilstm_mixup_regression.pt", help="Path to BiLSTM simplicity regressor")
    parser.add_argument("--reward_vocab_path", default="data/vocabs/mixup_vocab.json", help="Path to simplicity vocab")
    parser.add_argument("--prompt_prefix", default="", help="Prompt prefix for Seq2Seq inference")
    parser.add_argument("--sbert_model_name", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="SBERT model for semantic evaluation")
    parser.add_argument("--output_summary", default="results/evaluation/metric_weights_comparison_summary.csv")
    parser.add_argument("--output_details", default="results/evaluation/metric_weights_comparison_details.csv")
    parser.add_argument("--output_plot", default="results/plots/metric_weights_tradeoff_curve.png")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional max sample limit for evaluation")
    args = parser.parse_args()

    # Load Test Data
    print(f"Loading test data from {args.test_data_path}...")
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if args.max_samples and len(test_samples) > args.max_samples:
        test_samples = test_samples[:args.max_samples]

    as_texts = [item.get("source_text", item.get("as_text", item.get("source", ""))) for item in test_samples]
    ls_ref_texts = [item.get("target_text", item.get("ls_text", item.get("target", ""))) for item in test_samples]

    # Initialize Evaluator Engine
    evaluator = MetricWeightsEvaluator(
        reward_model_path=args.reward_model_path,
        reward_vocab_path=args.reward_vocab_path,
        sbert_model_name=args.sbert_model_name,
        max_seq_len=256,
    )

    # Models to Evaluate
    models_to_evaluate = [
        {"name": "SFT Baseline", "path": args.sft_model_path},
        {"name": "DPO (0.5 Style / 0.5 Sem)", "path": args.dpo_w05_w05_path},
        {"name": "DPO (0.7 Style / 0.3 Sem)", "path": args.dpo_w07_w03_path},
        {"name": "DPO (1.0 Style / 0.0 Sem)", "path": args.dpo_w10_w00_path},
    ]

    all_summaries = []
    all_details = []

    for model_cfg in models_to_evaluate:
        m_path = model_cfg["path"]
        m_name = model_cfg["name"]

        if not os.path.exists(m_path):
            print(f"[WARNING] Skipping {m_name}: Path '{m_path}' does not exist yet.")
            continue

        summary, details = evaluator.evaluate_model(
            model_name_or_path=m_path,
            display_name=m_name,
            base_model_name=args.base_model_name,
            as_texts=as_texts,
            ls_ref_texts=ls_ref_texts,
            prompt_prefix=args.prompt_prefix,
            max_source_len=256,
            max_target_len=256,
            batch_size=8,
        )
        all_summaries.append(summary)
        all_details.append(details)

    if not all_summaries:
        print("[ERROR] No models were successfully evaluated.")
        return

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(args.output_summary, index=False)
    print(f"\nSaved summary results to: {args.output_summary}")

    full_details_df = pd.concat(all_details, ignore_index=True)
    full_details_df.to_csv(args.output_details, index=False)
    print(f"Saved detailed results to: {args.output_details}")

    # Also save JSON format summary
    json_summary_path = args.output_summary.replace(".csv", ".json")
    with open(json_summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON summary to: {json_summary_path}")

    # Generate Visualization Plots
    try:
        generate_comparison_plots(summary_df, args.output_plot)
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")

    # Print Formatted Markdown Table
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY RESULTS")
    print("=" * 90)
    print(summary_df[[
        "model_name",
        "r_style_mean",
        "r_sem_as_mean",
        "sim_ref_mean",
        "reward_05_05_mean",
        "reward_07_03_mean",
        "reward_10_00_mean",
        "bleu_mean",
        "rougeL_f1_mean",
        "avg_gen_tokens",
        "truncation_rate",
    ]].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
