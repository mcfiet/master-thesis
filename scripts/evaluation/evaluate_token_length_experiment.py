#!/usr/bin/env python3
"""
=============================================================================
Comparative Evaluation Script: Token Length Experiment (256 vs. 500 vs. 1000)
=============================================================================
Evaluates and benchmarks:
  1. Simplicity Regressor Models (BiLSTM 256 vs. 500 vs. 1000 tokens)
  2. SFT and DPO Seq2Seq Models (mBART-50 256 vs. 500 vs. 1000 tokens)

Metrics computed for Translation / Generation:
  - Simplicity / Style Score (via BiLSTM Regressors)
  - Semantic Preservation to Source AS (SBERT Cosine Similarity)
  - Semantic Similarity to Reference LS (SBERT Cosine Similarity)
  - Composite Reward (w_style * R_style + w_sem * R_sem)
  - Length & Compression Ratio (Output Tokens / Input Tokens)
  - Sentence Structure & Truncation Rate (Incomplete sentence endings)
  - Lexical Overlap (BLEU, ROUGE-1, ROUGE-2, ROUGE-L)

Metrics computed for Metric Regressors:
  - Mean Score on Leichte Sprache (LS) & Alltagssprache (AS)
  - Separation Margin (Score_LS - Score_AS)
  - Classification Accuracy (Score_LS > Score_AS)
  - Length Correlation (Pearson/Spearman correlation between length and score)
  - Stratified Performance on Short, Medium, and Long Texts
=============================================================================
"""

import os
import sys
import json
import argparse
import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import spacy
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
os.makedirs("results/evaluation", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

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
    if len(candidate_tokens) == 0:
        return 0.0
    if len(reference_tokens) == 0:
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
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def compute_rouge_l(ref_tokens: List[str], cand_tokens: List[str]) -> Dict[str, float]:
    m = len(ref_tokens)
    n = len(cand_tokens)
    if m == 0 or n == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    p = lcs_len / max(1, n)
    r = lcs_len / max(1, m)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


# ==============================================================================
# METRIC REGRESSOR EVALUATOR
# ==============================================================================
def evaluate_metric_models(
    metric_configs: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    device: torch.device,
    nlp,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates BiLSTM metric models on test pairs (ls_text vs. as_text).
    """
    print("\n" + "=" * 70)
    print("Starte Evaluation der Simplicity-Metrik-Modelle...")
    print("=" * 70)

    as_texts = [str(item.get("as_text") or "").strip() for item in test_samples]
    ls_texts = [str(item.get("ls_text") or "").strip() for item in test_samples]

    metric_summaries = []
    metric_details = []

    for cfg in metric_configs:
        name = cfg["name"]
        model_path = cfg["model_path"]
        vocab_path = cfg["vocab_path"]
        max_seq_len = cfg["max_seq_len"]

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            print(f"Überspringe Metrik-Modell (Datei fehlt): {name} ({model_path})")
            continue

        print(f"\n--- Evaluiere Metrik: {name} (max_len={max_seq_len}) ---")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            stoi = vocab_data.get("stoi", vocab_data)

        unk_idx = stoi.get("<unk>") or stoi.get("<UNK>") or 1
        model = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        def score_texts(texts: List[str], max_len: int) -> np.ndarray:
            scores = []
            for text in texts:
                doc = nlp(text)
                tokens = [t.text.lower() for t in doc if not t.is_space]
                indices = [stoi.get(t, unk_idx) for t in tokens[:max_len]]
                if len(indices) == 0:
                    indices = [0]
                inp = torch.tensor([indices], dtype=torch.long, device=device)
                with torch.no_grad():
                    s = model(inp).item()
                scores.append(s)
            return np.array(scores)

        scores_ls = score_texts(ls_texts, max_seq_len)
        scores_as = score_texts(as_texts, max_seq_len)

        margins = scores_ls - scores_as
        correct_order = scores_ls > scores_as
        accuracy = float(np.mean(correct_order))

        # Token length correlation
        ls_lens = [len([t for t in nlp(t_str) if not t.is_space]) for t_str in ls_texts]
        as_lens = [len([t for t in nlp(t_str) if not t.is_space]) for t_str in as_texts]

        corr_ls, _ = spearmanr(scores_ls, ls_lens) if len(scores_ls) > 1 else (0.0, 0.0)
        corr_as, _ = spearmanr(scores_as, as_lens) if len(scores_as) > 1 else (0.0, 0.0)

        # Stratified accuracy by length of AS
        short_mask = np.array(as_lens) < 200
        med_mask = (np.array(as_lens) >= 200) & (np.array(as_lens) <= 450)
        long_mask = np.array(as_lens) > 450

        acc_short = float(np.mean(correct_order[short_mask])) if np.sum(short_mask) > 0 else 0.0
        acc_med = float(np.mean(correct_order[med_mask])) if np.sum(med_mask) > 0 else 0.0
        acc_long = float(np.mean(correct_order[long_mask])) if np.sum(long_mask) > 0 else 0.0

        margin_short = float(np.mean(margins[short_mask])) if np.sum(short_mask) > 0 else 0.0
        margin_med = float(np.mean(margins[med_mask])) if np.sum(med_mask) > 0 else 0.0
        margin_long = float(np.mean(margins[long_mask])) if np.sum(long_mask) > 0 else 0.0

        summary = {
            "metric_model": name,
            "max_seq_len": max_seq_len,
            "mean_score_ls": float(np.mean(scores_ls)),
            "std_score_ls": float(np.std(scores_ls)),
            "mean_score_as": float(np.mean(scores_as)),
            "std_score_as": float(np.std(scores_as)),
            "separation_margin": float(np.mean(margins)),
            "accuracy_ls_gt_as": accuracy,
            "margin_short": margin_short,
            "margin_med": margin_med,
            "margin_long": margin_long,
            "acc_short": acc_short,
            "acc_med": acc_med,
            "acc_long": acc_long,
            "length_corr_ls": float(corr_ls) if not np.isnan(corr_ls) else 0.0,
            "length_corr_as": float(corr_as) if not np.isnan(corr_as) else 0.0,
        }
        metric_summaries.append(summary)

        for i, (as_t, ls_t, s_as, s_ls, m, cor) in enumerate(zip(as_texts, ls_texts, scores_as, scores_ls, margins, correct_order)):
            metric_details.append({
                "metric_model": name,
                "sample_idx": i,
                "as_text": as_t,
                "ls_text": ls_t,
                "as_tokens": as_lens[i],
                "ls_tokens": ls_lens[i],
                "score_as": float(s_as),
                "score_ls": float(s_ls),
                "margin": float(m),
                "correct_order": bool(cor),
            })

    df_metric_summary = pd.DataFrame(metric_summaries)
    df_metric_details = pd.DataFrame(metric_details)
    return df_metric_summary, df_metric_details


# ==============================================================================
# SEQ2SEQ MODEL EVALUATOR
# ==============================================================================
class TokenLengthEvaluator:
    def __init__(
        self,
        reward_model_path: str,
        reward_vocab_path: str,
        reward_max_seq_len: int = 500,
        sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.reward_max_seq_len = reward_max_seq_len
        # Load SBERT
        print(f"Lade SBERT: {sbert_model_name}")
        self.sbert = SentenceTransformer(sbert_model_name, trust_remote_code=True, device=self.device)
        if "jina" in sbert_model_name.lower():
            self.sbert.max_seq_length = min(self.reward_max_seq_len, 1024)
            print(f"Jina SBERT max_seq_length gesetzt auf: {self.sbert.max_seq_length}")

        # Load BiLSTM Simplicity Model
        print(f"Lade Simplicity Regressor: {reward_model_path}")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            self.stoi = json.load(f)
        self.unk_idx = self.stoi.get("<unk>") or self.stoi.get("<UNK>") or 1

        self.bilstm = BiLSTMRegressor(vocab_size=len(self.stoi)).to(self.device)
        self.bilstm.load_state_dict(torch.load(reward_model_path, map_location=self.device, weights_only=True))
        self.bilstm.eval()

        self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])

    def predict_simplicity(self, texts: List[str]) -> np.ndarray:
        scores = []
        for text in texts:
            doc = self.nlp(str(text or ""))
            tokens = [t.text.lower() for t in doc if not t.is_space]
            indices = [self.stoi.get(t, self.unk_idx) for t in tokens[:self.reward_max_seq_len]]
            if len(indices) == 0:
                indices = [0]
            inp_tensor = torch.tensor([indices], dtype=torch.long, device=self.device)
            with torch.no_grad():
                score = self.bilstm(inp_tensor).item()
            scores.append(score)
        return np.array(scores)

    def predict_semantic_sim(self, texts_a: List[str], texts_b: List[str]) -> np.ndarray:
        with torch.inference_mode():
            emb_a = self.sbert.encode(texts_a, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            emb_b = self.sbert.encode(texts_b, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            cos_sims = util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy()
        return cos_sims

    def evaluate_generation(
        self,
        model_name_or_path: str,
        base_model_name: str,
        test_samples: List[Dict[str, Any]],
        max_source_len: int = 500,
        max_target_len: int = 500,
        prompt_prefix: str = "",
        batch_size: int = 4,
        w_style: float = 0.5,
        w_sem: float = 0.5,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        if not os.path.exists(model_name_or_path):
            print(f"Modell nicht gefunden: {model_name_or_path}")
            return {}, pd.DataFrame()

        print(f"\n--- Evaluiere: {model_name_or_path} (max_len={max_source_len}/{max_target_len}, prefix='{prompt_prefix}') ---")
        
        # Load Tokenizer & Model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fix_mistral_regex=True)
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(self.device)
            model = PeftModel.from_pretrained(base_model, model_name_or_path).to(self.device)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)

        model.eval()

        as_texts = [str(item.get("as_text") or "").strip() for item in test_samples]
        ls_ref_texts = [str(item.get("ls_text") or "").strip() for item in test_samples]

        gen_texts = []
        num_batches = (len(as_texts) + batch_size - 1) // batch_size
        for b in tqdm(range(num_batches), desc="Generierung"):
            batch_src = as_texts[b * batch_size : (b + 1) * batch_size]
            prompts = [prompt_prefix + t for t in batch_src]
            inputs = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_source_len,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_target_len,
                    num_beams=4,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_texts.extend(decoded)

        # Free GPU VRAM from Seq2Seq model
        del model
        if "base_model" in locals():
            del base_model
        torch.cuda.empty_cache()

        # Compute Core Metrics
        r_style = self.predict_simplicity(gen_texts)
        r_sem_as = self.predict_semantic_sim(as_texts, gen_texts)
        r_sem_as_norm = np.clip((r_sem_as + 1.0) / 2.0, 0.0, 1.0)
        sim_ref = self.predict_semantic_sim(ls_ref_texts, gen_texts)
        sim_ref_norm = np.clip((sim_ref + 1.0) / 2.0, 0.0, 1.0)
        tot_reward = w_style * r_style + w_sem * r_sem_as_norm

        # Lexical & Length Analysis
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

            # Lexical scores
            bleu = compute_sentence_bleu(toks_ref, toks_gen)
            r1 = compute_rouge_n(toks_ref, toks_gen, n=1)["f1"]
            r2 = compute_rouge_n(toks_ref, toks_gen, n=2)["f1"]
            rl = compute_rouge_l(toks_ref, toks_gen)["f1"]

            bleu_scores.append(bleu)
            r1_scores.append(r1)
            r2_scores.append(r2)
            rl_scores.append(rl)

        summary = {
            "model": os.path.basename(model_name_or_path.rstrip("/")),
            "max_len": max_source_len,
            "r_style_mean": float(np.mean(r_style)),
            "r_sem_as_mean": float(np.mean(r_sem_as_norm)),
            "sim_ref_mean": float(np.mean(sim_ref_norm)),
            "composite_reward_mean": float(np.mean(tot_reward)),
            "bleu_mean": float(np.mean(bleu_scores)),
            "rouge1_f1_mean": float(np.mean(r1_scores)),
            "rouge2_f1_mean": float(np.mean(r2_scores)),
            "rougeL_f1_mean": float(np.mean(rl_scores)),
            "avg_gen_tokens": float(np.mean(gen_token_counts)),
            "compression_ratio_mean": float(np.mean(compression_ratios)),
            "truncation_rate": float(np.mean(is_truncated)),
        }

        details = pd.DataFrame({
            "model": os.path.basename(model_name_or_path.rstrip("/")),
            "as_text": as_texts,
            "ls_ref_text": ls_ref_texts,
            "generated_text": gen_texts,
            "r_style": r_style,
            "r_sem_as": r_sem_as_norm,
            "sim_ref": sim_ref_norm,
            "composite_reward": tot_reward,
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate Token Length Experiment (256 vs 500 vs 1000).")
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--base_model_name", default="facebook/mbart-large-50")
    parser.add_argument("--reward_model_path", default="results/models/token_length_exp/bilstm_mixup_regression_500.pt")
    parser.add_argument("--reward_vocab_path", default="data/token_length_exp/mixup_vocab_500.json")
    parser.add_argument("--prompt_prefix", default="", help="Prompt prefix for Seq2Seq inference (default: empty)")
    parser.add_argument("--sbert_model_name", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", help="SentenceTransformer model name")
    parser.add_argument("--output_summary", default="results/evaluation/token_length_comparison_summary.csv")
    parser.add_argument("--output_details", default="results/evaluation/token_length_comparison_details.csv")
    parser.add_argument("--output_metric_summary", default="results/evaluation/token_length_metric_comparison.csv")
    parser.add_argument("--output_metric_details", default="results/evaluation/token_length_metric_details.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])

    # Load Test Samples
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if args.max_samples is not None and len(test_samples) > args.max_samples:
        test_samples = test_samples[:args.max_samples]

    print(f"Geladene Test-Samples: {len(test_samples)}")

    # =========================================================================
    # 1. EVALUATE METRIC MODELS (BiLSTM 256, 500, 1000)
    # =========================================================================
    metric_configs = [
        {
            "name": "metric_mixup_256",
            "model_path": "results/models/token_length_exp/bilstm_mixup_regression_256.pt",
            "vocab_path": "data/token_length_exp/mixup_vocab_256.json",
            "max_seq_len": 256,
        },
        {
            "name": "metric_mixup_500",
            "model_path": "results/models/token_length_exp/bilstm_mixup_regression_500.pt",
            "vocab_path": "data/token_length_exp/mixup_vocab_500.json",
            "max_seq_len": 500,
        },
        {
            "name": "metric_mixup_1000",
            "model_path": "results/models/token_length_exp/bilstm_mixup_regression_1000.pt",
            "vocab_path": "data/token_length_exp/mixup_vocab_1000.json",
            "max_seq_len": 1000,
        },
        # Master baseline if present
        {
            "name": "metric_mixup_master",
            "model_path": "results/models/bilstm_mixup_regression.pt",
            "vocab_path": "data/vocabs/mixup_vocab.json",
            "max_seq_len": 256,
        },
    ]

    df_metric_summary, df_metric_details = evaluate_metric_models(metric_configs, test_samples, device, nlp)

    if not df_metric_summary.empty:
        df_metric_summary.to_csv(args.output_metric_summary, index=False)
        print(f"Metrik-Zusammenfassung gespeichert: {args.output_metric_summary}")
        if not df_metric_details.empty:
            df_metric_details.to_csv(args.output_metric_details, index=False)
            print(f"Metrik-Details gespeichert: {args.output_metric_details}")

        print("\n--- Zusammenfassung der Simplicity-Metrik-Modelle ---")
        try:
            print(df_metric_summary.to_markdown(index=False))
        except Exception:
            print(df_metric_summary.to_string(index=False))

    # =========================================================================
    # 2. EVALUATE SEQ2SEQ MODELS (SFT & DPO)
    # =========================================================================
    rm_path = args.reward_model_path
    rv_path = args.reward_vocab_path
    if not os.path.exists(rm_path):
        rm_path = "results/models/bilstm_mixup_regression.pt"
        rv_path = "data/vocabs/mixup_vocab.json"

    evaluator = TokenLengthEvaluator(
        reward_model_path=rm_path,
        reward_vocab_path=rv_path,
        reward_max_seq_len=1000,
        sbert_model_name=args.sbert_model_name,
        device=device,
    )

    models_to_eval = [
        # (model_dir, max_source_len, max_target_len)
        ("results/models/token_length_exp/sft_len256", 256, 256),
        ("results/models/token_length_exp/sft_len500", 500, 500),
        ("results/models/token_length_exp/sft_len1000", 1000, 1000),
        ("results/models/token_length_exp/dpo_len256", 256, 256),
        ("results/models/token_length_exp/dpo_len500", 500, 500),
        ("results/models/token_length_exp/dpo_len1000", 1000, 1000),
        ("results/models/token_length_jina_exp/dpo_len256_jina", 256, 256),
        ("results/models/token_length_jina_exp/dpo_len500_jina", 500, 500),
        ("results/models/token_length_jina_exp/dpo_len1000_jina", 1000, 1000),
    ]

    summaries = []
    all_details = []

    for model_path, max_src, max_tgt in models_to_eval:
        if not os.path.exists(model_path):
            print(f"Überspringe nicht vorhandenes Modell: {model_path}")
            continue

        summary, details = evaluator.evaluate_generation(
            model_name_or_path=model_path,
            base_model_name=args.base_model_name,
            test_samples=test_samples,
            max_source_len=max_src,
            max_target_len=max_tgt,
            prompt_prefix=args.prompt_prefix,
            batch_size=4 if max_src <= 500 else 2,
        )

        if summary:
            summaries.append(summary)
            all_details.append(details)

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(args.output_summary, index=False)
        print(f"\nErfolgreich gespeichert: {args.output_summary}")

    if all_details:
        df_all_details = pd.concat(all_details, ignore_index=True)
        df_all_details.to_csv(args.output_details, index=False)
        print(f"Detailergebnisse gespeichert: {args.output_details}")

    if summaries:
        print("\n--- Zusammenfassung der Übersetzungs-Modelle (SFT & DPO) ---")
        try:
            print(df_summary.to_markdown(index=False))
        except Exception:
            print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
