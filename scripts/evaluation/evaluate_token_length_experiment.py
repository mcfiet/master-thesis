#!/usr/bin/env python3
"""
=============================================================================
Comparative Evaluation Script: Token Length Experiment (256 vs. 500 vs. 1000)
=============================================================================
Evaluates and benchmarks all trained SFT and DPO models across different
sequence lengths on a standardized test dataset (Lebenshilfe & Corpus).

Metrics computed:
  1. Simplicity / Style Score (via BiLSTM Mixup Regressors)
  2. Semantic Preservation to Source AS (SBERT Cosine Similarity)
  3. Semantic Similarity to Reference LS (SBERT Cosine Similarity)
  4. Composite Reward (w_style * R_style + w_sem * R_sem)
  5. Length & Compression Ratio (Output Tokens / Input Tokens)
  6. Sentence Structure & Truncation Rate (Incomplete sentence endings)
  7. Lexical Overlap (BLEU, ROUGE-1, ROUGE-2, ROUGE-L)
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

    # LCS dynamic programming
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
# MODEL EVALUATOR
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
        self.sbert = SentenceTransformer(sbert_model_name, device=self.device)

        # Load BiLSTM Simplicity Model
        print(f"Lade Simplicity Regressor: {reward_model_path}")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            self.stoi = vocab_data.get("stoi", vocab_data)

        self.unk_idx = self.stoi.get("<unk>") or self.stoi.get("<UNK>") or 1
        self.bilstm = BiLSTMRegressor(vocab_size=len(self.stoi), embed_dim=128, hidden_dim=128).to(self.device)
        self.bilstm.load_state_dict(torch.load(reward_model_path, map_location=self.device))
        self.bilstm.eval()

        self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])

    def predict_simplicity(self, texts: List[str]) -> np.ndarray:
        scores = []
        for text in texts:
            doc = self.nlp(text)
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
        emb_a = self.sbert.encode(texts_a, convert_to_tensor=True, show_progress_bar=False)
        emb_b = self.sbert.encode(texts_b, convert_to_tensor=True, show_progress_bar=False)
        cos_sims = util.cos_sim(emb_a, emb_b).diagonal().cpu().numpy()
        return cos_sims

    def evaluate_generation(
        self,
        model_name_or_path: str,
        base_model_name: str,
        test_samples: List[Dict[str, Any]],
        max_source_len: int = 500,
        max_target_len: int = 500,
        batch_size: int = 4,
        w_style: float = 0.5,
        w_sem: float = 0.5,
    ) -> Tuple[Dict[str, float], pd.DataFrame]:
        if not os.path.exists(model_name_or_path):
            print(f"Modell nicht gefunden: {model_name_or_path}")
            return {}, pd.DataFrame()

        print(f"\n--- Evaluiere: {model_name_or_path} (max_len={max_source_len}/{max_target_len}) ---")
        
        # Load Tokenizer & Model
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
            prompts = ["Übersetze in Leichte Sprache: " + t for t in batch_src]
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
    parser.add_argument("--output_summary", default="results/evaluation/token_length_comparison_summary.csv")
    parser.add_argument("--output_details", default="results/evaluation/token_length_comparison_details.csv")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Load Test Samples
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    if args.max_samples is not None and len(test_samples) > args.max_samples:
        test_samples = test_samples[:args.max_samples]

    print(f"Geladene Test-Samples: {len(test_samples)}")

    # Fallback to master metric if token_length_exp metric is not yet trained
    rm_path = args.reward_model_path
    rv_path = args.reward_vocab_path
    if not os.path.exists(rm_path):
        rm_path = "results/models/bilstm_mixup_regression.pt"
        rv_path = "data/vocabs/mixup_vocab.json"

    evaluator = TokenLengthEvaluator(
        reward_model_path=rm_path,
        reward_vocab_path=rv_path,
        reward_max_seq_len=1000,
    )

    models_to_eval = [
        # (model_dir, max_source_len, max_target_len)
        ("results/models/token_length_exp/sft_len256", 256, 256),
        ("results/models/token_length_exp/sft_len500", 500, 500),
        ("results/models/token_length_exp/sft_len1000", 1000, 1000),
        ("results/models/token_length_exp/dpo_len256", 256, 256),
        ("results/models/token_length_exp/dpo_len500", 500, 500),
        ("results/models/token_length_exp/dpo_len1000", 1000, 1000),
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
            batch_size=4 if max_src <= 500 else 2,
        )

        if summary:
            summaries.append(summary)
            all_details.append(details)

    if summaries:
        df_summary = pd.DataFrame(summaries)
        df_summary.to_csv(args.output_summary, index=False)
        print(f"\nErfolgreich gespeichert: {args.output_summary}")
        print("\n--- Zusammenfassung der Ergebnisse ---")
        print(df_summary.to_markdown(index=False))

    if all_details:
        df_all_details = pd.concat(all_details, ignore_index=True)
        df_all_details.to_csv(args.output_details, index=False)
        print(f"Detailergebnisse gespeichert: {args.output_details}")


if __name__ == "__main__":
    main()
