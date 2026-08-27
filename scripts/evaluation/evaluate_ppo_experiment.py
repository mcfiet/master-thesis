#!/usr/bin/env python3
"""
=============================================================================
7-Way Master Benchmark Evaluation Script (Few-Shot, SFT, DPO, PPO)
=============================================================================
Evaluates and benchmarks all modeling approaches on the unseen Lebenshilfe
test set across 3 essential dimensions for German Leichte Sprache:
  1. Simplification & Readability (BiLSTM Score, LIX, Flesch-DE, Wiener Sachtextformel)
  2. Faithfulness & Preservation (SBERT/Jina Cosine Sim, BLEU-4, ROUGE-1/2/L)
  3. Structural & Rule Adherence (Compression Ratio, Passiv-Ratio, Genitiv-Ratio, Nominalstil)

Models compared:
  - Few-Shot Baseline (Qwen 2.5 1.5B)
  - Decoder-Only SFT (Qwen 2.5)
  - Decoder-Only DPO (Qwen 2.5)
  - Decoder-Only PPO (Qwen 2.5)
  - Encoder-Decoder SFT (mBART-50)
  - Encoder-Decoder DPO (mBART-50)
  - Encoder-Decoder PPO (mBART-50)
=============================================================================
"""

import os
import sys
import json
import re
import math
import time
import argparse
import datetime
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    MBart50TokenizerFast,
)
from peft import PeftModel


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger(object):
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# BILSTM MIXUP REGRESSOR
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
# READABILITY METRICS
# ==============================================================================
def calculate_lix(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    long_words = sum(1 for w in words if len(w) > 6)
    return (num_words / num_sentences) + (long_words * 100.0 / num_words)


def calculate_flesch_de(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    num_syllables = sum(max(1, sum(1 for char in w if char in vowels)) for w in words)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    return 180.0 - asl - (58.5 * asw)


def calculate_wiener(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    ms = sum(1 for w in words if sum(1 for c in w if c in vowels) >= 3) * 100.0 / num_words
    sl = num_words / num_sentences
    iw = sum(1 for w in words if len(w) > 6) * 100.0 / num_words
    es = sum(1 for w in words if sum(1 for c in w if c in vowels) == 1) * 100.0 / num_words
    return (0.1935 * ms) + (0.1672 * sl) + (0.1297 * iw) - (0.0327 * es) - 0.8733


# ==============================================================================
# LEXICAL METRICS (BLEU & ROUGE)
# ==============================================================================
def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_sentence_bleu(ref_toks: List[str], cand_toks: List[str], max_n: int = 4) -> float:
    if not cand_toks or not ref_toks:
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
    if not ref_toks or not cand_toks:
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
    f1 = 2.0 * p * r / max(1e-9, (p + r))
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


# ==============================================================================
# RULE ADHERENCE ANALYZER
# ==============================================================================
class RuleAdherenceAnalyzer:
    def __init__(self, nlp):
        self.nlp = nlp
        self.nominal_suffixes = re.compile(r"(ung|keit|heit|schaft|ismus|tion|tät)$", re.IGNORECASE)

    def analyze(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
        words = [t for t in doc if not t.is_punct and not t.is_space]
        num_words = len(words)
        num_sents = max(1, len(list(doc.sents)))

        if num_words == 0:
            return {"passiv_ratio": 0.0, "genitiv_ratio": 0.0, "nominal_ratio": 0.0, "avg_sent_len": 0.0}

        passives = sum(1 for t in doc if t.dep_ in ["sbp", "svp"] or (t.lemma_ in ["werden", "sein"] and any(c.tag_ == "VVPP" for c in t.children)))
        genitives = sum(1 for t in doc if "Gen" in t.morph.get("Case", []))
        nominals = sum(1 for t in words if self.nominal_suffixes.search(t.text))

        return {
            "passiv_ratio": passives / num_sents,
            "genitiv_ratio": genitives / num_words,
            "nominal_ratio": nominals / num_words,
            "avg_sent_len": num_words / num_sents,
        }


# ==============================================================================
# MAIN BENCHMARK ENGINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="7-Way Master Benchmark for Leichte Sprache.")
    parser.add_argument("--test_data_path", type=str, default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--reward_model_path", type=str, default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--reward_vocab_path", type=str, default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", type=str, default="jinaai/jina-embeddings-v2-base-de")
    
    # Model Paths
    parser.add_argument("--qwen_base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft_decoder_path", type=str, default="results/models/decoder_only/sft")
    parser.add_argument("--dpo_decoder_path", type=str, default="results/models/decoder_only/dpo")
    parser.add_argument("--ppo_decoder_path", type=str, default="results/models/decoder_only/ppo")
    
    parser.add_argument("--mbart_base_model", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--sft_mbart_path", type=str, default="results/models/sft")
    parser.add_argument("--dpo_mbart_path", type=str, default="results/models/dpo")
    parser.add_argument("--ppo_mbart_path", type=str, default="results/models/ppo/seq2seq")

    # Outputs
    parser.add_argument("--output_csv", type=str, default="results/evaluation/benchmark_ppo_vs_dpo_vs_sft_7way.csv")
    parser.add_argument("--output_summary", type=str, default="results/evaluation/master_benchmark_summary_7way.csv")
    parser.add_argument("--plot_dir", type=str, default="results/plots/experiments/ppo")
    parser.add_argument("--log_dir", type=str, default="results/logs/experiments/ppo")
    parser.add_argument("--max_test_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"benchmark_7way_{timestamp}.log")
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"Starting 7-Way Master Benchmark at {timestamp} on {device}")
    print("=" * 80)

    # 1. Load Test Dataset
    print(f"Loading Test Set from: {args.test_data_path}")
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_raw = json.load(f)
        if isinstance(test_raw, dict) and "data" in test_raw:
            test_raw = test_raw["data"]

    test_data = []
    for item in test_raw:
        as_t = str(item.get("as_text") or item.get("as") or item.get("source_text") or item.get("source") or item.get("prompt") or "").strip()
        ls_t = str(item.get("ls_text") or item.get("ls") or item.get("target_text") or item.get("target") or "").strip()
        if as_t and ls_t:
            test_data.append({"as": as_t, "ls": ls_t})

    if args.max_test_samples and len(test_data) > args.max_test_samples:
        test_data = test_data[:args.max_test_samples]
    print(f"Total Test Samples: {len(test_data)}")

    # 2. Load Evaluation Models (BiLSTM, SpaCy, SBERT)
    try:
        nlp = spacy.load("de_core_news_sm")
    except Exception:
        nlp = spacy.blank("de")
        nlp.add_pipe("sentencizer")

    rule_analyzer = RuleAdherenceAnalyzer(nlp)

    print(f"Loading BiLSTM Regressor from: {args.reward_model_path}")
    with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data.get("stoi", vocab_data)
    unk_idx = stoi.get("<unk>") or stoi.get("<UNK>") or 1

    bilstm = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(device)
    raw_state = torch.load(args.reward_model_path, map_location=device)
    if isinstance(raw_state, dict):
        raw_state = raw_state.get("model_state_dict", raw_state.get("state_dict", raw_state))
    bilstm.load_state_dict(raw_state)
    bilstm.eval()

    def get_bilstm_scores(texts: List[str]) -> np.ndarray:
        batch_idxs = []
        max_l = 0
        docs = list(nlp.pipe(texts, batch_size=len(texts))) if len(texts) > 1 else [nlp(texts[0])]
        for doc in docs:
            tokens = [t.text.lower() for t in doc if not t.is_space]
            idxs = [stoi.get(t, unk_idx) for t in tokens[:500]] or [0]
            batch_idxs.append(idxs)
            max_l = max(max_l, len(idxs))
        padded = np.zeros((len(batch_idxs), max(max_l, 1)), dtype=np.int64)
        for i, idxs in enumerate(batch_idxs):
            padded[i, :len(idxs)] = idxs
        with torch.inference_mode():
            return bilstm(torch.tensor(padded, dtype=torch.long, device=device)).squeeze(-1).cpu().numpy()

    print(f"Loading SBERT Evaluator: {args.sbert_model_name}")
    sbert = SentenceTransformer(args.sbert_model_name, trust_remote_code=True, device=str(device))
    if "jina" in args.sbert_model_name.lower():
        sbert.max_seq_length = 512

    # 3. Model Inference Setup
    models_to_evaluate: Dict[str, Dict[str, Any]] = {}

    # Decoder-Only Models
    if os.path.exists(args.qwen_base_model) or "qwen" in args.qwen_base_model.lower():
        qwen_tok = AutoTokenizer.from_pretrained(args.qwen_base_model, trust_remote_code=True)
        if qwen_tok.pad_token is None:
            qwen_tok.pad_token = qwen_tok.eos_token
        qwen_tok.padding_side = "left"

        models_to_evaluate["Few-Shot Baseline (Qwen)"] = {
            "type": "causal_few_shot",
            "base_name": args.qwen_base_model,
            "adapter_path": None,
            "tok": qwen_tok,
        }
        if os.path.exists(args.sft_decoder_path):
            models_to_evaluate["Decoder-Only SFT (Qwen)"] = {
                "type": "causal_peft",
                "base_name": args.qwen_base_model,
                "adapter_path": args.sft_decoder_path,
                "tok": qwen_tok,
            }
        if os.path.exists(args.dpo_decoder_path):
            models_to_evaluate["Decoder-Only DPO (Qwen)"] = {
                "type": "causal_peft",
                "base_name": args.qwen_base_model,
                "adapter_path": args.dpo_decoder_path,
                "tok": qwen_tok,
            }
        if os.path.exists(args.ppo_decoder_path):
            models_to_evaluate["Decoder-Only PPO (Qwen)"] = {
                "type": "causal_peft",
                "base_name": args.qwen_base_model,
                "adapter_path": args.ppo_decoder_path,
                "tok": qwen_tok,
            }

    # Seq2Seq Models (mBART-50)
    if os.path.exists(args.mbart_base_model) or "mbart" in args.mbart_base_model.lower():
        try:
            mbart_tok = AutoTokenizer.from_pretrained(args.mbart_base_model, use_fast=False)
            mbart_tok.src_lang = "de_DE"
            mbart_tok.tgt_lang = "de_DE"
        except Exception:
            mbart_tok = None

        if mbart_tok:
            if os.path.exists(args.sft_mbart_path):
                models_to_evaluate["Encoder-Decoder SFT (mBART)"] = {
                    "type": "seq2seq_peft",
                    "base_name": args.mbart_base_model,
                    "adapter_path": args.sft_mbart_path,
                    "tok": mbart_tok,
                }
            if os.path.exists(args.dpo_mbart_path):
                models_to_evaluate["Encoder-Decoder DPO (mBART)"] = {
                    "type": "seq2seq_peft",
                    "base_name": args.mbart_base_model,
                    "adapter_path": args.dpo_mbart_path,
                    "tok": mbart_tok,
                }
            if os.path.exists(args.ppo_mbart_path):
                models_to_evaluate["Encoder-Decoder PPO (mBART)"] = {
                    "type": "seq2seq_peft",
                    "base_name": args.mbart_base_model,
                    "adapter_path": args.ppo_mbart_path,
                    "tok": mbart_tok,
                }

    print(f"Found {len(models_to_evaluate)} models to benchmark: {list(models_to_evaluate.keys())}")

    # 4. Generate Predictions
    translations_dict: Dict[str, List[str]] = {}
    sources = [s["as"] for s in test_data]
    references = [s["ls"] for s in test_data]

    for model_name, cfg in models_to_evaluate.items():
        print(f"\n--- Running Generations for: {model_name} ---")
        tok = cfg["tok"]
        mtype = cfg["type"]

        if mtype in ["causal_few_shot", "causal_peft"]:
            adapter_cfg = os.path.join(cfg["adapter_path"], "adapter_config.json") if cfg["adapter_path"] else None
            if adapter_cfg and os.path.exists(adapter_cfg):
                lm = AutoModelForCausalLM.from_pretrained(
                    cfg["base_name"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(device)
                lm = PeftModel.from_pretrained(lm, cfg["adapter_path"]).to(device)
            elif cfg["adapter_path"]:
                lm = AutoModelForCausalLM.from_pretrained(
                    cfg["adapter_path"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(device)
            else:
                lm = AutoModelForCausalLM.from_pretrained(
                    cfg["base_name"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(device)
            lm.eval()

            preds = []
            for src in tqdm(sources, desc=model_name):
                msgs = [
                    {"role": "system", "content": "Du bist ein professioneller Übersetzer für deutsche Leichte Sprache nach den offiziellen Regeln."},
                    {"role": "user", "content": f"Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n{src}"}
                ]
                prompt_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tok(prompt_str, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = lm.generate(
                        **enc,
                        max_new_tokens=256,
                        do_sample=False,
                        repetition_penalty=1.2,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id,
                    )
                gen_text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                preds.append(gen_text)
            translations_dict[model_name] = preds
            del lm
            torch.cuda.empty_cache()

        elif mtype == "seq2seq_peft":
            adapter_cfg = os.path.join(cfg["adapter_path"], "adapter_config.json") if cfg["adapter_path"] else None
            if adapter_cfg and os.path.exists(adapter_cfg):
                lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg["base_name"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ).to(device)
                lm = PeftModel.from_pretrained(lm, cfg["adapter_path"]).to(device)
            elif cfg["adapter_path"]:
                lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg["adapter_path"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ).to(device)
            else:
                lm = AutoModelForSeq2SeqLM.from_pretrained(
                    cfg["base_name"],
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                ).to(device)
            lm.eval()

            de_id = None
            if hasattr(tok, "lang_code_to_id") and tok.lang_code_to_id and "de_DE" in tok.lang_code_to_id:
                de_id = tok.lang_code_to_id["de_DE"]
            elif hasattr(tok, "convert_tokens_to_ids"):
                de_id = tok.convert_tokens_to_ids("de_DE")
            if de_id is None or de_id == getattr(tok, "unk_token_id", None):
                de_id = 250003
            print(f"  -> Konfigurierte Ziel-Sprach-ID (de_DE): {de_id}")
            preds = []
            for src in tqdm(sources, desc=model_name):
                enc = tok(src, max_length=256, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    gen_kw = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "max_length": 256,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.92,
                        "top_k": 50,
                        "repetition_penalty": 1.35,
                        "no_repeat_ngram_size": 3,
                        "pad_token_id": tok.pad_token_id,
                        "eos_token_id": tok.eos_token_id,
                    }
                    if de_id:
                        gen_kw["forced_bos_token_id"] = de_id
                    out = lm.generate(**gen_kw)
                gen_text = tok.decode(out[0], skip_special_tokens=True).strip()
                preds.append(gen_text)
            translations_dict[model_name] = preds
            del lm
            torch.cuda.empty_cache()

    # 5. Evaluate Metrics
    print("\n" + "=" * 80)
    print("Computing Readability, Faithfulness, and Rule Adherence Metrics...")
    print("=" * 80)

    results_rows = []
    summary_rows = []

    # Pre-encode sources & references
    src_embeddings = sbert.encode(sources, convert_to_tensor=True, batch_size=32)
    ref_embeddings = sbert.encode(references, convert_to_tensor=True, batch_size=32)

    for model_name, preds in translations_dict.items():
        print(f"Evaluating {model_name}...")
        pred_embeddings = sbert.encode(preds, convert_to_tensor=True, batch_size=32)
        sbert_sims = torch.diagonal(util.cos_sim(src_embeddings, pred_embeddings)).clamp(0.0, 1.0).cpu().numpy()
        bilstm_scores = get_bilstm_scores(preds)

        m_flesch, m_lix, m_wiener, m_bleu, m_rouge, m_comp, m_ratio = [], [], [], [], [], [], []
        m_passiv, m_genitiv, m_nominal, m_asl = [], [], [], []

        for i in range(len(sources)):
            src = sources[i]
            ref = references[i]
            cand = preds[i]

            fl = calculate_flesch_de(cand)
            lx = calculate_lix(cand)
            wn = calculate_wiener(cand)
            comp = (0.5 * bilstm_scores[i]) + (0.5 * sbert_sims[i])
            cr = len(cand) / max(1, len(src))

            ref_toks = [t.text.lower() for t in nlp(ref) if not t.is_space]
            cand_toks = [t.text.lower() for t in nlp(cand) if not t.is_space]

            bleu = compute_sentence_bleu(ref_toks, cand_toks)
            rouge = compute_rouge_l(ref_toks, cand_toks)["f1"]
            rules = rule_analyzer.analyze(cand)

            m_flesch.append(fl)
            m_lix.append(lx)
            m_wiener.append(wn)
            m_bleu.append(bleu)
            m_rouge.append(rouge)
            m_comp.append(comp)
            m_ratio.append(cr)
            m_passiv.append(rules["passiv_ratio"])
            m_genitiv.append(rules["genitiv_ratio"])
            m_nominal.append(rules["nominal_ratio"])
            m_asl.append(rules["avg_sent_len"])

            results_rows.append({
                "model": model_name,
                "sample_idx": i,
                "source_text": src,
                "reference_ls": ref,
                "generated_ls": cand,
                "bilstm_style": bilstm_scores[i],
                "sbert_sim": sbert_sims[i],
                "composite_reward": comp,
                "flesch_de": fl,
                "lix": lx,
                "wiener_sachtext": wn,
                "bleu_4": bleu,
                "rouge_l": rouge,
                "char_compression": cr,
                "passiv_ratio": rules["passiv_ratio"],
                "genitiv_ratio": rules["genitiv_ratio"],
                "nominal_ratio": rules["nominal_ratio"],
                "avg_sent_len": rules["avg_sent_len"],
            })

        summary_rows.append({
            "Model": model_name,
            "BiLSTM Simplicity": f"{np.mean(bilstm_scores):.3f} ± {np.std(bilstm_scores):.2f}",
            "SBERT Similarity": f"{np.mean(sbert_sims):.3f} ± {np.std(sbert_sims):.2f}",
            "Composite Reward": f"{np.mean(m_comp):.3f} ± {np.std(m_comp):.2f}",
            "Flesch-DE (↑)": f"{np.mean(m_flesch):.1f} ± {np.std(m_flesch):.1f}",
            "LIX (↓)": f"{np.mean(m_lix):.1f} ± {np.std(m_lix):.1f}",
            "Wiener Sachtext (↓)": f"{np.mean(m_wiener):.2f} ± {np.std(m_wiener):.2f}",
            "BLEU-4": f"{np.mean(m_bleu):.3f} ± {np.std(m_bleu):.2f}",
            "ROUGE-L F1": f"{np.mean(m_rouge):.3f} ± {np.std(m_rouge):.2f}",
            "Compression Ratio": f"{np.mean(m_ratio):.2f} ± {np.std(m_ratio):.2f}",
            "Passiv-Ratio": f"{np.mean(m_passiv):.3f}",
            "Genitiv-Ratio": f"{np.mean(m_genitiv):.3f}",
            "Nominal-Ratio": f"{np.mean(m_nominal):.3f}",
        })

    # 6. Save DataFrames
    df_results = pd.DataFrame(results_rows)
    df_results.to.csv(args.output_csv, index=False) if hasattr(df_results, "to") else df_results.to_csv(args.output_csv, index=False)
    print(f"\nDetailed sample benchmark saved to: {args.output_csv}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.output_summary, index=False)
    print(f"Summary table saved to: {args.output_summary}")

    print("\n" + "=" * 80)
    print("MASTER BENCHMARK SUMMARY TABLE:")
    print("=" * 80)
    print(df_summary.to_string(index=False))

    # 7. Generate Publication Plots
    print("\nGenerating Benchmark Plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Pareto Front (Simplicity vs Semantic Similarity)
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df_results,
        x="bilstm_style",
        y="sbert_sim",
        hue="model",
        style="model",
        alpha=0.6,
        s=70,
    )
    plt.title("Pareto-Trade-Off: Einfachheit ($R_{style}$) vs. Inhaltstreue ($R_{sem}$)", fontsize=13, fontweight="bold")
    plt.xlabel("BiLSTM Simplicity Score (↑)", fontsize=11)
    plt.ylabel("SBERT Cosine Similarity (↑)", fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    pareto_path = os.path.join(args.plot_dir, "pareto_style_vs_semantic_7way.png")
    plt.savefig(pareto_path, dpi=300)
    plt.close()

    # Plot 2: Boxplots for Readability
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df_results, x="model", y="flesch_de", ax=axes[0], palette="Set2")
    axes[0].set_title("Flesch-Reading-Ease (DE) (↑)", fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    sns.boxplot(data=df_results, x="model", y="lix", ax=axes[1], palette="Set2")
    axes[1].set_title("LIX Lesbarkeitsindex (↓)", fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

    sns.boxplot(data=df_results, x="model", y="composite_reward", ax=axes[2], palette="Set2")
    axes[2].set_title("Composite Reward Score (↑)", fontweight="bold")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    box_path = os.path.join(args.plot_dir, "readability_boxplots_7way.png")
    plt.savefig(box_path, dpi=300)
    plt.close()

    print(f"Benchmark plots saved to: {args.plot_dir}")
    print("=" * 80)
    print("7-Way Master Benchmark Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
