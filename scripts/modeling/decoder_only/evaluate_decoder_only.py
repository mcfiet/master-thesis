#!/usr/bin/env python3
"""
=============================================================================
Comprehensive Evaluation for Decoder-Only Models (SFT & DPO)
=============================================================================
Evaluates model outputs on the Lebenshilfe Gold-Standard Benchmark:
  1. Reward Model (BiLSTM MixUp Regressor Style-Score R_style)
  2. Semantic Preservation (SBERT to AS R_sem, SBERT to Gold Sim(Ref))
  3. Readability Metrics (Flesch-DE, Wiener Sachtextformel, LIX)
  4. Translation Quality (BLEU-4, ROUGE-L against Gold Reference)
  5. Rule Adherence / Leitplanken (Passiv, Genitiv, Nominalstil, Wort- & Satzlängen)
=============================================================================
"""

import os
import sys
import json
import re
import math
import argparse
import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from prompts import SYSTEM_PROMPT_LEICHTE_SPRACHE, USER_INSTRUCTION_PREFIX, create_chat_messages


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
log_dir = "results/logs/experiments/decoder_only"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("results/evaluation", exist_ok=True)

script_name = os.path.basename(__file__).replace(".py", "")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Logger(log_file)
sys.stderr = sys.stdout


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


def load_reward_model(device: str = "cpu", vocab_path: str = "data/vocabs/mixup_vocab.json", model_path: str = "results/models/lstm_article_mixup_regr/best_model.pt"):
    if not os.path.exists(vocab_path):
        vocab_path = "data/vocabs/synthetic_vocab.json"
    if not os.path.exists(vocab_path):
        return None, None

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    model = BiLSTMRegressor(len(vocab) + 2).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    elif os.path.exists("results/models/experiments/synthetic_regressor/bilstm_regressor_synthetic.pt"):
        model.load_state_dict(torch.load("results/models/experiments/synthetic_regressor/bilstm_regressor_synthetic.pt", map_location=device, weights_only=True))
    else:
        return None, None
    model.eval()
    return model, vocab


def score_text_style(text: str, model: nn.Module, vocab: Dict[str, int], device: str, max_len: int = 500) -> float:
    if model is None or vocab is None:
        return 0.5
    tokens = re.findall(r"\b\w+\b", str(text).lower())
    indices = [vocab.get(t, 1) for t in tokens][:max_len]
    if not indices:
        return 0.0
    padded = indices + [0] * (max_len - len(indices))
    tensor = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        score = model(tensor).item()
    return float(score)


# ==============================================================================
# RULE ADHERENCE & N-GRAM METRICS
# ==============================================================================
class LeichteSpracheRuleAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except Exception:
            self.nlp = spacy.blank("de")

        self.nominal_endings = re.compile(r"(ung|keit|heit|schaft|ismus|tion|tät)$", re.IGNORECASE)

    def analyze_text(self, text: str) -> Dict[str, float]:
        doc = self.nlp(str(text or ""))
        total_tokens = len([t for t in doc if not t.is_punct and not t.is_space])
        total_sentences = len(list(doc.sents)) or 1

        if total_tokens == 0:
            return {
                "avg_sentence_len": 0.0,
                "avg_word_len": 0.0,
                "passive_count": 0,
                "passive_ratio": 0.0,
                "genitive_count": 0,
                "genitive_ratio": 0.0,
                "nominal_count": 0,
                "nominal_ratio": 0.0,
                "long_words_ratio": 0.0,
            }

        words = [t.text for t in doc if not t.is_punct and not t.is_space]
        avg_word_len = sum(len(w) for w in words) / total_tokens
        avg_sent_len = total_tokens / total_sentences

        passive_count = 0
        genitive_count = 0
        nominal_count = 0
        long_words = sum(1 for w in words if len(w) > 6)

        for token in doc:
            if token.pos_ == "AUX" and token.lemma_ == "werden":
                passive_count += 1
            if "Case=Gen" in str(token.morph) or (token.text.endswith("s") and token.pos_ in ["NOUN", "PROPN"]):
                genitive_count += 1
            if token.pos_ == "NOUN" and self.nominal_endings.search(token.text):
                nominal_count += 1

        return {
            "avg_sentence_len": float(avg_sent_len),
            "avg_word_len": float(avg_word_len),
            "passive_count": int(passive_count),
            "passive_ratio": float(passive_count / total_sentences),
            "genitive_count": int(genitive_count),
            "genitive_ratio": float(genitive_count / total_tokens),
            "nominal_count": int(nominal_count),
            "nominal_ratio": float(nominal_count / total_tokens),
            "long_words_ratio": float(long_words / total_tokens),
        }


def compute_bleu_and_rouge(ref_text: str, hyp_text: str) -> Dict[str, float]:
    ref_tokens = re.findall(r"\b\w+\b", str(ref_text).lower())
    hyp_tokens = re.findall(r"\b\w+\b", str(hyp_text).lower())
    if not ref_tokens or not hyp_tokens:
        return {"bleu": 0.0, "rouge_l": 0.0}

    # BLEU-4
    precisions = []
    for n in range(1, 5):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            precisions.append(0.0)
            continue
        hyp_ng = [tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)]
        ref_ng = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
        ref_counts = {}
        for ng in ref_ng:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        matched = sum(1 for ng in hyp_ng if ref_counts.get(ng, 0) > 0)
        precisions.append(matched / max(1, len(hyp_ng)))

    smoothed_p = [max(p, 1e-4) for p in precisions]
    geo_mean = math.exp(sum(0.25 * math.log(p) for p in smoothed_p))
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens)))
    bleu = bp * geo_mean

    # ROUGE-L
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    rouge_l = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {"bleu": float(bleu), "rouge_l": float(rouge_l)}


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Decoder-Only SFT/DPO Models on Lebenshilfe Benchmark")
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--model_path", required=True, help="Path to adapter checkpoint or merged model")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_target_len", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sbert_model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Tokenizer & Model
    print(f"Loading Tokenizer from {args.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading Model from {args.model_path}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base_model, args.model_path)
    elif os.path.exists(os.path.join(args.model_path, "model.safetensors")) or os.path.exists(os.path.join(args.model_path, "pytorch_model.bin")):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        model = base_model
    model.eval()

    # 2. Load Evaluators & Reward Model
    print("Loading Reward Model (BiLSTM) & SBERT...")
    reward_model, vocab = load_reward_model(device=device)
    rule_analyzer = LeichteSpracheRuleAnalyzer()
    sbert = SentenceTransformer(args.sbert_model, trust_remote_code=True, device=device)

    # 3. Load Test Data
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples:
        data = data[:args.max_samples]
    print(f"Evaluating {len(data)} test samples from {args.test_data_path}...")

    generated_texts = []
    source_texts = []
    reference_texts = []

    for item in tqdm(data, desc="Generating Simplifications"):
        as_text = str(item.get("as_text") or "").strip()
        ls_ref = str(item.get("ls_text") or item.get("target_text") or "").strip()
        if not as_text:
            continue

        msgs = create_chat_messages(as_text=as_text, ls_text=None)
        prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=args.max_target_len,
                temperature=args.temperature,
                do_sample=False if args.temperature == 0.0 else True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        source_texts.append(as_text)
        reference_texts.append(ls_ref)
        generated_texts.append(gen_text)

    # 4. Compute Metrics
    print()
    print("Computing semantic similarities, style rewards, and readability metrics...")
    eff_len = getattr(sbert, "max_seq_length", 8192)
    sbert_bs = 2 if eff_len > 4096 else (4 if eff_len > 1024 else (8 if eff_len > 512 else 16))

    emb_src = sbert.encode(source_texts, batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)
    emb_gen = sbert.encode(generated_texts, batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)
    cosine_sims_as = util.cos_sim(emb_src, emb_gen).diagonal().cpu().numpy()

    if any(reference_texts):
        emb_ref = sbert.encode([r if r else " " for r in reference_texts], batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)
        cosine_sims_ref = util.cos_sim(emb_ref, emb_gen).diagonal().cpu().numpy()
    else:
        cosine_sims_ref = np.zeros(len(generated_texts))

    eval_rows = []
    for i in range(len(generated_texts)):
        gen = generated_texts[i]
        src = source_texts[i]
        ref = reference_texts[i]

        r_style = score_text_style(gen, reward_model, vocab, device)
        r_sem_as = float(cosine_sims_as[i])
        sim_ref = float(cosine_sims_ref[i])
        composite = 0.5 * r_style + 0.5 * r_sem_as

        rules = rule_analyzer.analyze_text(gen)
        flesch = textstat.flesch_reading_ease(gen)
        lix = textstat.lix(gen)
        wstf = textstat.wiener_sachtextformel(gen, variant=1) if hasattr(textstat, "wiener_sachtextformel") else 0.0
        n_gram_res = compute_bleu_and_rouge(ref, gen)

        eval_rows.append({
            "source_text": src,
            "reference_text": ref,
            "generated_text": gen,
            "r_style": r_style,
            "r_sem_as": r_sem_as,
            "sim_ref": sim_ref,
            "composite_reward": composite,
            "flesch_score": float(flesch),
            "lix_score": float(lix),
            "wiener_score": float(wstf),
            "bleu": n_gram_res["bleu"],
            "rouge_l": n_gram_res["rouge_l"],
            **rules,
        })

    df = pd.DataFrame(eval_rows)
    print()
    print("=" * 65)
    print("EVALUATION SUMMARY REPORT (LEBENSHILFE BENCHMARK)")
    print("=" * 65)
    print(f"Model: {args.model_path}")
    print(f"Total Samples: {len(df)}")
    print(f"Style Simplicity Reward (BiLSTM): {df["r_style"].mean():.4f} ± {df["r_style"].std():.4f}")
    print(f"SBERT Quelltreue (zu AS):         {df["r_sem_as"].mean():.4f} ± {df["r_sem_as"].std():.4f}")
    print(f"SBERT Ähnlichkeit (zu Gold Ref):  {df["sim_ref"].mean():.4f} ± {df["sim_ref"].std():.4f}")
    print(f"Composite Reward (0.5/0.5):       {df["composite_reward"].mean():.4f} ± {df["composite_reward"].std():.4f}")
    print(f"Flesch Reading Ease (DE):         {df["flesch_score"].mean():.2f} ± {df["flesch_score"].std():.2f}")
    print(f"LIX Lesbarkeitsindex:             {df["lix_score"].mean():.2f} ± {df["lix_score"].std():.2f}")
    print(f"BLEU-4 (vs. Gold):                {df["bleu"].mean():.4f} ± {df["bleu"].std():.4f}")
    print(f"ROUGE-L F1 (vs. Gold):            {df["rouge_l"].mean():.4f} ± {df["rouge_l"].std():.4f}")
    print(f"Genitive Ratio (per token):       {df["genitive_ratio"].mean():.4f}")
    print(f"Nominalization Ratio:             {df["nominal_ratio"].mean():.4f}")
    print("=" * 65)

    # 5. Save Report
    out_file = args.output_file or f"results/evaluation/eval_decoder_{timestamp}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Detailed evaluation saved to: {out_file}")


if __name__ == "__main__":
    main()
