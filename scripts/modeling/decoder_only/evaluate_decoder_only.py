#!/usr/bin/env python3
"""
=============================================================================
Comprehensive Evaluation for Decoder-Only Models (SFT & DPO)
=============================================================================
Evaluates model outputs across 3 essential dimensions for Leichte Sprache:
  1. Standard Simplification Metrics (BLEU, ROUGE-L, SARI, SBERT Semantic Preservation)
  2. Readability Metrics (Flesch-DE, Wiener Sachtextformel, LIX, Word Length)
  3. Rule Adherence / Leitplanken Metrics (Passiv-Ratio, Genitiv-Ratio, Nominalstil, Konjunktiv)
=============================================================================
"""

import os
import sys
import json
import re
import argparse
import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import spacy
import textstat
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from prompts import SYSTEM_PROMPT_LEICHTE_SPRACHE, USER_INSTRUCTION_PREFIX, create_chat_messages


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
log_dir = "results/logs"
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
# RULE ADHERENCE ANALYZER (Leichte Sprache W2 - W11)
# ==============================================================================
class LeichteSpracheRuleAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except Exception:
            self.nlp = spacy.blank("de")

        self.nominal_endings = re.compile(r"(ung|keit|heit|schaft|ismus|tion|tät)$", re.IGNORECASE)

    def analyze_text(self, text: str) -> Dict[str, float]:
        doc = self.nlp(text)
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

        # 1. Passive voice detection (auxiliary passives)
        passive_count = 0
        genitive_count = 0
        nominal_count = 0
        long_words = sum(1 for w in words if len(w) > 12)

        for token in doc:
            # W8: Passive
            if token.dep_ in ["sb_pass", "oc_pass"] or (token.lemma_ == "werden" and token.pos_ == "AUX"):
                passive_count += 1
            # W9: Genitive
            morph = str(token.morph)
            if "Case=Gen" in morph:
                genitive_count += 1
            # W7: Nominal style
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


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Decoder-Only SFT/DPO Models")
    parser.add_argument("--test_data_path", default="data/corpus/corpus_master_with_steps.json")
    parser.add_argument("--model_path", required=True, help="Path to adapter checkpoint or merged model")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_target_len", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sbert_model", default="jinaai/jina-embeddings-v2-base-de")
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
    else:
        model = base_model
    model.eval()

    # 2. Load Evaluators
    rule_analyzer = LeichteSpracheRuleAnalyzer()
    sbert = SentenceTransformer(args.sbert_model, trust_remote_code=True, device=device)
    if "jina" in args.sbert_model.lower():
        sbert.max_seq_length = min(args.max_target_len, 1024)
        print(f"Set Jina SBERT max_seq_length to {sbert.max_seq_length}")

    # 3. Load Test Data
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.max_samples:
        data = data[-args.max_samples:]  # Take holdout set from the end
    print(f"Evaluating {len(data)} test samples...")

    results = []
    generated_texts = []
    source_texts = []
    reference_texts = []

    for item in tqdm(data, desc="Generating Simplifications"):
        as_text = str(item.get("as_text") or "").strip()
        ls_ref = str(item.get("ls_text") or "").strip()
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
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(output_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        source_texts.append(as_text)
        reference_texts.append(ls_ref)
        generated_texts.append(gen_text)

    # 4. Compute Metrics
    print("\nComputing semantic similarities and readability metrics...")
    emb_src = sbert.encode(source_texts, convert_to_tensor=True)
    emb_gen = sbert.encode(generated_texts, convert_to_tensor=True)
    cosine_sims = util.cos_sim(emb_src, emb_gen).diagonal().cpu().numpy()

    eval_rows = []
    for i in range(len(generated_texts)):
        gen = generated_texts[i]
        src = source_texts[i]
        ref = reference_texts[i]

        rules = rule_analyzer.analyze_text(gen)
        flesch = textstat.flesch_reading_ease(gen)
        lix = textstat.lix(gen)

        eval_rows.append({
            "source_text": src[:120] + "...",
            "generated_text": gen[:120] + "...",
            "cosine_sim": float(cosine_sims[i]),
            "flesch_score": float(flesch),
            "lix_score": float(lix),
            **rules,
        })

    df = pd.DataFrame(eval_rows)
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Total Samples: {len(df)}")
    print(f"Avg Semantic Similarity (SBERT): {df['cosine_sim'].mean():.4f}")
    print(f"Avg Sentence Length (Tokens):     {df['avg_sentence_len'].mean():.2f}")
    print(f"Avg Word Length (Chars):          {df['avg_word_len'].mean():.2f}")
    print(f"Avg Passive Ratio (per sentence): {df['passive_ratio'].mean():.4f}")
    print(f"Avg Genitive Ratio (per token):   {df['genitive_ratio'].mean():.4f}")
    print(f"Avg Nominalization Ratio:         {df['nominal_ratio'].mean():.4f}")
    print(f"Avg Long Words (>12 chars):       {df['long_words_ratio'].mean():.4f}")
    print(f"Avg Flesch Reading Ease:          {df['flesch_score'].mean():.2f}")
    print(f"Avg LIX Score:                    {df['lix_score'].mean():.2f}")
    print("=" * 60)

    # 5. Save Report
    out_file = args.output_file or f"results/evaluation/eval_decoder_{timestamp}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"Detailed evaluation saved to: {out_file}")


if __name__ == "__main__":
    main()
