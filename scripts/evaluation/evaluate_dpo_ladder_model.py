#!/usr/bin/env python3
"""
=============================================================================
Evaluation Script: Temperature Ladder DPO Model (500 Tokens)
=============================================================================
Evaluates and compares:
  1. Base SFT Model (500 Tokens)
  2. Trained Temperature Ladder DPO Model (500 Tokens)
on the unseen Lebenshilfe benchmark dataset.
=============================================================================
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from peft import PeftModel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("EvaluateDPOLadder")


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


def load_model_and_tokenizer(model_path: str, base_model_name: str, device: str):
    is_seq2seq = True
    try:
        cfg = AutoConfig.from_pretrained(model_path)
        is_seq2seq = cfg.is_encoder_decoder
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(base_model_name)
            is_seq2seq = cfg.is_encoder_decoder
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    dtype = torch.float16 if device == "cuda" else torch.float32
    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_seq2seq:
        if has_adapter:
            base_m = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
            peft_m = PeftModel.from_pretrained(base_m, model_path)
            model = peft_m.merge_and_unload()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
    else:
        if has_adapter:
            base_m = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
            peft_m = PeftModel.from_pretrained(base_m, model_path)
            model = peft_m.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    if hasattr(tokenizer, "src_lang") or hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

    model = model.to(device)
    model.eval()
    return model, tokenizer, is_seq2seq


def main():
    parser = argparse.ArgumentParser(description="Evaluate 512-token DPO Ladder Model vs SFT Baseline.")
    parser.add_argument("--test_data_path", type=str, default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--sft_model_path", type=str, default="results/models/sft" if os.path.exists("results/models/sft") else "results/models/token_length_exp/sft_len512")
    parser.add_argument("--dpo_model_path", type=str, default="results/models/dpo" if os.path.exists("results/models/dpo") else "results/models/temperature_ladder_500/dpo_w05_w05")
    parser.add_argument("--base_model_name", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--reward_model_path", type=str, default="results/models/bilstm_mixup_regression.pt" if os.path.exists("results/models/bilstm_mixup_regression.pt") else "results/models/token_length_exp/bilstm_mixup_regression_512.pt")
    parser.add_argument("--reward_vocab_path", type=str, default="data/vocabs/mixup_vocab.json" if os.path.exists("data/vocabs/mixup_vocab.json") else "data/token_length_exp/mixup_vocab_512.json")
    parser.add_argument("--sbert_model_name", type=str, default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--sbert_max_seq_len", type=int, default=8192, help="Max sequence length for SBERT (default: 8192)")
    parser.add_argument("--output_summary", type=str, default="results/evaluation/dpo_ladder_summary.csv")
    parser.add_argument("--output_details", type=str, default="results/evaluation/dpo_ladder_details.csv")
    parser.add_argument("--max_source_len", type=int, default=1024)
    parser.add_argument("--max_target_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Test Data
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)
    logger.info(f"Loaded {len(test_samples)} test samples from {args.test_data_path}")

    # Load Reward Model & SBERT
    with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data.get("stoi", vocab_data)
    unk_idx = stoi.get("<unk>") or stoi.get("<UNK>") or 1

    bilstm = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(device)
    raw_st = torch.load(args.reward_model_path, map_location=device)
    if isinstance(raw_st, dict) and "model_state_dict" in raw_st:
        raw_st = raw_st["model_state_dict"]
    bilstm.load_state_dict(raw_st)
    bilstm.eval()

    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        nlp = spacy.blank("de")
        nlp.add_pipe("sentencizer")

    sbert = SentenceTransformer(args.sbert_model_name, device=device, trust_remote_code=True)
    if args.sbert_max_seq_len and hasattr(sbert, "max_seq_length"):
        sbert.max_seq_length = args.sbert_max_seq_len

    def score_texts(as_list, cand_list, ref_list):
        # Simplicity
        style_scores = []
        for text in cand_list:
            doc = nlp(str(text or ""))
            tokens = [t.text.lower() for t in doc if not t.is_space]
            indices = [stoi.get(t, unk_idx) for t in tokens[:args.max_target_len]]
            if len(indices) == 0:
                indices = [0]
            inp = torch.tensor([indices], dtype=torch.long, device=device)
            with torch.no_grad():
                style_scores.append(bilstm(inp).item())
        style_scores = np.array(style_scores)

        # Semantics
        effective_len = getattr(sbert, "max_seq_length", args.sbert_max_seq_len)
        if effective_len > 4096:
            sbert_bs = 2
        elif effective_len > 1024:
            sbert_bs = 4
        elif effective_len > 512:
            sbert_bs = 8
        else:
            sbert_bs = 16

        with torch.inference_mode():
            emb_as = sbert.encode(as_list, batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)
            emb_cand = sbert.encode(cand_list, batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)
            emb_ref = sbert.encode(ref_list, batch_size=sbert_bs, convert_to_tensor=True, show_progress_bar=False)

            sim_as = util.cos_sim(emb_as, emb_cand).diagonal().cpu().numpy()
            sim_ref = util.cos_sim(emb_ref, emb_cand).diagonal().cpu().numpy()

        sim_as_norm = np.clip((sim_as + 1.0) / 2.0, 0.0, 1.0)
        tot_reward = 0.5 * style_scores + 0.5 * sim_as_norm
        return style_scores, sim_as, sim_ref, tot_reward

    models_to_evaluate = [
        ("SFT", args.sft_model_path),
        ("DPO Ladder", args.dpo_model_path),
    ]

    summary_records = []
    detail_records = []

    as_texts = [str(s.get("as_text") or "").strip() for s in test_samples]
    ref_texts = [str(s.get("ls_text") or "").strip() for s in test_samples]

    for model_name, path in models_to_evaluate:
        if not os.path.exists(path):
            logger.warning(f"Model path does not exist, skipping: {path}")
            continue

        logger.info(f"Generating simplifications for: {model_name} ({path})...")
        model, tokenizer, is_seq2seq = load_model_and_tokenizer(path, args.base_model_name, device)

        translations = []
        num_batches = (len(as_texts) + args.batch_size - 1) // args.batch_size
        for b in tqdm(range(num_batches), desc=f"Evaluating {model_name}"):
            batch_prompts = as_texts[b * args.batch_size : (b + 1) * args.batch_size]
            inp = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_source_len,
                return_tensors="pt",
            ).to(device)

            if is_seq2seq:
                gen_kwargs = {
                    "input_ids": inp["input_ids"],
                    "attention_mask": inp.get("attention_mask"),
                    "max_length": args.max_target_len,
                    "num_beams": 4,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "early_stopping": True,
                    "length_penalty": 1.0,
                }
            else:
                gen_kwargs = {
                    "input_ids": inp["input_ids"],
                    "attention_mask": inp.get("attention_mask"),
                    "max_new_tokens": args.max_target_len,
                    "do_sample": False,
                    "num_beams": 1,
                    "repetition_penalty": 1.1,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }

            with torch.no_grad():
                out = model.generate(**gen_kwargs)
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            translations.extend([d.strip() for d in decoded])

        # Evaluate translations
        style_sc, sim_as, sim_ref, comp_r = score_texts(as_texts, translations, ref_texts)

        comp_ratios = [len(t.split()) / max(1, len(a.split())) for t, a in zip(translations, as_texts)]

        summary_records.append({
            "Model": model_name,
            "Style (Simplicity)": f"{style_sc.mean():.4f} ± {style_sc.std():.4f}",
            "SBERT to AS": f"{sim_as.mean():.4f} ± {sim_as.std():.4f}",
            "SBERT to Ref (Gold)": f"{sim_ref.mean():.4f} ± {sim_ref.std():.4f}",
            "Composite Reward": f"{comp_r.mean():.4f} ± {comp_r.std():.4f}",
            "Compression Ratio": f"{np.mean(comp_ratios):.3f}",
        })

        for i in range(len(as_texts)):
            detail_records.append({
                "Model": model_name,
                "as_text": as_texts[i],
                "ls_reference": ref_texts[i],
                "translation": translations[i],
                "style_score": float(style_sc[i]),
                "sbert_to_as": float(sim_as[i]),
                "sbert_to_ref": float(sim_ref[i]),
                "composite_reward": float(comp_r[i]),
                "compression_ratio": float(comp_ratios[i]),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)), exist_ok=True)
    df_sum = pd.DataFrame(summary_records)
    df_sum.to_csv(args.output_summary, index=False)
    logger.info(f"Saved summary to {args.output_summary}")

    df_det = pd.DataFrame(detail_records)
    df_det.to_csv(args.output_details, index=False)
    logger.info(f"Saved details to {args.output_details}")

    print("\n=================== EVALUATION RESULTS ===================")
    print(df_sum.to_string(index=False))
    print("==========================================================\n")


if __name__ == "__main__":
    main()
