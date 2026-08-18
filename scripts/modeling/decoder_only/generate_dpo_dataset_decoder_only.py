#!/usr/bin/env python3
"""
=============================================================================
DPO Preference Dataset Generator for Decoder-Only LLMs
=============================================================================
This script generates preference pairs (prompt, chosen, rejected) for DPO training
using a fine-tuned Decoder-Only SFT model and a Composite Reward Model:
  1. Prompts the SFT model using the standardized Leichte-Sprache Chat Template.
  2. Samples multiple candidate simplifications per source text.
  3. Evaluates each candidate with:
     - BiLSTM Simplicity / Style Regressor (R_style)
     - Sentence-BERT Semantic Preservation (R_sem)
     - Composite Reward = w_style * R_style + w_sem * R_sem_norm
  4. Identifies the best (chosen) and worst (rejected) candidate.
  5. Filters by minimum score margin and saves train/validation JSONL datasets.
=============================================================================
"""

import os
import sys
import json
import random
import argparse
import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel

from prompts import SYSTEM_PROMPT_LEICHTE_SPRACHE, USER_INSTRUCTION_PREFIX, create_chat_messages


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
log_dir = "results/logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("data/dpo", exist_ok=True)

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
# REWARD MODEL DEFINITION (BiLSTM Regressor)
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


class CompositeRewardEvaluator:
    def __init__(
        self,
        reward_model_path: str,
        reward_vocab_path: str,
        sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        w_style: float = 0.5,
        w_sem: float = 0.5,
        max_seq_len: int = 256,
        device: str = "cuda",
    ):
        self.w_style = w_style
        self.w_sem = w_sem
        self.max_seq_len = max_seq_len
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load Vocabulary
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.unk_idx = self.vocab.get("<UNK>", 1)
        self.pad_idx = self.vocab.get("<PAD>", 0)

        # Load SpaCy Tokenizer
        try:
            self.nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
        except Exception:
            self.nlp = spacy.blank("de")

        # Load BiLSTM Model
        self.reward_model = BiLSTMRegressor(vocab_size=len(self.vocab))
        state_dict = torch.load(reward_model_path, map_location=self.device)
        self.reward_model.load_state_dict(state_dict)
        self.reward_model.to(self.device)
        self.reward_model.eval()

        # Load SBERT
        print(f"Loading SBERT model: {sbert_model_name}")
        self.sbert = SentenceTransformer(sbert_model_name, trust_remote_code=True, device=str(self.device))
        if "jina" in sbert_model_name.lower():
            self.sbert.max_seq_length = min(self.max_seq_len, 1024)
            print(f"Set Jina SBERT max_seq_length to {self.sbert.max_seq_length}")

    def _tokenize(self, text: str) -> List[int]:
        tokens = [token.text.lower() for token in self.nlp(text)]
        token_ids = [self.vocab.get(t, self.unk_idx) for t in tokens][:self.max_seq_len]
        return token_ids

    def predict_simplicity(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        batch_ids = [self._tokenize(t) for t in texts]
        max_l = max((len(ids) for ids in batch_ids), default=1)
        padded = np.zeros((len(texts), max(max_l, 1)), dtype=np.int64)
        for i, ids in enumerate(batch_ids):
            padded[i, :len(ids)] = ids
        tensor_x = torch.tensor(padded, dtype=torch.long, device=self.device)
        with torch.no_grad():
            scores = self.reward_model(tensor_x).squeeze(-1).cpu().numpy()
        return np.atleast_1d(scores)

    def predict_similarity(self, source_texts: List[str], cand_texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            emb_src = self.sbert.encode(source_texts, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            emb_cand = self.sbert.encode(cand_texts, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            sims = util.cos_sim(emb_src, emb_cand).diagonal().cpu().numpy()
        return np.atleast_1d(sims)

    def compute_rewards(self, source_texts: List[str], cand_texts: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_style = self.predict_simplicity(cand_texts)
        r_sem = self.predict_similarity(source_texts, cand_texts)
        r_sem_norm = np.clip((r_sem + 1.0) / 2.0, 0.0, 1.0)
        total = self.w_style * r_style + self.w_sem * r_sem_norm
        return total, r_style, r_sem_norm


# ==============================================================================
# MAIN GENERATION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate DPO pairs from SFT Decoder-Only Model")
    parser.add_argument("--corpus_path", default="data/corpus/corpus_master_with_steps.json")
    parser.add_argument("--sft_model_path", required=True, help="Path to trained SFT adapter or checkpoint")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--reward_model_path", default="results/models/token_length_exp/bilstm_mixup_regression_1000.pt")
    parser.add_argument("--reward_vocab_path", default="data/token_length_exp/mixup_vocab_1000.json")
    parser.add_argument("--sbert_model_name", default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--output_file", default="data/dpo/dpo_preference_pairs_decoder_only.jsonl")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--min_score_margin", type=float, default=0.05)
    parser.add_argument("--max_source_len", type=int, default=1000)
    parser.add_argument("--max_target_len", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--w_style", type=float, default=0.7)
    parser.add_argument("--w_sem", type=float, default=0.3)
    parser.add_argument("--val_split_ratio", type=float, default=0.15)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Tokenizer & Model
    print(f"Loading Tokenizer from: {args.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batched autoregressive generation

    print(f"Loading SFT Model with adapter from: {args.sft_model_path}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if os.path.exists(os.path.join(args.sft_model_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base_model, args.sft_model_path)
    else:
        model = base_model
    model.eval()

    # 2. Load Reward Evaluator
    print("Loading Composite Reward Evaluator...")
    evaluator = CompositeRewardEvaluator(
        reward_model_path=args.reward_model_path,
        reward_vocab_path=args.reward_vocab_path,
        sbert_model_name=args.sbert_model_name,
        w_style=args.w_style,
        w_sem=args.w_sem,
        max_seq_len=args.max_target_len,
        device=device,
    )

    # 3. Load Data
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    if args.max_samples:
        corpus = corpus[:args.max_samples]
    print(f"Total samples to process: {len(corpus)}")

    # 4. Generate & Rank
    preference_pairs = []
    dropped_margin = 0
    dropped_identical = 0

    batch_size = args.batch_size
    for b_idx in tqdm(range(0, len(corpus), batch_size), desc="Sampling Candidates"):
        batch = corpus[b_idx : b_idx + batch_size]
        as_texts = [str(item.get("as_text") or "").strip() for item in batch]
        
        # Build prompt chat templates
        formatted_prompts = []
        for text in as_texts:
            msgs = create_chat_messages(as_text=text, ls_text=None)
            prompt_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(prompt_str)

        # Tokenize batch
        inputs = tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=args.max_source_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_target_len,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                num_return_sequences=args.num_candidates,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice out prompt tokens
        input_len = inputs["input_ids"].shape[1]
        completions_tokens = gen_tokens[:, input_len:]
        decoded_cands = tokenizer.batch_decode(completions_tokens, skip_special_tokens=True)

        for i, item in enumerate(batch):
            as_text = as_texts[i]
            cands = [c.strip() for c in decoded_cands[i * args.num_candidates : (i + 1) * args.num_candidates]]
            
            # Ground-truth addition
            gt_ls = str(item.get("ls_text") or "").strip()
            if gt_ls and gt_ls not in cands:
                cands.append(gt_ls)

            cands = list(set([c for c in cands if len(c) > 10]))
            if len(cands) < 2:
                dropped_identical += 1
                continue

            # Compute rewards
            source_rep = [as_text] * len(cands)
            total_r, style_r, sem_r = evaluator.compute_rewards(source_rep, cands)

            ranked_idx = np.argsort(-total_r)
            best_idx = ranked_idx[0]
            worst_idx = ranked_idx[-1]

            chosen = cands[best_idx]
            rejected = cands[worst_idx]
            margin = float(total_r[best_idx] - total_r[worst_idx])

            if chosen == rejected:
                dropped_identical += 1
                continue
            if margin < args.min_score_margin:
                dropped_margin += 1
                continue

            pair = {
                "prompt": formatted_prompts[i],
                "as_text": as_text,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": float(total_r[best_idx]),
                "rejected_score": float(total_r[worst_idx]),
                "score_margin": margin,
                "chosen_style": float(style_r[best_idx]),
                "chosen_sem": float(sem_r[best_idx]),
                "rejected_style": float(style_r[worst_idx]),
                "rejected_sem": float(sem_r[worst_idx]),
                "source": item.get("source"),
            }
            preference_pairs.append(pair)

    print(f"\nGenerated {len(preference_pairs)} DPO pairs.")
    print(f"Dropped (low margin < {args.min_score_margin}): {dropped_margin}")
    print(f"Dropped (identical/insufficient): {dropped_identical}")

    # 5. Save Splits
    random.shuffle(preference_pairs)
    split_idx = int((1.0 - args.val_split_ratio) * len(preference_pairs))
    train_pairs = preference_pairs[:split_idx]
    val_pairs = preference_pairs[split_idx:]

    base_name, ext = os.path.splitext(args.output_file)
    eval_file = f"{base_name}_eval{ext}"

    with open(args.output_file, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Train DPO pairs saved to: {args.output_file} ({len(train_pairs)} pairs)")

    with open(eval_file, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Validation DPO pairs saved to: {eval_file} ({len(val_pairs)} pairs)")


if __name__ == "__main__":
    main()
