#!/usr/bin/env python3
"""
=============================================================================
Progressive Temperature Ladder DPO Dataset Generator (500 Tokens)
=============================================================================
Generates high-contrast DPO preference pairs (prompt, chosen, rejected) strictly
from SFT model generation without ground-truth reference texts:
  1. Starts candidate sampling at a base temperature (e.g. T = 0.7).
  2. Evaluates candidates using the 500-Token BiLSTM Regressor & SBERT.
  3. If candidate pool margin (max_reward - min_reward) < min_score_margin (0.05),
     escalates temperature through a ladder (0.7 -> 0.8 -> 0.9 -> 1.0) and
     samples additional candidates until the target margin is satisfied.
  4. Exports train and validation splits to JSONL format.
=============================================================================
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

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
    set_seed,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("GenerateDPOLadder")


# ---------------------------------------------------------------------------
# BiLSTM Regressor Definition (500-Token Compatible)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Composite Reward Evaluator
# ---------------------------------------------------------------------------
class CompositeRewardEvaluator:
    def __init__(
        self,
        reward_model_path: str,
        reward_vocab_path: str,
        sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        w_style: float = 0.5,
        w_sem: float = 0.5,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        max_seq_len: int = 500,
        device: str = "cpu",
    ):
        self.w_style = w_style
        self.w_sem = w_sem
        self.max_seq_len = max_seq_len
        self.device = torch.device(device)

        # 1. Load Vocabulary
        logger.info(f"Loading Reward Model vocabulary from: {reward_vocab_path}")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            self.stoi = vocab_data.get("stoi", vocab_data)

        self.unk_idx = self.stoi.get("<unk>") or self.stoi.get("<UNK>") or 1

        # 2. Load BiLSTM Simplicity Model
        logger.info(f"Loading BiLSTM Regressor weights from: {reward_model_path}")
        self.bilstm_model = BiLSTMRegressor(
            vocab_size=len(self.stoi),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        raw_state = torch.load(reward_model_path, map_location=self.device)
        if isinstance(raw_state, dict):
            if "model_state_dict" in raw_state:
                raw_state = raw_state["model_state_dict"]
            elif "state_dict" in raw_state:
                raw_state = raw_state["state_dict"]

        self.bilstm_model.load_state_dict(raw_state)
        self.bilstm_model.eval()

        # 3. Load SpaCy Tokenizer
        try:
            self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
        except Exception:
            self.nlp = spacy.blank("de")
            self.nlp.add_pipe("sentencizer")

        # 4. Load SBERT Model
        logger.info(f"Loading SBERT model for semantic evaluation: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name, trust_remote_code=True, device=self.device)
        if "jina" in sbert_model_name.lower():
            self.sbert_model.max_seq_length = min(max(self.max_seq_len, 256), 1024)

    def predict_simplicity_scores(self, texts: List[str]) -> np.ndarray:
        scores = []
        for text in texts:
            doc = self.nlp(str(text or ""))
            tokens = [t.text.lower() for t in doc if not t.is_space]
            indices = [self.stoi.get(t, self.unk_idx) for t in tokens[: self.max_seq_len]]
            if len(indices) == 0:
                indices = [0]
            inp_tensor = torch.tensor([indices], dtype=torch.long, device=self.device)
            with torch.no_grad():
                score = self.bilstm_model(inp_tensor).item()
            scores.append(score)
        return np.array(scores)

    def predict_semantic_similarity(self, source_texts: List[str], candidate_texts: List[str]) -> np.ndarray:
        with torch.inference_mode():
            emb_src = self.sbert_model.encode(source_texts, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            emb_cand = self.sbert_model.encode(candidate_texts, batch_size=8, convert_to_tensor=True, show_progress_bar=False)
            cosine_sims = util.cos_sim(emb_src, emb_cand).diagonal().cpu().numpy()
        return cosine_sims

    def compute_rewards(
        self, source_texts: List[str], candidate_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_style = self.predict_simplicity_scores(candidate_texts)
        r_sem = self.predict_semantic_similarity(source_texts, candidate_texts)
        r_sem_norm = np.clip((r_sem + 1.0) / 2.0, 0.0, 1.0)
        total_rewards = self.w_style * r_style + self.w_sem * r_sem_norm
        return total_rewards, r_style, r_sem_norm


# ---------------------------------------------------------------------------
# Corpus Loading
# ---------------------------------------------------------------------------
def load_corpus_data(corpus_path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading corpus from: {corpus_path}")
    if corpus_path.endswith(".json"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif corpus_path.endswith(".csv"):
        df = pd.read_csv(corpus_path)
        raw_data = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format: {corpus_path}")

    cleaned = []
    for item in raw_data:
        as_text = str(item.get("as_text") or item.get("text") or "").strip()
        if len(as_text) > 20:
            cleaned.append({
                "as_text": as_text,
                "source": item.get("source", "unknown"),
                "as_tokens": item.get("as_tokens", len(as_text.split())),
            })

    logger.info(f"Loaded {len(cleaned)} valid records from corpus.")
    if max_samples and len(cleaned) > max_samples:
        logger.info(f"Subsampling to {max_samples} records.")
        cleaned = cleaned[:max_samples]
    return cleaned


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_sft_model_and_tokenizer(
    model_name_or_path: str,
    base_model_name: str = "facebook/mbart-large-50",
    device: str = "cuda",
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, bool]:
    logger.info(f"Loading SFT model/tokenizer from: {model_name_or_path}")

    # Determine Seq2Seq architecture
    is_seq2seq = True
    try:
        cfg = AutoConfig.from_pretrained(model_name_or_path)
        is_seq2seq = cfg.is_encoder_decoder
    except Exception:
        try:
            cfg = AutoConfig.from_pretrained(base_model_name)
            is_seq2seq = cfg.is_encoder_decoder
        except Exception:
            pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    dtype = torch.float16 if device == "cuda" else torch.float32

    # Check for weights/adapters
    has_adapter = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))

    if is_seq2seq:
        if has_adapter:
            logger.info(f"Merging SFT LoRA adapter into base Seq2Seq model ({base_model_name})...")
            base_m = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
            peft_m = PeftModel.from_pretrained(base_m, model_name_or_path)
            model = peft_m.merge_and_unload()
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    else:
        if has_adapter:
            logger.info(f"Merging SFT LoRA adapter into base CausalLM model ({base_model_name})...")
            base_m = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
            peft_m = PeftModel.from_pretrained(base_m, model_name_or_path)
            model = peft_m.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    if hasattr(tokenizer, "src_lang") or hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"
        logger.info(f"Configured multilingual tokenizer with src_lang='de_DE' and tgt_lang='de_DE'")

    model = model.to(device)
    model.eval()
    return model, tokenizer, is_seq2seq


# ---------------------------------------------------------------------------
# Candidate Generation Helper
# ---------------------------------------------------------------------------
def sample_candidates_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    is_seq2seq: bool,
    num_candidates: int,
    temperature: float,
    top_p: float = 0.92,
    top_k: int = 50,
    max_source_len: int = 500,
    max_target_len: int = 500,
    device: str = "cuda",
) -> List[List[str]]:
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_source_len,
        return_tensors="pt",
    ).to(device)

    forced_bos_token_id = None
    if is_seq2seq and hasattr(tokenizer, "lang_code_to_id") and "de_DE" in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]

    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": 1.2,
        "num_return_sequences": num_candidates,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if is_seq2seq:
        gen_kwargs["max_length"] = max_target_len
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
    else:
        gen_kwargs["max_new_tokens"] = max_target_len

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    if not is_seq2seq:
        input_len = inputs["input_ids"].shape[1]
        outputs = outputs[:, input_len:]

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    result = []
    for i in range(len(prompts)):
        cands = decoded[i * num_candidates : (i + 1) * num_candidates]
        result.append([c.strip() for c in cands])
    return result


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs using Progressive Temperature Ladder (500 Tokens)."
    )

    # Data
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="data/temperature_ladder_500/corpus_10kgnad_len500_as.json",
        help="Path to input corpus JSON/CSV with as_text.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max corpus records to process.",
    )

    # SFT Model
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="results/models/token_length_exp/sft_len500",
        help="Path to SFT model directory or checkpoint.",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="facebook/mbart-large-50",
        help="Base model architecture (default: facebook/mbart-large-50).",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="",
        help="Optional prompt prefix.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for candidate generation.",
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=500,
        help="Max source sequence length (default: 500).",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=500,
        help="Max target sequence length (default: 500).",
    )

    # Temperature Ladder & Sampling
    parser.add_argument(
        "--temperature_ladder",
        type=float,
        nargs="+",
        default=[0.7, 0.8, 0.9, 1.0],
        help="Progressive temperature ladder steps (default: 0.7 0.8 0.9 1.0).",
    )
    parser.add_argument(
        "--candidates_per_step",
        type=int,
        default=3,
        help="Number of candidates sampled at each temperature step (default: 3).",
    )
    parser.add_argument(
        "--max_total_candidates",
        type=int,
        default=12,
        help="Maximum cumulative candidate pool per item (default: 12).",
    )
    parser.add_argument(
        "--min_score_margin",
        type=float,
        default=0.05,
        help="Minimum required margin (chosen_score - rejected_score >= min_margin, default: 0.05).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.92,
        help="Top-p sampling parameter (default: 0.92).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50).",
    )

    # Reward Model
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="results/models/token_length_exp/bilstm_mixup_regression_500.pt",
        help="Path to 500-token BiLSTM weights (.pt).",
    )
    parser.add_argument(
        "--reward_vocab_path",
        type=str,
        default="data/token_length_exp/mixup_vocab_500.json",
        help="Path to 500-token BiLSTM vocabulary JSON.",
    )
    parser.add_argument(
        "--reward_max_seq_len",
        type=int,
        default=500,
        help="Max sequence length for Reward Model (default: 500).",
    )
    parser.add_argument(
        "--sbert_model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="SBERT model for semantic evaluation.",
    )
    parser.add_argument(
        "--w_style",
        type=float,
        default=0.5,
        help="Weight for style score in composite reward (default: 0.5).",
    )
    parser.add_argument(
        "--w_sem",
        type=float,
        default=0.5,
        help="Weight for semantic similarity in composite reward (default: 0.5).",
    )

    # Output
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/temperature_ladder_500/dpo_pairs_w05_w05.jsonl",
        help="Output path for DPO dataset.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Active compute device: {device}")
    logger.info(f"Configured Temperature Ladder: {args.temperature_ladder}")
    logger.info(f"Target Score Margin: >= {args.min_score_margin}")

    # 1. Load Corpus
    corpus_records = load_corpus_data(corpus_path=args.corpus_path, max_samples=args.max_samples)
    if len(corpus_records) == 0:
        logger.error("No valid corpus records found.")
        return

    # 2. Load SFT Model
    sft_model, sft_tokenizer, is_seq2seq = load_sft_model_and_tokenizer(
        model_name_or_path=args.sft_model_path,
        base_model_name=args.base_model_name,
        device=device,
    )

    # 3. Load Reward Evaluator
    reward_evaluator = CompositeRewardEvaluator(
        reward_model_path=args.reward_model_path,
        reward_vocab_path=args.reward_vocab_path,
        sbert_model_name=args.sbert_model_name,
        w_style=args.w_style,
        w_sem=args.w_sem,
        max_seq_len=args.reward_max_seq_len,
        device=device,
    )

    # 4. Process Batches through Temperature Ladder
    logger.info("Starting Progressive Temperature Ladder generation...")
    dpo_pairs: List[Dict[str, Any]] = []
    dropped_low_margin = 0
    dropped_identical = 0
    temp_resolution_counts = {t: 0 for t in args.temperature_ladder}

    batch_size = args.batch_size
    num_batches = (len(corpus_records) + batch_size - 1) // batch_size

    for b_idx in tqdm(range(num_batches), desc="Processing Batches"):
        batch_items = corpus_records[b_idx * batch_size : (b_idx + 1) * batch_size]
        as_texts = [item["as_text"] for item in batch_items]
        prompts = [args.prompt_prefix + t for t in as_texts]

        # Tracking state per item in this batch
        batch_state = []
        for item, as_text, prompt in zip(batch_items, as_texts, prompts):
            batch_state.append({
                "item": item,
                "as_text": as_text,
                "prompt": prompt,
                "candidate_pool": [],
                "reward_cache": {},  # cand_str -> (total_r, style_r, sem_r)
                "resolved": False,
                "resolved_temp": None,
            })

        # Iterate through temperature ladder
        for temp in args.temperature_ladder:
            active_indices = [
                i for i, s in enumerate(batch_state)
                if not s["resolved"] and len(s["candidate_pool"]) < args.max_total_candidates
            ]

            if len(active_indices) == 0:
                break  # All items in batch already resolved!

            active_prompts = [batch_state[i]["prompt"] for i in active_indices]

            # Sample candidates for active items at current temperature
            sampled = sample_candidates_batch(
                model=sft_model,
                tokenizer=sft_tokenizer,
                prompts=active_prompts,
                is_seq2seq=is_seq2seq,
                num_candidates=args.candidates_per_step,
                temperature=temp,
                top_p=args.top_p,
                top_k=args.top_k,
                max_source_len=args.max_source_len,
                max_target_len=args.max_target_len,
                device=device,
            )

            # Update candidate pools and evaluate new candidates
            for idx_in_active, global_idx in enumerate(active_indices):
                state = batch_state[global_idx]
                new_cands = sampled[idx_in_active]

                # Hygiene filtering: no empty, no extreme fragments, no exact echo of input
                clean_new = []
                for c in new_cands:
                    c_clean = c.strip()
                    if len(c_clean) >= 10 and c_clean != state["as_text"] and c_clean not in state["candidate_pool"]:
                        clean_new.append(c_clean)

                if len(clean_new) > 0:
                    # Compute rewards for newly generated candidates
                    source_rep = [state["as_text"]] * len(clean_new)
                    tot_r, style_r, sem_r = reward_evaluator.compute_rewards(source_rep, clean_new)
                    for c_str, tr, sr, semr in zip(clean_new, tot_r, style_r, sem_r):
                        state["reward_cache"][c_str] = (float(tr), float(sr), float(semr))
                        state["candidate_pool"].append(c_str)

                # Check if pool satisfies margin
                if len(state["candidate_pool"]) >= 2:
                    pool = state["candidate_pool"]
                    scores = [state["reward_cache"][c][0] for c in pool]
                    max_sc = max(scores)
                    min_sc = min(scores)
                    margin = max_sc - min_sc

                    if margin >= args.min_score_margin:
                        state["resolved"] = True
                        state["resolved_temp"] = temp

        # Extract preference pairs from batch state
        for state in batch_state:
            pool = state["candidate_pool"]
            if len(pool) < 2:
                dropped_identical += 1
                continue

            # Rank all pooled candidates by total reward
            ranked_cands = sorted(pool, key=lambda c: state["reward_cache"][c][0], reverse=True)
            chosen_cand = ranked_cands[0]
            rejected_cand = ranked_cands[-1]

            chosen_score, chosen_style, chosen_sem = state["reward_cache"][chosen_cand]
            rejected_score, rejected_style, rejected_sem = state["reward_cache"][rejected_cand]
            margin = chosen_score - rejected_score

            if chosen_cand == rejected_cand:
                dropped_identical += 1
                continue

            if margin < args.min_score_margin:
                dropped_low_margin += 1
                continue

            # Success
            res_temp = state["resolved_temp"] or args.temperature_ladder[-1]
            temp_resolution_counts[res_temp] = temp_resolution_counts.get(res_temp, 0) + 1

            dpo_pairs.append({
                "prompt": state["prompt"],
                "as_text": state["as_text"],
                "chosen": chosen_cand,
                "rejected": rejected_cand,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": margin,
                "chosen_style": chosen_style,
                "chosen_sem": chosen_sem,
                "rejected_style": rejected_style,
                "rejected_sem": rejected_sem,
                "resolved_temp": res_temp,
                "pool_size": len(pool),
                "source": state["item"].get("source", "10kgnad"),
            })

    # Summary Stats
    total_processed = len(corpus_records)
    logger.info(f"=== Temperature Ladder Generation Summary ===")
    logger.info(f"Total Input Prompts: {total_processed}")
    logger.info(f"Valid DPO Pairs Generated: {len(dpo_pairs)} ({len(dpo_pairs)/max(1, total_processed)*100:.1f}% retention)")
    logger.info(f"Dropped (Margin < {args.min_score_margin}): {dropped_low_margin}")
    logger.info(f"Dropped (Identical/Insufficient): {dropped_identical}")
    logger.info(f"Resolution Breakdown by Temperature:")
    for t in args.temperature_ladder:
        count = temp_resolution_counts.get(t, 0)
        logger.info(f"  - Resolved at T={t}: {count} pairs ({count/max(1, len(dpo_pairs))*100:.1f}%)")

    if len(dpo_pairs) == 0:
        logger.error("No valid pairs generated! Consider adjusting --min_score_margin.")
        return

    margins = [p["score_margin"] for p in dpo_pairs]
    chosen_scs = [p["chosen_score"] for p in dpo_pairs]
    rejected_scs = [p["rejected_score"] for p in dpo_pairs]
    logger.info(
        f"Score Summary -> Chosen Avg: {np.mean(chosen_scs):.4f} | "
        f"Rejected Avg: {np.mean(rejected_scs):.4f} | Avg Margin: {np.mean(margins):.4f}"
    )

    # 5. Save Output Splits
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    random.shuffle(dpo_pairs)

    if args.val_split_ratio > 0 and len(dpo_pairs) > 10:
        split_idx = int((1.0 - args.val_split_ratio) * len(dpo_pairs))
        train_pairs = dpo_pairs[:split_idx]
        val_pairs = dpo_pairs[split_idx:]

        base_name, ext = os.path.splitext(args.output_file)
        eval_output_file = f"{base_name}_eval{ext}"

        _save_pairs(train_pairs, args.output_file)
        logger.info(f"Saved {len(train_pairs)} training pairs to: {args.output_file}")

        _save_pairs(val_pairs, eval_output_file)
        logger.info(f"Saved {len(val_pairs)} validation pairs to: {eval_output_file}")
    else:
        _save_pairs(dpo_pairs, args.output_file)
        logger.info(f"Saved all {len(dpo_pairs)} pairs to: {args.output_file}")

    logger.info("=== Temperature Ladder DPO Generation Complete ===")


def _save_pairs(pairs: List[Dict[str, Any]], filepath: str):
    if filepath.endswith(".jsonl"):
        with open(filepath, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
