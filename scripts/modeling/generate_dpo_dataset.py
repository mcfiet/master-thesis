#!/usr/bin/env python3
"""
=============================================================================
DPO Preference Dataset Generator
=============================================================================
This script creates a preference dataset (prompt, chosen, rejected) for DPO
training by:
  1. Loading and filtering `corpus_master.json` based on metadata (e.g.
     semantic similarity, token counts, sources).
  2. Generating multiple candidate simplifications using the fine-tuned SFT
     model (from `5_train_sft.py`). Supports checkpoints (.pt/.pth/directory).
  3. Evaluating each candidate using:
     - The BiLSTM Simplicity / Style Regressor (from `3_regression_train_mixup.py`
       or `4_regression_train_synthetic.py`).
     - Semantic similarity via Sentence-BERT (meaning preservation).
     - Composite Reward = w_style * R_style + w_sem * R_sem_norm.
  4. Ranking candidates into winner (chosen) and loser (rejected) pairs.
  5. Filtering by minimum score margin and exporting ready-to-train DPO datasets.
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

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("GenerateDPODataset")


# ---------------------------------------------------------------------------
# BiLSTM Regressor Definition (compatible with 3_regression_train_mixup & 4)
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
        max_seq_len: int = 150,
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

        # 3. Load SpaCy for tokenizer
        try:
            self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
        except Exception:
            self.nlp = spacy.blank("de")
            self.nlp.add_pipe("sentencizer")

        # 4. Load SBERT Model
        logger.info(f"Loading SBERT model for semantic evaluation: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name, trust_remote_code=True, device=self.device)
        if "jina" in sbert_model_name.lower():
            # Align Jina context window directly with reward_max_seq_len to prevent OOM
            target_seq_len = max(self.max_seq_len, 256)
            self.sbert_model.max_seq_length = min(target_seq_len, 1024)
            logger.info(f"Set Jina SBERT max_seq_length to {self.sbert_model.max_seq_length} (aligned with max_target_len)")

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
        """
        Computes composite reward, style simplicity score, and normalized semantic similarity.
        """
        r_style = self.predict_simplicity_scores(candidate_texts)
        r_sem = self.predict_semantic_similarity(source_texts, candidate_texts)
        # Normalize semantic cosine similarity from [-1, 1] to [0, 1]
        r_sem_norm = np.clip((r_sem + 1.0) / 2.0, 0.0, 1.0)
        total_rewards = self.w_style * r_style + self.w_sem * r_sem_norm
        return total_rewards, r_style, r_sem_norm


# ---------------------------------------------------------------------------
# Corpus Loading and Filtering
# ---------------------------------------------------------------------------
def load_and_filter_corpus(
    corpus_path: str,
    min_sim: Optional[float] = None,
    max_sim: Optional[float] = None,
    min_as_tokens: Optional[int] = None,
    max_as_tokens: Optional[int] = None,
    sources: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Loads corpus_master.json / .csv and filters records based on metadata.
    """
    logger.info(f"Loading corpus from: {corpus_path}")

    if corpus_path.endswith(".json"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif corpus_path.endswith(".csv"):
        df = pd.read_csv(corpus_path)
        raw_data = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported corpus file type: {corpus_path}. Must be .json or .csv.")

    logger.info(f"Total raw records in corpus: {len(raw_data)}")

    filtered_records = []
    for item in raw_data:
        as_text = str(item.get("as_text") or "").strip()
        ls_text = str(item.get("ls_text") or "").strip()

        if not as_text or not ls_text:
            continue

        # 1. Similarity filter
        sim = item.get("semantic_similarity_8192")
        if sim is not None:
            try:
                sim_val = float(sim)
                if min_sim is not None and sim_val < min_sim:
                    continue
                if max_sim is not None and sim_val > max_sim:
                    continue
            except (ValueError, TypeError):
                pass

        # 2. Token length filters
        as_tokens = item.get("as_tokens")
        if as_tokens is not None:
            try:
                tokens_val = int(as_tokens)
                if min_as_tokens is not None and tokens_val < min_as_tokens:
                    continue
                if max_as_tokens is not None and tokens_val > max_as_tokens:
                    continue
            except (ValueError, TypeError):
                pass

        # 3. Source filter
        if sources is not None and len(sources) > 0:
            item_source = item.get("source")
            if item_source not in sources:
                continue

        filtered_records.append(item)

    logger.info(f"Records remaining after metadata filtering: {len(filtered_records)}")

    if max_samples is not None and len(filtered_records) > max_samples:
        logger.info(f"Subsampling to {max_samples} records.")
        filtered_records = filtered_records[:max_samples]

    return filtered_records


# ---------------------------------------------------------------------------
# SFT Model Loading & Generation
# ---------------------------------------------------------------------------
def load_sft_model_and_tokenizer(
    model_name_or_path: str,
    base_model_name: str = "facebook/mbart-large-50",
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    is_seq2seq: Optional[bool] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, bool]:
    """
    Loads SFT model and tokenizer from directory (or Hugging Face Hub model ID).
    """
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"

    # Enforce directory path (no raw .pt files)
    if os.path.isfile(model_name_or_path) or model_name_or_path.endswith((".pt", ".pth", ".bin")):
        raise ValueError(
            f"Ungültiger Pfad '{model_name_or_path}': Es muss ein Modell-Ordnerpfad übergeben werden "
            f"(z.B. 'results/models/new_pipeline/sft' oder ein HuggingFace Hub Modell-Name), keine .pt/.pth Datei."
        )

    logger.info(f"Loading Hugging Face SFT Model and Tokenizer from directory: {model_name_or_path}")

    # Robustly identify whether the architecture is Encoder-Decoder (Seq2Seq) or Decoder-Only (Causal LM)
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        detected_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    except Exception:
        name_lower = str(model_name_or_path).lower()
        detected_seq2seq = any(k in name_lower for k in ["mbart", "bart", "t5", "marian", "pegasus"])

    if is_seq2seq is not None:
        detected_seq2seq = is_seq2seq

    logger.info(f"Model architecture mode: {'Seq2Seq (Encoder-Decoder)' if detected_seq2seq else 'Causal LM (Decoder-Only)'}")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    except Exception:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        except Exception:
            logger.warning(f"Could not load tokenizer directly from {model_name_or_path}. Falling back to base_model_name: {base_model_name}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    # Load Model
    if detected_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    check_name = str(model_name_or_path).lower()
    if "mbart" in check_name or "facebook/mbart" in str(getattr(model.config, "_name_or_path", "")).lower():
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

    model = model.to(device)
    model.eval()
    return model, tokenizer, detected_seq2seq


def generate_candidates_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    is_seq2seq: bool,
    num_candidates: int = 3,
    max_source_len: int = 512,
    max_target_len: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.92,
    top_k: int = 50,
    device: str = "cuda",
) -> List[List[str]]:
    """
    Generates `num_candidates` sample simplifications for each prompt.
    """
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
        # Cut off prompt from output for Causal LM
        input_len = inputs["input_ids"].shape[1]
        outputs = outputs[:, input_len:]

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    result = []
    for i in range(len(prompts)):
        cands = decoded[i * num_candidates : (i + 1) * num_candidates]
        # Clean candidates
        cands = [c.strip() for c in cands]
        result.append(cands)
    return result


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO preference pairs (chosen/rejected) using SFT and Reward Models."
    )

    # --- Corpus & Filtering Arguments ---
    corpus_group = parser.add_argument_group("Corpus & Filtering")
    corpus_group.add_argument(
        "--corpus_path",
        type=str,
        default="data/new_pipeline/analysis/corpus_master.json",
        help="Path to corpus_master.json or .csv.",
    )
    corpus_group.add_argument(
        "--min_sim",
        type=float,
        default=0.75,
        help="Minimum semantic_similarity_8192 to include (default: 0.75).",
    )
    corpus_group.add_argument(
        "--max_sim",
        type=float,
        default=1.0,
        help="Maximum semantic_similarity_8192 to include (default: 1.0).",
    )
    corpus_group.add_argument(
        "--min_as_tokens",
        type=int,
        default=None,
        help="Minimum token count for as_text.",
    )
    corpus_group.add_argument(
        "--max_as_tokens",
        type=int,
        default=None,
        help="Maximum token count for as_text.",
    )
    corpus_group.add_argument(
        "--sources",
        type=str,
        nargs="+",
        default=None,
        help="Filter by specific sources (e.g., 'behindertenbeauftragter', 'bmas', 'mdr').",
    )
    corpus_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of corpus samples to process (for debugging or partial runs).",
    )

    # --- SFT Model Arguments ---
    sft_group = parser.add_argument_group("SFT Model")
    sft_group.add_argument(
        "--sft_model_path",
        type=str,
        required=True,
        help="Path to SFT model weights (.pt/.pth) or HF model directory.",
    )
    sft_group.add_argument(
        "--model_name",
        "--base_model_name",
        dest="base_model_name",
        type=str,
        default="facebook/mbart-large-50",
        help="Base pretrained model name/architecture if sft_model_path is a .pt weights file (default: 'facebook/mbart-large-50').",
    )
    sft_group.add_argument(
        "--prompt_prefix",
        type=str,
        default="",
        help="Optional prompt instruction prefix prepended to as_text (default: '').",
    )
    sft_group.add_argument(
        "--num_candidates",
        type=int,
        default=3,
        help="Number of candidate simplifications to generate per text (default: 3).",
    )
    sft_group.add_argument(
        "--include_ground_truth",
        action="store_true",
        default=True,
        help="Include the ground-truth ls_text from the corpus as an additional candidate for evaluation.",
    )
    sft_group.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for SFT candidate generation (default: 4).",
    )
    sft_group.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (default: 0.8).",
    )
    sft_group.add_argument(
        "--top_p",
        type=float,
        default=0.92,
        help="Top-p nucleus sampling parameter (default: 0.92).",
    )
    sft_group.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50).",
    )
    sft_group.add_argument(
        "--max_source_len",
        type=int,
        default=512,
        help="Max source token sequence length (default: 512).",
    )
    sft_group.add_argument(
        "--max_target_len",
        type=int,
        default=512,
        help="Max target token sequence length (default: 512).",
    )

    # --- Reward Model Arguments ---
    reward_group = parser.add_argument_group("Reward Model & Evaluation")
    reward_group.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="Path to trained BiLSTM Regressor weights (.pt).",
    )
    reward_group.add_argument(
        "--reward_vocab_path",
        type=str,
        required=True,
        help="Path to vocabulary JSON for BiLSTM Regressor.",
    )
    reward_group.add_argument(
        "--sbert_model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="SentenceTransformer model for semantic preservation scoring.",
    )
    reward_group.add_argument(
        "--w_style",
        type=float,
        default=0.5,
        help="Weight for simplicity/style score in composite reward (default: 0.5).",
    )
    reward_group.add_argument(
        "--w_sem",
        type=float,
        default=0.5,
        help="Weight for semantic similarity in composite reward (default: 0.5).",
    )
    reward_group.add_argument(
        "--reward_max_seq_len",
        type=int,
        default=None,
        help="Max sequence token length for BiLSTM simplicity reward scoring (defaults to max_target_len).",
    )
    reward_group.add_argument(
        "--min_score_margin",
        type=float,
        default=0.05,
        help="Minimum reward score margin (chosen_score - rejected_score) to keep a pair (default: 0.05).",
    )

    # --- Output Arguments ---
    out_group = parser.add_argument_group("Output & General")
    out_group.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output file path for generated DPO dataset (.jsonl or .json).",
    )
    out_group.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of generated pairs to save into a separate validation set (default: 0.1).",
    )
    out_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
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

    # 1. Load and filter corpus
    corpus_records = load_and_filter_corpus(
        corpus_path=args.corpus_path,
        min_sim=args.min_sim,
        max_sim=args.max_sim,
        min_as_tokens=args.min_as_tokens,
        max_as_tokens=args.max_as_tokens,
        sources=args.sources,
        max_samples=args.max_samples,
    )

    if len(corpus_records) == 0:
        logger.error("No records remaining after filtering! Check your filtering thresholds.")
        return

    # 2. Load SFT Model
    sft_model, sft_tokenizer, is_seq2seq = load_sft_model_and_tokenizer(
        model_name_or_path=args.sft_model_path,
        base_model_name=args.base_model_name,
        device=device,
    )

    # 3. Load Reward Evaluator
    reward_max_len = args.reward_max_seq_len if args.reward_max_seq_len is not None else args.max_target_len
    reward_evaluator = CompositeRewardEvaluator(
        reward_model_path=args.reward_model_path,
        reward_vocab_path=args.reward_vocab_path,
        sbert_model_name=args.sbert_model_name,
        w_style=args.w_style,
        w_sem=args.w_sem,
        max_seq_len=reward_max_len,
        device=device,
    )

    # 4. Generate & Score Candidates in Batches
    logger.info("Starting SFT candidate generation and reward scoring...")
    dpo_pairs: List[Dict[str, Any]] = []
    dropped_low_margin = 0
    dropped_identical = 0

    batch_size = args.batch_size
    num_batches = (len(corpus_records) + batch_size - 1) // batch_size

    for b_idx in tqdm(range(num_batches), desc="Processing Corpus"):
        batch_items = corpus_records[b_idx * batch_size : (b_idx + 1) * batch_size]
        as_texts = [str(item.get("as_text") or "").strip() for item in batch_items]
        prompts = [args.prompt_prefix + text for text in as_texts]

        # Generate candidates from SFT model
        generated_candidates = generate_candidates_batch(
            model=sft_model,
            tokenizer=sft_tokenizer,
            prompts=prompts,
            is_seq2seq=is_seq2seq,
            num_candidates=args.num_candidates,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )

        for item, as_text, prompt, cands in zip(batch_items, as_texts, prompts, generated_candidates):
            all_cands = list(set([c for c in cands if len(c) > 0]))

            # Optionally include ground-truth LS text
            if args.include_ground_truth:
                gt_ls = str(item.get("ls_text") or "").strip()
                if gt_ls and gt_ls not in all_cands:
                    all_cands.append(gt_ls)

            if len(all_cands) < 2:
                dropped_identical += 1
                continue

            # Compute rewards for all candidates against original as_text
            source_rep = [as_text] * len(all_cands)
            tot_rewards, style_scores, sem_scores = reward_evaluator.compute_rewards(source_rep, all_cands)

            # Rank candidates by composite reward (descending)
            ranked_indices = np.argsort(-tot_rewards)
            best_idx = ranked_indices[0]
            worst_idx = ranked_indices[-1]

            chosen_text = all_cands[best_idx]
            rejected_text = all_cands[worst_idx]

            chosen_score = float(tot_rewards[best_idx])
            rejected_score = float(tot_rewards[worst_idx])
            margin = chosen_score - rejected_score

            if chosen_text == rejected_text:
                dropped_identical += 1
                continue

            if margin < args.min_score_margin:
                dropped_low_margin += 1
                continue

            pair = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "score_margin": margin,
                "chosen_style": float(style_scores[best_idx]),
                "chosen_sem": float(sem_scores[best_idx]),
                "rejected_style": float(style_scores[worst_idx]),
                "rejected_sem": float(sem_scores[worst_idx]),
                "source": item.get("source"),
                "as_sim": item.get("semantic_similarity_8192"),
            }
            dpo_pairs.append(pair)

    logger.info(f"Generated {len(dpo_pairs)} valid DPO preference pairs.")
    logger.info(f"Dropped due to low margin (< {args.min_score_margin}): {dropped_low_margin}")
    logger.info(f"Dropped due to identical/insufficient candidates: {dropped_identical}")

    if len(dpo_pairs) == 0:
        logger.error("No valid DPO pairs generated! Lower `--min_score_margin` or check model generation.")
        return

    # Statistics
    margins = [p["score_margin"] for p in dpo_pairs]
    chosen_scs = [p["chosen_score"] for p in dpo_pairs]
    rejected_scs = [p["rejected_score"] for p in dpo_pairs]
    logger.info(f"Score Summary -> Chosen Avg: {np.mean(chosen_scs):.4f} | Rejected Avg: {np.mean(rejected_scs):.4f} | Avg Margin: {np.mean(margins):.4f}")

    # 5. Save Output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    random.shuffle(dpo_pairs)

    if args.val_split_ratio > 0 and len(dpo_pairs) > 10:
        split_idx = int((1.0 - args.val_split_ratio) * len(dpo_pairs))
        train_pairs = dpo_pairs[:split_idx]
        val_pairs = dpo_pairs[split_idx:]

        # Determine train and eval output paths
        base_name, ext = os.path.splitext(args.output_file)
        eval_output_file = f"{base_name}_eval{ext}"

        # Save Train
        _save_pairs(train_pairs, args.output_file)
        logger.info(f"Saved {len(train_pairs)} training pairs to: {args.output_file}")

        # Save Eval
        _save_pairs(val_pairs, eval_output_file)
        logger.info(f"Saved {len(val_pairs)} validation pairs to: {eval_output_file}")
    else:
        _save_pairs(dpo_pairs, args.output_file)
        logger.info(f"Saved all {len(dpo_pairs)} pairs to: {args.output_file}")

    logger.info("=== DPO Preference Dataset Generation Completed Successfully ===")


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
