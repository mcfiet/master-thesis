#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Similarity Threshold Experiment: SFT Translation Model Training & Eval
=============================================================================
Untersucht den Einfluss des Ähnlichkeits-Schwellenwerts s_min in {0.60, 0.70, 0.80}
(bei s_max = 0.98) auf das Supervised Fine-Tuning (SFT) von mBART-50.

Nutzung der exakten Hyperparameter aus run_pipeline:
- base_model_name = "facebook/mbart-large-50" (de_DE Initialisierung)
- max_source_len = 1024, max_target_len = 1024
- batch_size = 2, accumulation_steps = 8 (effektive Batch Size 16)
- lr = 1e-4, warmup_ratio = 0.10, epochs = 30, patience = 10
- LoRA: r = 16, alpha = 32, dropout = 0.05 (Target Modules: Attention + FC)
- Beam Search Inferenz: num_beams = 4, repetition_penalty = 1.2, no_repeat_ngram_size = 3

Evaluation:
1. In-Domain Held-Out Testset (10% Split): Best Val Loss, Final Loss.
2. Out-of-Domain Lebenshilfe Benchmark (data/lebenshilfe/lebenshilfe_dataset_clean.json):
   R_style (Simplicity), R_sem (AS-Semantik), Sim_ref (LS-Treue), Composite Reward,
   BLEU, ROUGE-L, Truncation Rate (%), Gen Tokens.
=============================================================================
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("SFTSimilarityExperiment")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def compute_ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def compute_sentence_bleu(ref_toks: List[str], cand_toks: List[str], max_n: int = 4) -> float:
    if len(cand_toks) == 0 or len(ref_toks) == 0:
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


def compute_rouge_l(ref_toks: List[str], cand_toks: List[str]) -> float:
    if len(ref_toks) == 0 or len(cand_toks) == 0:
        return 0.0
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
    return float(2 * p * r / max(1e-9, (p + r)))


class TranslationDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_src_len: int = 1024, max_tgt_len: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = str(item["as_text"])
        tgt_text = str(item["ls_text"])

        inputs = self.tokenizer(
            src_text, max_length=self.max_src_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=tgt_text, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"]

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "as_text": src_text,
            "ls_text": tgt_text,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Similarity Threshold Experiment: SFT mBART-50")
    parser.add_argument('--corpus_path', default="data/analysis/corpus_master.csv", help="Path to master corpus")
    parser.add_argument('--lh_dataset_path', default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Path to Lebenshilfe dataset")
    parser.add_argument('--min_sim', type=float, default=0.70, help="Min cosine similarity threshold")
    parser.add_argument('--max_sim', type=float, default=0.98, help="Max cosine similarity threshold")
    parser.add_argument('--base_model_name', type=str, default="facebook/mbart-large-50")
    parser.add_argument('--reward_model_path', type=str, default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument('--reward_vocab_path', type=str, default="data/vocabs/mixup_vocab.json")
    parser.add_argument('--sbert_model_name', type=str, default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument('--max_source_len', type=int, default=1024)
    parser.add_argument('--max_target_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default="results/experiments/similarity_threshold")
    parser.add_argument('--experiment_name', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.experiment_name is None:
        sim_str = f"{int(round(args.min_sim * 100)):02d}"
        args.experiment_name = f"sft_sim_{sim_str}"

    model_save_dir = os.path.join(args.output_dir, args.experiment_name)
    temp_adapter_dir = os.path.join(model_save_dir, "temp_adapter")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(temp_adapter_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    logger.info(f"=== Starting SFT Similarity Threshold Experiment: {args.experiment_name} ===")
    logger.info(f"Similarity Filter: [{args.min_sim:.2f}, {args.max_sim:.2f}] | Device: {device} | Base: {args.base_model_name}")

    # 1. Daten laden und filtern
    if not os.path.exists(args.corpus_path):
        raise FileNotFoundError(f"Corpus not found: {args.corpus_path}")

    if args.corpus_path.endswith(".json"):
        with open(args.corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        pairs = []
        for r in raw_data:
            sim = r.get("semantic_similarity_8192", 1.0)
            if sim is not None and args.min_sim <= sim <= args.max_sim:
                as_t = str(r.get("as_text") or "").strip()
                ls_t = str(r.get("ls_text") or "").strip()
                if as_t and ls_t:
                    pairs.append({"as_text": as_t, "ls_text": ls_t})
    else:
        df = pd.read_csv(args.corpus_path)
        filtered = df[
            (df["semantic_similarity_8192"] >= args.min_sim) & 
            (df["semantic_similarity_8192"] <= args.max_sim)
        ].dropna(subset=["as_text", "ls_text"])
        pairs = [{"as_text": str(r["as_text"]).strip(), "ls_text": str(r["ls_text"]).strip()} for _, r in filtered.iterrows()]

    logger.info(f"Gefilterte Gesamtartikelpaare: {len(pairs)}")

    # Deterministischer Split (80% Train, 10% Val, 10% Test)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    n_total = len(pairs)
    n_val = int(0.10 * n_total)
    n_test = int(0.10 * n_total)
    n_train = n_total - n_val - n_test

    train_data = pairs[:n_train]
    val_data = pairs[n_train : n_train + n_val]
    heldout_test_data = pairs[n_train + n_val :]

    logger.info(f"Split-Übersicht: Train={len(train_data)}, Val={len(val_data)}, Held-Out Test={len(heldout_test_data)}")

    # 2. Tokenizer & Basismodell
    logger.info(f"Initialisiere Tokenizer ({args.base_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=False)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Lade Basismodell {args.base_model_name}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=dtype).to(device)

    logger.info(f"Konfiguriere LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # 3. DataLoader
    train_dataset = TranslationDataset(train_data, tokenizer, args.max_source_len, args.max_target_len)
    val_dataset = TranslationDataset(val_data, tokenizer, args.max_source_len, args.max_target_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = (len(train_loader) // args.accumulation_steps + 1) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # 5. Training Loop mit Early Stopping
    start_time = time.time()
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    logger.info("--- Starte SFT Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoche {epoch+1}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.accumulation_steps
            loss.backward()

            total_train_loss += outputs.loss.item()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        logger.info(f"Epoche {epoch+1:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model.save_pretrained(temp_adapter_dir)
            tokenizer.save_pretrained(temp_adapter_dir)
            logger.info(f"   [+] Bester Val Loss: {best_val_loss:.4f} -> Checkpoint gespeichert.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"   [!] Early Stopping nach {epoch+1} Epochen ausgelöst.")
                break

    train_duration = time.time() - start_time
    logger.info(f"Training abgeschlossen in {train_duration:.2f} Sekunden.")

    # 6. Adapter verschmelzen (merge_and_unload)
    logger.info("Verschmelze LoRA-Adapter mit dem Basismodell...")
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_m = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=dtype).to(device)
    peft_m = PeftModel.from_pretrained(base_m, temp_adapter_dir)
    merged_model = peft_m.merge_and_unload()

    merged_model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    torch.save(merged_model.state_dict(), os.path.join(model_save_dir, "sft.pt"))

    # Cleanup temp adapter
    if os.path.exists(temp_adapter_dir):
        shutil.rmtree(temp_adapter_dir)

    # 7. Out-of-Domain Evaluation auf Lebenshilfe Benchmark
    logger.info("=== Starte Out-of-Domain Evaluation (Lebenshilfe Benchmark) ===")
    with open(args.lh_dataset_path, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    as_texts = [item.get("source_text") or item.get("as_text", "") for item in lh_data]
    ls_ref_texts = [item.get("target_text") or item.get("ls_text", "") for item in lh_data]

    merged_model.eval()
    gen_texts = []
    infer_batch_size = 4

    gen_kwargs = {
        "max_length": args.max_target_len,
        "num_beams": 4,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "length_penalty": 1.0,
    }

    for i in tqdm(range(0, len(as_texts), infer_batch_size), desc="Generierung"):
        b_src = as_texts[i : i + infer_batch_size]
        inp = tokenizer(b_src, max_length=args.max_source_len, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = merged_model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                **gen_kwargs
            )
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        gen_texts.extend(decoded)

    del merged_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 8. Metriken berechnen (Reward Regressor + SBERT + Lexikalisch)
    logger.info("Berechne quantitative Bewertungsmetriken...")
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])

    # Simplicity via Reward Model
    r_style = np.zeros(len(gen_texts))
    if os.path.exists(args.reward_vocab_path) and os.path.exists(args.reward_model_path):
        with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            if "stoi" in vocab:
                vocab = vocab["stoi"]

        reward_model = BiLSTMRegressor(vocab_size=len(vocab)).to(device)
        state_dict = torch.load(args.reward_model_path, map_location=device, weights_only=False)
        if "model_state_dict" in state_dict:
            reward_model.load_state_dict(state_dict["model_state_dict"])
        else:
            reward_model.load_state_dict(state_dict)
        reward_model.eval()

        all_s = []
        with torch.no_grad():
            for t in gen_texts:
                doc = nlp(t)
                ids = [vocab.get(tok.text.lower(), 1) for tok in doc if not tok.is_space][:1024]
                if not ids:
                    ids = [0]
                tensor = torch.tensor([ids], dtype=torch.long, device=device)
                p = reward_model(tensor).squeeze().item()
                all_s.append(float(p))
        r_style = np.array(all_s)

    # SBERT Semantic Similarity
    sbert = SentenceTransformer(args.sbert_model_name, device=device, trust_remote_code=True)
    if hasattr(sbert, "max_seq_length"):
        sbert.max_seq_length = 512

    def predict_sbert_sim(t1: List[str], t2: List[str]) -> np.ndarray:
        all_sim = []
        for i in range(0, len(t1), 8):
            e1 = sbert.encode(t1[i : i + 8], convert_to_tensor=True, show_progress_bar=False)
            e2 = sbert.encode(t2[i : i + 8], convert_to_tensor=True, show_progress_bar=False)
            sims = util.cos_sim(e1, e2).diag().cpu().numpy()
            all_sim.extend(sims.tolist() if isinstance(sims, np.ndarray) and sims.ndim > 0 else [float(sims)])
        return np.array(all_sim)

    r_sem_raw = predict_sbert_sim(as_texts, gen_texts)
    r_sem_as = np.clip((r_sem_raw + 1.0) / 2.0, 0.0, 1.0)
    sim_ref_raw = predict_sbert_sim(ls_ref_texts, gen_texts)
    sim_ref = np.clip((sim_ref_raw + 1.0) / 2.0, 0.0, 1.0)
    composite_reward = 0.5 * r_style + 0.5 * r_sem_as

    bleu_list, rouge_l_list = [], []
    gen_tokens_list = []
    truncation_list = []
    valid_ends = {".", "!", "?", '."', '!"', '?"', ".'", "!'", "?'"}

    for ref_s, gen_s in zip(ls_ref_texts, gen_texts):
        t_ref = [t.text.lower() for t in nlp(ref_s) if not t.is_space]
        t_gen = [t.text.lower() for t in nlp(gen_s) if not t.is_space]
        gen_tokens_list.append(len(t_gen))
        bleu_list.append(compute_sentence_bleu(t_ref, t_gen))
        rouge_l_list.append(compute_rouge_l(t_ref, t_gen))

        trimmed = gen_s.strip()
        ends_clean = any(trimmed.endswith(end) for end in valid_ends) if len(trimmed) > 0 else False
        truncation_list.append(not ends_clean)

    # 9. Speichern der Ergebnisse
    metrics_summary = {
        "experiment_name": args.experiment_name,
        "model_type": "SFT mBART-50 LoRA",
        "min_sim": float(args.min_sim),
        "max_sim": float(args.max_sim),
        "num_total_filtered_pairs": len(pairs),
        "num_train_pairs": len(train_data),
        "num_val_pairs": len(val_data),
        "num_test_pairs": len(heldout_test_data),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history["train_loss"][-1]),
        "training_time_seconds": float(train_duration),
        "r_style_mean": float(np.mean(r_style)),
        "r_sem_as_mean": float(np.mean(r_sem_as)),
        "sim_ref_mean": float(np.mean(sim_ref)),
        "composite_reward_mean": float(np.mean(composite_reward)),
        "bleu_mean": float(np.mean(bleu_list)),
        "rouge_l_mean": float(np.mean(rouge_l_list)),
        "avg_gen_tokens": float(np.mean(gen_tokens_list)),
        "truncation_rate_pct": float(np.mean(truncation_list) * 100),
        "model_dir": model_save_dir
    }

    json_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    df_det = pd.DataFrame({
        "experiment_name": args.experiment_name,
        "min_sim": args.min_sim,
        "as_text": as_texts,
        "ls_ref_text": ls_ref_texts,
        "generated_text": gen_texts,
        "r_style": r_style,
        "r_sem_as": r_sem_as,
        "sim_ref": sim_ref,
        "composite_reward": composite_reward,
        "bleu": bleu_list,
        "rouge_l": rouge_l_list,
        "gen_tokens": gen_tokens_list,
        "is_truncated": truncation_list,
    })
    det_csv = os.path.join(args.output_dir, f"{args.experiment_name}_details.csv")
    df_det.to_csv(det_csv, index=False)

    logger.info(f"[ERFOLG] SFT Experiment {args.experiment_name} abgeschlossen!")
    logger.info(f"Simplicity: {metrics_summary['r_style_mean']:.4f} | Semantik: {metrics_summary['r_sem_as_mean']:.4f} | Treue: {metrics_summary['sim_ref_mean']:.4f} | Comp: {metrics_summary['composite_reward_mean']:.4f}")


if __name__ == "__main__":
    main()
