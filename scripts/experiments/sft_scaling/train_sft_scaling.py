#!/usr/bin/env python3
"""
SFT Data Scaling & Empirical Learning Curve Experiment
Masterarbeit: Automatische Übersetzung von Alltagssprache (AS) in Leichte Sprache (LS)

Untersucht das Skalierungsverhalten des Supervised Fine-Tuning (SFT) Modells
(facebook/mbart-large-50 + LoRA) entlang der Datenmenge:
Fractions F in {0.10, 0.25, 0.50, 0.75, 1.00} der verfügbaren Trainingsartikel.
"""

import os
import sys
import time
import json
import shutil
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


# ==============================================================================
# SEED CONFIGURATION
# ==============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# REWARD MODEL & METRICS (BILSTM REGRESSOR & SBERT)
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


# ==============================================================================
# TRANSLATION DATASET
# ==============================================================================
class TranslationDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_src_len: int = 256, max_tgt_len: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = item["as_text"]
        tgt_text = item["ls_text"]

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


# ==============================================================================
# MAIN TRAINING & EVALUATION PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SFT Data Scaling on mBART-50.")
    parser.add_argument("--corpus_path", type=str, default="data/analysis/corpus_master.csv")
    parser.add_argument("--test_file", type=str, default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--output_dir", type=str, default="results/experiments/sft_scaling")
    parser.add_argument("--base_model_name", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--reward_model_path", type=str, default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--reward_vocab_path", type=str, default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", type=str, default="jinaai/jina-embeddings-v2-base-de")
    
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training corpus to use (e.g. 0.10, 0.25, 0.50, 0.75, 1.00)")
    parser.add_argument("--experiment_name", type=str, default=None)
    
    parser.add_argument("--min_sim", type=float, default=0.70)
    parser.add_argument("--max_sim", type=float, default=0.98)
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    if args.experiment_name is None:
        frac_str = str(args.train_fraction).replace(".", "")
        args.experiment_name = f"sft_scale_f{frac_str}"
        
    model_save_dir = os.path.join(args.output_dir, args.experiment_name)
    temp_adapter_dir = os.path.join(model_save_dir, "temp_adapter")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(temp_adapter_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"\n{'='*75}")
    print(f"SFT DATA SCALING EXPERIMENT: {args.experiment_name}")
    print(f"Train-Fraction: {args.train_fraction:.2f} | Device: {device} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"Zielverzeichnis: {model_save_dir}")
    print(f"{'='*75}\n")
    
    # 1. Daten laden und deterministisch splitten (80/10/10)
    print(f"Lade Datensatz aus {args.corpus_path}...")
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
        
    print(f"Gefilterte Gesamtartikelpaare: {len(pairs)}")
    
    # Deterministischer Shuffle für strikte Reproduzierbarkeit
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    
    n_total = len(pairs)
    n_val = int(0.10 * n_total)
    n_test = int(0.10 * n_total)
    n_train_total = n_total - n_val - n_test
    
    train_full = pairs[:n_train_total]
    val_data = pairs[n_train_total : n_train_total + n_val]
    heldout_test_data = pairs[n_train_total + n_val :]
    
    # Subsampling der Trainingsdaten entsprechend train_fraction
    n_train_sub = max(1, int(round(args.train_fraction * len(train_full))))
    train_data = train_full[:n_train_sub]
    
    print(f"Split-Übersicht:")
    print(f"  -> Trainingsdaten ({args.train_fraction*100:.0f}%): {len(train_data)} Artikelpaare (aus {len(train_full)} Pool)")
    print(f"  -> Validierungsdaten (fest 10%):  {len(val_data)} Artikelpaare")
    print(f"  -> Held-Out Testset (fest 10%):   {len(heldout_test_data)} Artikelpaare")
    
    # 2. Tokenizer & Modell initialisieren
    print(f"\nInitialisiere Tokenizer ({args.base_model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=False)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Lade Basismodell {args.base_model_name}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=dtype).to(device)
    
    print(f"Konfiguriere LoRA Adapter (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
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
    
    print("\n--- Starte SFT Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoche {epoch+1}/{args.epochs} [Train]")
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
                
            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})
            
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
        
        print(f"-> Epoche {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Speichere temporären LoRA Adapter
            model.save_pretrained(temp_adapter_dir)
            tokenizer.save_pretrained(temp_adapter_dir)
            print(f"   [+] Bester Val Loss verbessert auf {best_val_loss:.4f}. Adapter-Checkpoint gespeichert.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"   [!] Early Stopping getriggert nach {epoch+1} Epochen.")
                break
                
    train_duration = time.time() - start_time
    print(f"\nTraining abgeschlossen in {train_duration:.2f} Sekunden.")
    
    # 6. Adapter sauber verschmelzen (merge_and_unload) & Bereinigen
    print("\nLade besten LoRA-Adapter und verschmelze fest mit Basismodell (merge_and_unload)...")
    del model, base_model
    torch.cuda.empty_cache()
    
    base_m = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=dtype).to(device)
    peft_m = PeftModel.from_pretrained(base_m, temp_adapter_dir)
    merged_model = peft_m.merge_and_unload()
    
    # Speichere das vollständig fusionierte Standalone-Modell
    merged_model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)
    torch.save(merged_model.state_dict(), os.path.join(model_save_dir, "sft.pt"))
    
    # Save training history JSON
    try:
        hist_path = os.path.join(model_save_dir, "training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass
    
    # Bereinige temporäres Adapter-Verzeichnis und entferne jegliche adapter_config.json Reste
    if os.path.exists(temp_adapter_dir):
        shutil.rmtree(temp_adapter_dir)
    for leftover in ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]:
        leftover_path = os.path.join(model_save_dir, leftover)
        if os.path.exists(leftover_path):
            os.remove(leftover_path)
            
    print(f"[ERFOLG] Vollstaendig fusioniertes Standalone-Modell (ohne Adapter-Reste) unter {model_save_dir} gespeichert!")
    
    # 7. Vollständige Quantitative Evaluation auf dem Lebenshilfe-Testset mit forced_bos_token_id
    print("\n" + "="*75)
    print("STARTE QUANTITATIVE EVALUATION (Lebenshilfe-Benchmark)")
    print("="*75)
    
    with open(args.test_file, "r", encoding="utf-8") as f:
        lh_data = json.load(f)
    as_texts = [item.get("source_text", item.get("as_text", item.get("source", ""))) for item in lh_data]
    ls_ref_texts = [item.get("target_text", item.get("ls_text", item.get("target", ""))) for item in lh_data]
    
    # Inferenz mit de_DE Sprach-Token Verankerung
    merged_model.eval()
    gen_texts = []
    infer_batch_size = 8
    
    gen_kwargs = {
        "max_length": args.max_target_len,
        "num_beams": 4,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }
    if hasattr(tokenizer, "lang_code_to_id") and "de_DE" in tokenizer.lang_code_to_id:
        gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["de_DE"]
    elif hasattr(merged_model.config, "forced_bos_token_id") and merged_model.config.forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = merged_model.config.forced_bos_token_id
        
    for i in tqdm(range(0, len(as_texts), infer_batch_size), desc="Generierung"):
        b_src = as_texts[i : i + infer_batch_size]
        inp = tokenizer(b_src, max_length=args.max_source_len, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = merged_model.generate(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                **gen_kwargs
            )
        decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
        gen_texts.extend(decoded)
        
    del merged_model
    torch.cuda.empty_cache()
    
    # Metriken berechnen
    print("Lade Reward-Modelle & NLP-Pipelines...")
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    
    with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    reward_model = BiLSTMRegressor(vocab_size=len(vocab)).to(device)
    state_dict = torch.load(args.reward_model_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        reward_model.load_state_dict(state_dict["model_state_dict"])
    else:
        reward_model.load_state_dict(state_dict)
    reward_model.eval()
    
    sbert = SentenceTransformer(args.sbert_model_name, device=device, trust_remote_code=True)
    if hasattr(sbert, "max_seq_length"):
        sbert.max_seq_length = 512
    
    # Simplicity
    def predict_simplicity(texts: List[str]) -> np.ndarray:
        all_s = []
        for i in range(0, len(texts), 64):
            b_txt = texts[i : i + 64]
            b_ids = []
            for t in b_txt:
                doc = nlp(t)
                ids = [vocab.get(tok.text.lower(), 1) for tok in doc if not tok.is_space]
                ids = ids[:256] + [0] * max(0, 256 - len(ids))
                b_ids.append(ids)
            tensor = torch.tensor(b_ids, dtype=torch.long, device=device)
            with torch.no_grad():
                preds = reward_model(tensor).squeeze(-1).cpu().numpy()
            all_s.extend(preds.tolist() if isinstance(preds, np.ndarray) and preds.ndim > 0 else [float(preds)])
        return np.array(all_s)
        
    def predict_sbert_sim(t1: List[str], t2: List[str]) -> np.ndarray:
        effective_len = getattr(sbert, "max_seq_length", 8192)
        if effective_len > 4096:
            sbert_bs = 2
        elif effective_len > 1024:
            sbert_bs = 4
        elif effective_len > 512:
            sbert_bs = 8
        else:
            sbert_bs = 16

        all_sim = []
        for i in range(0, len(t1), sbert_bs):
            e1 = sbert.encode(t1[i : i + sbert_bs], convert_to_tensor=True, batch_size=sbert_bs, show_progress_bar=False)
            e2 = sbert.encode(t2[i : i + sbert_bs], convert_to_tensor=True, batch_size=sbert_bs, show_progress_bar=False)
            sims = util.cos_sim(e1, e2).diag().cpu().numpy()
            all_sim.extend(sims.tolist() if isinstance(sims, np.ndarray) and sims.ndim > 0 else [float(sims)])
        return np.array(all_sim)
        
    r_style = predict_simplicity(gen_texts)
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
        
    # Zusammenfassung
    metrics_summary = {
        "experiment_name": args.experiment_name,
        "train_fraction": float(args.train_fraction),
        "num_train_pairs": len(train_data),
        "total_available_train_pairs": len(train_full),
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
    
    # Save Metrics JSON
    json_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
    # Save Details CSV
    df_det = pd.DataFrame({
        "experiment_name": args.experiment_name,
        "train_fraction": args.train_fraction,
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
    
    # Save Loss Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
        plt.title(f"SFT Lernkurve: {args.experiment_name} (F={args.train_fraction})")
        plt.xlabel("Epoche")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = os.path.join(args.output_dir, f"{args.experiment_name}_loss_curve.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  -> Plot: {plot_path}")
    except Exception as e:
        print(f"  [!] Plot-Erstellung fehlgeschlagen: {e}")

    print(f"\n[ERFOLG] Ergebnisse gespeichert:")
    print(f"  -> JSON: {json_path}")
    print(f"  -> CSV:  {det_csv}")
    print(f"\nKennzahlen für {args.experiment_name} (N={len(train_data)}):")
    print(f"  Simplicity (R_style): {metrics_summary['r_style_mean']:.4f}")
    print(f"  Semantik (R_sem_as):  {metrics_summary['r_sem_as_mean']:.4f}")
    print(f"  Treue LS (Sim_ref):   {metrics_summary['sim_ref_mean']:.4f}")
    print(f"  Composite Reward:     {metrics_summary['composite_reward_mean']:.4f}")
    print(f"  BLEU / ROUGE-L:       {metrics_summary['bleu_mean']:.4f} / {metrics_summary['rouge_l_mean']:.4f}")
    print(f"  Truncation Rate:      {metrics_summary['truncation_rate_pct']:.2f}%")
    print(f"  Ø Gen Tokens:         {metrics_summary['avg_gen_tokens']:.1f}")


if __name__ == "__main__":
    main()
