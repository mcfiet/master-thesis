#!/usr/bin/env python3
"""
=============================================================================
Native PyTorch DPO Training Script for Seq2Seq (Encoder-Decoder / mBART)
=============================================================================
This script performs Direct Preference Optimization (DPO) starting from a
pre-trained Supervised Fine-Tuned (SFT) Seq2Seq model (e.g. facebook/mbart-large-50).

Features:
  - 100% native PyTorch implementation tailored specifically for Encoder-Decoder architectures
  - Exact Seq2Seq token log-likelihood computation (prompt -> encoder, completion -> decoder)
  - Zero-VRAM reference model evaluation via PEFT adapter disabling (model.disable_adapter())
  - Support for pre-generated offline preference pairs (data/dpo_preference_pairs.jsonl)
  - Real-time tracking of DPO Loss, Implicit Reward Margin, and Reward Accuracy
  - Evaluation loop with Early Stopping based on validation loss
  - Model checkpoint saving (LoRA adapters + Tokenizer) directly to output directory
  - Automatic training curve plot generation (Loss, Reward Margin, Accuracy)
=============================================================================
"""

import os
import sys
import json
import datetime
import random
import argparse
import contextlib
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# LOGGING CLASS
# ==============================================================================
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
    print(f"Globaler Seed auf {seed} gesetzt.")

# ==============================================================================
# CLI ARGUMENT PARSING
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Native PyTorch DPO Training for Seq2Seq / mBART Models."
    )
    # Model & Directories
    parser.add_argument(
        "--model_name_or_path", "--sft_model_path",
        dest="model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained SFT model directory (e.g. results/models/new_pipeline/sft)."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Path to training preference pairs JSON/JSONL file (e.g. data/dpo_preference_pairs.jsonl)."
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to validation preference pairs JSON/JSONL file (e.g. data/dpo_preference_pairs_eval.jsonl)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where trained DPO model & tokenizer will be saved."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results/logs/run_pipeline",
        help="Directory where training log file will be saved."
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="results/plots/run_pipeline",
        help="Directory where loss/accuracy plots will be saved."
    )

    # Sequence Lengths
    parser.add_argument("--max_source_len", type=int, default=256, help="Max source prompt length (tokens).")
    parser.add_argument("--max_target_len", type=int, default=256, help="Max target completion length (tokens).")

    # Hyperparameters
    parser.add_argument("--loss_type", type=str, default="sum", choices=["sum", "mean"], help="DPO loss reduction: 'sum' (classic sequence-level log-prob sum, Rafailov et al. 2023) or 'mean' (length-normalized ablation).")
    parser.add_argument("--beta", type=float, default=0.01, help="DPO temperature parameter beta (default: 0.01, optimal setting identified in Beta sweep).")
    parser.add_argument("--learning_rate", "--lr", dest="lr", type=float, default=5e-6, help="Learning rate (default: 5e-6).")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2).")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps (default: 8).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3).")
    parser.add_argument("--warmup_ratio", type=float, default=0.10, help="Linear warmup ratio (default: 0.10).")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW (default: 0.01).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm clipping (default: 1.0).")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in epochs (default: 3).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # LoRA / PEFT
    parser.add_argument("--prompt_prefix", type=str, default="", help="Prompt prefix for source text (e.g. 'Vereinfache zu Leichter Sprache: ').")
    parser.add_argument("--use_peft", action="store_true", default=True, help="Use LoRA parameter-efficient training (default: True).")
    parser.add_argument("--no_peft", action="store_false", dest="use_peft", help="Disable LoRA and fine-tune all parameters.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank r (default: 16).")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling (default: 32).")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate (default: 0.05).")

    return parser.parse_args()

# ==============================================================================
# DATASET & DATALOADER FOR DPO PREFERENCE PAIRS
# ==============================================================================
class DPOPreferenceDataset(Dataset):
    """
    Dataset storing (prompt, chosen, rejected) text pairs for Seq2Seq DPO training.
    """
    def __init__(
        self,
        data_file: str,
        tokenizer,
        max_source_len: int = 256,
        max_target_len: int = 256,
        prompt_prefix: str = "",
    ):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.prompt_prefix = prompt_prefix
        self.records: List[Dict[str, Any]] = []

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Datensatz-Datei nicht gefunden: {data_file}")

        print(f"Lade DPO-Präferenzdatensatz aus: {data_file}")
        if data_file.endswith(".jsonl"):
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        elif data_file.endswith(".json"):
            with open(data_file, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    self.records = content
                elif isinstance(content, dict) and "data" in content:
                    self.records = content["data"]
                else:
                    self.records = [content]
        else:
            raise ValueError(f"Nicht unterstütztes Dateiformat: {data_file}")

        print(f"Erfolgreich geladen: {len(self.records)} Präferenz-Paare.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        prompt = self.prompt_prefix + str(item.get("prompt", "")).strip()
        chosen = str(item.get("chosen", "")).strip()
        rejected = str(item.get("rejected", "")).strip()

        # 1. Tokenize Prompt (Encoder)
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Tokenize Chosen Completion (Decoder)
        chosen_enc = self.tokenizer(
            text_target=chosen,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        chosen_labels = chosen_enc["input_ids"].clone().squeeze(0)
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100

        # 3. Tokenize Rejected Completion (Decoder)
        rejected_enc = self.tokenizer(
            text_target=rejected,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        rejected_labels = rejected_enc["input_ids"].clone().squeeze(0)
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = -100

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels,
            "rejected_labels": rejected_labels,
            "raw_prompt": prompt,
            "raw_chosen": chosen,
            "raw_rejected": rejected,
        }

# ==============================================================================
# EXACT SEQ2SEQ DPO LOG-LIKELIHOOD & LOSS COMPUTATION
# ==============================================================================
def get_seq2seq_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str = "sum"
) -> torch.Tensor:
    """
    Computes exact sequence-level log-probabilities for Seq2Seq target tokens.
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) with -100 for pad tokens
        loss_type: 'sum' (classic sequence-level log-prob sum) or 'mean' (length-normalized ablation)
    Returns:
        log_probabilities per sequence in batch: (batch_size,)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss_mask = (labels != -100)
    labels_clamped = labels.clone()
    labels_clamped[~loss_mask] = 0

    # Gather log prob of target token at each position
    per_token_logps = torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)
    
    # Sum log probabilities over non-padding tokens
    sum_logps = (per_token_logps * loss_mask).sum(dim=-1)

    if loss_type == "mean":
        token_counts = loss_mask.sum(dim=-1).clamp(min=1)
        return sum_logps / token_counts
    return sum_logps

def compute_dpo_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    beta: float = 0.1,
    loss_type: str = "sum",
    is_peft: bool = True,
    ref_model: nn.Module = None,
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Executes a forward pass for both policy and reference models,
    computing the DPO loss, implicit chosen/rejected rewards, and reward accuracy.
    """
    prompt_ids = batch["prompt_input_ids"].to(device)
    prompt_mask = batch["prompt_attention_mask"].to(device)
    chosen_labels = batch["chosen_labels"].to(device)
    rejected_labels = batch["rejected_labels"].to(device)

    # 1. Forward Pass with Policy Model π_θ
    policy_chosen_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=chosen_labels)
    policy_rejected_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=rejected_labels)

    policy_chosen_logps = get_seq2seq_logps(policy_chosen_out.logits, chosen_labels, loss_type=loss_type)
    policy_rejected_logps = get_seq2seq_logps(policy_rejected_out.logits, rejected_labels, loss_type=loss_type)

    # 2. Forward Pass with Reference Model π_ref
    with torch.no_grad():
        if is_peft:
            # Zero-VRAM: Disable LoRA adapters to evaluate the frozen base SFT model
            with model.disable_adapter():
                ref_chosen_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=chosen_labels)
                ref_rejected_out = model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=rejected_labels)
                ref_chosen_logps = get_seq2seq_logps(ref_chosen_out.logits, chosen_labels, loss_type=loss_type)
                ref_rejected_logps = get_seq2seq_logps(ref_rejected_out.logits, rejected_labels, loss_type=loss_type)
        else:
            ref_chosen_out = ref_model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=chosen_labels)
            ref_rejected_out = ref_model(input_ids=prompt_ids, attention_mask=prompt_mask, labels=rejected_labels)
            ref_chosen_logps = get_seq2seq_logps(ref_chosen_out.logits, chosen_labels, loss_type=loss_type)
            ref_rejected_logps = get_seq2seq_logps(ref_rejected_out.logits, rejected_labels, loss_type=loss_type)

    # 3. Compute DPO Loss & Implicit Rewards
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    # L_DPO = -E[log sigmoid(beta * (log(pi_w/ref_w) - log(pi_l/ref_l)))]
    losses = -F.logsigmoid(beta * logits)
    loss = losses.mean()

    # Reward metrics
    chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
    reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()

    return loss, reward_acc, reward_margin, loss.item()

# ==============================================================================
# EVALUATION LOOP
# ==============================================================================
def evaluate_dpo(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    loss_type: str = "sum",
    is_peft: bool = True,
    ref_model: nn.Module = None,
) -> Tuple[float, float, float]:
    """
    Evaluates DPO loss, accuracy, and margin on the validation split.
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    num_batches = len(dataloader)

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="DPO Evaluation", leave=False):
            loss, acc, margin, loss_val = compute_dpo_step(
                model=model,
                batch=batch,
                device=device,
                beta=beta,
                loss_type=loss_type,
                is_peft=is_peft,
                ref_model=ref_model,
            )
            total_loss += loss_val
            total_acc += acc
            total_margin += margin

    return total_loss / num_batches, total_acc / num_batches, total_margin / num_batches

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    log_dir = args.log_dir
    plot_dir = args.plot_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    script_name = os.path.basename(__file__).replace(".py", "")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    print(f"Log file initialized at: {log_file}")
    print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Nutze Device: {device}")

    # Enforce directory path for SFT model
    if os.path.isfile(args.model_name_or_path) or args.model_name_or_path.endswith((".pt", ".pth", ".bin")):
        raise ValueError(
            f"Ungültiger Pfad '{args.model_name_or_path}': Es muss ein Modell-Ordnerpfad übergeben werden "
            f"(z.B. 'results/models/new_pipeline/sft'), keine .pt Datei."
        )

    print(f"Lade Basis-SFT-Modell und Tokenizer aus: {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if hasattr(tokenizer, "src_lang") or hasattr(tokenizer, "lang_code_to_id") or "mbart" in str(getattr(tokenizer, "name_or_path", "")).lower() or "mbart" in args.model_name_or_path.lower():
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"
        print("=" * 80)
        print("[SPRACHCODE-KONTROLLE] Tokenizer für mBART erfolgreich konfiguriert:")
        print(f"  -> src_lang: {tokenizer.src_lang}")
        print(f"  -> tgt_lang: {tokenizer.tgt_lang}")
        if hasattr(tokenizer, "lang_code_to_id"):
            print(f"  -> de_DE Token-ID: {tokenizer.lang_code_to_id.get('de_DE')}")
        test_toks = tokenizer(text_target="Test")["input_ids"]
        print(f"  -> Test-Target Enkodierung (Token IDs): {test_toks}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("[SPRACHCODE-KONTROLLE] Kein multilingualer Tokenizer erkannt (Standard Monolingual).")
        print("=" * 80)

    # Load Seq2Seq Model with Robust SFT Adapter Merging
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    adapter_cfg_file = os.path.join(args.model_name_or_path, "adapter_config.json")
    
    if os.path.exists(adapter_cfg_file):
        with open(adapter_cfg_file, "r", encoding="utf-8") as f:
            acfg = json.load(f)
        base_model_id = acfg.get("base_model_name_or_path", "facebook/mbart-large-50")
        print(f"Lade Basismodell '{base_model_id}' fuer SFT-Adapter...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, torch_dtype=dtype)
        
        print(f"Lade und verschmelze (merge_and_unload) SFT-LoRA-Adapter aus: {args.model_name_or_path}...")
        sft_peft_model = PeftModel.from_pretrained(base_model, args.model_name_or_path)
        base_model = sft_peft_model.merge_and_unload()
        print("[ERFOLG] SFT-Adapter erfolgreich in Basisgewichte integriert (SFT ist nun die Basis fuer DPO)!")
        model = base_model.to(device)
    else:
        print(f"Lade regulaeres Seq2Seq-Modell direkt aus: {args.model_name_or_path}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
        ).to(device)

    # LoRA / PEFT Configuration for DPO
    ref_model = None
    if args.use_peft:
        # Determine base model architecture
        is_t5_model = "t5" in args.model_name_or_path.lower() or (os.path.exists(adapter_cfg_file) and "t5" in base_model_id.lower())
        if is_t5_model:
            target_modules = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"] if "mt5" in (args.model_name_or_path.lower() + (base_model_id.lower() if os.path.exists(adapter_cfg_file) else "")) else ["q", "v", "k", "o", "wi", "wo"]
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]

        print(f"Konfiguriere DPO-LoRA auf Basis des SFT-Modells (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, targets={target_modules})...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Full Parameter Fine-Tuning: Erstelle gefrorene Kopie als Referenzmodell...")
        import copy
        ref_model = copy.deepcopy(model).to(device)
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

    # Datasets and DataLoaders
    train_dataset = DPOPreferenceDataset(
        data_file=args.train_file,
        tokenizer=tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        prompt_prefix=args.prompt_prefix,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.eval_file and os.path.exists(args.eval_file):
        val_dataset = DPOPreferenceDataset(
            data_file=args.eval_file,
            tokenizer=tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
            prompt_prefix=args.prompt_prefix,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        print(f"Validierungsset initialisiert: {len(val_dataset)} Paare ({len(val_loader)} Batches).")

    # Optimizer and LR Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_training_steps = (len(train_loader) // args.accumulation_steps) * args.epochs
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(total_training_steps, 1),
    )

    print("\n" + "=" * 80)
    print(f"=== Starte DPO Training ({args.epochs} Epochen | Beta={args.beta} | LR={args.lr}) ===")
    print("=" * 80 + "\n")

    history = {
        "train_loss": [], "train_acc": [], "train_margin": [],
        "val_loss": [], "val_acc": [], "val_margin": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0

    def _save_checkpoint(m, path, tok):
        os.makedirs(path, exist_ok=True)
        if hasattr(m, "merge_and_unload"):
            try:
                import copy
                temp_m = copy.deepcopy(m)
                merged = temp_m.merge_and_unload()
                merged.save_pretrained(path)
                del temp_m, merged
                print(f"Modell fusioniert und gespeichert nach: {path}")
            except Exception as e:
                print(f"Hinweis beim Speichern des fusionierten Modells: {e}")
                m.save_pretrained(path)
        else:
            m.save_pretrained(path)
        tok.save_pretrained(path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"DPO Epoche {epoch}/{args.epochs}")
        for step, batch in enumerate(pbar):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else contextlib.nullcontext():
                loss, acc, margin, loss_val = compute_dpo_step(
                    model=model,
                    batch=batch,
                    device=device,
                    beta=args.beta,
                    loss_type=args.loss_type,
                    is_peft=args.use_peft,
                    ref_model=ref_model,
                )
                scaled_loss = loss / args.accumulation_steps

            scaled_loss.backward()

            epoch_loss += loss_val
            epoch_acc += acc
            epoch_margin += margin

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "Loss": f"{loss_val:.4f}",
                "Acc": f"{acc:.2f}",
                "Margin": f"{margin:.3f}",
            })

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_acc = epoch_acc / len(train_loader)
        avg_train_margin = epoch_margin / len(train_loader)

        history["train_loss"].append(avg_train_loss)
        history["train_acc"].append(avg_train_acc)
        history["train_margin"].append(avg_train_margin)

        print(f"\n--- Epoche {epoch}/{args.epochs} Zusammenfassung ---")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Train Margin: {avg_train_margin:.4f}")

        # Validation Step
        if val_loader:
            avg_val_loss, avg_val_acc, avg_val_margin = evaluate_dpo(
                model=model,
                dataloader=val_loader,
                device=device,
                beta=args.beta,
                loss_type=args.loss_type,
                is_peft=args.use_peft,
                ref_model=ref_model,
            )
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(avg_val_acc)
            history["val_margin"].append(avg_val_margin)

            print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f} | Val Margin:   {avg_val_margin:.4f}")

            # Early Stopping and Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"Neuer bester Val Loss ({best_val_loss:.4f})! Speichere Modell nach {args.output_dir}...")
                _save_checkpoint(model, args.output_dir, tokenizer)
            else:
                patience_counter += 1
                print(f"Keine Verbesserung (Patience: {patience_counter}/{args.patience})")
                if patience_counter >= args.patience:
                    print(f"Early Stopping ausgelöst nach Epoche {epoch}.")
                    break
        else:
            # Save final checkpoint if no validation set is used
            _save_checkpoint(model, args.output_dir, tokenizer)

    print("\n" + "=" * 80)
    print(f"DPO Training erfolgreich beendet! Modell gespeichert in: {args.output_dir}")
    print("=" * 80)

    # Generate Training Curves Plot
    try:
        epochs_range = range(1, len(history["train_loss"]) + 1)
        plt.figure(figsize=(15, 5))

        # Plot Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="royalblue")
        if val_loader and history["val_loss"]:
            plt.plot(epochs_range, history["val_loss"], "s--", label="Val Loss", color="darkorange")
        plt.title("DPO Loss")
        plt.xlabel("Epoche")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Plot Reward Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, history["train_acc"], "o-", label="Train Acc", color="forestgreen")
        if val_loader and history["val_acc"]:
            plt.plot(epochs_range, history["val_acc"], "s--", label="Val Acc", color="crimson")
        plt.title("DPO Reward Accuracy")
        plt.xlabel("Epoche")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Plot Reward Margin
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, history["train_margin"], "o-", label="Train Margin", color="purple")
        if val_loader and history["val_margin"]:
            plt.plot(epochs_range, history["val_margin"], "s--", label="Val Margin", color="teal")
        plt.title("DPO Implicit Reward Margin")
        plt.xlabel("Epoche")
        plt.ylabel("Margin (Chosen - Rejected)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plot_path = os.path.join(plot_dir, f"dpo_training_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"DPO Trainingskurven-Plot gespeichert unter: {plot_path}")
    except Exception as e:
        print(f"Hinweis: Plot konnte nicht erstellt werden ({e})")

    # Save training history JSON
    try:
        hist_path = os.path.join(args.output_dir, "training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"DPO Trainingshistorie gespeichert unter: {hist_path}")
    except Exception as e:
        print(f"Hinweis: Historie konnte nicht gespeichert werden ({e})")

if __name__ == "__main__":
    main()

