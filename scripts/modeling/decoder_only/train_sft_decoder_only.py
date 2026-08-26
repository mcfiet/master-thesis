#!/usr/bin/env python3
"""
=============================================================================
SFT Training for Decoder-Only LLMs with Hugging Face TRL & PEFT
=============================================================================
This script performs Supervised Fine-Tuning (SFT) on German text simplification
corpora (Leichte Sprache) using modern Decoder-Only architectures (e.g. Qwen 2.5,
Llama 3.1) and the Hugging Face TRL `SFTTrainer`.

Features:
  - System prompt integration containing official Leichte Sprache guidelines (W2-W11)
  - Dataset formatting in standardized Hugging Face Chat Template
  - Native loss calculation restricted to assistant completions (assistant_only_loss)
  - LoRA / QLoRA parameter-efficient fine-tuning via PEFT
  - Dynamic GPU / bfloat16 / float16 precision handling
  - Automatic logging, checkpointing, and training curve plotting
=============================================================================
"""

import os
import sys
import json
import random
import re
import datetime
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

from prompts import SYSTEM_PROMPT_LEICHTE_SPRACHE, USER_INSTRUCTION_PREFIX, create_chat_messages


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
log_dir = "results/logs"
plot_dir = "results/plots"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

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
print(f"Log file initialized at: {log_file}")
print("Working directory:", os.getcwd())


# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================
def load_corpus_pairs(
    corpus_path: str,
    min_sim: float = 0.70,
    max_sim: float = 1.0,
    max_samples: int = None,
) -> List[Dict[str, str]]:
    print(f"Loading corpus from: {corpus_path}")
    if corpus_path.endswith(".json"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif corpus_path.endswith(".csv"):
        df = pd.read_csv(corpus_path)
        raw_data = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file format for {corpus_path}")

    pairs = []
    for row in raw_data:
        sim = row.get("semantic_similarity_8192")
        if sim is not None and not (min_sim <= float(sim) <= max_sim):
            continue
        as_text = str(row.get("as_text") or "").strip()
        ls_text = str(row.get("ls_text") or "").strip()
        if as_text and ls_text and len(as_text) > 20 and len(ls_text) > 10:
            pairs.append({
                "as_text": as_text,
                "ls_text": ls_text,
                "source": str(row.get("source") or "unknown"),
            })

    print(f"Total valid pairs loaded ({min_sim} <= sim <= {max_sim}): {len(pairs)}")
    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]
        print(f"Subsampled to {len(pairs)} records.")
    return pairs


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train SFT model (Decoder-Only) using TRL")
    parser.add_argument("--corpus_path", default="data/analysis/corpus_master.json", help="Path to corpus file")
    parser.add_argument("--lh_dataset_path", default=None, help="Optional additional lebenshilfe dataset path")
    parser.add_argument("--output_dir", default="results/models/decoder_only/sft", help="Output directory for model checkpoint")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct", help="Base decoder-only model identifier")
    parser.add_argument("--min_sim", type=float, default=0.70)
    parser.add_argument("--max_sim", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for LoRA adapters")
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--val_split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_peft", action="store_true", default=True, help="Use LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.10, help="LoRA dropout")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device selected: {device}")

    # 1. Load Data
    all_pairs = load_corpus_pairs(args.corpus_path, min_sim=args.min_sim, max_sim=args.max_sim, max_samples=args.max_samples)
    if args.lh_dataset_path and os.path.exists(args.lh_dataset_path):
        print(f"Appending secondary dataset: {args.lh_dataset_path}")
        lh_pairs = load_corpus_pairs(args.lh_dataset_path, min_sim=0.0, max_sim=1.0)
        all_pairs.extend(lh_pairs)

    random.shuffle(all_pairs)
    split_idx = int((1.0 - args.val_split) * len(all_pairs))
    train_records = all_pairs[:split_idx]
    val_records = all_pairs[split_idx:]
    print(f"Dataset split: {len(train_records)} Train | {len(val_records)} Validation")

    # Instruction Variations against template-overfitting
    INSTRUCTION_VARIATIONS = [
        "Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n",
        "Übersetze folgenden deutschen Text nach den offiziellen Regeln der Leichten Sprache:\n\n",
        "Schreibe diesen Text in einfacher, klarer und leicht verständlicher Sprache:\n\n",
        "Übertrage den folgenden Text in Leichte Sprache (kurze Sätze, einfache Wörter):\n\n",
        "Formuliere den folgenden schweren Text in barrierefreie deutsche Leichte Sprache um:\n\n",
    ]

    # Format into chat messages with header cleanup and instruction jittering
    def format_chat_records(records, is_train=True):
        formatted = []
        for r in records:
            # Header cleanup: remove isolated state/portal header lines
            clean_ls = re.sub(r"^(Sachsen-Anhalt|Hamburg|Schleswig-Holstein|Bremen|Niedersachsen|Pinneberg|Kiel)\s*\n+", "", r["ls_text"], flags=re.IGNORECASE).strip()
            prefix = random.choice(INSTRUCTION_VARIATIONS) if is_train else USER_INSTRUCTION_PREFIX
            messages = create_chat_messages(
                as_text=r["as_text"],
                ls_text=clean_ls,
                system_prompt=SYSTEM_PROMPT_LEICHTE_SPRACHE,
                instruction_prefix=prefix,
            )
            formatted.append({"messages": messages})
        return formatted

    train_dataset = Dataset.from_list(format_chat_records(train_records, is_train=True))
    val_dataset = Dataset.from_list(format_chat_records(val_records, is_train=False)) if len(val_records) > 0 else None

    # 2. Load Tokenizer
    print(f"Loading Tokenizer from: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFTTrainer handles padding

    # 3. Load Model
    print(f"Loading Model: {args.model_name}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # 4. LoRA Setup
    peft_config = None
    if args.use_peft:
        print(f"Configuring PEFT/LoRA (rank={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

    # 5. SFT Configuration
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        save_total_limit=2,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        report_to="none",
    )

    # 6. SFTTrainer Initialization & Training
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if val_dataset else []
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        args=sft_config,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("Starting SFT Training with TRL...")
    print("=" * 60)
    train_result = trainer.train()

    # 7. Save Merged Standalone Model & Tokenizer
    print(f"\nSaving merged standalone SFT model to: {args.output_dir}")
    try:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Fallback saving with trainer: {e}")
        trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Save Metrics & Plot
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Plot Loss Curve
    log_history = trainer.state.log_history
    train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

    if train_losses:
        plt.figure(figsize=(9, 5))
        plt.plot(train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
        if eval_losses:
            eval_steps = np.linspace(0, len(train_losses) - 1, len(eval_losses))
            plt.plot(eval_steps, eval_losses, label="Eval Loss", color="#ff7f0e", marker="o", linewidth=2)
        plt.title(f"SFT Training Loss Curve ({args.model_name})")
        plt.xlabel("Logging Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = os.path.join(plot_dir, f"sft_decoder_loss_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Loss plot saved to: {plot_path}")

    # Save training history JSON
    try:
        hist_path = os.path.join(args.output_dir, "training_history.json")
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2)
        print(f"SFT Trainingshistorie gespeichert unter: {hist_path}")
    except Exception as e:
        print(f"Hinweis: Historie konnte nicht gespeichert werden ({e})")

    print("\n" + "=" * 60)
    print("SFT Training Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
