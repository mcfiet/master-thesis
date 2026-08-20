#!/usr/bin/env python3
"""
=============================================================================
DPO Training for Decoder-Only LLMs with Hugging Face TRL & PEFT
=============================================================================
This script performs Direct Preference Optimization (DPO) starting from a
fine-tuned Decoder-Only SFT model (e.g. Qwen 2.5, Llama 3.1) using `trl.DPOTrainer`.

Features:
  - Hugging Face TRL DPOTrainer integration with DPOConfig
  - LoRA parameter-efficient training with shared Reference Model (zero extra VRAM)
  - Tracking of DPO Loss, Implicit Reward Margins, and Accuracy
  - Early stopping and evaluation on validation preference pairs
  - Automatic loss and margin curve generation in `results/plots/`
=============================================================================
"""

import os
import sys
import json
import random
import datetime
import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed,
)
from peft import PeftModel, LoraConfig, TaskType
from trl import DPOTrainer, DPOConfig

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
def load_dpo_dataset(file_path: str, max_samples: int = None) -> List[Dict[str, str]]:
    print(f"Loading DPO preference dataset from: {file_path}")
    pairs = []
    if file_path.endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    pairs.append({
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                    })
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                pairs.append({
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print(f"Total DPO pairs loaded: {len(pairs)}")
    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]
        print(f"Subsampled to {len(pairs)} pairs.")
    return pairs


# ==============================================================================
# MAIN DPO TRAINING
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train DPO Decoder-Only Model with TRL")
    parser.add_argument("--dpo_train_file", default="data/dpo/dpo_preference_pairs_decoder_only.jsonl")
    parser.add_argument("--dpo_eval_file", default=None)
    parser.add_argument("--sft_model_path", required=True, help="Path to SFT adapter or checkpoint")
    parser.add_argument("--base_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="results/models/decoder_only/dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta (0.05 to 0.2)")
    parser.add_argument("--lr", type=float, default=2e-6, help="Learning rate for DPO")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.10)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Data
    train_pairs = load_dpo_dataset(args.dpo_train_file, max_samples=args.max_samples)
    eval_pairs = []

    if args.dpo_eval_file and os.path.exists(args.dpo_eval_file):
        eval_pairs = load_dpo_dataset(args.dpo_eval_file, max_samples=args.max_samples)
    elif args.val_split > 0 and len(train_pairs) > 10:
        random.shuffle(train_pairs)
        split_idx = int((1.0 - args.val_split) * len(train_pairs))
        eval_pairs = train_pairs[split_idx:]
        train_pairs = train_pairs[:split_idx]

    print(f"DPO Split: {len(train_pairs)} Train | {len(eval_pairs)} Eval")

    train_ds = Dataset.from_list(train_pairs)
    eval_ds = Dataset.from_list(eval_pairs) if len(eval_pairs) > 0 else None

    # 2. Load Tokenizer
    print(f"Loading Tokenizer from: {args.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3. Load Model
    print(f"Loading Model: {args.base_model_name}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # Check if loading existing PEFT SFT adapter or training fresh LoRA
    if os.path.exists(os.path.join(args.sft_model_path, "adapter_config.json")):
        print(f"Loading SFT LoRA adapter from: {args.sft_model_path}...")
        model = PeftModel.from_pretrained(base_model, args.sft_model_path, is_trainable=True)
        peft_config = None
    else:
        print("Using base model and applying fresh LoRA config for DPO...")
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = base_model

    # 4. DPO Configuration
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=5,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch" if eval_ds else "no",
        load_best_model_at_end=True if eval_ds else False,
        metric_for_best_model="eval_loss" if eval_ds else None,
        greater_is_better=False,
        save_total_limit=2,
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        max_length=args.max_length,
        report_to="none",
    )

    # 5. DPOTrainer Initialization
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if eval_ds else []
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT automatically handles reference disabling (0 extra VRAM)
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        args=dpo_config,
        callbacks=callbacks,
    )

    print("\n" + "=" * 60)
    print("Starting DPO Training with TRL...")
    print("=" * 60)
    train_result = trainer.train()

    # 6. Save Model & Tokenizer
    print(f"\nSaving DPO model adapter to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 7. Plot Loss and Reward Metrics
    log_history = trainer.state.log_history
    dpo_losses = [entry["loss"] for entry in log_history if "loss" in entry]
    reward_margins = [entry.get("rewards/margins") for entry in log_history if "rewards/margins" in entry]
    reward_accuracies = [entry.get("rewards/accuracies") for entry in log_history if "rewards/accuracies" in entry]

    if dpo_losses:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(dpo_losses, label="DPO Loss", color="#d62728", linewidth=2)
        axes[0].set_title(f"DPO Training Loss ({args.base_model_name})")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend()

        if reward_margins and any(m is not None for m in reward_margins):
            axes[1].plot([m for m in reward_margins if m is not None], label="Reward Margin", color="#2ca02c", linewidth=2)
            if reward_accuracies and any(a is not None for a in reward_accuracies):
                axes[1].plot([a for a in reward_accuracies if a is not None], label="Accuracy", color="#1f77b4", linestyle="--")
            axes[1].set_title("DPO Reward Margin & Accuracy")
            axes[1].set_xlabel("Step")
            axes[1].grid(True, linestyle="--", alpha=0.6)
            axes[1].legend()

        plot_path = os.path.join(plot_dir, f"dpo_decoder_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"DPO training plot saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("DPO Training Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
