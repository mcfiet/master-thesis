#!/usr/bin/env python3
"""
=============================================================================
DPO Training Script with Hugging Face TRL (DPOTrainer)
=============================================================================
This script performs Direct Preference Optimization (DPO) starting from a
pre-trained Supervised Fine-Tuned (SFT) model.

It uses the official Hugging Face `trl` library (DPOTrainer, DPOConfig) and
supports:
  - Causal LM (e.g. Llama, Mistral, Qwen) and Seq2Seq LM (e.g. mBART, T5)
  - Loading from HuggingFace directories/repos or PyTorch checkpoint files (.pt)
  - Full Parameter Fine-Tuning or Parameter-Efficient Fine-Tuning (LoRA / QLoRA)
  - Standard Preference Dataset format (prompt, chosen, rejected)
  - Automatic reference model handling (zero-VRAM reference via PEFT adapter disabling)
  - Mixed precision training (bfloat16 / fp16) and gradient checkpointing
  - Flexible evaluation and checkpoint saving
  - Dynamic backward/forward compatibility across all TRL & Transformers versions

Documentation Reference:
  - Hugging Face TRL DPO: https://huggingface.co/docs/trl/dpo_trainer
=============================================================================
"""

import argparse
import dataclasses
import inspect
import logging
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Set, Union

import datasets
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

# ---------------------------------------------------------------------------
# Setup Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("TrainDPO")


# ---------------------------------------------------------------------------
# Helper: Extract Valid Class Parameters for Dynamic Compatibility
# ---------------------------------------------------------------------------
def get_class_fields_and_params(cls: Any) -> Set[str]:
    fields: Set[str] = set()
    if dataclasses.is_dataclass(cls):
        for f in dataclasses.fields(cls):
            fields.add(f.name)
    try:
        sig = inspect.signature(cls.__init__)
        fields.update(sig.parameters.keys())
    except Exception:
        pass
    return fields


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a language model using Direct Preference Optimization (DPO) with HF TRL."
    )

    # --- Model Arguments ---
    model_group = parser.add_argument_group("Model Arguments")
    model_group.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pre-trained SFT model checkpoint (.pt or dir) or Hugging Face Hub model ID.",
    )
    model_group.add_argument(
        "--base_model_name",
        "--model_name",
        dest="base_model_name",
        type=str,
        default="facebook/mbart-large-50",
        help="Base model identifier if --model_name_or_path is a .pt weights file (default: 'facebook/mbart-large-50').",
    )
    model_group.add_argument(
        "--ref_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Optional path to reference model. If None and LoRA is used, TRL automatically "
            "uses the base model without adapters as the reference model (saving VRAM). "
            "If None and full fine-tuning is used, the base model is copied as reference."
        ),
    )
    model_group.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Torch data type for model weights (default: bfloat16).",
    )
    model_group.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["sdpa", "flash_attention_2", "eager"],
        help="Attention implementation to use (e.g. 'flash_attention_2' or 'sdpa').",
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow custom code from the model repository on Hugging Face Hub.",
    )

    # --- PEFT / LoRA Arguments ---
    peft_group = parser.add_argument_group("PEFT / LoRA Arguments")
    peft_group.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Whether to use LoRA for parameter-efficient fine-tuning (default: True).",
    )
    peft_group.add_argument(
        "--no_peft",
        dest="use_peft",
        action="store_false",
        help="Disable LoRA and perform full-parameter DPO fine-tuning.",
    )
    peft_group.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA attention dimension (rank r, default: 16).",
    )
    peft_group.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha parameter for scaling (default: 32).",
    )
    peft_group.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout probability (default: 0.05).",
    )
    peft_group.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="List of module names to target with LoRA (e.g., q_proj k_proj v_proj o_proj gate_proj up_proj down_proj). "
             "If None, PEFT default or all-linear modules are targeted.",
    )
    peft_group.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load model in 4-bit precision (QLoRA) using bitsandbytes.",
    )
    peft_group.add_argument(
        "--use_8bit",
        action="store_true",
        help="Load model in 8-bit precision using bitsandbytes.",
    )

    # --- Dataset Arguments ---
    data_group = parser.add_argument_group("Dataset Arguments")
    data_group.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face Hub dataset identifier (e.g., 'Anthropic/hh-rlhf').",
    )
    data_group.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="Path to local training dataset file (.json, .jsonl, .csv, .parquet).",
    )
    data_group.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Path to local validation dataset file (.json, .jsonl, .csv, .parquet).",
    )
    data_group.add_argument(
        "--eval_split_ratio",
        type=float,
        default=0.05,
        help="Fraction of train data to split into validation set if --eval_file is not given (default: 0.05).",
    )
    data_group.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name for the prompt text (default: 'prompt').",
    )
    data_group.add_argument(
        "--chosen_column",
        type=str,
        default="chosen",
        help="Column name for the preferred/chosen response (default: 'chosen').",
    )
    data_group.add_argument(
        "--rejected_column",
        type=str,
        default="rejected",
        help="Column name for the dispreferred/rejected response (default: 'rejected').",
    )
    data_group.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate the number of training examples to this value (for debugging).",
    )
    data_group.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate the number of evaluation examples to this value (for debugging).",
    )

    # --- DPO Hyperparameters ---
    dpo_group = parser.add_argument_group("DPO Hyperparameters")
    dpo_group.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter beta (KL penalty strength, typically 0.01 - 0.5, default: 0.1).",
    )
    dpo_group.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "hinge", "ipo", "kto_pair", "bco_pair", "sppo", "robust"],
        help="DPO loss formulation to use (default: 'sigmoid').",
    )
    dpo_group.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (default: 0.0, Conservative DPO uses > 0.0).",
    )
    dpo_group.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum total token length (prompt + completion, default: 1024).",
    )
    dpo_group.add_argument(
        "--max_prompt_length",
        type=int,
        default=512,
        help="Maximum token length for the prompt portion (default: 512).",
    )
    dpo_group.add_argument(
        "--max_target_length",
        type=int,
        default=None,
        help="Maximum token length for the target/completion portion (optional).",
    )

    # --- Standard Training Arguments ---
    train_group = parser.add_argument_group("Training Arguments")
    train_group.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where model checkpoints and logs will be written.",
    )
    train_group.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Total number of training epochs to perform (default: 3.0).",
    )
    train_group.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate for AdamW optimizer (default: 5e-6 for LoRA, 5e-7 for full FT).",
    )
    train_group.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU/device for training (default: 2).",
    )
    train_group.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2,
        help="Batch size per GPU/device for evaluation (default: 2).",
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of update steps to accumulate before backward/update pass (default: 8).",
    )
    train_group.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup ratio over total training steps (default: 0.1).",
    )
    train_group.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="The learning rate scheduler type (default: 'cosine').",
    )
    train_group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer (default: 0.01).",
    )
    train_group.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce VRAM usage (default: True).",
    )
    train_group.add_argument(
        "--no_gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    train_group.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every X update steps (default: 10).",
    )
    train_group.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Evaluation strategy to adopt during training (default: 'steps').",
    )
    train_group.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Run evaluation every X update steps (default: 50).",
    )
    train_group.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
        help="Checkpoint save strategy (default: 'steps').",
    )
    train_group.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X update steps (default: 100).",
    )
    train_group.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Limit the total amount of checkpoints. Deletes older checkpoints (default: 3).",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization and reproducibility (default: 42).",
    )
    train_group.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="List of integrations to report results/logs to (e.g., 'none', 'tensorboard', 'wandb'). Default: 'none'.",
    )
    train_group.add_argument(
        "--dataset_num_proc",
        type=int,
        default=4,
        help="Number of worker processes for dataset tokenization/preprocessing (default: 4).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset Loading and Standardizing
# ---------------------------------------------------------------------------
def load_and_prepare_dataset(args: argparse.Namespace) -> DatasetDict:
    """
    Loads raw dataset from Hugging Face Hub or local files, standardizes column names
    to ('prompt', 'chosen', 'rejected') as expected by TRL DPOTrainer.
    """
    logger.info("Loading preference dataset...")

    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name)
    elif args.train_file is not None:
        data_files = {"train": args.train_file}
        if args.eval_file is not None:
            data_files["eval"] = args.eval_file

        extension = args.train_file.split(".")[-1]
        if extension in ["json", "jsonl"]:
            raw_datasets = load_dataset("json", data_files=data_files)
        elif extension == "csv":
            raw_datasets = load_dataset("csv", data_files=data_files)
        elif extension == "parquet":
            raw_datasets = load_dataset("parquet", data_files=data_files)
        else:
            raise ValueError(f"Unsupported file format: .{extension}. Supported: json, jsonl, csv, parquet.")
    else:
        raise ValueError("Must provide either --dataset_name or --train_file.")

    # Create validation split if only train dataset is provided
    if "eval" not in raw_datasets and "validation" not in raw_datasets and "test" not in raw_datasets:
        if args.eval_split_ratio > 0:
            logger.info(f"Splitting training data with validation ratio: {args.eval_split_ratio}")
            split_dataset = raw_datasets["train"].train_test_split(
                test_size=args.eval_split_ratio,
                seed=args.seed,
            )
            raw_datasets = DatasetDict({
                "train": split_dataset["train"],
                "eval": split_dataset["test"],
            })
        else:
            raw_datasets = DatasetDict({"train": raw_datasets["train"]})
    elif "eval" not in raw_datasets:
        # Standardize split naming to "train" and "eval"
        eval_key = "validation" if "validation" in raw_datasets else "test"
        raw_datasets = DatasetDict({
            "train": raw_datasets["train"],
            "eval": raw_datasets[eval_key],
        })

    # Standardize column names to 'prompt', 'chosen', 'rejected'
    for split_name in raw_datasets.keys():
        dataset_split = raw_datasets[split_name]
        rename_dict = {}

        if args.prompt_column != "prompt" and args.prompt_column in dataset_split.column_names:
            rename_dict[args.prompt_column] = "prompt"
        if args.chosen_column != "chosen" and args.chosen_column in dataset_split.column_names:
            rename_dict[args.chosen_column] = "chosen"
        if args.rejected_column != "rejected" and args.rejected_column in dataset_split.column_names:
            rename_dict[args.rejected_column] = "rejected"

        if rename_dict:
            raw_datasets[split_name] = dataset_split.rename_columns(rename_dict)

        # Validate required columns
        cols = raw_datasets[split_name].column_names
        for required_col in ["prompt", "chosen", "rejected"]:
            if required_col not in cols:
                raise ValueError(
                    f"Dataset split '{split_name}' is missing required column '{required_col}'. "
                    f"Available columns: {cols}. Use --prompt_column, --chosen_column, --rejected_column to map."
                )

    # Subsample if requested
    if args.max_train_samples is not None and len(raw_datasets["train"]) > args.max_train_samples:
        logger.info(f"Subsampling training set to {args.max_train_samples} samples.")
        raw_datasets["train"] = raw_datasets["train"].select(range(args.max_train_samples))

    if "eval" in raw_datasets and args.max_eval_samples is not None and len(raw_datasets["eval"]) > args.max_eval_samples:
        logger.info(f"Subsampling eval set to {args.max_eval_samples} samples.")
        raw_datasets["eval"] = raw_datasets["eval"].select(range(args.max_eval_samples))

    logger.info(f"Prepared Dataset: {raw_datasets}")
    return raw_datasets


# ---------------------------------------------------------------------------
# Ensure LoRA / PEFT classes report is_encoder_decoder for Seq2Seq Models
# ---------------------------------------------------------------------------
LoraConfig.is_encoder_decoder = True
PeftModel.is_encoder_decoder = True


# ---------------------------------------------------------------------------
# Model and Tokenizer Initialization
# ---------------------------------------------------------------------------
def setup_model_and_tokenizer(args: argparse.Namespace):
    """
    Initializes tokenizer and model with appropriate precision, quantization, and LoRA configuration.
    Supports both Causal LM and Seq2Seq LM architectures, from directories or raw .pt weight files.
    """
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    quantization_config = None

    if args.use_4bit:
        logger.info("Configuring 4-bit QLoRA quantization (bitsandbytes)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.use_8bit:
        logger.info("Configuring 8-bit quantization (bitsandbytes)...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "quantization_config": quantization_config,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    # Enforce directory path (no raw .pt files)
    if os.path.isfile(args.model_name_or_path) or args.model_name_or_path.endswith((".pt", ".pth", ".bin")):
        raise ValueError(
            f"Ungültiger Pfad '{args.model_name_or_path}': Es muss ein Modell-Ordnerpfad übergeben werden "
            f"(z.B. 'results/models/new_pipeline/sft' oder ein HuggingFace Hub Modell-Name), keine .pt/.pth Datei."
        )

    logger.info(f"Loading Model and Tokenizer from directory: {args.model_name_or_path}")

    # Robustly identify whether the architecture is Encoder-Decoder (Seq2Seq) or Decoder-Only (Causal LM)
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    except Exception:
        name_lower = str(args.model_name_or_path).lower()
        is_seq2seq = any(k in name_lower for k in ["mbart", "bart", "t5", "marian", "pegasus"])

    logger.info(f"Model architecture mode: {'Seq2Seq (Encoder-Decoder)' if is_seq2seq else 'Causal LM (Decoder-Only)'}")

    # Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, use_fast=True)

    # Load Model
    if is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("Setting tokenizer.pad_token to tokenizer.eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Added special [PAD] token to tokenizer")

    check_name = str(args.model_name_or_path).lower()
    if "mbart" in check_name or "facebook/mbart" in str(getattr(model.config, "_name_or_path", "")).lower():
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

    # Reference Model
    ref_model = None
    if args.ref_model_name_or_path is not None:
        logger.info(f"Loading separate reference model from: {args.ref_model_name_or_path}")
        if is_seq2seq:
            ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.ref_model_name_or_path, **model_kwargs)
        else:
            ref_model = AutoModelForCausalLM.from_pretrained(args.ref_model_name_or_path, **model_kwargs)
    elif not args.use_peft:
        logger.info("Full fine-tuning selected: DPOTrainer will manage reference model copies.")

    # PEFT / LoRA Config
    peft_config = None
    if args.use_peft:
        task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM
        logger.info(f"Configuring LoRA (task_type={task_type}, r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules or "all-linear",
            bias="none",
            task_type=task_type,
        )
        setattr(peft_config, "is_encoder_decoder", is_seq2seq)

    return model, ref_model, tokenizer, peft_config, is_seq2seq


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Check tensorboard availability if requested
    if args.report_to and "tensorboard" in args.report_to.lower():
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            try:
                import tensorboardX  # noqa: F401
            except ImportError:
                logger.warning("tensorboard/tensorboardX is not installed. Disabling report_to (setting to 'none').")
                args.report_to = "none"

    logger.info("=== Starting DPO Training with Hugging Face TRL ===")
    logger.info(f"Target Output Directory: {args.output_dir}")

    # 1. Prepare Datasets
    raw_datasets = load_and_prepare_dataset(args)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets.get("eval", None)

    # 2. Prepare Model, Tokenizer, and PEFT
    model, ref_model, tokenizer, peft_config, is_seq2seq = setup_model_and_tokenizer(args)

    # 3. Dynamically Configure DPOConfig (Compatible with all TRL / Transformers releases)
    eval_strat_val = args.eval_strategy if eval_dataset is not None else "no"
    is_enc_dec = getattr(model.config, "is_encoder_decoder", is_seq2seq)

    # Force is_encoder_decoder on model configs
    model.config.is_encoder_decoder = is_enc_dec
    if ref_model is not None:
        ref_model.config.is_encoder_decoder = is_enc_dec

    all_candidate_dpo_args: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_target_length": args.max_target_length,
        "max_completion_length": args.max_target_length,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps,
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "report_to": args.report_to.split(",") if args.report_to != "none" else "none",
        "dataset_num_proc": args.dataset_num_proc,
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
        "remove_unused_columns": False,
        "is_encoder_decoder": is_enc_dec,
    }

    config_valid_fields = get_class_fields_and_params(DPOConfig)

    # Handle eval strategy name differences
    if "eval_strategy" in config_valid_fields:
        all_candidate_dpo_args["eval_strategy"] = eval_strat_val
        if eval_dataset is not None:
            all_candidate_dpo_args["eval_steps"] = args.eval_steps
    elif "evaluation_strategy" in config_valid_fields:
        all_candidate_dpo_args["evaluation_strategy"] = eval_strat_val
        if eval_dataset is not None:
            all_candidate_dpo_args["eval_steps"] = args.eval_steps

    # Filter to valid keys or use resilient instantiation
    dpo_config_kwargs = {}
    for k, v in all_candidate_dpo_args.items():
        if v is not None and k in config_valid_fields:
            dpo_config_kwargs[k] = v

    # Resilient DPOConfig initialization loop
    while True:
        try:
            dpo_config = DPOConfig(**dpo_config_kwargs)
            break
        except TypeError as e:
            err_msg = str(e)
            match = re.search(r"unexpected keyword argument '([^']+)'", err_msg)
            if match:
                bad_key = match.group(1)
                logger.info(f"Filtering out unsupported DPOConfig argument: '{bad_key}'")
                dpo_config_kwargs.pop(bad_key, None)
            else:
                raise e

    # Explicitly set is_encoder_decoder on dpo_config object
    setattr(dpo_config, "is_encoder_decoder", is_enc_dec)

    # 4. Initialize DPOTrainer
    logger.info(f"Initializing DPOTrainer (is_encoder_decoder={is_enc_dec})...")
    trainer_valid_fields = get_class_fields_and_params(DPOTrainer)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "is_encoder_decoder": is_enc_dec,
    }

    # Pass tokenizer / processing_class based on TRL version
    if "processing_class" in trainer_valid_fields:
        trainer_kwargs["processing_class"] = tokenizer
    if "tokenizer" in trainer_valid_fields or "processing_class" not in trainer_valid_fields:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass length parameters to trainer if they were not in DPOConfig
    for param_name, val in [
        ("max_length", args.max_length),
        ("max_prompt_length", args.max_prompt_length),
        ("max_target_length", args.max_target_length),
    ]:
        if val is not None and param_name not in dpo_config_kwargs and param_name in trainer_valid_fields:
            trainer_kwargs[param_name] = val

    # Resilient DPOTrainer initialization loop
    while True:
        try:
            trainer = DPOTrainer(**trainer_kwargs)
            break
        except TypeError as e:
            err_msg = str(e)
            match = re.search(r"unexpected keyword argument '([^']+)'", err_msg)
            if match:
                bad_key = match.group(1)
                logger.info(f"Filtering out unsupported DPOTrainer argument: '{bad_key}'")
                trainer_kwargs.pop(bad_key, None)
            else:
                raise e

    # Ensure trainer instance has is_encoder_decoder explicitly set
    setattr(trainer, "is_encoder_decoder", is_enc_dec)

    # 5. Execute Training
    logger.info("Starting DPO training loop...")
    train_result = trainer.train()

    # 6. Save Model and Artifacts
    logger.info(f"Saving final trained model and tokenizer to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Run final evaluation if eval dataset is present
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("=== DPO Training Completed Successfully ===")


if __name__ == "__main__":
    main()
