#!/usr/bin/env python3
"""
Generate translations using the SFT Baseline and DPO Ladder models on test datasets.
Saves the results to CSV for downstream linguistic rule adherence auditing.
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Generate translations using SFT and DPO Ladder models.")
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="data/lebenshilfe/lebenshilfe_dataset.json",
        help="Path to test dataset JSON/JSONL"
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="data/temperature_ladder_500/dpo_pairs_w05_w05_eval.jsonl",
        help="Optional additional eval JSONL split"
    )
    parser.add_argument(
        "--sft_model_path",
        type=str,
        default="results/models/sft",
        help="Path to SFT model directory"
    )
    parser.add_argument(
        "--dpo_model_path",
        type=str,
        default="results/models/seq2seq_dpo_w05_w05_filtered",
        help="Path to DPO Ladder model directory"
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="facebook/mbart-large-50",
        help="Base HuggingFace Seq2Seq model"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="results/evaluation/ladder_generations_eval.csv",
        help="Path to save generated outputs CSV"
    )
    parser.add_argument(
        "--max_samples_eval",
        type=int,
        default=50,
        help="Number of samples from eval_jsonl to include"
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=500,
        help="Max source token length"
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=500,
        help="Max target generation length"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Inference batch size"
    )
    return parser.parse_args()


def load_model(model_path: str, base_model_name: str, device: torch.device):
    print(f"Loading model from {model_path}...")
    dtype = torch.float32  # Stable on Mac MPS/CPU
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = "de_DE"
    if hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = "de_DE"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        
    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if has_adapter:
        base_m = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
        peft_m = PeftModel.from_pretrained(base_m, model_path)
        model = peft_m.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
        
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_batch(model, tokenizer, texts, device, max_source_len=500, max_target_len=500):
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and "de_DE" in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]

    inputs = tokenizer(
        texts,
        max_length=max_source_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_length": max_target_len,
        "num_beams": 4,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }
    if forced_bos_token_id is not None:
        gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        gen_kwargs["decoder_start_token_id"] = forced_bos_token_id

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
        
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [t.strip() for t in decoded]


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Collect Samples
    samples = []
    
    # Load Lebenshilfe dataset
    if os.path.exists(args.test_dataset):
        with open(args.test_dataset, "r", encoding="utf-8") as f:
            lh_data = json.load(f)
        for item in lh_data:
            as_text = str(item.get("as_text") or "").strip()
            ls_text = str(item.get("ls_text") or "").strip()
            if as_text and ls_text:
                samples.append({
                    "sample_id": f"lebenshilfe_{len(samples)}",
                    "domain": "lebenshilfe_gold",
                    "as_text": as_text,
                    "ref_ls_text": ls_text
                })
        print(f"Loaded {len(samples)} samples from Lebenshilfe dataset.")

    # Load Eval JSONL samples
    if os.path.exists(args.eval_jsonl) and args.max_samples_eval > 0:
        eval_count = 0
        with open(args.eval_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                as_text = str(item.get("as_text") or "").strip()
                ref_ls = str(item.get("chosen") or "").strip()
                if as_text:
                    samples.append({
                        "sample_id": f"eval_ladder_{eval_count}",
                        "domain": item.get("source", "eval_split"),
                        "as_text": as_text,
                        "ref_ls_text": ref_ls
                    })
                    eval_count += 1
                    if eval_count >= args.max_samples_eval:
                        break
        print(f"Loaded {eval_count} samples from eval JSONL split.")

    print(f"Total evaluation samples to generate: {len(samples)}")
    as_texts = [s["as_text"] for s in samples]

    # 2. SFT Inference
    sft_model, sft_tokenizer = load_model(args.sft_model_path, args.base_model_name, device)
    sft_translations = []
    
    num_batches = (len(as_texts) + args.batch_size - 1) // args.batch_size
    for b in tqdm(range(num_batches), desc="Generating SFT translations"):
        batch_texts = as_texts[b * args.batch_size : (b + 1) * args.batch_size]
        outs = generate_batch(sft_model, sft_tokenizer, batch_texts, device, args.max_source_len, args.max_target_len)
        sft_translations.extend(outs)
        
    del sft_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 3. DPO Ladder Inference
    dpo_model, dpo_tokenizer = load_model(args.dpo_model_path, args.base_model_name, device)
    dpo_translations = []
    
    for b in tqdm(range(num_batches), desc="Generating DPO Ladder translations"):
        batch_texts = as_texts[b * args.batch_size : (b + 1) * args.batch_size]
        outs = generate_batch(dpo_model, dpo_tokenizer, batch_texts, device, args.max_source_len, args.max_target_len)
        dpo_translations.extend(outs)
        
    del dpo_model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 4. Save to DataFrame & CSV
    df = pd.DataFrame(samples)
    df["sft_translation"] = sft_translations
    df["dpo_translation"] = dpo_translations
    
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\nSaved generated translations for {len(df)} samples to {args.output_csv}")


if __name__ == "__main__":
    main()
