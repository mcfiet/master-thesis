#!/usr/bin/env python3
"""
scripts/tests/test_inference_parameters.py

Systematischer Vergleich von:
- num_beams (1, 3, 5, 8)
- length_penalty (0.8, 1.0, 1.3, 1.6)
- repetition_penalty (1.0, 1.15, 1.25, 1.40)
- no_repeat_ngram_size (0, 2, 3, 4)
"""

import sys
import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def load_model_and_tok(model_path, base_model, device):
    print("=" * 80)
    print(f"📦 Lade Modell: {model_path}")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("  -> Lade Basismodell und verschmelze LoRA-Adapter...")
        bm = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
        model = PeftModel.from_pretrained(bm, model_path).merge_and_unload().to(device)
    else:
        print("  -> Lade Standalone-Modell...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def generate_with_config(model, tokenizer, text, device, gen_kwargs):
    inputs = tokenizer(
        [text],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_length": 256,
        **gen_kwargs,
    }

    with torch.no_grad():
        outs = model.generate(**kwargs)
    return tokenizer.decode(outs[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Inference Parameter Tuning")
    parser.add_argument("--model_path", default="results/models/test_single_dpo", help="Pfad zum Modell")
    parser.add_argument("--base_model", default="results/models/sft", help="Pfad zum Basismodell / Tokenizer")
    parser.add_argument("--test_file", default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Testset JSON")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index des Testbeispiels (0 bis 4)")
    parser.add_argument("--sweep", choices=["beams", "length", "rep", "ngram", "sweet_spots", "custom"], default="sweet_spots")

    # Custom Args
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    model, tokenizer = load_model_and_tok(args.model_path, args.base_model, device)

    with open(args.test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    sample = data[args.sample_idx]
    src = sample.get("source_text", sample.get("as_text", ""))
    ref = sample.get("target_text", sample.get("ls_text", ""))

    print("\n" + "=" * 80)
    print(f"📌 [TESTBEISPIEL {args.sample_idx + 1}]")
    print(f"▶️ QUELLTEXT (AS): {src[:160]}...")
    print(f"✅ REFERENZ (LS):  {ref[:160]}...")
    print("=" * 80 + "\n")

    if args.sweep == "custom":
        cfg = {
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
        }
        print(f"⚙️ Config: {cfg}")
        out = generate_with_config(model, tokenizer, src, device, cfg)
        print(f"🤖 Output:\n{out}\n")

    elif args.sweep == "beams":
        print("🔎 Vergleiche Einfluss von: num_beams (1, 3, 5, 8)\n")
        for b in [1, 3, 5, 8]:
            cfg = {"num_beams": b, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": (b > 1)}
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🔹 num_beams = {b}")
            print(f"🤖 {out}\n")

    elif args.sweep == "length":
        print("🔎 Vergleiche Einfluss von: length_penalty (0.8, 1.0, 1.3, 1.6)\n")
        for lp in [0.8, 1.0, 1.3, 1.6]:
            cfg = {"num_beams": 4, "length_penalty": lp, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3, "early_stopping": True}
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🔹 length_penalty = {lp}")
            print(f"🤖 {out}\n")

    elif args.sweep == "rep":
        print("🔎 Vergleiche Einfluss von: repetition_penalty (1.0, 1.15, 1.25, 1.45)\n")
        for rp in [1.0, 1.15, 1.25, 1.45]:
            cfg = {"num_beams": 4, "repetition_penalty": rp, "no_repeat_ngram_size": 3, "length_penalty": 1.0, "early_stopping": True}
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🔹 repetition_penalty = {rp}")
            print(f"🤖 {out}\n")

    elif args.sweep == "ngram":
        print("🔎 Vergleiche Einfluss von: no_repeat_ngram_size (0, 2, 3, 4)\n")
        for ng in [0, 2, 3, 4]:
            cfg = {"num_beams": 4, "repetition_penalty": 1.2, "no_repeat_ngram_size": ng, "length_penalty": 1.0, "early_stopping": True}
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🔹 no_repeat_ngram_size = {ng}")
            print(f"🤖 {out}\n")

    elif args.sweep == "sweet_spots":
        configs = [
            ("A. Kompakt & Präzise (Beam 4, LP 0.9, Rep 1.2, nGram 3)", {
                "num_beams": 4, "length_penalty": 0.9, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3, "early_stopping": True
            }),
            ("B. Standard Ausgewogen (Beam 4, LP 1.0, Rep 1.2, nGram 3)", {
                "num_beams": 4, "length_penalty": 1.0, "repetition_penalty": 1.2, "no_repeat_ngram_size": 3, "early_stopping": True
            }),
            ("C. Ausführlich mit Erklärung (Beam 5, LP 1.3, Rep 1.25, nGram 3)", {
                "num_beams": 5, "length_penalty": 1.3, "repetition_penalty": 1.25, "no_repeat_ngram_size": 3, "early_stopping": True
            }),
            ("D. Maximale Satz-Vielfalt (Beam 5, LP 1.1, Rep 1.35, nGram 4)", {
                "num_beams": 5, "length_penalty": 1.1, "repetition_penalty": 1.35, "no_repeat_ngram_size": 4, "early_stopping": True
            }),
        ]
        for label, cfg in configs:
            print("-" * 80)
            print(f"🔹 {label}")
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🤖 {out}\n")


if __name__ == "__main__":
    main()
