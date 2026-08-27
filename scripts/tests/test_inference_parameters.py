#!/usr/bin/env python3
"""
scripts/tests/test_inference_parameters.py

Interaktiver Vergleich verschiedener Inferenz- und Decoding-Parameter
(Beam Search, Sampling, Length Penalty, Repetition Penalty, n-Gram Penalty)
auf echten Lebenshilfe-Testbeispielen.
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
    parser = argparse.ArgumentParser(description="Inference Parameter Playground")
    parser.add_argument("--model_path", default="results/models/test_single_dpo", help="Pfad zum Modell")
    parser.add_argument("--base_model", default="results/models/sft", help="Pfad zum Basismodell / Tokenizer")
    parser.add_argument("--test_file", default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Testset JSON")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index des Testbeispiels (0 bis N)")
    parser.add_argument("--mode", default="grid", choices=["grid", "custom"], help="Modus: 'grid' (Presets vergleichen) oder 'custom'")
    
    # Custom Args
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

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
    print(f"▶️ QUELLTEXT (AS): {src[:200]}...")
    print(f"✅ REFERENZ (LS):  {ref[:200]}...")
    print("=" * 80 + "\n")

    if args.mode == "custom":
        gen_kw = {
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "length_penalty": args.length_penalty,
        }
        if args.do_sample:
            gen_kw["temperature"] = args.temperature
            gen_kw["top_p"] = args.top_p
            gen_kw["top_k"] = args.top_k
        else:
            gen_kw["num_beams"] = args.num_beams
            gen_kw["early_stopping"] = True

        out = generate_with_config(model, tokenizer, src, device, gen_kw)
        print(f"⚙️ Parameter: {gen_kw}")
        print(f"🤖 GENERIERT:\n{out}\n")

    else:
        # GRID COMPARISON: Verschiedene Strategien im direkten Vergleich
        configs = [
            ("1. Standard Beam 4 (Aktuell)", {
                "num_beams": 4,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 1.0,
            }),
            ("2. Beam 5 mit Length Penalty 1.2 (Längere Ausführungen)", {
                "num_beams": 5,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 1.2,
            }),
            ("3. Beam 4 mit höherer Repetition Penalty (1.35)", {
                "num_beams": 4,
                "repetition_penalty": 1.35,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 1.0,
            }),
            ("4. Greedy Search (Schnell, Deterministisch)", {
                "num_beams": 1,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
            }),
            ("5. Sampling (Temp=0.7, Top-p=0.92, Kreativer)", {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.92,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }),
            ("6. Konservatives Sampling (Temp=0.3, Top-p=0.9)", {
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }),
        ]

        for name, cfg in configs:
            print("-" * 80)
            print(f"🔹 {name}")
            print(f"   Config: {cfg}")
            out = generate_with_config(model, tokenizer, src, device, cfg)
            print(f"🤖 Output:\n{out}\n")


if __name__ == "__main__":
    main()
