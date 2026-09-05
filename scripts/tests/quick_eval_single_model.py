#!/usr/bin/env python3
"""
scripts/tests/quick_eval_single_model.py

Schnelltest für ein einzelnes trainiertes DPO-Modell auf Lebenshilfe-Beispielen.
Gibt die Übersetzungen direkt farbig / formatiert im Terminal aus.
"""

import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "results/models/test_single_dpo"
    base_model = sys.argv[2] if len(sys.argv) > 2 else "results/models/sft"
    test_file = sys.argv[3] if len(sys.argv) > 3 else "data/lebenshilfe/lebenshilfe_dataset_clean.json"

    print("=" * 80)
    print(f"Schnelltest für Modell: {model_path}")
    print(f"   Basis-Modell: {base_model}")
    print(f"   Test-Datei: {test_file}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Nutze Device: {device}")

    # 1. Tokenizer laden
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    # 2. Modell laden
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("Lade Basismodell und merge LoRA-Adapter...")
        bm = AutoModelForSeq2SeqLM.from_pretrained(base_model).to(device)
        model = PeftModel.from_pretrained(bm, model_path).merge_and_unload().to(device)
    else:
        print("Lade Standalone-Modell...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    # 3. Testdaten laden
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data[:5]  # Erste 5 Beispiele
    print(f"\nGeneriere Übersetzungen für {len(samples)} Testbeispiele...\n")

    for idx, item in enumerate(samples):
        src = item.get("source_text", item.get("as_text", ""))
        ref = item.get("target_text", item.get("ls_text", ""))

        inputs = tokenizer(
            [src],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_length": 256,
            "num_beams": 4,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

        with torch.no_grad():
            outs = model.generate(**gen_kwargs)
        pred = tokenizer.decode(outs[0], skip_special_tokens=True).strip()

        print("-" * 80)
        print(f"[BEISPIEL {idx + 1}]")
        print(f"QUELLTEXT (AS): {src[:150]}...")
        print(f"REFERENZ (LS):  {ref[:150]}...")
        print(f"GENERIERT:     {pred}")
        print("-" * 80)


if __name__ == "__main__":
    main()
