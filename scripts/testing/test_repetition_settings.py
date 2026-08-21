#!/usr/bin/env python3
"""
=============================================================================
Quick Test Script: Repetition Penalty & N-Gram Blocking Comparison
=============================================================================
Compares different decoding configurations side-by-side on a sample text to
measure the direct impact of:
  - repetition_penalty (1.2 vs 1.4 vs 1.6)
  - no_repeat_ngram_size (None vs 3)
  - temperature (0.9 vs 0.8 vs 0.7)
=============================================================================
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from peft import PeftModel

DEFAULT_SAMPLE_TEXT = (
    "Viel zusätzliche Förderung für Flüchtlingskinder und andere \"Seiteneinsteiger\". "
    "Wien – In der Debatte um die Integration von Flüchtlings- und Migrantenkindern gilt Hamburg als eines der Vorbilder. "
    "Das deutsche Bundesland setzt dabei unter anderem auf einen Sozialindex für Schulen, nach dem etwa Lehrerstunden vergeben werden, "
    "sowie zeitlich begrenzte Vorbereitungsklassen für Quereinsteiger, schilderte die Erziehungswissenschafterin Ursula Neumann (Uni Hamburg) "
    "am Donnerstagabend bei einer Diskussionsveranstaltung auf Einladung der Initiative Bildung grenzenlos und dem Institut für Germanistik der Universität Wien. "
    "Die Situation in Hamburg sei mit jener in Wien etwa vergleichbar, befand Neumann: In Wien haben knapp 40 Prozent der Schüler eine andere Muttersprache als Deutsch, "
    "in Hamburg verfügen 45 Prozent über Migrationshintergrund. Die Sprachförderung an den Schulen beruhe auf zwei Meilensteinen: "
    "Einerseits auf dem 2006 verabschiedeten Hamburger Sprachförderkonzept sowie anderseits auf der Erarbeitung eines schulspezifischen Sozialindex: "
    "Je nach Zusammensetzung der Schülerschaft entscheidet dieser etwa über die Größe der Klassen beziehungsweise die Zuteilung von Lehrerstunden und Unterstützungspersonal."
)


def load_model_and_tokenizer(model_path: str, base_model_name: str, device: str):
    print(f"Loading model from: {model_path}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if hasattr(tokenizer, "src_lang") or hasattr(tokenizer, "lang_code_to_id"):
        tokenizer.src_lang = "de_DE"
        tokenizer.tgt_lang = "de_DE"

    has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if has_adapter:
        print(f"Merging LoRA adapter onto {base_model_name}...")
        base_m = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
        peft_m = PeftModel.from_pretrained(base_m, model_path)
        model = peft_m.merge_and_unload()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)

    model = model.to(device)
    model.eval()
    return model, tokenizer


def analyze_repetition(text: str):
    words = text.lower().split()
    if len(words) < 3:
        return 1.0, 0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    unique_trigrams = len(set(trigrams))
    total_trigrams = len(trigrams)
    ratio = unique_trigrams / max(1, total_trigrams)

    # Count sentence repeats
    sents = [s.strip() for s in text.replace("\n", ".").split(".") if len(s.strip()) > 10]
    repeated_sents = len(sents) - len(set(sents))
    return ratio, repeated_sents


def test_configurations(model, tokenizer, prompt: str, device: str):
    # Test configurations
    configs = [
        {
            "name": "1. Aktuelle Einstellung (Problem-Setting)",
            "temp": 0.9,
            "rep_penalty": 1.2,
            "no_repeat_ngram": 0,
            "top_p": 0.92,
        },
        {
            "name": "2. Nur Repetition Penalty erhöht (1.5)",
            "temp": 0.8,
            "rep_penalty": 1.5,
            "no_repeat_ngram": 0,
            "top_p": 0.92,
        },
        {
            "name": "3. Nur Repetition Penalty stark erhöht (1.8)",
            "temp": 0.8,
            "rep_penalty": 1.8,
            "no_repeat_ngram": 0,
            "top_p": 0.92,
        },
        {
            "name": "4. Mit No-Repeat-Ngram (no_repeat_ngram_size=3)",
            "temp": 0.8,
            "rep_penalty": 1.2,
            "no_repeat_ngram": 3,
            "top_p": 0.92,
        },
        {
            "name": "5. Kombinierte Best-Practice (rep=1.35 + ngram=3 + temp=0.75)",
            "temp": 0.75,
            "rep_penalty": 1.35,
            "no_repeat_ngram": 3,
            "top_p": 0.92,
        },
    ]

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=500,
    ).to(device)

    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and "de_DE" in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]

    print("\n" + "=" * 80)
    print("PROMPT (Eingabetext):")
    print(prompt[:300] + "...\n" + "=" * 80)

    for cfg in configs:
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "do_sample": True,
            "temperature": cfg["temp"],
            "top_p": cfg["top_p"],
            "top_k": 50,
            "repetition_penalty": cfg["rep_penalty"],
            "max_length": 500,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if cfg["no_repeat_ngram"] > 0:
            gen_kwargs["no_repeat_ngram_size"] = cfg["no_repeat_ngram"]
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

        with torch.no_grad():
            output = model.generate(**gen_kwargs)

        decoded = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        trigram_ratio, rep_sents = analyze_repetition(decoded)

        print(f"\n[{cfg['name']}]")
        print(f"Parameter: Temp={cfg['temp']} | Rep-Penalty={cfg['rep_penalty']} | No-Repeat-Ngram={cfg['no_repeat_ngram']}")
        print(f"Qualität: Wörter={len(decoded.split())} | Trigram-Einzigartigkeit={trigram_ratio*100:.1f}% | Satz-Wiederholungen={rep_sents}")
        print("-" * 60)
        print(decoded)
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test repetition settings on SFT model.")
    parser.add_argument("--model_path", type=str, default="results/models/token_length_exp/sft_len500")
    parser.add_argument("--base_model_name", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--sample_text", type=str, default=DEFAULT_SAMPLE_TEXT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model_name, device)
    test_configurations(model, tokenizer, args.sample_text, device)


if __name__ == "__main__":
    main()
