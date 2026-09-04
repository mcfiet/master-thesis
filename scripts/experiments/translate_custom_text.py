#!/usr/bin/env python3
"""
scripts/experiments/translate_custom_text.py

Skript zur Uebersetzung eines individuellen Eingabetextes in Leichte Sprache (LS).
Unterstuetzt frei waehlbare Modelle (Seq2Seq wie mBART oder Decoder-Only / CausalLM),
LoRA-Adapter sowie Basemodelle.

Standardmodell:
  mBART DPO len1024 (wie in der DPO Experten-Evaluierung verwendet)
  Pfad: results/models/token_length_exp/dpo_len1024
  Basis: facebook/mbart-large-50
"""

import os
import sys
import argparse
from typing import Any, Tuple
import torch


def calculate_lix(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    long_words = sum(1 for w in words if len(w) > 6)
    return round((num_words / num_sentences) + (long_words * 100 / num_words), 2)


def calculate_flesch_de(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyaeoeueAEIOUYAEIOUY"
    num_syllables = 0
    for word in words:
        count = sum(1 for char in word if char in vowels)
        num_syllables += max(1, count)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    return round(180.0 - asl - (58.5 * asw), 2)


def load_model(
    model_path: str,
    base_model_name: str,
    model_type: str,
    device: torch.device
) -> Tuple[Any, Any, str]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
    from peft import PeftModel

    print(f"\n[INFO] Lade Tokenizer und Modell:")
    print(f"       Modell-Pfad : {model_path}")
    print(f"       Basis-Modell: {base_model_name}")

    # Tokenizer laden
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    if "mbart" in model_path.lower() or "mbart" in base_model_name.lower():
        if hasattr(tokenizer, "src_lang"):
            tokenizer.src_lang = "de_DE"
        if hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = "de_DE"

    # Bestimme Modelltyp automatisch falls 'auto'
    resolved_type = model_type
    if resolved_type == "auto":
        is_seq2seq = False
        try:
            cfg = AutoConfig.from_pretrained(model_path if os.path.exists(model_path) else base_model_name)
            is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))
        except Exception:
            if "mbart" in model_path.lower() or "mbart" in base_model_name.lower() or "t5" in base_model_name.lower():
                is_seq2seq = True
        resolved_type = "seq2seq" if is_seq2seq else "causal_lm"

    print(f"       Erkannter Modell-Typ: {resolved_type}")

    # Modell laden
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if resolved_type == "seq2seq":
        has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json")) if os.path.exists(model_path) else False
        if has_adapter:
            print("       -> Lade Basis-Modell und LoRA-Adapter...")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        else:
            try:
                print("       -> Versuche direktes Laden als Standalone Seq2SeqLM...")
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype)
            except Exception as e:
                print(f"       -> Fallback auf Basis-Modell + PEFT: {e}")
                base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=dtype)
                model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
    else:
        # CausalLM / Decoder-Only
        has_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json")) if os.path.exists(model_path) else False
        if has_adapter:
            print("       -> Lade Decoder-Only Basis-Modell und LoRA-Adapter...")
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()
        else:
            try:
                print("       -> Versuche direktes Laden als Standalone CausalLM...")
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
            except Exception as e:
                print(f"       -> Fallback auf Basis-Modell + PEFT: {e}")
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=dtype)
                model = PeftModel.from_pretrained(base_model, model_path).merge_and_unload()

    model.to(device)
    model.eval()
    print("       -> Modell erfolgreich geladen und einsatzbereit.\n")
    return model, tokenizer, resolved_type


def translate_text(
    model,
    tokenizer,
    model_type: str,
    source_text: str,
    device: torch.device,
    max_source_len: int = 1024,
    max_target_len: int = 1024,
    num_beams: int = 4,
    repetition_penalty: float = 1.2,
    no_repeat_ngram_size: int = 3,
    length_penalty: float = 1.0,
    temperature: float = 1.0,
    do_sample: bool = False
) -> str:
    source_text = source_text.strip()
    if not source_text:
        return ""

    if model_type == "seq2seq":
        inputs = tokenizer(
            source_text,
            padding=True,
            truncation=True,
            max_length=max_source_len,
            return_tensors="pt"
        ).to(device)

        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_length": max_target_len,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
            "early_stopping": True,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded
    else:
        # Decoder-Only Prompting
        prompt = (
            f"Uebersetze den folgenden Text aus der Standardsprache in Leichte Sprache (A2 Niveau).\n\n"
            f"Originaltext:\n{source_text}\n\n"
            f"Uebersetzung in Leichte Sprache:\n"
        )
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_source_len,
            return_tensors="pt"
        ).to(device)

        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "max_new_tokens": max_target_len,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["num_beams"] = num_beams

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        # Schneide Prompt ab
        input_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
        return decoded


def main():
    parser = argparse.ArgumentParser(
        description="Uebersetzt benutzerdefinierten Text in Leichte Sprache mit waehlbarem Modell."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Der zu uebersetzende Text (Alltagssprache / AS)."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/experiments/translation/master_thesis_abstract.txt" if os.path.exists("data/experiments/translation/master_thesis_abstract.txt") else None,
        help="Pfad zu einer Textdatei mit dem zu uebersetzenden Text (Standard: data/experiments/translation/master_thesis_abstract.txt)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/models/token_length_exp/dpo_len1024",
        help="Pfad zum Modell/Checkpoint (Standard: DPO Modell aus der Expertenevaluierung: results/models/token_length_exp/dpo_len1024)."
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="facebook/mbart-large-50",
        help="HuggingFace Basis-Modell (z.B. facebook/mbart-large-50)."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["auto", "seq2seq", "causal_lm"],
        default="auto",
        help="Modell-Architektur ('auto', 'seq2seq' oder 'causal_lm')."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optionaler Pfad zum Speichern der Uebersetzung als Textdatei."
    )
    parser.add_argument(
        "--max_source_len",
        type=int,
        default=1024,
        help="Maximale Quelltext-Tokenlaenge (Standard: 1024)."
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=1024,
        help="Maximale Zieltext-Tokenlaenge (Standard: 1024)."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam-Search Weite (Standard: 4)."
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Repetition Penalty (Standard: 1.2)."
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=3,
        help="No Repeat N-Gram Size (Standard: 3)."
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length Penalty (Standard: 1.0)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling Temperature (Standard: 1.0)."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Aktiviert stochastisches Sampling statt deterministischer Beam-Search."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
        help="Ausfuehrungs-Device ('cuda', 'mps' oder 'cpu')."
    )

    args = parser.parse_args()

    # Eingabetext beschaffen
    input_text = ""
    if args.text:
        input_text = args.text
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"[FEHLER] Eingabedatei existiert nicht: {args.input_file}")
            sys.exit(1)
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        print("=" * 70)
        print("Kein Text uebergeben. Bitte Text eingeben (beende Eingabe mit EOF / Strg+D):")
        print("=" * 70)
        try:
            input_text = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nAbgebrochen.")
            sys.exit(0)

    input_text = input_text.strip()
    if not input_text:
        print("[FEHLER] Kein Text zur Uebersetzung angegeben.")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[INFO] Verwende Device: {device}")

    # Modell laden
    model, tokenizer, model_type = load_model(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        model_type=args.model_type,
        device=device
    )

    # Uebersetzung generieren
    print("[INFO] Generiere Uebersetzung...")
    translation = translate_text(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        source_text=input_text,
        device=device,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        temperature=args.temperature,
        do_sample=args.do_sample
    )

    # Statistiken berechnen
    words_orig = len(input_text.split())
    words_trans = len(translation.split())
    flesch_orig = calculate_flesch_de(input_text)
    flesch_trans = calculate_flesch_de(translation)
    lix_orig = calculate_lix(input_text)
    lix_trans = calculate_lix(translation)

    # Ausgabe
    separator = "=" * 70
    print("\n" + separator)
    print("QUELLTEXT (Alltagssprache):")
    print(separator)
    print(input_text)
    print("\n" + separator)
    print("UEBERSETZUNG (Leichte Sprache):")
    print(separator)
    print(translation)
    print("\n" + separator)
    print("STATISTIKEN & LESBARKEIT:")
    print(separator)
    print(f"  Wortanzahl       : {words_orig} -> {words_trans} (Verhaeltnis: {words_trans/max(1, words_orig):.2f})")
    print(f"  Flesch Reading DE: {flesch_orig} -> {flesch_trans} (Hoeher = Leichter)")
    print(f"  LIX Index        : {lix_orig} -> {lix_trans} (Niedriger = Leichter)")
    print(separator + "\n")

    # Speichern falls gewuenscht
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(translation + "\n")
        print(f"[INFO] Uebersetzung gespeichert in: {args.output_file}")


if __name__ == "__main__":
    main()
