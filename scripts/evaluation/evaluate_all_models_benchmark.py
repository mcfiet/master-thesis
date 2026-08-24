#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_all_models_benchmark.py

5-Wege-Benchmark Evaluation:
Vergleicht 5 Modellierungsansätze auf dem ungesehenen Lebenshilfe-Testset:
1. Few-Shot Baseline (Qwen/Qwen2.5-1.5B-Instruct)
2. Decoder-Only SFT (Qwen)
3. Decoder-Only DPO (Qwen)
4. Encoder-Decoder SFT (mBART-50)
5. Encoder-Decoder DPO (mBART-50)

Speichert:
- results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv (Breites Format für compare_fewshot_sft_dpo_models.ipynb)
- results/evaluation/master_benchmark_summary.csv (Aggregierte Übersicht)
"""

import os
import sys
import json
import argparse
import time
import math
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoConfig
)
from peft import PeftModel


# ==============================================================================
# REWARD MODEL ARCHITECTURE (BILSTM MIXUP REGRESSOR)
# ==============================================================================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out)


# ==============================================================================
# READABILITY & SIMPLIFICATION METRICS
# ==============================================================================
def calculate_lix(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    long_words = sum(1 for w in words if len(w) > 6)
    return (num_words / num_sentences) + (long_words * 100 / num_words)


def calculate_flesch_de(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    num_syllables = 0
    for word in words:
        count = sum(1 for char in word if char in vowels)
        num_syllables += max(1, count)
    asl = num_words / num_sentences
    asw = num_syllables / num_words
    return 180 - asl - (58.5 * asw)


def calculate_wiener(text: str) -> float:
    words = [w for w in text.split() if w.strip()]
    if not words:
        return 0.0
    num_words = len(words)
    num_sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    ms = sum(1 for w in words if sum(1 for c in w if c in vowels) >= 3) * 100 / num_words
    sl = num_words / num_sentences
    iw = sum(1 for w in words if len(w) > 6) * 100 / num_words
    return 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 2.4943


def compute_sentence_bleu(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens or not hyp_tokens:
        return 0.0
    precisions = []
    for n in range(1, 5):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            precisions.append(0.0)
            continue
        hyp_ngrams = [tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]
        ref_counts = {}
        for ng in ref_ngrams:
            ref_counts[ng] = ref_counts.get(ng, 0) + 1
        matched = 0
        for ng in hyp_ngrams:
            if ref_counts.get(ng, 0) > 0:
                matched += 1
                ref_counts[ng] -= 1
        precisions.append(matched / len(hyp_ngrams))

    if any(p == 0.0 for p in precisions):
        smoothed_p = [max(p, 1e-4) for p in precisions]
        geo_mean = math.exp(sum(0.25 * math.log(p) for p in smoothed_p))
    else:
        geo_mean = math.exp(sum(0.25 * math.log(p) for p in precisions))

    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens)))
    return bp * geo_mean


def compute_rouge_l(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == hyp_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


# ==============================================================================
# PROMPTS & IN-CONTEXT FEW SHOT
# ==============================================================================
FEW_SHOT_SYSTEM_PROMPT = """Du bist ein professioneller Übersetzer für deutsche Leichte Sprache. Deine einzige Aufgabe ist es, schwere deutsche Texte in verständliche Leichte Sprache nach den offiziellen Regeln zu übertragen.

WICHTIGE GRUNDREGELN ZUR INHALTSTREUE:
- W1 (Strikte Inhaltstreue): Übersetze NUR die Informationen, die tatsächlich im Ausgangstext stehen.
- KEINE ERFINDUNGEN: Erfinde unter keinen Umständen neue Angebote, Vereine, Broschüren, Orte, E-Mail-Adressen, Telefonnummern, Websites oder Kontaktboxen.
- Behalte alle Namen, Zahlen, Fakten und Daten aus dem Ausgangstext unverändert bei.

REGELN DER LEICHTEN SPRACHE:
- W2 (Einfache Wörter): Benutze einfache und genaue Wörter.
- W5 (Kurze Wörter): Benutze kurze, bekannte Wörter.
- W6 (Keine Abkürzungen): Schreibe alle Wörter vollständig aus.
- W7 (Verbalstil): Verwende Verben und vermeide Nominalstil.
- W8 (Aktiv): Schreibe im Aktiv.
- W9 (Kein Genitiv): Vermeide den 2. Fall (Genitiv).
- W10 (Kein Konjunktiv): Verwende keinen Konjunktiv.
- W11 (Positiv formulieren): Formuliere Aussagen positiv.

Erstelle ausschließlich die vereinfachte Übersetzung des Textes ohne zusätzliche Kommentare."""

FEW_SHOT_EXAMPLES = [
    {
        "as": "Die Bundesregierung hat beschlossen, die Fördermittel für barrierefreie Wohnungen um zwanzig Prozent zu erhöhen.",
        "ls": "Die Bundesregierung gibt mehr Geld für barrierefreie Wohnungen. Das Geld wird um 20 Prozent mehr."
    },
    {
        "as": "Aufgrund von umfangreichen Wartungsarbeiten an den Gleisanlagen ist der Zugverkehr zwischen Hamburg und Hannover bis kommenden Montag vollständig unterbrochen.",
        "ls": "An den Gleisen werden Bauarbeiten gemacht. Deshalb fahren zwischen Hamburg und Hannover keine Züge. Das dauert bis nächsten Montag."
    }
]


def create_few_shot_prompt(as_text: str, tokenizer: Any) -> str:
    messages = [{"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": f"Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n{ex['as']}"})
        messages.append({"role": "assistant", "content": ex['ls']})
    messages.append({"role": "user", "content": f"Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n{as_text}"})
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{FEW_SHOT_SYSTEM_PROMPT}\n\nText: {as_text}\n\nLeichte Sprache:"


def create_sft_dpo_prompt(as_text: str, tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Vereinfache folgenden Text in verständliche deutsche Leichte Sprache:\n\n{as_text}"}
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{FEW_SHOT_SYSTEM_PROMPT}\n\nText: {as_text}\n\nLeichte Sprache:"


# ==============================================================================
# MAIN BENCHMARK RUNNER
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="5-Way Master Benchmark Evaluation.")
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--sft_mbart_path", default="results/models/sft")
    parser.add_argument("--dpo_mbart_path", default="results/models/dpo")
    parser.add_argument("--qwen_base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sft_decoder_path", default="results/models/decoder_only/sft")
    parser.add_argument("--dpo_decoder_path", default="results/models/decoder_only/dpo")
    parser.add_argument("--reward_model_path", default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--reward_vocab_path", default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--output_csv", default="results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv")
    parser.add_argument("--output_summary", default="results/evaluation/master_benchmark_summary.csv")
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    print(f"=== 5-Wege Benchmark Runner ({args.device}) ===")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)

    # 1. Testdaten laden
    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Test-Datensatz nicht gefunden: {args.test_data_path}")

    with open(args.test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"Geladene Test-Beispiele: {len(test_data)}")

    ids = [item.get("id", i) for i, item in enumerate(test_data)]
    as_texts = [item.get("source_text", item.get("as_text", item.get("source", ""))) for item in test_data]
    ls_texts = [item.get("target_text", item.get("ls_text", item.get("target", ""))) for item in test_data]

    # 2. Reward-Modell & SBERT initialisieren
    print("Lade Reward-Modell (BiLSTM) & SBERT...")
    with open(args.reward_vocab_path, "r", encoding="utf-8") as f:
        vocab_raw = json.load(f)
        stoi = vocab_raw.get("stoi", vocab_raw)

    bilstm = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(args.device)
    raw_weights = torch.load(args.reward_model_path, map_location=args.device, weights_only=False)
    if isinstance(raw_weights, dict) and "model_state_dict" in raw_weights:
        raw_weights = raw_weights["model_state_dict"]
    bilstm.load_state_dict(raw_weights)
    bilstm.eval()

    sbert = SentenceTransformer(args.sbert_model_name, device=args.device, trust_remote_code=True)
    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    except Exception:
        nlp = spacy.blank("de")

    def calc_r_style(texts: List[str]) -> np.ndarray:
        scores = []
        for t in texts:
            doc = nlp(t)
            tokens = [tok.text.lower() for tok in doc if not tok.is_space]
            encoded = [stoi.get(tok, stoi.get("<unk>", 1)) for tok in tokens][:256]
            if len(encoded) < 256:
                encoded = encoded + [0] * (256 - len(encoded))
            tensor_in = torch.tensor([encoded], dtype=torch.long, device=args.device)
            with torch.no_grad():
                pred = bilstm(tensor_in).squeeze().item()
            scores.append(pred)
        return np.array(scores)

    def calc_r_sem(src_list: List[str], hyp_list: List[str]) -> np.ndarray:
        sims = []
        for i in range(0, len(src_list), 32):
            e_src = sbert.encode(src_list[i : i + 32], convert_to_tensor=True, show_progress_bar=False)
            e_hyp = sbert.encode(hyp_list[i : i + 32], convert_to_tensor=True, show_progress_bar=False)
            c = util.cos_sim(e_src, e_hyp).diag().cpu().numpy()
            c_norm = np.clip((c + 1.0) / 2.0, 0.0, 1.0)
            sims.extend(c_norm.tolist() if isinstance(c_norm, np.ndarray) and c_norm.ndim > 0 else [float(c_norm)])
        return np.array(sims)

    results_dict: Dict[str, Any] = {
        "id": ids,
        "source_text": as_texts,
        "target_text": ls_texts,
    }
    summary_rows = []

    def evaluate_model_outputs(model_name_tag: str, gen_texts: List[str], prefix: str):
        print(f"Berechne Metriken für {model_name_tag}...")
        r_style = calc_r_style(gen_texts)
        r_sem = calc_r_sem(as_texts, gen_texts)
        composite = 0.5 * r_style + 0.5 * r_sem

        lix_scores = [calculate_lix(t) for t in gen_texts]
        flesch_scores = [calculate_flesch_de(t) for t in gen_texts]
        wiener_scores = [calculate_wiener(t) for t in gen_texts]

        bleu_scores = []
        rouge_scores = []
        for ref, gen in zip(ls_texts, gen_texts):
            r_tokens = [tok.text.lower() for tok in nlp(ref) if not tok.is_space]
            g_tokens = [tok.text.lower() for tok in nlp(gen) if not tok.is_space]
            bleu_scores.append(compute_sentence_bleu(r_tokens, g_tokens))
            rouge_scores.append(compute_rouge_l(r_tokens, g_tokens))

        results_dict[f"gen_{prefix}"] = gen_texts
        results_dict[f"r_style_{prefix}"] = r_style
        results_dict[f"r_sem_as_{prefix}"] = r_sem
        results_dict[f"composite_{prefix}"] = composite
        results_dict[f"lix_{prefix}"] = lix_scores
        results_dict[f"flesch_{prefix}"] = flesch_scores
        results_dict[f"wiener_{prefix}"] = wiener_scores
        results_dict[f"bleu_{prefix}"] = bleu_scores
        results_dict[f"rouge_l_{prefix}"] = rouge_scores

        summary_rows.append({
            "Modell / Paradigma": model_name_tag,
            "Simplicity (R_style)": f"{np.mean(r_style):.4f} ± {np.std(r_style):.4f}",
            "SBERT-Quelltreue (R_sem)": f"{np.mean(r_sem):.4f} ± {np.std(r_sem):.4f}",
            "Composite Total Reward": f"{np.mean(composite):.4f} ± {np.std(composite):.4f}",
            "LIX Index (↓)": f"{np.mean(lix_scores):.4f} ± {np.std(lix_scores):.4f}",
            "Flesch DE (↑)": f"{np.mean(flesch_scores):.4f} ± {np.std(flesch_scores):.4f}",
            "Wiener Sachtext (↓)": f"{np.mean(wiener_scores):.4f} ± {np.std(wiener_scores):.4f}",
            "BLEU (↑)": f"{np.mean(bleu_scores):.4f} ± {np.std(bleu_scores):.4f}",
            "ROUGE-L (↑)": f"{np.mean(rouge_scores):.4f} ± {np.std(rouge_scores):.4f}",
        })

    # ==========================================================================
    # MODELL 1: Few-Shot Baseline (Qwen/Qwen2.5-1.5B-Instruct)
    # ==========================================================================
    print("\n--- [1/5] Inferenz: Few-Shot Baseline (Qwen-1.5B) ---")
    try:
        qwen_tok = AutoTokenizer.from_pretrained(args.qwen_base_model)
        qwen_m = AutoModelForCausalLM.from_pretrained(
            args.qwen_base_model,
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map=args.device if args.device != "cpu" else None
        )
        qwen_m.eval()
        gen_fs = []
        for src in tqdm(as_texts, desc="Few-Shot Inferenz"):
            prompt = create_few_shot_prompt(src, qwen_tok)
            inputs = qwen_tok(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                out = qwen_m.generate(
                    **inputs,
                    max_new_tokens=args.max_target_len,
                    do_sample=False,
                    repetition_penalty=1.15,
                    pad_token_id=qwen_tok.eos_token_id
                )
            gen_text = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            gen_fs.append(gen_text)
        del qwen_m
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"Fehler bei Few-Shot Inferenz: {e}. Verwende Fallback/Dummy.")
        gen_fs = [f"[Fehler Few-Shot]: {t}" for t in as_texts]

    evaluate_model_outputs("1. Few-Shot Baseline (Qwen-1.5B)", gen_fs, "dec_fs")

    # ==========================================================================
    # MODELL 2: Decoder-Only SFT (Qwen)
    # ==========================================================================
    print("\n--- [2/5] Inferenz: Decoder-Only SFT (Qwen) ---")
    try:
        if os.path.exists(args.sft_decoder_path):
            base_m = AutoModelForCausalLM.from_pretrained(
                args.qwen_base_model,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                device_map=args.device if args.device != "cpu" else None
            )
            sft_dec_m = PeftModel.from_pretrained(base_m, args.sft_decoder_path)
            sft_dec_m = sft_dec_m.merge_and_unload() if hasattr(sft_dec_m, "merge_and_unload") else sft_dec_m
            sft_dec_m.eval()
            gen_dec_sft = []
            for src in tqdm(as_texts, desc="Decoder SFT Inferenz"):
                prompt = create_sft_dpo_prompt(src, qwen_tok)
                inputs = qwen_tok(prompt, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    out = sft_dec_m.generate(
                        **inputs,
                        max_new_tokens=args.max_target_len,
                        do_sample=False,
                        repetition_penalty=1.15,
                        pad_token_id=qwen_tok.eos_token_id
                    )
                gen_text = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                gen_dec_sft.append(gen_text)
            del sft_dec_m, base_m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"Decoder SFT Modellpfad {args.sft_decoder_path} nicht gefunden.")
            gen_dec_sft = gen_fs
    except Exception as e:
        print(f"Fehler bei Decoder SFT Inferenz: {e}")
        gen_dec_sft = gen_fs

    evaluate_model_outputs("2. Decoder-Only SFT (Qwen-1.5B)", gen_dec_sft, "dec_sft")

    # ==========================================================================
    # MODELL 3: Decoder-Only DPO (Qwen)
    # ==========================================================================
    print("\n--- [3/5] Inferenz: Decoder-Only DPO (Qwen) ---")
    try:
        if os.path.exists(args.dpo_decoder_path):
            base_m = AutoModelForCausalLM.from_pretrained(
                args.qwen_base_model,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
                device_map=args.device if args.device != "cpu" else None
            )
            dpo_dec_m = PeftModel.from_pretrained(base_m, args.dpo_decoder_path)
            dpo_dec_m = dpo_dec_m.merge_and_unload() if hasattr(dpo_dec_m, "merge_and_unload") else dpo_dec_m
            dpo_dec_m.eval()
            gen_dec_dpo = []
            for src in tqdm(as_texts, desc="Decoder DPO Inferenz"):
                prompt = create_sft_dpo_prompt(src, qwen_tok)
                inputs = qwen_tok(prompt, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    out = dpo_dec_m.generate(
                        **inputs,
                        max_new_tokens=args.max_target_len,
                        do_sample=False,
                        repetition_penalty=1.15,
                        pad_token_id=qwen_tok.eos_token_id
                    )
                gen_text = qwen_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                gen_dec_dpo.append(gen_text)
            del dpo_dec_m, base_m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"Decoder DPO Modellpfad {args.dpo_decoder_path} nicht gefunden.")
            gen_dec_dpo = gen_dec_sft
    except Exception as e:
        print(f"Fehler bei Decoder DPO Inferenz: {e}")
        gen_dec_dpo = gen_dec_sft

    evaluate_model_outputs("3. Decoder-Only DPO (Qwen-1.5B)", gen_dec_dpo, "dec_dpo")

    # ==========================================================================
    # MODELL 4: Encoder-Decoder SFT (mBART-50)
    # ==========================================================================
    print("\n--- [4/5] Inferenz: Encoder-Decoder SFT (mBART-50) ---")
    try:
        mbart_tok = AutoTokenizer.from_pretrained("facebook/mbart-large-50")
        mbart_tok.src_lang = "de_DE"
        mbart_tok.tgt_lang = "de_DE"

        if os.path.exists(args.sft_mbart_path):
            sft_mbart_m = AutoModelForSeq2SeqLM.from_pretrained(
                args.sft_mbart_path,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
            ).to(args.device)
            sft_mbart_m.eval()
            gen_enc_sft = []
            for i in tqdm(range(0, len(as_texts), args.batch_size), desc="mBART SFT Inferenz"):
                batch_src = as_texts[i : i + args.batch_size]
                inputs = mbart_tok(batch_src, max_length=args.max_source_len, padding=True, truncation=True, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    outs = sft_mbart_m.generate(
                        **inputs,
                        max_length=args.max_target_len,
                        num_beams=4,
                        forced_bos_token_id=mbart_tok.lang_code_to_id.get("de_DE", None),
                        repetition_penalty=1.2,
                        early_stopping=True
                    )
                gen_enc_sft.extend(mbart_tok.batch_decode(outs, skip_special_tokens=True))
            del sft_mbart_m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"mBART SFT Modellpfad {args.sft_mbart_path} nicht gefunden.")
            gen_enc_sft = [f"[mBART SFT nicht gefunden]: {t}" for t in as_texts]
    except Exception as e:
        print(f"Fehler bei mBART SFT Inferenz: {e}")
        gen_enc_sft = [f"[Fehler mBART SFT]: {t}" for t in as_texts]

    evaluate_model_outputs("4. Encoder-Decoder SFT (mBART-50)", gen_enc_sft, "enc_sft")

    # ==========================================================================
    # MODELL 5: Encoder-Decoder DPO (mBART-50)
    # ==========================================================================
    print("\n--- [5/5] Inferenz: Encoder-Decoder DPO (mBART-50) ---")
    try:
        if os.path.exists(args.dpo_mbart_path):
            dpo_mbart_m = AutoModelForSeq2SeqLM.from_pretrained(
                args.dpo_mbart_path,
                torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
            ).to(args.device)
            dpo_mbart_m.eval()
            gen_enc_dpo = []
            for i in tqdm(range(0, len(as_texts), args.batch_size), desc="mBART DPO Inferenz"):
                batch_src = as_texts[i : i + args.batch_size]
                inputs = mbart_tok(batch_src, max_length=args.max_source_len, padding=True, truncation=True, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    outs = dpo_mbart_m.generate(
                        **inputs,
                        max_length=args.max_target_len,
                        num_beams=4,
                        forced_bos_token_id=mbart_tok.lang_code_to_id.get("de_DE", None),
                        repetition_penalty=1.2,
                        early_stopping=True
                    )
                gen_enc_dpo.extend(mbart_tok.batch_decode(outs, skip_special_tokens=True))
            del dpo_mbart_m
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"mBART DPO Modellpfad {args.dpo_mbart_path} nicht gefunden.")
            gen_enc_dpo = gen_enc_sft
    except Exception as e:
        print(f"Fehler bei mBART DPO Inferenz: {e}")
        gen_enc_dpo = gen_enc_sft

    evaluate_model_outputs("5. Encoder-Decoder DPO (mBART-50)", gen_enc_dpo, "enc_dpo")

    # ==========================================================================
    # DATEN EXPORTIEREN
    # ==========================================================================
    df_eval = pd.DataFrame(results_dict)
    df_eval.to_csv(args.output_csv, index=False)
    print(f"\n[ERFOLG] 5-Wege Benchmark Details gespeichert in: {args.output_csv}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.output_summary, index=False)
    print(f"[ERFOLG] Master Benchmark Summary gespeichert in: {args.output_summary}")

    print("\n" + "=" * 90)
    print("MASTER 5-WEGE-BENCHMARK GESAMT-ÜBERSICHT")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)


if __name__ == "__main__":
    main()

