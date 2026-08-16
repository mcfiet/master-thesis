#!/usr/bin/env python3
"""
=============================================================================
DPO Tokenisierungs- & Grenz-Analyse (TRL vs. Nativ Seq2Seq / mBART)
=============================================================================
Dieses Skript analysiert und visualisiert das Verhalten des Tokenizers bei
der Aufbereitung von DPO-Trainingsdaten (Alltagssprache -> Leichte Sprache).

Es zeigt:
  1. Warum das EOS-Token (</s>) beim separaten Tokenisieren des Prompts
     zu einem 1-Token-Versatz (Off-by-One) in TRL führt.
  2. Welches Wort am Anfang der Antwort (Chosen / Rejected) durch TRL
     abgeschnitten bzw. wegmaskiert wird.
  3. Wie die native Seq2Seq-Verarbeitung in 6_train_dpo.py dieses Problem
     vollständig vermeidet.
=============================================================================
"""

import os
import sys
import argparse
from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM

# ANSI Farbcodes für übersichtliche Terminal-Ausgabe
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"

SAMPLE_PAIRS = [
    {
        "name": "Beispiel 1 (Bundestag / Parlament)",
        "prompt": "Der Bundestag hat heute nach langer Debatte ein neues Gesetz zur Förderung erneuerbarer Energien beschlossen.",
        "chosen": "Das Parlament hat heute über ein neues Gesetz gesprochen. Das Gesetz soll mehr Öko-Strom fördern.",
        "rejected": "Der Bundestag beschloss heute debattiert ein erneuerbares Gesetz."
    },
    {
        "name": "Beispiel 2 (Bundesregierung / Natur-Schutz)",
        "prompt": "Die Bundesregierung plant umfassende Maßnahmen zur Bekämpfung des weltweiten Klimawandels.",
        "chosen": "Die Politik möchte mehr für den Natur-Schutz tun. Alle Länder sollen zusammen helfen.",
        "rejected": "Die Regierung plant Maßnahmen gegen das Klima."
    },
    {
        "name": "Beispiel 3 (Gerichtsurteil / Richter)",
        "prompt": "Das Bundesverfassungsgericht erklärte die bisherige Regelung für verfassungswidrig und nichtig.",
        "chosen": "Ein hohes Gericht hat entschieden: Das alte Gesetz war nicht in Ordnung.",
        "rejected": "Das Gericht erklärt die Regelung für nichtig."
    }
]

def print_banner(title: str):
    print("\n" + "=" * 85)
    print(f"{C_BOLD}{C_CYAN}{title.center(85)}{C_RESET}")
    print("=" * 85)

def analyze_pair(tokenizer, sample: Dict[str, str], pair_idx: int):
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    rejected = sample["rejected"]
    
    print_banner(f"{sample['name']}")
    print(f"{C_BOLD}Prompt (Alltagssprache):{C_RESET}    \"{prompt}\"")
    print(f"{C_BOLD}Chosen (Leichte Sprache):{C_RESET}   \"{chosen}\"")
    print(f"{C_BOLD}Rejected (Schlechte Var.):{C_RESET} \"{rejected}\"")
    print("-" * 85)

    # 1. Separate Tokenisierung
    tok_prompt = tokenizer(prompt)["input_ids"]
    tok_chosen = tokenizer(chosen)["input_ids"]
    
    # 2. TRL-Konkatenierung
    tok_concat = tokenizer(prompt + " " + chosen)["input_ids"]
    
    prompt_tokens = tokenizer.convert_ids_to_tokens(tok_prompt)
    chosen_tokens = tokenizer.convert_ids_to_tokens(tok_chosen)
    concat_tokens = tokenizer.convert_ids_to_tokens(tok_concat)
    
    p_len = len(tok_prompt)
    
    print(f"\n{C_BOLD}1. Längen-Analyse & Special-Tokens:{C_RESET}")
    print(f"  • Token-Anzahl Prompt:            {C_YELLOW}{p_len}{C_RESET} Tokens")
    print(f"  • Letztes Token des Prompts:      Index {p_len - 1} = {C_RED}'{prompt_tokens[-1]}' (ID: {tok_prompt[-1]}){C_RESET}  <-- EOS-Token (</s>)")
    print(f"  • Erstes Token von Chosen:        Index 1  = {C_GREEN}'{chosen_tokens[1]}' (ID: {tok_chosen[1]}){C_RESET}  <-- Erstes echtes Inhaltswort")

    print(f"\n{C_BOLD}2. Der TRL Off-by-One Mismatch an der Schnittstelle:{C_RESET}")
    print(f"  • TRL berechnet: {C_YELLOW}prompt_length = {p_len}{C_RESET} (inkl. EOS-Token </s>)")
    print(f"  • TRL-Annahme:   {C_YELLOW}Maskiere Index 0 bis {p_len - 1} als Prompt, starte Loss ab Index {p_len}.{C_RESET}")
    
    lost_token_idx = p_len - 1
    lost_token_id = tok_concat[lost_token_idx]
    lost_token_str = concat_tokens[lost_token_idx]
    
    trl_start_token_id = tok_concat[p_len]
    trl_start_token_str = concat_tokens[p_len]

    print(f"  • Im konkatenierten String an Index {lost_token_idx}: {C_RED}'{lost_token_str}' (ID: {lost_token_id}){C_RESET}")
    print(f"  • Tatsächlicher TRL-Loss-Start ab Index {p_len}:    {C_BLUE}'{trl_start_token_str}' (ID: {trl_start_token_id}){C_RESET}")
    
    print(f"\n  {C_RED}{C_BOLD}❌ FEHLERBEWEIS:{C_RESET} Das Wort {C_RED}'{lost_token_str}'{C_RESET} wird von TRL als Teil des Prompts wegmaskiert!")
    print(f"     -> Das Modell erhält für das allererste Wort {C_RED}'{lost_token_str}'{C_RESET} keinen Loss-Gradienten.")

    # 3. Detaillierte Token-Tabelle an der Schnittstelle
    print(f"\n{C_BOLD}3. Detaillierte Token-Gegenüberstellung an der Nahtstelle:{C_RESET}")
    print(f"{'Index':<7} | {'Prompt einzeln':<22} | {'Konkateniert (TRL)':<22} | {'TRL Maske':<15} | {'Nativ 6_train_dpo.py'}")
    print("-" * 95)
    
    start_view = max(0, p_len - 4)
    end_view = min(len(tok_concat), p_len + 4)
    
    for idx in range(start_view, end_view):
        p_tok = prompt_tokens[idx] if idx < p_len else "-"
        c_tok = concat_tokens[idx] if idx < len(concat_tokens) else "-"
        
        if idx < p_len - 1:
            trl_status = f"{C_YELLOW}Prompt (0){C_RESET}"
            native_status = f"{C_YELLOW}Encoder Input{C_RESET}"
        elif idx == p_len - 1:
            trl_status = f"{C_RED}{C_BOLD}Maskiert (0) ❌{C_RESET}"
            native_status = f"{C_GREEN}{C_BOLD}Decoder Target 1 ✅{C_RESET}"
        elif idx == p_len:
            trl_status = f"{C_GREEN}Loss Start (1){C_RESET}"
            native_status = f"{C_GREEN}Decoder Target 2 ✅{C_RESET}"
        else:
            trl_status = f"{C_GREEN}Loss (1){C_RESET}"
            native_status = f"{C_GREEN}Decoder Target ✅{C_RESET}"
            
        highlight = C_BOLD if idx in [p_len - 1, p_len] else ""
        print(f"{highlight}{idx:<7}{C_RESET} | {highlight}{p_tok:<22}{C_RESET} | {highlight}{c_tok:<22}{C_RESET} | {trl_status:<24} | {native_status}")

    print("-" * 95)

def run_forward_loss_comparison(tokenizer, sample: Dict[str, str]):
    print_banner("4. Mathematischer Log-Likelihood & Gradienten-Check")
    print("Erstelle temporäres mBART-Modell für exakten Forward-Pass-Vergleich...")
    
    model_name = "facebook/mbart-large-50"
    config = AutoConfig.from_pretrained(model_name)
    config.encoder_layers = 2
    config.decoder_layers = 2
    config.d_model = 256
    config.encoder_ffn_dim = 512
    config.decoder_ffn_dim = 512
    config.encoder_attention_heads = 4
    config.decoder_attention_heads = 4
    config.is_encoder_decoder = True
    
    model = AutoModelForSeq2SeqLM.from_config(config)
    model.eval()
    
    prompt = sample["prompt"]
    chosen = sample["chosen"]
    
    # Nativer Vorwärtspass (6_train_dpo.py)
    enc_prompt = tokenizer(prompt, return_tensors="pt")
    dec_chosen = tokenizer(chosen, return_tensors="pt")
    
    labels = dec_chosen["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        native_out = model(
            input_ids=enc_prompt["input_ids"],
            attention_mask=enc_prompt["attention_mask"],
            labels=labels
        )
        logits = native_out.logits # (1, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        
        mask = (labels != -100)
        labels_clamped = labels.clone()
        labels_clamped[~mask] = 0
        per_token_logps = torch.gather(log_probs, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)[0]
        
        # Erstes Token des Zielsatzes
        first_token_id = labels[0, 1].item()
        first_token_str = tokenizer.convert_ids_to_tokens([first_token_id])[0]
        first_token_logp = per_token_logps[1].item()
        
        total_native_logp = (per_token_logps * mask[0]).sum().item()
        
        # TRL abgeschnittener Log-Likelihood (ohne Token 1)
        trl_logp_missing_first = (per_token_logps[2:] * mask[0, 2:]).sum().item()

    print(f"\n  • Gesamter Log-Likelihood im nativen Seq2Seq:       {C_GREEN}{total_native_logp:.4f}{C_RESET}")
    print(f"  • Log-Likelihood des ersten Tokens ({C_BOLD}'{first_token_str}'{C_RESET}):        {C_YELLOW}{first_token_logp:.4f}{C_RESET}")
    print(f"  • Von TRL erfasster Log-Likelihood (ab Token 2):    {C_RED}{trl_logp_missing_first:.4f}{C_RESET}")
    print(f"  • {C_RED}{C_BOLD}Verlorener Wahrscheinlichkeits-Anteil:{C_RESET}            {C_RED}{abs(first_token_logp):.4f} ({abs(first_token_logp/total_native_logp)*100:.1f} % des Gesamtsatzes){C_RESET}")

def main():
    parser = argparse.ArgumentParser(description="DPO Tokenisierungs- und Grenz-Analyse")
    parser.add_argument("--model_name", type=str, default="facebook/mbart-large-50", help="Modell/Tokenizer Name")
    args = parser.parse_args()

    print_banner(f"DPO TOKENISIERUNGS- & GRENZ-ANALYSE ({args.model_name})")
    print(f"Lade Tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

    for i, sample in enumerate(SAMPLE_PAIRS):
        analyze_pair(tokenizer, sample, i + 1)

    # Führe numerischen Vorwärtspass für das erste Beispiel durch
    run_forward_loss_comparison(tokenizer, SAMPLE_PAIRS[0])

if __name__ == "__main__":
    main()
