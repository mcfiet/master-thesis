#!/usr/bin/env python3
"""
=============================================================================
Quantitative Rule-Adherence Auditor for Leichte Sprache (LS)
=============================================================================
Calculates comprehensive linguistic rule metrics on both Source (AS) and
Target (LS / Model translations), along with exact relative and absolute
reduction rates:
- Syntax: Sentence length, long sentences, subordination ratio, SPO order
- Grammar: Passive voice, genitive case, subjunctive mood, nominalization,
  verb-to-noun ratio, negation density
- Lexicon: Word length, polysyllable ratio, compound hyphenation, abbreviations,
  digit ratio, lexical diversity (MATTR)
- Readability: Wiener Sachtextformel, Flesch Reading Ease (DE), LIX
=============================================================================
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import spacy
import pyphen
import textstat
from tqdm import tqdm

# Ensure German configuration
textstat.set_lang('de')


class LeichteSpracheRuleAuditor:
    def __init__(self, spacy_model: str = "de_core_news_lg"):
        print(f"Loading SpaCy model: {spacy_model}...")
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception:
            print(f"Warning: {spacy_model} not found, falling back to de_core_news_sm or blank.")
            try:
                self.nlp = spacy.load("de_core_news_sm")
            except Exception:
                self.nlp = spacy.blank("de")
                self.nlp.add_pipe("sentencizer")

        self.dic = pyphen.Pyphen(lang='de_DE')
        self.nominal_suffixes = re.compile(r"(ung|keit|heit|schaft|tion|tät|ismus|ment|nis)$", re.IGNORECASE)
        self.abbr_pattern = re.compile(r"\b(z\.\s*B\.|bzw\.|d\.\s*h\.|u\.\s*a\.|usw\.|ca\.|evtl\.|vgl\.|inkl\.|ggf\.|dr\.|prof\.)\b", re.IGNORECASE)
        self.acronym_pattern = re.compile(r"\b[A-ZÄÖÜ]{2,}\b")
        self.negation_pattern = re.compile(r"\b(nicht|kein[a-z]*|nie|niemals|weder|nirgends|niemand[a-z]*)\b", re.IGNORECASE)
        self.written_numbers_pattern = re.compile(r"\b(eins|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|dreizehn|vierzehn|fünfzehn|sechzehn|siebzehn|achtzehn|neunzehn|zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig|hundert|tausend|million[a-z]*|milliarde[a-z]*)\b", re.IGNORECASE)
        self.digit_pattern = re.compile(r"\b\d+\b")
        self.hyphen_compound_pattern = re.compile(r"[A-ZÄÖÜa-zäöüß]+[-·][A-ZÄÖÜa-zäöüß]+")

    def count_syllables(self, word: str) -> int:
        clean_word = re.sub(r"[^\w]", "", word)
        if not clean_word:
            return 1
        hyphenated = self.dic.inserted(clean_word)
        return max(1, hyphenated.count('-') + 1)

    def analyze_doc(self, doc) -> Dict[str, float]:
        total_sents = len(list(doc.sents)) or 1
        tokens_no_punct = [t for t in doc if not t.is_punct and not t.is_space]
        total_tokens = len(tokens_no_punct)
        
        if total_tokens == 0:
            return {
                "sent_count": 0, "token_count": 0, "avg_sent_len": 0.0, "long_sent_ratio": 0.0,
                "subord_ratio": 0.0, "subject_initial_ratio": 0.0, "passive_count": 0,
                "passive_ratio": 0.0, "genitive_count": 0, "genitive_ratio": 0.0,
                "subjunctive_count": 0, "subjunctive_ratio": 0.0, "nominal_count": 0,
                "nominal_ratio": 0.0, "verb_to_noun_ratio": 0.0, "negation_density": 0.0,
                "avg_word_len": 0.0, "polysyllable_ratio": 0.0, "hyphen_compound_ratio": 0.0,
                "abbr_ratio": 0.0, "digit_ratio": 0.0, "flesch_score": 0.0, "wstf_score": 0.0,
                "lix_score": 0.0, "mattr_50": 0.0
            }

        words = [t.text for t in tokens_no_punct]
        
        # 1. Sentence Complexity
        long_sents = 0
        subject_initial_sents = 0
        subord_clauses = 0

        for sent in doc.sents:
            sent_tokens = [t for t in sent if not t.is_punct and not t.is_space]
            if len(sent_tokens) > 12:
                long_sents += 1
            if sent_tokens:
                # Check subject-initial
                first_tok = sent_tokens[0]
                if first_tok.dep_ in ["sb", "sb_pass", "sp"] or any(child.dep_ in ["sb", "sb_pass"] for child in first_tok.children):
                    subject_initial_sents += 1

        for token in doc:
            if token.pos_ == "SCONJ" or token.dep_ in ["relcl", "advcl", "ccomp"]:
                subord_clauses += 1

        avg_sent_len = total_tokens / total_sents
        long_sent_ratio = long_sents / total_sents
        subord_ratio = subord_clauses / total_sents
        subject_initial_ratio = subject_initial_sents / total_sents

        # 2. Grammar & Morphosyntax
        passive_count = 0
        genitive_count = 0
        subjunctive_count = 0
        nominal_count = 0
        total_verbs = 0
        total_nouns = 0
        total_nouns_and_pron = 0
        hyphen_compounds = 0

        for token in doc:
            if token.is_punct or token.is_space:
                continue
            
            morph_str = str(token.morph)
            
            # Verbs
            if token.pos_ in ["VERB", "AUX"]:
                total_verbs += 1
                if "Mood=Sub" in morph_str:
                    subjunctive_count += 1
                    
            # Passives
            if token.dep_ in ["sb_pass", "oc_pass"]:
                passive_count += 1
            elif token.lemma_ == "werden" and token.pos_ == "AUX":
                # Check if auxiliary is part of passive
                if any(child.pos_ == "VERB" and ("VerbForm=Part" in str(child.morph) or child.tag_ == "VVPP") for child in token.head.children) or (token.head.pos_ == "VERB" and "VerbForm=Part" in str(token.head.morph)):
                    passive_count += 1

            # Nouns / Pronouns & Genitives
            if token.pos_ in ["NOUN", "PROPN"]:
                total_nouns += 1
                total_nouns_and_pron += 1
                if "Case=Gen" in morph_str:
                    genitive_count += 1
                if self.nominal_suffixes.search(token.text):
                    nominal_count += 1
                if self.hyphen_compound_pattern.search(token.text):
                    hyphen_compounds += 1
            elif token.pos_ == "PRON":
                total_nouns_and_pron += 1
                if "Case=Gen" in morph_str:
                    genitive_count += 1

        passive_ratio = passive_count / total_sents
        genitive_ratio = genitive_count / max(1, total_nouns_and_pron)
        subjunctive_ratio = subjunctive_count / max(1, total_verbs)
        nominal_ratio = nominal_count / total_tokens
        verb_to_noun_ratio = total_verbs / max(1, total_nouns)
        hyphen_compound_ratio = hyphen_compounds / max(1, total_nouns)

        # 3. Lexicon & Words
        raw_text = doc.text
        negations = len(self.negation_pattern.findall(raw_text))
        negation_density = negations / total_sents

        abbrevs = len(self.abbr_pattern.findall(raw_text)) + len(self.acronym_pattern.findall(raw_text))
        abbr_ratio = abbrevs / max(1, len(words))

        digits = len(self.digit_pattern.findall(raw_text))
        written_nums = len(self.written_numbers_pattern.findall(raw_text))
        digit_ratio = digits / max(1, (digits + written_nums)) if (digits + written_nums) > 0 else 1.0

        avg_word_len = sum(len(w) for w in words) / max(1, len(words))
        
        # Syllables
        syllable_counts = [self.count_syllables(w) for w in words]
        polysyllable_words = sum(1 for sc in syllable_counts if sc >= 3)
        polysyllable_ratio = polysyllable_words / max(1, len(words))

        # MATTR-50
        lemmas = [token.lemma_.lower() for token in tokens_no_punct]
        mattr_50 = self.calculate_mattr(lemmas, window_size=50)

        # 4. Classical Readability
        try:
            flesch_score = float(textstat.flesch_reading_ease(raw_text))
        except Exception:
            flesch_score = 0.0
        try:
            wstf_score = float(textstat.wiener_sachtextformel(raw_text, variant=1))
        except Exception:
            wstf_score = 12.0
        try:
            lix_score = float(textstat.lix(raw_text))
        except Exception:
            lix_score = 50.0

        return {
            "sent_count": total_sents,
            "token_count": total_tokens,
            "avg_sent_len": float(avg_sent_len),
            "long_sent_ratio": float(long_sent_ratio),
            "subord_ratio": float(subord_ratio),
            "subject_initial_ratio": float(subject_initial_ratio),
            "passive_count": int(passive_count),
            "passive_ratio": float(passive_ratio),
            "genitive_count": int(genitive_count),
            "genitive_ratio": float(genitive_ratio),
            "subjunctive_count": int(subjunctive_count),
            "subjunctive_ratio": float(subjunctive_ratio),
            "nominal_count": int(nominal_count),
            "nominal_ratio": float(nominal_ratio),
            "verb_to_noun_ratio": float(verb_to_noun_ratio),
            "negation_density": float(negation_density),
            "avg_word_len": float(avg_word_len),
            "polysyllable_ratio": float(polysyllable_ratio),
            "hyphen_compound_ratio": float(hyphen_compound_ratio),
            "abbr_ratio": float(abbr_ratio),
            "digit_ratio": float(digit_ratio),
            "flesch_score": float(flesch_score),
            "wstf_score": float(wstf_score),
            "lix_score": float(lix_score),
            "mattr_50": float(mattr_50)
        }

    def calculate_mattr(self, tokens: List[str], window_size: int = 50) -> float:
        if len(tokens) == 0:
            return 0.0
        if len(tokens) <= window_size:
            return len(set(tokens)) / len(tokens)
        
        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i : i + window_size]
            ttrs.append(len(set(window)) / window_size)
        return float(np.mean(ttrs))

    @staticmethod
    def calculate_reduction_delta(as_metrics: Dict[str, float], tgt_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculates exact delta (tgt - as) and reduction % ((as - tgt) / as * 100)."""
        eps = 1e-6
        results = {}
        
        # Features where reduction (decrease) is desired
        reduction_keys = [
            ("avg_sent_len", "red_sent_len_pct"),
            ("long_sent_ratio", "red_long_sent_pct"),
            ("subord_ratio", "red_subord_pct"),
            ("passive_ratio", "red_passive_pct"),
            ("genitive_ratio", "red_genitive_pct"),
            ("subjunctive_ratio", "red_subjunctive_pct"),
            ("nominal_ratio", "red_nominal_pct"),
            ("polysyllable_ratio", "red_polysyllable_pct"),
            ("abbr_ratio", "red_abbr_pct"),
            ("wstf_score", "red_wstf_pct"),
            ("lix_score", "red_lix_pct"),
            ("mattr_50", "red_mattr_pct")
        ]
        
        # Features where increase is desired
        increase_keys = [
            ("flesch_score", "gain_flesch_pts"),
            ("subject_initial_ratio", "gain_subject_initial_pct"),
            ("verb_to_noun_ratio", "gain_vnr_pct"),
            ("hyphen_compound_ratio", "gain_hyphen_pct"),
            ("digit_ratio", "gain_digit_pct")
        ]

        for base_key, out_key in reduction_keys:
            v_as = as_metrics.get(base_key, 0.0)
            v_tgt = tgt_metrics.get(base_key, 0.0)
            diff = v_as - v_tgt
            pct = (diff / (v_as + eps)) * 100.0 if v_as > eps else 0.0
            results[f"delta_{base_key}"] = float(v_tgt - v_as)
            results[out_key] = float(pct)

        for base_key, out_key in increase_keys:
            v_as = as_metrics.get(base_key, 0.0)
            v_tgt = tgt_metrics.get(base_key, 0.0)
            diff = v_tgt - v_as
            pct = (diff / (v_as + eps)) * 100.0 if v_as > eps else 0.0
            results[f"delta_{base_key}"] = float(diff)
            results[out_key] = float(pct if "pct" in out_key else diff)

        return results


def process_corpus(input_csv: str, output_csv: str, auditor: LeichteSpracheRuleAuditor):
    print(f"Auditing full corpus from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    as_texts = df["as_text"].fillna("").astype(str).tolist()
    ls_texts = df["ls_text"].fillna("").astype(str).tolist()
    
    print("Parsing AS texts with SpaCy...")
    as_docs = list(tqdm(auditor.nlp.pipe(as_texts, batch_size=64), total=len(as_texts), desc="AS parsing"))
    print("Parsing LS texts with SpaCy...")
    ls_docs = list(tqdm(auditor.nlp.pipe(ls_texts, batch_size=64), total=len(ls_texts), desc="LS parsing"))

    records = []
    for i in tqdm(range(len(df)), desc="Calculating metrics & reductions"):
        source = df.iloc[i].get("source", "unknown")
        as_res = auditor.analyze_doc(as_docs[i])
        ls_res = auditor.analyze_doc(ls_docs[i])
        red_res = auditor.calculate_reduction_delta(as_res, ls_res)
        
        row_dict = {
            "source": source,
            "as_tokens": as_res["token_count"],
            "ls_tokens": ls_res["token_count"],
            "compression_ratio": ls_res["token_count"] / max(1, as_res["token_count"]),
        }
        for k, v in as_res.items():
            row_dict[f"as_{k}"] = v
        for k, v in ls_res.items():
            row_dict[f"ls_{k}"] = v
        for k, v in red_res.items():
            row_dict[f"red_{k}"] = v
            
        records.append(row_dict)

    res_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Successfully saved audited corpus results to {output_csv}")
    return res_df


def process_model_comparisons(input_csv: str, output_csv: str, auditor: LeichteSpracheRuleAuditor):
    print(f"Auditing model generations from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Identify format
    as_col = "as_text" if "as_text" in df.columns else "source_text"
    ref_col = "ls_reference" if "ls_reference" in df.columns else ("ls_ref_text" if "ls_ref_text" in df.columns else None)
    gen_col = "translation" if "translation" in df.columns else ("generated_text" if "generated_text" in df.columns else None)
    model_col = "model_name" if "model_name" in df.columns else ("Model" if "Model" in df.columns else ("model" if "model" in df.columns else ("experiment_name" if "experiment_name" in df.columns else None)))

    as_texts = df[as_col].fillna("").astype(str).tolist()
    gen_texts = df[gen_col].fillna("").astype(str).tolist() if gen_col else []
    ref_texts = df[ref_col].fillna("").astype(str).tolist() if ref_col else []

    print("Parsing AS texts...")
    as_docs = list(tqdm(auditor.nlp.pipe(as_texts, batch_size=64), total=len(as_texts), desc="AS docs"))
    print("Parsing Generated texts...")
    gen_docs = list(tqdm(auditor.nlp.pipe(gen_texts, batch_size=64), total=len(gen_texts), desc="Gen docs")) if gen_texts else []
    print("Parsing Reference texts...")
    ref_docs = list(tqdm(auditor.nlp.pipe(ref_texts, batch_size=64), total=len(ref_texts), desc="Ref docs")) if ref_texts else []

    records = []
    for i in tqdm(range(len(df)), desc="Auditing model comparisons"):
        model_name = df.iloc[i].get(model_col, "Unknown Model") if model_col else "Default"
        as_res = auditor.analyze_doc(as_docs[i])
        gen_res = auditor.analyze_doc(gen_docs[i]) if gen_docs else {}
        ref_res = auditor.analyze_doc(ref_docs[i]) if ref_docs else {}
        
        red_gen = auditor.calculate_reduction_delta(as_res, gen_res) if gen_res else {}
        red_ref = auditor.calculate_reduction_delta(as_res, ref_res) if ref_res else {}

        row_dict = {
            "model": model_name,
            "as_tokens": as_res["token_count"],
            "gen_tokens": gen_res.get("token_count", 0),
            "ref_tokens": ref_res.get("token_count", 0),
            "compression_ratio": gen_res.get("token_count", 0) / max(1, as_res["token_count"]),
        }
        for k, v in as_res.items():
            row_dict[f"as_{k}"] = v
        for k, v in gen_res.items():
            row_dict[f"gen_{k}"] = v
        for k, v in red_gen.items():
            row_dict[f"gen_{k}"] = v
        for k, v in ref_res.items():
            row_dict[f"ref_{k}"] = v
        for k, v in red_ref.items():
            row_dict[f"ref_{k}"] = v

        records.append(row_dict)

    res_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    res_df.to_csv(output_csv, index=False)
    print(f"Successfully saved audited model results to {output_csv}")
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Audit Rule Adherence of Leichte Sprache")
    parser.add_argument("--mode", choices=["corpus", "models", "all"], default="all")
    parser.add_argument("--corpus_csv", default="data/analysis/corpus_master.csv")
    parser.add_argument("--models_csv", default="results/evaluation/token_length_comparison_detailed.csv")
    parser.add_argument("--output_corpus", default="data/analysis/rule_adherence_corpus.csv")
    parser.add_argument("--output_models", default="data/analysis/rule_adherence_models_seq2seq.csv")
    parser.add_argument("--spacy_model", default="de_core_news_lg")
    args = parser.parse_args()

    auditor = LeichteSpracheRuleAuditor(spacy_model=args.spacy_model)

    if args.mode in ["corpus", "all"] and os.path.exists(args.corpus_csv):
        process_corpus(args.corpus_csv, args.output_corpus, auditor)

    if args.mode in ["models", "all"] and os.path.exists(args.models_csv):
        process_model_comparisons(args.models_csv, args.output_models, auditor)


if __name__ == "__main__":
    main()
