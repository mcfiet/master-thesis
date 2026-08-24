#!/usr/bin/env python3
"""
scripts/evaluation/evaluate_all_models_benchmark.py

5-Wege-Benchmark Evaluation:
Speichert:
- results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv (Wide Format für Master-Notebook)
- results/evaluation/master_benchmark_summary.csv (Aggregierte Übersicht)
"""

import os
import sys
import json
import argparse
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoModel
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--output_csv", default="results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv")
    parser.add_argument("--output_summary", default="results/evaluation/master_benchmark_summary.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"=== 5-Wege Benchmark Runner ({args.device}) ===")

if __name__ == "__main__":
    main()
