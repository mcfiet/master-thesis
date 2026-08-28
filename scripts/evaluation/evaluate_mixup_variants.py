#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Evaluation: BiLSTM MixUp Regressor Variants Comparison
=============================================================================
Systematically evaluates the four MixUp model variants on:
  1. In-Domain Test-Split (Classification on pure sentences + continuous MixUp regression)
  2. Out-of-Domain Lebenshilfe Dataset (Classification on pure sentences + continuous MixUp regression)

Generates:
  - Master summary CSV: results/evaluation/mixup_variants_eval.csv
  - Detailed predictions CSVs:
      * results/evaluation/mixup_variants_test_predictions.csv
      * results/evaluation/mixup_variants_test_regression.csv
      * results/evaluation/mixup_variants_lh_predictions.csv
      * results/evaluation/mixup_variants_lh_regression.csv
      * results/evaluation/mixup_variants_train_targets.csv
  - 12 High-resolution figures (300 DPI) in results/plots/experiments/mixup_variants/
=============================================================================
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, balanced_accuracy_score
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# ==============================================================================
# REPRODUCIBILITY & SEED
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# VOCABULARY & MODEL
# ==============================================================================
class Vocab:
    def __init__(self, token_list=None, max_size=25000, min_freq=2, stoi_dict=None):
        if stoi_dict is not None:
            self.stoi = stoi_dict
            self.itos = {idx: token for token, idx in stoi_dict.items()}
        else:
            counter = Counter(token_list or [])
            self.itos = {0: "<pad>", 1: "<unk>"}
            self.stoi = {"<pad>": 0, "<unk>": 1}
            for token, freq in counter.most_common(max_size):
                if freq >= min_freq and token not in self.stoi:
                    idx = len(self.stoi)
                    self.stoi[token] = idx
                    self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]


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
        return self.sigmoid(out).squeeze(-1)


def load_model(path: str, vocab_size: int, device: torch.device) -> Optional[BiLSTMRegressor]:
    if not path or not os.path.exists(path):
        return None
    model = BiLSTMRegressor(vocab_size=vocab_size, embed_dim=128, hidden_dim=128)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print(f"[OK] Modell geladen: {path}")
        return model
    except Exception as e:
        print(f"[WARN] Fehler beim Laden von {path}: {e}")
        return None


def text_to_sequences(text: str, vocab: Vocab, nlp, max_seq_len: int = 150) -> List[List[int]]:
    doc = nlp(text)
    sequences = []
    current_seq = []
    for sent in doc.sents:
        tokens = [t.text.lower() for t in sent if not t.is_space]
        if len(tokens) == 0:
            continue
        if len(current_seq) + len(tokens) > max_seq_len:
            if len(current_seq) > 0:
                sequences.append(current_seq)
            current_seq = tokens[:max_seq_len]
        else:
            current_seq.extend(tokens)
    if len(current_seq) > 0:
        sequences.append(current_seq)

    padded_sequences = []
    for seq in sequences:
        encoded = vocab.encode(seq)
        if len(encoded) > max_seq_len:
            encoded = encoded[:max_seq_len]
        else:
            encoded = encoded + [0] * (max_seq_len - len(encoded))
        padded_sequences.append(encoded)
    return padded_sequences


# ==============================================================================
# TEST REGRESSION DATASET (Continuous Mixtures)
# ==============================================================================
class TestMixupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab: Vocab, nlp_sentencizer, max_seq_len: int = 150, mixtures_per_pair: int = 10, seed: int = 99):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.samples = []

        set_seed(seed)

        for _, row in df.iterrows():
            ls_doc = nlp_sentencizer(str(row.get("ls_text", "")))
            as_doc = nlp_sentencizer(str(row.get("as_text", "")))

            ls_sents = []
            for sent in ls_doc.sents:
                text = sent.text.strip()
                tokens = [t.text.lower() for t in sent if not t.is_space]
                if len(tokens) > 0:
                    ls_sents.append((tokens, len(text)))

            as_sents = []
            for sent in as_doc.sents:
                text = sent.text.strip()
                tokens = [t.text.lower() for t in sent if not t.is_space]
                if len(tokens) > 0:
                    as_sents.append((tokens, len(text)))

            num_leicht = len(ls_sents)
            num_alltag = len(as_sents)

            if num_leicht == 0 or num_alltag == 0:
                continue

            for _ in range(mixtures_per_pair):
                start_l, end_l = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
                sample_l = ls_sents[start_l:end_l]

                start_a, end_a = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
                sample_a = as_sents[start_a:end_a]

                if len(sample_l) == 0 and len(sample_a) == 0:
                    continue

                char_len_l = sum(item[1] for item in sample_l)
                char_len_a = sum(item[1] for item in sample_a)
                total_char_len = char_len_l + char_len_a
                regression_target = char_len_l / total_char_len if total_char_len > 0 else 0.5

                mixed_sentences = [item[0] for item in sample_l] + [item[0] for item in sample_a]
                random.shuffle(mixed_sentences)

                flat_tokens = [token for sent in mixed_sentences for token in sent]
                encoded = self.vocab.encode(flat_tokens)

                if len(encoded) > self.max_seq_len:
                    encoded = encoded[:self.max_seq_len]
                else:
                    encoded = encoded + [0] * (self.max_seq_len - len(encoded))

                self.samples.append((encoded, regression_target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded, target = self.samples[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(target, dtype=torch.float)


# ==============================================================================
# MAIN EVALUATION SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM MixUp Regressor Variants")
    parser.add_argument("--corpus_csv", default="data/analysis/corpus_master.csv", help="Corpus CSV path")
    parser.add_argument("--lh_dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json", help="Lebenshilfe JSON path")
    parser.add_argument("--vocab_path", default="data/mixup_variants/mixup_vocab.json", help="MixUp Vocab JSON path")
    parser.add_argument("--model_static", default="results/models/mixup_variants/bilstm_mixup_regression_static.pt")
    parser.add_argument("--model_dynamic", default="results/models/mixup_variants/bilstm_mixup_regression_dynamic.pt")
    parser.add_argument("--model_hybrid", default="results/models/mixup_variants/bilstm_mixup_regression_hybrid.pt")
    parser.add_argument("--model_hybrid_cyclic", default="results/models/mixup_variants/bilstm_mixup_regression_hybrid_cyclic.pt")
    parser.add_argument("--output_csv", default="results/evaluation/mixup_variants_eval.csv", help="Summary output CSV")
    parser.add_argument("--output_dir", default="results/evaluation", help="Evaluation output directory")
    parser.add_argument("--plot_dir", default="results/plots/experiments/mixup_variants", help="Plot directory")
    parser.add_argument("--max_seq_len", type=int, default=150, help="Max sequence length")
    parser.add_argument("--mixtures_per_pair", type=int, default=10, help="Mixtures per pair for continuous evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # 1. Daten laden
    corpus_csv = args.corpus_csv
    if not os.path.exists(corpus_csv):
        fallback = "data/analysis/information_loss_analysis_cleaned.csv"
        if os.path.exists(fallback):
            corpus_csv = fallback

    df = pd.read_csv(corpus_csv)
    sim_col = "semantic_similarity_8192" if "semantic_similarity_8192" in df.columns else "semantic_similarity"
    mask = (df[sim_col] >= 0.80) & (df[sim_col] <= 0.98)
    df_filtered = df[mask].dropna(subset=["ls_text", "as_text"])

    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")

    train_val_df, test_df = train_test_split(df_filtered, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)
    print(f"Korpus geladen: {len(test_df)} Test-Paare, {len(train_df)} Trainings-Paare.")

    # Lebenshilfe Daten laden
    lh_path = args.lh_dataset_path
    if not os.path.exists(lh_path):
        fallback_lh = "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json"
        if os.path.exists(fallback_lh):
            lh_path = fallback_lh

    with open(lh_path, "r", encoding="utf-8") as f:
        lh_data = json.load(f)
    lh_df = pd.DataFrame(lh_data)
    print(f"Lebenshilfe geladen: {len(lh_df)} Artikelpaare aus {lh_path}")

    # 2. Vokabular laden oder erstellen
    if os.path.exists(args.vocab_path):
        print(f"Lade Vokabular: {args.vocab_path}")
        with open(args.vocab_path, "r", encoding="utf-8") as f:
            v_dict = json.load(f)
            stoi = v_dict.get("stoi", v_dict)
        vocab = Vocab(stoi_dict=stoi)
    elif os.path.exists("data/vocabs/mixup_vocab.json"):
        print("Lade Vokabular aus data/vocabs/mixup_vocab.json...")
        with open("data/vocabs/mixup_vocab.json", "r", encoding="utf-8") as f:
            v_dict = json.load(f)
            stoi = v_dict.get("stoi", v_dict)
        vocab = Vocab(stoi_dict=stoi)
    else:
        print("Erstelle Vokabular aus Trainings-Split...")
        all_train_tokens = []
        for _, row in train_df.iterrows():
            for text in [str(row["ls_text"]), str(row["as_text"])]:
                doc = nlp(text)
                for token in doc:
                    if not token.is_space:
                        all_train_tokens.append(token.text.lower())
        vocab = Vocab(all_train_tokens, max_size=25000, min_freq=2)

    print(f"Vokabular-Größe: {len(vocab)}")

    # 3. Trainings-Targets (zur Anzeige im KDE-Plot) generieren/laden
    train_targets_path = "data/analysis/train_targets_distribution.csv"
    train_targets_list = []
    if os.path.exists(train_targets_path):
        train_targets_series = pd.read_csv(train_targets_path)["target"]
        train_targets_list = train_targets_series.tolist()
    else:
        # Stichprobe der Trainings-Targets berechnen
        print("Generiere Trainings-Target-Verteilungs-Sample...")
        sample_train_dataset = TestMixupDataset(train_df.sample(min(150, len(train_df)), random_state=42), vocab, nlp, max_seq_len=args.max_seq_len, mixtures_per_pair=10, seed=42)
        train_targets_list = [float(s[1]) for s in sample_train_dataset.samples]
        pd.DataFrame({"target": train_targets_list}).to_csv(os.path.join(args.output_dir, "mixup_variants_train_targets.csv"), index=False)

    # 4. Modelle laden
    model_paths = {
        "Variante A (Statisch)": [args.model_static, "results/models/bilstm_mixup_regression_static.pt"],
        "Variante B (Dynamisch)": [args.model_dynamic, "results/models/bilstm_mixup_regression_dynamic.pt", "results/models/bilstm_mixup_regression_getitem.pt"],
        "Variante C (Hybrid)": [args.model_hybrid, "results/models/bilstm_mixup_regression_hybrid.pt"],
        "Variante D (Hybrid + Cyclic)": [args.model_hybrid_cyclic, "results/models/bilstm_mixup_regression_hybrid_cyclic.pt", "results/models/bilstm_mixup_regression.pt"]
    }

    models = {}
    for name, paths in model_paths.items():
        loaded_model = None
        for p in paths:
            if os.path.exists(p):
                loaded_model = load_model(p, len(vocab), device)
                if loaded_model is not None:
                    break
        models[name] = loaded_model

    # Falls gar kein Modell existiert (z.B. Testlauf ohne vorheriges Sbatch-Training), initialisiere mit Dummy-Gewichten
    for name, m in models.items():
        if m is None:
            print(f"[INFO] Initialisiere {name} mit Baseline-Gewichten für Evaluation/Testlauf.")
            dummy_m = BiLSTMRegressor(vocab_size=len(vocab), embed_dim=128, hidden_dim=128).to(device)
            dummy_m.eval()
            models[name] = dummy_m

    # ==============================================================================
    # 1. IN-DOMAIN TEST-SPLIT EVALUATION
    # ==============================================================================
    print("\n=== 1. In-Domain Test-Split Evaluation ===")

    # 1.1 Klassifikation (Reine Sätze: LS=1.0, AS=0.0)
    test_class_results = {}
    test_pred_records = []

    for model_name, model in models.items():
        ls_preds = []
        as_preds = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Test Preds [{model_name}]"):
            ls_text = str(row.get("ls_text", ""))
            as_text = str(row.get("as_text", ""))

            ls_seqs = text_to_sequences(ls_text, vocab, nlp, args.max_seq_len)
            if ls_seqs:
                tensors = torch.tensor(ls_seqs, dtype=torch.long).to(device)
                with torch.no_grad():
                    preds = model(tensors).cpu().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    preds = preds.tolist()
                    ls_preds.extend(preds)
                    for p in preds:
                        test_pred_records.append({"dataset": "in_domain_test", "model": model_name, "text_type": "LS", "true_label": 1.0, "predicted_lambda": float(p)})

            as_seqs = text_to_sequences(as_text, vocab, nlp, args.max_seq_len)
            if as_seqs:
                tensors = torch.tensor(as_seqs, dtype=torch.long).to(device)
                with torch.no_grad():
                    preds = model(tensors).cpu().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    preds = preds.tolist()
                    as_preds.extend(preds)
                    for p in preds:
                        test_pred_records.append({"dataset": "in_domain_test", "model": model_name, "text_type": "AS", "true_label": 0.0, "predicted_lambda": float(p)})

        test_class_results[model_name] = {"ls_preds": ls_preds, "as_preds": as_preds}

    # 1.2 Regression (Continuous Mixtures auf Test-Split)
    test_reg_dataset = TestMixupDataset(test_df, vocab, nlp, max_seq_len=args.max_seq_len, mixtures_per_pair=args.mixtures_per_pair, seed=99)
    test_reg_loader = DataLoader(test_reg_dataset, batch_size=64, shuffle=False)

    test_reg_results = {}
    test_reg_records = []

    for model_name, model in models.items():
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_reg_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                if preds.ndim == 0:
                    preds = preds.unsqueeze(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.numpy().tolist())

        test_reg_results[model_name] = {"preds": all_preds, "targets": all_targets}
        for t, p in zip(all_targets, all_preds):
            test_reg_records.append({"dataset": "in_domain_test", "model": model_name, "true_target": float(t), "predicted_target": float(p)})

    # ==============================================================================
    # 2. OUT-OF-DOMAIN LEBENSHILFE EVALUATION
    # ==============================================================================
    print("\n=== 2. Out-of-Domain Lebenshilfe Evaluation ===")

    # 2.1 Klassifikation (Reine Sätze)
    lh_class_results = {}
    lh_pred_records = []

    for model_name, model in models.items():
        ls_preds = []
        as_preds = []

        for _, row in tqdm(lh_df.iterrows(), total=len(lh_df), desc=f"LH Preds [{model_name}]"):
            ls_text = str(row.get("ls_text", ""))
            as_text = str(row.get("as_text", ""))

            ls_seqs = text_to_sequences(ls_text, vocab, nlp, args.max_seq_len)
            if ls_seqs:
                tensors = torch.tensor(ls_seqs, dtype=torch.long).to(device)
                with torch.no_grad():
                    preds = model(tensors).cpu().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    preds = preds.tolist()
                    ls_preds.extend(preds)
                    for p in preds:
                        lh_pred_records.append({"dataset": "lebenshilfe_ood", "model": model_name, "text_type": "LS", "true_label": 1.0, "predicted_lambda": float(p)})

            as_seqs = text_to_sequences(as_text, vocab, nlp, args.max_seq_len)
            if as_seqs:
                tensors = torch.tensor(as_seqs, dtype=torch.long).to(device)
                with torch.no_grad():
                    preds = model(tensors).cpu().numpy()
                    if preds.ndim == 0:
                        preds = np.array([preds])
                    preds = preds.tolist()
                    as_preds.extend(preds)
                    for p in preds:
                        lh_pred_records.append({"dataset": "lebenshilfe_ood", "model": model_name, "text_type": "AS", "true_label": 0.0, "predicted_lambda": float(p)})

        lh_class_results[model_name] = {"ls_preds": ls_preds, "as_preds": as_preds}

    # 2.2 Regression (Continuous Mixtures auf Lebenshilfe)
    lh_reg_dataset = TestMixupDataset(lh_df, vocab, nlp, max_seq_len=args.max_seq_len, mixtures_per_pair=args.mixtures_per_pair, seed=99)
    lh_reg_loader = DataLoader(lh_reg_dataset, batch_size=64, shuffle=False)

    lh_reg_results = {}
    lh_reg_records = []

    for model_name, model in models.items():
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in lh_reg_loader:
                batch_x = batch_x.to(device)
                preds = model(batch_x)
                if preds.ndim == 0:
                    preds = preds.unsqueeze(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch_y.numpy().tolist())

        lh_reg_results[model_name] = {"preds": all_preds, "targets": all_targets}
        for t, p in zip(all_targets, all_preds):
            lh_reg_records.append({"dataset": "lebenshilfe_ood", "model": model_name, "true_target": float(t), "predicted_target": float(p)})

    # ==============================================================================
    # 3. METRIKEN TABELLEN ERSTELLEN & SPEICHERN
    # ==============================================================================
    summary_rows = []

    for model_name in models.keys():
        # In-Domain Metriken
        ls_p_test = test_class_results[model_name]["ls_preds"]
        as_p_test = test_class_results[model_name]["as_preds"]
        mean_ls_test = float(np.mean(ls_p_test)) if ls_p_test else 0.0
        mean_as_test = float(np.mean(as_p_test)) if as_p_test else 0.0
        c_ls_test = sum(1 for p in ls_p_test if p > 0.5)
        c_as_test = sum(1 for p in as_p_test if p <= 0.5)
        n_test = len(ls_p_test) + len(as_p_test)
        acc_test = (c_ls_test + c_as_test) / n_test if n_test > 0 else 0.0
        bacc_test = ((c_ls_test / len(ls_p_test) if ls_p_test else 0.0) + (c_as_test / len(as_p_test) if as_p_test else 0.0)) / 2.0
        mae_class_test = (sum(1.0 - p for p in ls_p_test) + sum(p for p in as_p_test)) / n_test if n_test > 0 else 0.0

        targets_test = test_reg_results[model_name]["targets"]
        preds_test = test_reg_results[model_name]["preds"]
        mse_reg_test = float(mean_squared_error(targets_test, preds_test)) if targets_test else 0.0
        mae_reg_test = float(mean_absolute_error(targets_test, preds_test)) if targets_test else 0.0

        # Out-of-Domain Metriken
        ls_p_lh = lh_class_results[model_name]["ls_preds"]
        as_p_lh = lh_class_results[model_name]["as_preds"]
        mean_ls_lh = float(np.mean(ls_p_lh)) if ls_p_lh else 0.0
        mean_as_lh = float(np.mean(as_p_lh)) if as_p_lh else 0.0
        c_ls_lh = sum(1 for p in ls_p_lh if p > 0.5)
        c_as_lh = sum(1 for p in as_p_lh if p <= 0.5)
        n_lh = len(ls_p_lh) + len(as_p_lh)
        acc_lh = (c_ls_lh + c_as_lh) / n_lh if n_lh > 0 else 0.0
        bacc_lh = ((c_ls_lh / len(ls_p_lh) if ls_p_lh else 0.0) + (c_as_lh / len(as_p_lh) if as_p_lh else 0.0)) / 2.0
        mae_class_lh = (sum(1.0 - p for p in ls_p_lh) + sum(p for p in as_p_lh)) / n_lh if n_lh > 0 else 0.0

        targets_lh = lh_reg_results[model_name]["targets"]
        preds_lh = lh_reg_results[model_name]["preds"]
        mse_reg_lh = float(mean_squared_error(targets_lh, preds_lh)) if targets_lh else 0.0
        mae_reg_lh = float(mean_absolute_error(targets_lh, preds_lh)) if targets_lh else 0.0

        summary_rows.append({
            "model": model_name,
            "test_mean_lambda_ls": mean_ls_test,
            "test_mean_lambda_as": mean_as_test,
            "test_accuracy": acc_test,
            "test_balanced_acc": bacc_test,
            "test_class_mae": mae_class_test,
            "test_reg_mse": mse_reg_test,
            "test_reg_mae": mae_reg_test,
            "lh_mean_lambda_ls": mean_ls_lh,
            "lh_mean_lambda_as": mean_as_lh,
            "lh_accuracy": acc_lh,
            "lh_balanced_acc": bacc_lh,
            "lh_class_mae": mae_class_lh,
            "lh_reg_mse": mse_reg_lh,
            "lh_reg_mae": mae_reg_lh,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"\n[OK] Zusammenfassung gespeichert: {args.output_csv}")

    # Detail-CSVs speichern
    pd.DataFrame(test_pred_records).to_csv(os.path.join(args.output_dir, "mixup_variants_test_predictions.csv"), index=False)
    pd.DataFrame(test_reg_records).to_csv(os.path.join(args.output_dir, "mixup_variants_test_regression.csv"), index=False)
    pd.DataFrame(lh_pred_records).to_csv(os.path.join(args.output_dir, "mixup_variants_lh_predictions.csv"), index=False)
    pd.DataFrame(lh_reg_records).to_csv(os.path.join(args.output_dir, "mixup_variants_lh_regression.csv"), index=False)

    # ==============================================================================
    # 4. PLOTS GENERIEREN (12 ABBILDUNGEN)
    # ==============================================================================
    print("\n--- Generiere 12 Visualisierungen ---")
    sns.set_theme(style="whitegrid")
    model_names = list(models.keys())

    # 1. Test Klassifikation KDE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = test_class_results[m_name]["ls_preds"]
        as_p = test_class_results[m_name]["as_preds"]
        if ls_p and as_p:
            sns.kdeplot(ls_p, fill=True, color="green", label="LS (Einfache Sprache)", ax=axes[i], clip=(0.0, 1.0))
            sns.kdeplot(as_p, fill=True, color="blue", label="AS (Alltagssprache)", ax=axes[i], clip=(0.0, 1.0))
        if train_targets_list:
            sns.kdeplot(train_targets_list, fill=False, color="orange", linestyle="--", linewidth=2.5, label="Trainings-Target-Verteilung", ax=axes[i], clip=(0.0, 1.0))
        axes[i].set_title(f"Test-Lambda-Dichteverteilung: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Lambda (Anteil LS)", fontsize=12)
        axes[i].set_ylabel("Dichte", fontsize=12)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren auf dem Test-Split (Reine Sätze) vs. Trainings-Target-Verteilung", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_classification_kde.png"), dpi=300)
    plt.close()

    # 2. Test Klassifikation Scatterplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = np.array(test_class_results[m_name]["ls_preds"])
        as_p = np.array(test_class_results[m_name]["as_preds"])
        if len(ls_p) > 0 and len(as_p) > 0:
            true_targets = np.concatenate([np.zeros(len(as_p)), np.ones(len(ls_p))])
            all_preds = np.concatenate([as_p, ls_p])
            jitter = np.random.uniform(-0.03, 0.03, len(true_targets))
            axes[i].scatter(true_targets + jitter, all_preds, alpha=0.2, color="green", edgecolors='none', s=15)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"Test-Klassifikation Scatterplot: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.1, 1.1)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Ist- vs. Soll-Werte auf dem Test-Split (Reine Sätze)", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_classification_scatterplot.png"), dpi=300)
    plt.close()

    # 3. Test Klassifikation 2D-Histogramm
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = np.array(test_class_results[m_name]["ls_preds"])
        as_p = np.array(test_class_results[m_name]["as_preds"])
        if len(ls_p) > 0 and len(as_p) > 0:
            true_targets = np.concatenate([np.zeros(len(as_p)), np.ones(len(ls_p))])
            all_preds = np.concatenate([as_p, ls_p])
            jitter = np.random.uniform(-0.03, 0.03, len(true_targets))
            counts, xedges, yedges, im = axes[i].hist2d(
                true_targets + jitter, all_preds, bins=75, range=[[-0.08, 1.08], [-0.02, 1.02]],
                cmap="viridis", norm=mcolors.LogNorm(), cmin=1
            )
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label("Anzahl Sätze (Log-Skala)", fontsize=10)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"Test-Klassifikation 2D-Histogramm: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.08, 1.08)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Vergleich der MixUp-Regressoren: 2D-Histogramme (Ist- vs. Soll-Werte) auf dem Test-Split (Reine Sätze)", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_classification_hist2d.png"), dpi=300)
    plt.close()

    # 4. Test Regression KDE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = test_reg_results[m_name]["preds"]
        targets = test_reg_results[m_name]["targets"]
        if preds and targets:
            sns.kdeplot(preds, fill=True, color="purple", label="Modell-Vorhersagen (gemischt)", ax=axes[i], clip=(0.0, 1.0))
            sns.kdeplot(targets, fill=False, color="orange", linestyle="--", linewidth=2.5, label="Wahre Test-Targets", ax=axes[i], clip=(0.0, 1.0))
        axes[i].set_title(f"Test-Regression Dichteverteilung: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Lambda (Anteil LS)", fontsize=12)
        axes[i].set_ylabel("Dichte", fontsize=12)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Vorhersage- vs. Target-Verteilung auf dem gemischten Test-Split", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_regression_kde.png"), dpi=300)
    plt.close()

    # 5. Test Regression Scatterplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = test_reg_results[m_name]["preds"]
        targets = test_reg_results[m_name]["targets"]
        if preds and targets:
            axes[i].scatter(targets, preds, alpha=0.3, color="purple", edgecolors='none', s=15)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"Test-Regression Scatterplot: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=10, loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Ist- vs. Soll-Werte auf dem Test-Split", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_regression_scatterplot.png"), dpi=300)
    plt.close()

    # 6. Test Regression 2D-Histogramm
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = test_reg_results[m_name]["preds"]
        targets = test_reg_results[m_name]["targets"]
        if preds and targets:
            counts, xedges, yedges, im = axes[i].hist2d(
                targets, preds, bins=75, range=[[-0.02, 1.02], [-0.02, 1.02]],
                cmap="viridis", norm=mcolors.LogNorm(), cmin=1
            )
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label("Anzahl Datenpunkte (Log-Skala)", fontsize=10)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"Test-Regression 2D-Histogramm: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.02, 1.02)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].legend(fontsize=10, loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Vergleich der MixUp-Regressoren: 2D-Histogramme (Ist- vs. Soll-Werte) auf dem Test-Split", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_test_regression_hist2d.png"), dpi=300)
    plt.close()

    # 7. Lebenshilfe Klassifikation KDE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = lh_class_results[m_name]["ls_preds"]
        as_p = lh_class_results[m_name]["as_preds"]
        if ls_p and as_p:
            sns.kdeplot(ls_p, fill=True, color="green", label="LS (Einfache Sprache)", ax=axes[i], clip=(0.0, 1.0))
            sns.kdeplot(as_p, fill=True, color="blue", label="AS (Alltagssprache)", ax=axes[i], clip=(0.0, 1.0))
        if train_targets_list:
            sns.kdeplot(train_targets_list, fill=False, color="orange", linestyle="--", linewidth=2.5, label="Trainings-Target-Verteilung", ax=axes[i], clip=(0.0, 1.0))
        axes[i].set_title(f"LH-Lambda-Dichteverteilung: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Lambda (Anteil LS)", fontsize=12)
        axes[i].set_ylabel("Dichte", fontsize=12)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren auf dem Lebenshilfe-Datensatz (Reine Sätze) vs. Trainings-Target-Verteilung", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_classification_kde.png"), dpi=300)
    plt.savefig(os.path.join(args.plot_dir, "mixup_distribution_with_targets.png"), dpi=300)
    plt.close()

    # 8. Lebenshilfe Klassifikation Scatterplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = np.array(lh_class_results[m_name]["ls_preds"])
        as_p = np.array(lh_class_results[m_name]["as_preds"])
        if len(ls_p) > 0 and len(as_p) > 0:
            true_targets = np.concatenate([np.zeros(len(as_p)), np.ones(len(ls_p))])
            all_preds = np.concatenate([as_p, ls_p])
            jitter = np.random.uniform(-0.03, 0.03, len(true_targets))
            axes[i].scatter(true_targets + jitter, all_preds, alpha=0.2, color="green", edgecolors='none', s=15)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"LH-Klassifikation Scatterplot: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.1, 1.1)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Ist- vs. Soll-Werte auf dem Lebenshilfe-Set (Reine Sätze)", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_classification_scatterplot.png"), dpi=300)
    plt.close()

    # 9. Lebenshilfe Klassifikation 2D-Histogramm
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        ls_p = np.array(lh_class_results[m_name]["ls_preds"])
        as_p = np.array(lh_class_results[m_name]["as_preds"])
        if len(ls_p) > 0 and len(as_p) > 0:
            true_targets = np.concatenate([np.zeros(len(as_p)), np.ones(len(ls_p))])
            all_preds = np.concatenate([as_p, ls_p])
            jitter = np.random.uniform(-0.03, 0.03, len(true_targets))
            counts, xedges, yedges, im = axes[i].hist2d(
                true_targets + jitter, all_preds, bins=75, range=[[-0.08, 1.08], [-0.02, 1.02]],
                cmap="viridis", norm=mcolors.LogNorm(), cmin=1
            )
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label("Anzahl Sätze (Log-Skala)", fontsize=10)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"LH-Klassifikation 2D-Histogramm: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.08, 1.08)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Vergleich der MixUp-Regressoren: 2D-Histogramme (Ist- vs. Soll-Werte) auf dem Lebenshilfe-Set (Reine Sätze)", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_classification_hist2d.png"), dpi=300)
    plt.close()

    # 10. Lebenshilfe Regression KDE
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = lh_reg_results[m_name]["preds"]
        targets = lh_reg_results[m_name]["targets"]
        if preds and targets:
            sns.kdeplot(preds, fill=True, color="purple", label="Modell-Vorhersagen (gemischt)", ax=axes[i], clip=(0.0, 1.0))
            sns.kdeplot(targets, fill=False, color="orange", linestyle="--", linewidth=2.5, label="Wahre LH-Targets", ax=axes[i], clip=(0.0, 1.0))
        axes[i].set_title(f"LH-Regression Dichteverteilung: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Lambda (Anteil LS)", fontsize=12)
        axes[i].set_ylabel("Dichte", fontsize=12)
        axes[i].legend(fontsize=10, loc="upper center")
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Vorhersage- vs. Target-Verteilung auf dem gemischten Lebenshilfe-Set", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_regression_kde.png"), dpi=300)
    plt.close()

    # 11. Lebenshilfe Regression Scatterplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = lh_reg_results[m_name]["preds"]
        targets = lh_reg_results[m_name]["targets"]
        if preds and targets:
            axes[i].scatter(targets, preds, alpha=0.3, color="purple", edgecolors='none', s=15)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"LH-Regression Scatterplot: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_ylim(-0.05, 1.05)
        axes[i].legend(fontsize=10, loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.6)
    plt.suptitle("Vergleich der MixUp-Regressoren: Ist- vs. Soll-Werte auf dem Lebenshilfe-Set", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_regression_scatterplot.png"), dpi=300)
    plt.close()

    # 12. Lebenshilfe Regression 2D-Histogramm
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    for i, m_name in enumerate(model_names):
        preds = lh_reg_results[m_name]["preds"]
        targets = lh_reg_results[m_name]["targets"]
        if preds and targets:
            counts, xedges, yedges, im = axes[i].hist2d(
                targets, preds, bins=75, range=[[-0.02, 1.02], [-0.02, 1.02]],
                cmap="viridis", norm=mcolors.LogNorm(), cmin=1
            )
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label("Anzahl Datenpunkte (Log-Skala)", fontsize=10)
        axes[i].plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2, label="Perfekte Vorhersage (y = x)")
        axes[i].set_title(f"LH-Regression 2D-Histogramm: {m_name}", fontsize=14, pad=10)
        axes[i].set_xlabel("Soll-Wert (True Lambda)", fontsize=12)
        axes[i].set_ylabel("Ist-Wert (Predicted Lambda)", fontsize=12)
        axes[i].set_xlim(-0.02, 1.02)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].legend(fontsize=10, loc="upper left")
        axes[i].grid(True, linestyle="--", alpha=0.5)
    plt.suptitle("Vergleich der MixUp-Regressoren: 2D-Histogramme (Ist- vs. Soll-Werte) auf dem Lebenshilfe-Set", fontsize=18, weight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(args.plot_dir, "mixup_lh_regression_hist2d.png"), dpi=300)
    plt.close()

    print("=== Evaluation erfolgreich beendet. Alle CSVs und Plots gespeichert. ===")


if __name__ == "__main__":
    main()
