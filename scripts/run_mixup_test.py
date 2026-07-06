import pandas as pd
import random
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.utils.data import Dataset

CSV_PATH = "results/information_loss_analysis_cleaned.csv"
MIN_SIM = 0.8
MAX_SIM = 0.98

def get_contiguous_slice(sentences, k):
    num_sents = len(sentences)
    if num_sents == 0 or k <= 0:
        return []
    if num_sents <= k:
        return list(sentences)
    else:
        start = random.randint(0, num_sents - k)
        return sentences[start : start + k]

class NormalMixupDataset(Dataset):
    def __init__(self, df, nlp_sentencizer):
        self.ls_data = []
        self.as_data = []
        for _, row in df.iterrows():
            ls_sents = [s.text.strip() for s in nlp_sentencizer(str(row["ls_text"])).sents if s.text.strip()]
            as_sents = [s.text.strip() for s in nlp_sentencizer(str(row["as_text"])).sents if s.text.strip()]
            self.ls_data.append(ls_sents)
            self.as_data.append(as_sents)
            
    def __len__(self):
        return len(self.ls_data)
        
    def __getitem__(self, idx):
        leichte_saetze = self.ls_data[idx]
        alltags_saetze = self.as_data[idx]
        num_leicht = len(leichte_saetze)
        num_alltag = len(alltags_saetze)
        if num_leicht == 0 or num_alltag == 0:
            return "", 0.5
            
        start_leichte_saetze, ende_leichte_saetze = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
        sample_leicht = leichte_saetze[start_leichte_saetze:ende_leichte_saetze]
        
        start_alltags_saetze, ende_alltags_saetze = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
        sample_alltag = alltags_saetze[start_alltags_saetze:ende_alltags_saetze]
        
        kompletter_absatz = sample_leicht + sample_alltag
        random.shuffle(kompletter_absatz)
        
        str_sample_leicht = ''.join(sample_leicht)
        str_sample_alltag = ''.join(sample_alltag)
        len_sample_leicht = len(str_sample_leicht)
        len_sample_alltag = len(str_sample_alltag)
        
        total_len = len_sample_leicht + len_sample_alltag
        regression_target = len_sample_leicht / total_len if total_len > 0 else 0.5
        return ' '.join(kompletter_absatz), regression_target

class NewMixupDataset(Dataset):
    def __init__(self, df, nlp_sentencizer):
        self.ls_data = []
        self.as_data = []
        for _, row in df.iterrows():
            ls_sents = [s.text.strip() for s in nlp_sentencizer(str(row["ls_text"])).sents if s.text.strip()]
            as_sents = [s.text.strip() for s in nlp_sentencizer(str(row["as_text"])).sents if s.text.strip()]
            self.ls_data.append(ls_sents)
            self.as_data.append(as_sents)
            
    def __len__(self):
        return len(self.ls_data)
        
    def __getitem__(self, idx):
        leichte_saetze = self.ls_data[idx]
        alltags_saetze = self.as_data[idx]
        num_leicht = len(leichte_saetze)
        num_alltag = len(alltags_saetze)
        if num_leicht == 0 or num_alltag == 0:
            return "", 0.5
            
        lam = random.uniform(0.0, 1.0)
        N = random.randint(8, 15)
        num_ls = int(round(lam * N))
        num_as = N - num_ls
        
        sample_leicht = get_contiguous_slice(leichte_saetze, num_ls)
        sample_alltag = get_contiguous_slice(alltags_saetze, num_as)
        
        kompletter_absatz = sample_leicht + sample_alltag
        random.shuffle(kompletter_absatz)
        
        total_sents = len(sample_leicht) + len(sample_alltag)
        regression_target = len(sample_leicht) / total_sents if total_sents > 0 else 0.5
        return ' '.join(kompletter_absatz), regression_target

def main():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    mask = (df["semantic_similarity_8192"] >= MIN_SIM) & (df["semantic_similarity_8192"] <= MAX_SIM)
    df_filtered = df[mask].dropna(subset=["ls_text", "as_text"])
    print(f"Loaded {len(df_filtered)} pairs.")
    
    print("Initializing spaCy and datasets...")
    nlp = spacy.blank("de")
    nlp.add_pipe("sentencizer")
    
    normal_ds = NormalMixupDataset(df_filtered, nlp)
    new_ds = NewMixupDataset(df_filtered, nlp)
    
    print("Running simulation...")
    normal_targets = []
    new_targets = []
    
    for epoch in range(10):
        for i in range(len(df_filtered)):
            _, t_norm = normal_ds[i]
            _, t_new = new_ds[i]
            
            normal_targets.append(t_norm)
            new_targets.append(t_new)
            
    print("Generating plot...")
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(normal_targets, bins=25, kde=False, color="salmon", stat="probability")
    plt.title("1. Erste Variante (Zufällige Slices)")
    plt.xlabel("Regression Target (Anteil Leichte Sprache)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.xlim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    sns.histplot(new_targets, bins=25, kde=False, color="skyblue", stat="probability")
    plt.title("2. Zweite Variante (Gleichverteiltes Konzept)")
    plt.xlabel("Regression Target (Anteil Leichte Sprache)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.xlim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plot_path = "results/mixup_comparison_two_variants.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    
    # Save single plot with only the first variant
    plt.figure(figsize=(8, 6))
    sns.histplot(normal_targets, bins=25, kde=False, color="salmon", stat="probability")
    plt.title("Target-Verteilung (Erste Variante)")
    plt.xlabel("Regression Target (Anteil Leichte Sprache)")
    plt.ylabel("Wahrscheinlichkeit")
    plt.xlim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save to research/img/analysis to make it accessible to Marp
    os.makedirs("research/img/analysis", exist_ok=True)
    single_plot_path = "research/img/analysis/mixup_first_variant_distribution.png"
    plt.savefig(single_plot_path, dpi=300)
    print(f"Single plot saved to {single_plot_path}")

if __name__ == '__main__':
    main()
