import torch
import torch.nn as nn
import pandas as pd
import json
import os
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
import textstat
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json" 
MODEL_PATH = "results/models/lstm_article_sim_0.80_to_0.98.pt"
VOCAB_SOURCE_CSV = "data/analysis/information_loss_analysis_cleaned.csv"
VOCAB_SIM_RANGE = (0.8, 0.98)

MAX_SEQ_LEN = 512
EMBED_DIM = 128
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

class Vocab:
    def __init__(self, sentences, max_size=25000, min_freq=3):
        counter = Counter()
        for sent in sentences:
            counter.update(sent)
        
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": 0, "<unk>": 1}
        
        for token, freq in counter.most_common(max_size):
            if freq >= min_freq:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
                
    def __len__(self):
        return len(self.itos)
    
    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

def build_original_vocab():
    print(f"Reconstructing training vocab from {VOCAB_SOURCE_CSV}...")
    df = pd.read_csv(VOCAB_SOURCE_CSV)
    mask = (df["semantic_similarity_8192"] >= VOCAB_SIM_RANGE[0]) & (df["semantic_similarity_8192"] <= VOCAB_SIM_RANGE[1])
    df_filtered = df[mask]
    
    nlp = spacy.blank("de")
    X, y = [], []
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Vocab reconstruction"):
        ls_tokens = [t.text.lower() for t in nlp(str(row["ls_text"])) if not t.is_space]
        if len(ls_tokens) >= 10:
            X.append(ls_tokens)
            y.append(1)
        as_tokens = [t.text.lower() for t in nlp(str(row["as_text"])) if not t.is_space]
        if len(as_tokens) >= 10:
            X.append(as_tokens)
            y.append(0)
    
    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, _, _, _ = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)
    
    return Vocab(X_train)

def main(output_csv: str = "results/evaluation/eval_article_classifier.csv", output_summary: str = "results/evaluation/article_classifier_metrics.json"):
    print(f"Using device: {DEVICE}")
    vocab = build_original_vocab()
    print(f"Vocab size: {len(vocab)}")
    
    model = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path not found: {DATASET_PATH}")
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Evaluating {len(data)} pairs from {DATASET_PATH}...")
    
    nlp = spacy.blank("de")
    
    def predict(text):
        tokens = [t.text.lower() for t in nlp(text) if not t.is_space]
        encoded = vocab.encode(tokens)[:MAX_SEQ_LEN]
        padded = encoded + [0] * (MAX_SEQ_LEN - len(encoded))
        tensor = torch.tensor([padded], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            output = model(tensor).squeeze()
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
        return pred, prob

    results = []
    for item in tqdm(data, desc="Evaluating pairs"):
        ls_text = item.get("ls_text", "")
        as_text = item.get("as_text", "")
        
        if not ls_text or not as_text:
            continue
        
        ls_pred, ls_prob = predict(ls_text)
        as_pred, as_prob = predict(as_text)
        
        results.append({
            "LS_ID": item.get("ls_filename", "N/A"),
            "AS_ID": item.get("as_filename", "N/A"),
            "LS_Pred": "Simple" if ls_pred == 1 else "Normal",
            "LS_Conf": ls_prob if ls_pred == 1 else 1 - ls_prob,
            "AS_Pred": "Simple" if as_pred == 1 else "Normal",
            "AS_Conf": as_prob if as_pred == 1 else 1 - as_prob,
            "LS_Flesch": textstat.flesch_reading_ease(ls_text),
            "AS_Flesch": textstat.flesch_reading_ease(as_text),
            "LS_Wiener": textstat.wiener_sachtextformel(ls_text, 1),
            "AS_Wiener": textstat.wiener_sachtextformel(as_text, 1),
            "Correct": (ls_pred == 1 and as_pred == 0)
        })

    df_res = pd.DataFrame(results)
    
    y_true = [1] * len(df_res) + [0] * len(df_res) # LS=1, AS=0
    y_pred = list(df_res["LS_Pred"].map({"Simple": 1, "Normal": 0})) + list(df_res["AS_Pred"].map({"Simple": 1, "Normal": 0}))
    
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=["Normal (AS)", "Simple (LS)"], output_dict=True)
    
    print("\n" + "="*50)
    print(" ARTICLE-LEVEL CLASSIFICATION METRICS (Direct)")
    print("="*50)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {bacc*100:.2f}%")
    print(f"Perfect Pair Match: {df_res['Correct'].sum()} / {len(df_res)} ({df_res['Correct'].mean()*100:.2f}%)")
    print(f"LS articles correctly identified as Simple: {df_res[df_res['LS_Pred']=='Simple'].shape[0]} / {len(df_res)} ({df_res[df_res['LS_Pred']=='Simple'].shape[0]/len(df_res)*100:.2f}%)")
    print(f"AS articles correctly identified as Normal: {df_res[df_res['AS_Pred']=='Normal'].shape[0]} / {len(df_res)} ({df_res[df_res['AS_Pred']=='Normal'].shape[0]/len(df_res)*100:.2f}%)")
    print("\nClassification Report (Article-Level):")
    print(classification_report(y_true, y_pred, target_names=["Normal (AS)", "Simple (LS)"]))
    
    print("\n" + "="*50)
    print(" READABILITY METRICS")
    print("="*50)
    print(f"Avg LS Flesch: {df_res['LS_Flesch'].mean():.2f} (Higher = Easier, target LS: >80)")
    print(f"Avg AS Flesch: {df_res['AS_Flesch'].mean():.2f}")
    print(f"Flesch Gap: {df_res['LS_Flesch'].mean() - df_res['AS_Flesch'].mean():.2f}")
    print(f"Avg LS Wiener: {df_res['LS_Wiener'].mean():.2f} (Lower = Easier, target LS: <6)")
    print(f"Avg AS Wiener: {df_res['AS_Wiener'].mean():.2f}")

    # Speichern der Ergebnisse
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_res.to_csv(output_csv, index=False)
        print(f"\n[ERFOLG] Detail-Vorhersagen gespeichert unter: {output_csv}")

    if output_summary:
        os.makedirs(os.path.dirname(output_summary), exist_ok=True)
        summary_data = {
            "dataset_path": DATASET_PATH,
            "model_path": MODEL_PATH,
            "overall_accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "perfect_pair_match_ratio": float(df_res["Correct"].mean()),
            "avg_ls_flesch": float(df_res["LS_Flesch"].mean()),
            "avg_as_flesch": float(df_res["AS_Flesch"].mean()),
            "flesch_gap": float(df_res["LS_Flesch"].mean() - df_res["AS_Flesch"].mean()),
            "avg_ls_wiener": float(df_res["LS_Wiener"].mean()),
            "avg_as_wiener": float(df_res["AS_Wiener"].mean()),
            "classification_report": report_dict
        }
        with open(output_summary, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[ERFOLG] Metrik-Zusammenfassung gespeichert unter: {output_summary}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate article-level BiLSTM classifier")
    parser.add_argument("--dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json" if os.path.exists("data/lebenshilfe/lebenshilfe_dataset_clean.json") else "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json")
    parser.add_argument("--model_path", default="results/models/lstm_article_sim_0.80_to_0.98.pt")
    parser.add_argument("--vocab_source_csv", default="data/analysis/corpus_master.csv" if os.path.exists("data/analysis/corpus_master.csv") else "data/analysis/information_loss_analysis_cleaned.csv")
    parser.add_argument("--output_csv", default="results/evaluation/eval_article_classifier.csv")
    parser.add_argument("--output_summary", default="results/evaluation/article_classifier_metrics.json")
    args = parser.parse_args()
    
    if os.path.exists(args.dataset_path):
        DATASET_PATH = args.dataset_path
    if os.path.exists(args.model_path):
        MODEL_PATH = args.model_path
    if os.path.exists(args.vocab_source_csv):
        VOCAB_SOURCE_CSV = args.vocab_source_csv
    
    main(output_csv=args.output_csv, output_summary=args.output_summary)

