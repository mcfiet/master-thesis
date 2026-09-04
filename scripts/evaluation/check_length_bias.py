import torch
import torch.nn as nn
import pandas as pd
import json
import os
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np
from scipy.stats import pearsonr, spearmanr

# --- CONFIGURATION ---
DATASET_PATH = "data/lebenshilfe/lebenshilfe_dataset_clean.json" 
MODEL_PATH = "results/models/bilstm_article_classifier.pt"
VOCAB_PATH = "data/vocabs/article_vocab.json"
VOCAB_SOURCE_CSV = "data/analysis/corpus_master.csv"
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
    def __init__(self, sentences=None, max_size=25000, min_freq=3):
        if sentences is None:
            sentences = []
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

def load_or_build_vocab():
    if VOCAB_PATH and os.path.exists(VOCAB_PATH):
        print(f"Loading vocabulary from JSON: {VOCAB_PATH}...")
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        stoi = v_data.get("stoi", v_data)
        itos = v_data.get("itos", None)
        if itos is None:
            itos = [None] * len(stoi)
            for k, idx in stoi.items():
                if idx < len(itos):
                    itos[idx] = k
        vocab_obj = Vocab()
        vocab_obj.stoi = stoi
        vocab_obj.itos = itos
        return vocab_obj

    print(f"Reconstructing vocab from {VOCAB_SOURCE_CSV}...")
    df = pd.read_csv(VOCAB_SOURCE_CSV)
    sim_col = "semantic_similarity_8192" if "semantic_similarity_8192" in df.columns else "semantic_similarity"
    mask = (df[sim_col] >= VOCAB_SIM_RANGE[0]) & (df[sim_col] <= VOCAB_SIM_RANGE[1])
    df_filtered = df[mask]
    
    nlp = spacy.blank("de")
    X, y = [], []
    
    for _, row in df_filtered.iterrows():
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

def main(output_csv: str = "results/evaluation/length_bias_results.csv"):
    print(f"Using device: {DEVICE}")
    vocab = load_or_build_vocab()
    print(f"Vocab size: {len(vocab)}")
    
    model = BiLSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    raw_weights = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(raw_weights, dict) and "model_state_dict" in raw_weights:
        raw_weights = raw_weights["model_state_dict"]
    model.load_state_dict(raw_weights)
    model.eval()
    print("Model loaded successfully.")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    nlp = spacy.blank("de")
    
    # Collect statistics
    all_texts = []
    lengths = []
    probabilities = []
    true_labels = []
    
    # Dummy evaluation
    dummy_preds = []
    
    # Fixed-length evaluations
    fixed_len_preds_50 = []
    fixed_len_preds_100 = []
    
    dummy_token = vocab.itos[2] if len(vocab.itos) > 2 and vocab.itos[2] is not None else "der"
    print(f"Selected dummy token for replacement: '{dummy_token}'")
    
    for item in data:
        ls_text = item.get("ls_text", "")
        as_text = item.get("as_text", "")
        if not ls_text or not as_text:
            continue
            
        for text, label in [(ls_text, 1), (as_text, 0)]:
            tokens = [t.text.lower() for t in nlp(text) if not t.is_space]
            original_len = len(tokens)
            all_texts.append(text)
            lengths.append(original_len)
            true_labels.append(label)
            
            # 1. Standard prediction
            encoded = vocab.encode(tokens)[:MAX_SEQ_LEN]
            if not encoded:
                encoded = [0]
            tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                output = model(tensor).squeeze()
                prob = torch.sigmoid(output).item()
                probabilities.append(prob)
                
            # 2. Dummy representation prediction (constant tokens of original length)
            dummy_tokens = [dummy_token] * original_len
            dummy_encoded = vocab.encode(dummy_tokens)[:MAX_SEQ_LEN] or [0]
            dummy_tensor = torch.tensor([dummy_encoded], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                dummy_output = model(dummy_tensor).squeeze()
                dummy_prob = torch.sigmoid(dummy_output).item()
                dummy_preds.append(1 if dummy_prob > 0.5 else 0)
                
            # 3. Fixed length prediction: 50 tokens
            fixed_tokens_50 = tokens[:50]
            fixed_encoded_50 = vocab.encode(fixed_tokens_50) or [0]
            fixed_tensor_50 = torch.tensor([fixed_encoded_50], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                fixed_output_50 = model(fixed_tensor_50).squeeze()
                fixed_prob_50 = torch.sigmoid(fixed_output_50).item()
                fixed_len_preds_50.append(1 if fixed_prob_50 > 0.5 else 0)
                
            # 4. Fixed length prediction: 100 tokens
            fixed_tokens_100 = tokens[:100]
            fixed_encoded_100 = vocab.encode(fixed_tokens_100) or [0]
            fixed_tensor_100 = torch.tensor([fixed_encoded_100], dtype=torch.long).to(DEVICE)
            
            with torch.no_grad():
                fixed_output_100 = model(fixed_tensor_100).squeeze()
                fixed_prob_100 = torch.sigmoid(fixed_output_100).item()
                fixed_len_preds_100.append(1 if fixed_prob_100 > 0.5 else 0)

    lengths = np.array(lengths)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    preds = (probabilities > 0.5).astype(int)
    
    print("\n" + "="*50)
    print(" EXPERIMENT 1: CORRELATION ANALYSIS")
    print("="*50)
    pearson_r, p_val_p = pearsonr(lengths, probabilities)
    spearman_r, p_val_s = spearmanr(lengths, probabilities)
    print(f"Correlation between text length and predicted 'Simple' probability:")
    print(f" - Pearson r:  {pearson_r:.4f} (p-value: {p_val_p:.4e})")
    print(f" - Spearman r: {spearman_r:.4f} (p-value: {p_val_s:.4e})")
    print("Interpretation:")
    print(" - Positive r means longer texts get classified as Simple (LS).")
    print(" - Negative r means shorter texts get classified as Simple (LS).")
    print(" - Strong correlation (|r| > 0.6) suggests potential length bias.")
    
    print("\n" + "="*50)
    print(" EXPERIMENT 2: CONSTANT TOKEN / DUMMY TEXT TEST")
    print("="*50)
    print("Method: Replace every word with a dummy token, preserving original length.")
    print("If accuracy remains high, the model is just using sequence length/padding.")
    dummy_bacc = balanced_accuracy_score(true_labels, dummy_preds)
    print(f"Balanced Accuracy on Dummy Texts: {dummy_bacc*100:.2f}%")
    print("\nClassification Report (Dummy Texts):")
    print(classification_report(true_labels, dummy_preds, target_names=["AS", "LS"]))
    
    print("\n" + "="*50)
    print(" EXPERIMENT 3: FIXED-LENGTH TRUNCATION TEST")
    print("="*50)
    print("Method: Truncate all texts to exactly N tokens, removing length differences.")
    
    bacc_original = balanced_accuracy_score(true_labels, preds)
    bacc_50 = balanced_accuracy_score(true_labels, fixed_len_preds_50)
    bacc_100 = balanced_accuracy_score(true_labels, fixed_len_preds_100)
    
    print(f"Balanced Accuracy (Full sequences, max 512): {bacc_original*100:.2f}%")
    print(f"Balanced Accuracy (Truncated to exactly 50 tokens):  {bacc_50*100:.2f}%")
    print(f"Balanced Accuracy (Truncated to exactly 100 tokens): {bacc_100*100:.2f}%")
    
    print("\nClassification Report (100 tokens):")
    print(classification_report(true_labels, fixed_len_preds_100, target_names=["AS", "LS"]))

    # Save DataFrame for Notebook Consumption
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results = pd.DataFrame({
        "text": all_texts,
        "length": lengths,
        "probability": probabilities,
        "true_label": true_labels,
        "dummy_pred": dummy_preds,
        "fixed_50_pred": fixed_len_preds_50,
        "fixed_100_pred": fixed_len_preds_100,
    })
    df_results.to_csv(output_csv, index=False)
    print(f"\nErgebnisse erfolgreich gespeichert in: {output_csv}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check length bias in BiLSTM classifier")
    parser.add_argument("--dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json")
    parser.add_argument("--model_path", default="results/models/bilstm_article_classifier.pt")
    parser.add_argument("--vocab_path", default="data/vocabs/article_vocab.json")
    parser.add_argument("--vocab_source_csv", default="data/analysis/corpus_master.csv")
    parser.add_argument("--output_csv", default="results/evaluation/length_bias_results.csv")
    args = parser.parse_args()
    
    if args.dataset_path:
        DATASET_PATH = args.dataset_path
    if args.model_path:
        MODEL_PATH = args.model_path
    if args.vocab_path:
        VOCAB_PATH = args.vocab_path
    if args.vocab_source_csv:
        VOCAB_SOURCE_CSV = args.vocab_source_csv
        
    main(output_csv=args.output_csv)

