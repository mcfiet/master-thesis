import argparse
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
import numpy as np
import os

# Configuration defaults
CSV_PATH = "results/information_loss_analysis_cleaned.csv"
MAX_SEQ_LEN = 512 # Increased for full articles
BATCH_SIZE = 32 # Reduced batch size for longer sequences
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_article_data(min_sim, max_sim):
    print(f"Loading article data from {CSV_PATH} with {min_sim} <= similarity <= {max_sim}...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source file not found: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    
    # Filter by similarity
    mask = (df["semantic_similarity_8192"] >= min_sim) & (df["semantic_similarity_8192"] <= max_sim)
    df_filtered = df[mask]
    
    print(f"Found {len(df_filtered)} article pairs matching the criteria.")
    
    X = []
    y = []
    
    # Simple tokenizer (word-based) for speed on full articles
    # Using spacy tokenizer only (no linguistic features) for efficiency
    nlp = spacy.blank("de")
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Tokenizing articles"):
        ls_text = str(row["ls_text"])
        as_text = str(row["as_text"])
        
        # Process LS article
        ls_tokens = [t.text.lower() for t in nlp(ls_text) if not t.is_space]
        if len(ls_tokens) >= 10:
            X.append(ls_tokens)
            y.append(1) # Simple
            
        # Process AS article
        as_tokens = [t.text.lower() for t in nlp(as_text) if not t.is_space]
        if len(as_tokens) >= 10:
            X.append(as_tokens)
            y.append(0) # Normal
    
    print(f"Total articles loaded: {len(X)} ({y.count(1)} LS, {y.count(0)} AS)")
    
    if len(X) < 20:
        raise ValueError("Not enough data matching the similarity criteria.")
        
    return X, y

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

class ArticleDataset(Dataset):
    def __init__(self, X, y, vocab, max_len):
        self.X, self.y, self.vocab, self.max_len = X, y, vocab, max_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        tokens = self.X[idx]
        encoded = self.vocab.encode(tokens)[:self.max_len]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.4) # Increased dropout for smaller dataset
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

def train(min_sim, max_sim):
    try:
        X, y = load_article_data(min_sim, max_sim)
    except Exception as e:
        print(f"Error: {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)
    
    vocab = Vocab(X_train)
    print(f"Vocab size: {len(vocab)}")
    
    train_ds = ArticleDataset(X_train, y_train, vocab, MAX_SEQ_LEN)
    val_ds = ArticleDataset(X_val, y_val, vocab, MAX_SEQ_LEN)
    test_ds = ArticleDataset(X_test, y_test, vocab, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = BiLSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    patience = 7 # Slightly more patience for smaller dataset
    counter = 0
    
    model_save_path = f"results/lstm_article_sim_{min_sim:.2f}_to_{max_sim:.2f}.pt"
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx.to(DEVICE)).squeeze()
                val_preds.extend(torch.round(torch.sigmoid(out)).cpu().numpy())
                val_targets.extend(by.numpy())
        
        val_acc = balanced_accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, Val BAcc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    print("\nEvaluating on Test Set...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(DEVICE)).squeeze()
            test_preds.extend(torch.round(torch.sigmoid(out)).cpu().numpy())
            test_targets.extend(by.numpy())
            
    bacc = balanced_accuracy_score(test_targets, test_preds)
    print("\nTest Results:")
    print(classification_report(test_targets, test_preds, target_names=["Normal", "Simple"]))
    print(f"Balanced Accuracy: {bacc:.4f}")
    
    # Simple report line for parent script
    with open("results/article_experiments_summary.csv", "a") as f:
        f.write(f"{min_sim},{max_sim},{len(X)},{bacc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Article-level BiLSTM Classifier")
    parser.add_argument("--min-sim", type=float, default=0.0)
    parser.add_argument("--max-sim", type=float, default=1.0)
    args = parser.parse_args()
    
    train(args.min_sim, args.max_sim)
