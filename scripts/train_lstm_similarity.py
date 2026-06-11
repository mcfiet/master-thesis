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
MAX_SEQ_LEN = 100
MIN_SENT_LEN = 3
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
EPOCHS = 20
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(min_sim, max_sim):
    print(f"Loading data from {CSV_PATH} with {min_sim} <= similarity <= {max_sim}...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Source file not found: {CSV_PATH}")
        
    df = pd.read_csv(CSV_PATH)
    
    # Filter by similarity
    mask = (df["semantic_similarity_8192"] >= min_sim) & (df["semantic_similarity_8192"] <= max_sim)
    df_filtered = df[mask]
    
    print(f"Found {len(df_filtered)} article pairs matching the criteria (out of {len(df)} total).")
    
    ls_sentences = []
    as_sentences = []
    
    # Using small model for speed in sentence splitting
    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    except OSError:
        print("SpaCy model 'de_core_news_sm' not found. Please run 'python -m spacy download de_core_news_sm'")
        raise

    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Processing texts"):
        ls_text = str(row["ls_text"])
        as_text = str(row["as_text"])
        
        # Process LS text
        ls_doc = nlp(ls_text)
        for sent in ls_doc.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= MIN_SENT_LEN:
                ls_sentences.append(tokens)
        
        # Process AS text
        as_doc = nlp(as_text)
        for sent in as_doc.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= MIN_SENT_LEN:
                as_sentences.append(tokens)
    
    print(f"Total extracted: {len(ls_sentences)} LS sentences and {len(as_sentences)} AS sentences.")
    
    if len(ls_sentences) < 10 or len(as_sentences) < 10:
        raise ValueError("Not enough data matching the similarity criteria to perform training.")
        
    # Balance classes
    min_len = min(len(ls_sentences), len(as_sentences))
    random.seed(42) # Ensure reproducible class balancing
    random.shuffle(ls_sentences)
    random.shuffle(as_sentences)
    ls_sentences = ls_sentences[:min_len]
    as_sentences = as_sentences[:min_len]
    
    X = ls_sentences + as_sentences
    y = [1] * len(ls_sentences) + [0] * len(as_sentences)
    
    return X, y

class Vocab:
    def __init__(self, sentences, max_size=20000, min_freq=2):
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

class SentenceDataset(Dataset):
    def __init__(self, X, y, vocab, max_len):
        self.X = X
        self.y = y
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        tokens = self.X[idx]
        encoded = self.vocab.encode(tokens)[:self.max_len]
        padded = encoded + [0] * (self.max_len - len(encoded))
        return torch.tensor(padded, dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=1):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

def train(min_sim, max_sim):
    try:
        X, y = load_data(min_sim, max_sim)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=42, stratify=y_train_val)
    
    vocab = Vocab(X_train)
    print(f"Vocab size: {len(vocab)}")
    
    train_ds = SentenceDataset(X_train, y_train, vocab, MAX_SEQ_LEN)
    val_ds = SentenceDataset(X_val, y_val, vocab, MAX_SEQ_LEN)
    test_ds = SentenceDataset(X_test, y_test, vocab, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = BiLSTMClassifier(len(vocab), EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0
    patience = 5
    counter = 0
    
    model_save_path = f"results/lstm_baseline_sim_{min_sim:.2f}_to_{max_sim:.2f}.pt"
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                outputs = model(batch_x).squeeze()
                preds = torch.round(torch.sigmoid(outputs))
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(batch_y.numpy())
        
        val_acc = balanced_accuracy_score(val_targets, val_preds)
        print(f"Epoch {epoch+1} - Avg Train Loss: {train_loss/len(train_loader):.4f}, Val BAcc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    # Final Test
    print("\nEvaluating on Test Set...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x).squeeze()
            preds = torch.round(torch.sigmoid(outputs))
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(batch_y.numpy())
            
    print("\nTest Results:")
    print(classification_report(test_targets, test_preds, target_names=["Normal", "Simple"]))
    print(f"Balanced Accuracy: {balanced_accuracy_score(test_targets, test_preds):.4f}")
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM Baseline with Similarity Filtering")
    parser.add_argument("--min-sim", type=float, default=0.0, help="Minimum semantic similarity score (0.0 to 1.0)")
    parser.add_argument("--max-sim", type=float, default=1.0, help="Maximum semantic similarity score (0.0 to 1.0)")
    args = parser.parse_args()
    
    train(args.min_sim, args.max_sim)
