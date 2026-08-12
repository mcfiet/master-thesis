import os
import sys
import datetime
import random
import json
import argparse
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
log_dir = "results/logs"
plot_dir = "results/plots"
os.makedirs(log_dir, exist_ok=True)
os.makedirs("results/models", exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

script_name = os.path.basename(__file__).replace(".py", "")
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout
print(f"Log file initialized at: {log_file}")
print("Aktuelles Arbeitsverzeichnis:", os.getcwd())

# ==============================================================================
# SEED CONFIGURATION
# ==============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Globaler Seed auf {seed} gesetzt.")

set_seed(42)

# ==============================================================================
# ZENTRALE KONFIGURATION & PARAMS
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--corpus_with_steps_path', required=True)
parser.add_argument('--lh_with_steps_path', required=True)
parser.add_argument('--model_save_path', required=True)
parser.add_argument('--vocab_save_path', required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--max_seq_len', type=int, required=True)
args = parser.parse_args()

CORPUS_WITH_STEPS_PATH = args.corpus_with_steps_path
LH_WITH_STEPS_PATH = args.lh_with_steps_path
MODEL_SAVE_PATH = args.model_save_path
VOCAB_SAVE_PATH = args.vocab_save_path
EPOCHS = args.epochs
MAX_SEQ_LEN = args.max_seq_len

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Nutze Device: {device}")

# ==============================================================================
# DATA LOADING & FLAT SAMPLING
# ==============================================================================
def load_flattened_samples(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {json_path}")
        
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)
        
    samples = []
    for art_idx, art in enumerate(articles):
        ls_text = art.get("ls_text", "").strip()
        as_text = art.get("as_text", "").strip()
        steps = art.get("intermediate_steps", {})
        
        if ls_text:
            samples.append({"text": ls_text, "target": 1.0, "article_id": art_idx, "stage": "1.00 (LS)"})
        if as_text:
            samples.append({"text": as_text, "target": 0.0, "article_id": art_idx, "stage": "0.00 (AS)"})
            
        for step_str, step_text in steps.items():
            try:
                target_val = float(step_str)
                if step_text and step_text.strip():
                    samples.append({
                        "text": step_text.strip(),
                        "target": 1.0 - target_val,
                        "article_id": art_idx,
                        "stage": f"{1.0 - target_val:.2f}"
                    })
            except ValueError:
                continue
                
    return samples, articles

raw_samples, raw_articles = load_flattened_samples(CORPUS_WITH_STEPS_PATH)
print(f"Anzahl geladener Artikel: {len(raw_articles)}")
print(f"Anzahl ausgeflachter Text-Samples: {len(raw_samples)}")

stage_counts = Counter(s["stage"] for s in raw_samples)
for stage, count in sorted(stage_counts.items()):
    print(f"  Stufe {stage}: {count} Samples")

# ==============================================================================
# VOCABULARY SETUP
# ==============================================================================
nlp = spacy.blank("de")

def build_vocab(samples, min_freq=2):
    counter = Counter()
    for sample in tqdm(samples, desc="Baue Vokabular"):
        doc = nlp(sample["text"])
        tokens = [t.text.lower() for t in doc if not t.is_space]
        counter.update(tokens)
        
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = len(vocab)
    return vocab

vocab = build_vocab(raw_samples, min_freq=2)
print(f"Vokabulargröße: {len(vocab)} Wörter")

os.makedirs(os.path.dirname(VOCAB_SAVE_PATH), exist_ok=True)
with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
print(f"Vokabular gespeichert unter: {VOCAB_SAVE_PATH}")

# ==============================================================================
# PYTORCH DATASET & DATALOADER
# ==============================================================================
class SyntheticStepDataset(Dataset):
    def __init__(self, samples, vocab, nlp_spacy, max_len=MAX_SEQ_LEN):
        self.samples = samples
        self.vocab = vocab
        self.nlp = nlp_spacy
        self.max_len = max_len
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        doc = self.nlp(item["text"])
        tokens = [t.text.lower() for t in doc if not t.is_space][:self.max_len]
        
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        if not token_ids:
            token_ids = [self.vocab["<PAD>"]]
            
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(item["target"], dtype=torch.float32)

def pad_collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded_seqs, targets

unique_art_ids = list(set(s["article_id"] for s in raw_samples))
random.shuffle(unique_art_ids)
split_idx = int(0.8 * len(unique_art_ids))
train_art_ids = set(unique_art_ids[:split_idx])
val_art_ids = set(unique_art_ids[split_idx:])

train_samples = [s for s in raw_samples if s["article_id"] in train_art_ids]
val_samples = [s for s in raw_samples if s["article_id"] in val_art_ids]

print(f"Train-Samples: {len(train_samples)} (aus {len(train_art_ids)} Artikeln)")
print(f"Val-Samples: {len(val_samples)} (aus {len(val_art_ids)} Artikeln)")

train_ds = SyntheticStepDataset(train_samples, vocab, nlp, max_len=MAX_SEQ_LEN)
val_ds = SyntheticStepDataset(val_samples, vocab, nlp, max_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out).squeeze(-1)

model = BiLSTMRegressor(vocab_size=len(vocab), embed_dim=128, hidden_dim=128, dropout=0.3).to(device)
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
best_val_loss = float("inf")
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * x_batch.size(0)
        
    avg_train_loss = total_train_loss / len(train_ds)
    train_losses.append(avg_train_loss)
    
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            total_val_loss += loss.item() * x_batch.size(0)
            
    avg_val_loss = total_val_loss / len(val_ds)
    val_losses.append(avg_val_loss)
    scheduler.step()
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        saved_str = " -> [Modell gespeichert]"
    else:
        saved_str = ""
        
    print(f"Epoche {epoch:02d}/{EPOCHS:02d} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}{saved_str}")

# ==============================================================================
# EVALUATION & PLOTTING
# ==============================================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train MSE Loss", marker="o")
plt.plot(range(1, EPOCHS + 1), val_losses, label="Val MSE Loss", marker="s")
plt.title("Lernkurve des BiLSTM-Regressors (Synthetische Stufen)")
plt.xlabel("Epoche")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(os.path.join(plot_dir, "synthetic_training_loss.png"))
plt.close()

# Load best model and evaluate on out-of-domain Lebenshilfe set
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

lh_samples, lh_articles = load_flattened_samples(LH_WITH_STEPS_PATH)
lh_ds = SyntheticStepDataset(lh_samples, vocab, nlp, max_len=MAX_SEQ_LEN)
lh_loader = DataLoader(lh_ds, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)

all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in lh_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

mse = np.mean((all_preds - all_targets) ** 2)
mae = np.mean(np.abs(all_preds - all_targets))
pearson_r, _ = pearsonr(all_preds, all_targets)
spearman_rho, _ = spearmanr(all_preds, all_targets)

print("=" * 60)
print("OUT-OF-DOMAIN EVALUIERUNG: LEBENSHILFE SYNTHETIK-SET")
print("=" * 60)
print(f"Anzahl Evaluierungssamples: {len(all_targets)}")
print(f"MSE (Mean Squared Error):  {mse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"Pearson Korrelation r:    {pearson_r:.4f}")
print(f"Spearman Korrelation rho: {spearman_rho:.4f}")
print("=" * 60)

df_eval = pd.DataFrame({
    "Zielstufe": [f"{t:.2f}" for t in all_targets],
    "Vorhergesagter Score": all_preds
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_eval, x="Zielstufe", y="Vorhergesagter Score", palette="viridis")
sns.stripplot(data=df_eval, x="Zielstufe", y="Vorhergesagter Score", color="black", alpha=0.3, jitter=0.2)
plt.title("Vorhergesagte Komplexitäts-Scores auf den Lebenshilfe-Synthetik-Stufen")
plt.xlabel("Vorgegebene Zielstufe (LLM)")
plt.ylabel("Modellvorhersage (BiLSTM Regressor)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(plot_dir, "synthetic_eval_boxplot.png"))
plt.close()
