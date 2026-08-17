import os
import sys
import datetime
import random
import argparse
import numpy as np
import pandas as pd
import spacy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import Counter
from tqdm import tqdm
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
parser = argparse.ArgumentParser(description="Training eines Vanilla RNN / RNN Baseline Regressors")
parser.add_argument('--csv_path', default="data/analysis/corpus_master.csv")
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_sim', type=float, default=0.98)
parser.add_argument('--min_sim', type=float, default=0.8)
parser.add_argument('--max_seq_len', type=int, default=256)
parser.add_argument('--rnn_type', type=str, default="rnn", choices=["rnn", "gru", "lstm"], help="Typ der Rekurrenz: 'rnn' (Vanilla), 'gru', 'lstm'")
parser.add_argument('--bidirectional', action="store_true", help="Falls gesetzt, wird das RNN bidirektional trainiert (Default: unidirektional)")
parser.add_argument('--model_save_path', default="results/models/rnn_mixup_regression.pt")
parser.add_argument('--vocab_save_path', default="data/vocabs/mixup_vocab.json")
args = parser.parse_args()

CSV_PATH = args.csv_path
BATCH_SIZE = args.batch_size
EMBEDDING_DIM = args.embedding_dim
EPOCHS = args.epochs
HIDDEN_DIM = args.hidden_dim
LR = args.lr
MAX_SIM = args.max_sim
MIN_SIM = args.min_sim
MAX_SEQ_LEN = args.max_seq_len
RNN_TYPE = args.rnn_type
BIDIRECTIONAL = args.bidirectional
MODEL_SAVE_PATH = args.model_save_path
VOCAB_SAVE_PATH = args.vocab_save_path

print("Konfiguration:")
print(f"  - RNN-Typ: {RNN_TYPE} (Bidirektional: {BIDIRECTIONAL})")
print(f"  - Embedding-Dim: {EMBEDDING_DIM}, Hidden-Dim: {HIDDEN_DIM}")
print(f"  - Max Seq Len: {MAX_SEQ_LEN}, Batch Size: {BATCH_SIZE}, LR: {LR}, Epochs: {EPOCHS}")
print(f"  - Modell-Speicherpfad: {MODEL_SAVE_PATH}")
print(f"  - Vokabular-Pfad: {VOCAB_SAVE_PATH}")

# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================
df = pd.read_csv(CSV_PATH)
mask = (df["semantic_similarity_8192"] >= MIN_SIM) & (df["semantic_similarity_8192"] <= MAX_SIM)
df_filtered = df[mask].dropna(subset=["ls_text", "as_text"])
print(f"Gefundene Artikelpaare: {len(df_filtered)}")

nlp = spacy.blank("de")
nlp.add_pipe("sentencizer")

# Train-Val-Test Split
train_val_df, test_df = train_test_split(df_filtered, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1111, random_state=42)
print(f"Training: {len(train_df)} Paare, Validierung: {len(val_df)} Paare, Test: {len(test_df)} Paare")

# ==============================================================================
# VOCABULARY
# ==============================================================================
class Vocab:
    def __init__(self, token_list=None, stoi_dict=None, max_size=25000, min_freq=2):
        if stoi_dict is not None:
            self.stoi = stoi_dict
            self.itos = [None] * len(stoi_dict)
            for token, idx in stoi_dict.items():
                self.itos[idx] = token
        else:
            counter = Counter(token_list)
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

# Falls Vokabular bereits existiert, laden – ansonsten neu erstellen
if os.path.exists(VOCAB_SAVE_PATH):
    print(f"Lade bestehendes Vokabular aus {VOCAB_SAVE_PATH}...")
    with open(VOCAB_SAVE_PATH, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    vocab = Vocab(stoi_dict=stoi)
else:
    print("Sammle Tokens für neues Vokabular...")
    all_train_tokens = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Vokab-Tokens sammeln"):
        for text in [str(row["ls_text"]), str(row["as_text"])]:
            doc = nlp(text)
            for token in doc:
                if not token.is_space:
                    all_train_tokens.append(token.text.lower())

    vocab = Vocab(token_list=all_train_tokens, max_size=25000, min_freq=2)
    os.makedirs(os.path.dirname(VOCAB_SAVE_PATH), exist_ok=True)
    with open(VOCAB_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
    print(f"Vokabular gespeichert unter: {VOCAB_SAVE_PATH}")

print(f"Vokabular-Größe: {len(vocab)}")

# ==============================================================================
# PYTORCH DATASET
# ==============================================================================
class MixupPyTorchDataset(Dataset):
    def __init__(self, df, vocab, nlp_sentencizer, max_seq_len=MAX_SEQ_LEN, mixtures_per_pair=5, is_train=True):
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.current_epoch = 0
        self.total_epochs = EPOCHS
        
        self.ls_data = []
        self.as_data = []
        self.static_samples = []
        
        seed = 42 if is_train else 99
        set_seed(seed)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Segmentiere & Tokenisiere"):
            ls_doc = nlp_sentencizer(str(row["ls_text"]))
            as_doc = nlp_sentencizer(str(row["as_text"]))
            
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
                
            self.ls_data.append(ls_sents)
            self.as_data.append(as_sents)
            
            article_idx = len(self.ls_data) - 1
            for _ in range(mixtures_per_pair):
                start_l, end_l = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
                sample_l = ls_sents[start_l:end_l]
                
                start_a, end_a = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
                sample_a = as_sents[start_a:end_a]
                
                if len(sample_l) == 0 and len(sample_a) == 0:
                    regression_target = 0.5
                    encoded = [0] * self.max_seq_len
                else:
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
                        
                self.static_samples.append((article_idx, encoded, regression_target))
                
        print(f"Generiert: {len(self.static_samples)} statische Samples.")
        
    def set_epoch(self, epoch, total_epochs):
        self.current_epoch = epoch
        self.total_epochs = total_epochs
            
    def __len__(self):
        return len(self.static_samples)
        
    def __getitem__(self, idx):
        article_idx, static_encoded, static_target = self.static_samples[idx]
        
        if not self.is_train:
            return torch.tensor(static_encoded, dtype=torch.long), torch.tensor(static_target, dtype=torch.float)
            
        p_dynamic = self.current_epoch / max(1, self.total_epochs - 1)
        
        if random.random() < p_dynamic:
            ls_sents = self.ls_data[article_idx]
            as_sents = self.as_data[article_idx]
            num_leicht = len(ls_sents)
            num_alltag = len(as_sents)
            
            start_l, end_l = sorted([random.randint(0, num_leicht), random.randint(0, num_leicht)])
            sample_l = ls_sents[start_l:end_l]
            
            start_a, end_a = sorted([random.randint(0, num_alltag), random.randint(0, num_alltag)])
            sample_a = as_sents[start_a:end_a]
            
            if len(sample_l) == 0 and len(sample_a) == 0:
                regression_target = 0.5
                encoded = [0] * self.max_seq_len
            else:
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
                    
            return torch.tensor(encoded, dtype=torch.long), torch.tensor(regression_target, dtype=torch.float)
        else:
            return torch.tensor(static_encoded, dtype=torch.long), torch.tensor(static_target, dtype=torch.float)

print("Erstelle PyTorch Datasets...")
train_dataset = MixupPyTorchDataset(train_df, vocab, nlp, max_seq_len=MAX_SEQ_LEN, mixtures_per_pair=20, is_train=True)
val_dataset = MixupPyTorchDataset(val_df, vocab, nlp, max_seq_len=MAX_SEQ_LEN, mixtures_per_pair=20, is_train=False)

# ==============================================================================
# MODEL ARCHITECTURE (RNN BASELINE REGRESSOR)
# ==============================================================================
class RNNRegressor(nn.Module):
    """
    Klassisches / Vanilla RNN Modell zur Regression von Komplexitätsscores als Baseline.
    Unterstützt Vanilla RNN, GRU und unidirektionales / bidirektionales Setup.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3, rnn_type="rnn", bidirectional=False):
        super(RNNRegressor, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, nonlinearity="tanh")
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError(f"Ungültiger rnn_type: {rnn_type}")
            
        fc_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_in_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        
        if self.rnn_type == "lstm":
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)
            
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
            
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out)

# ==============================================================================
# TRAINING SETUP
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Nutze Device: {DEVICE}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RNNRegressor(
    vocab_size=len(vocab),
    embed_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    rnn_type=RNN_TYPE,
    bidirectional=BIDIRECTIONAL
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5)

history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
best_val_loss = float('inf')
patience = 8
counter = 0

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
for epoch in range(EPOCHS):
    train_dataset.set_epoch(epoch, EPOCHS)
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(batch_x).squeeze()
        
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
            
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            preds = model(batch_x).squeeze()
            if preds.ndim == 0:
                preds = preds.unsqueeze(0)
                
            loss = criterion(preds, batch_y)
            val_loss += loss.item()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    epoch_train_loss = epoch_loss / len(train_loader)
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_mae = mean_absolute_error(all_targets, all_preds)
    
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_mae'].append(epoch_val_mae)
    
    print(f"Epoch {epoch+1:02d} | Train Loss (MSE): {epoch_train_loss:.4f} | Val Loss (MSE): {epoch_val_loss:.4f} | Val MAE: {epoch_val_mae:.4f}")
    
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"=> Modell gespeichert (bester Val Loss) unter {MODEL_SAVE_PATH}")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping ausgelöst.")
            break

    scheduler.step()

# ==============================================================================
# EVALUATION & PLOTTING
# ==============================================================================
plot_prefix = f"rnn_{RNN_TYPE}_{'bi' if BIDIRECTIONAL else 'uni'}"
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss (MSE)')
plt.plot(history['val_loss'], label='Val Loss (MSE)')
plt.title(f'Trainingsverlauf - {RNN_TYPE.upper()} Baseline Regressor')
plt.xlabel('Epoche')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(plot_dir, f"{plot_prefix}_training_loss.png"))
plt.close()

# Load best model and evaluate
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()
val_preds = []
val_targets = []

with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(DEVICE)
        preds = model(batch_x).squeeze()
        if preds.ndim == 0:
            preds = preds.unsqueeze(0)
        val_preds.extend(preds.cpu().numpy())
        val_targets.extend(batch_y.numpy())

plt.figure(figsize=(8, 6))
sns.scatterplot(x=val_targets, y=val_preds, alpha=0.6, color="teal")
plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Perfekte Vorhersage")
plt.title(f"Echte vs. Vorhergesagte Targets ({RNN_TYPE.upper()} Baseline Regression)")
plt.xlabel("Echte Lambda-Werte")
plt.ylabel("Vorhergesagte Lambda-Werte")
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(plot_dir, f"{plot_prefix}_predictions_scatter.png"))
plt.close()

val_mse = mean_squared_error(val_targets, val_preds)
val_mae = mean_absolute_error(val_targets, val_preds)
print("\n" + "="*50)
print(f"Abschließende Validierungsmetriken des besten {RNN_TYPE.upper()}-Modells:")
print(f"- Mean Squared Error (MSE): {val_mse:.4f}")
print(f"- Mean Absolute Error (MAE): {val_mae:.4f}")
print("="*50)
