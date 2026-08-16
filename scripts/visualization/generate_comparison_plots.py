import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

# Set style
sns.set_theme(style="whitegrid")

# Paths
LH_WITH_STEPS_PATH = "data/lebenshilfe/lebenshilfe_dataset_with_steps.json"
MIXUP_MODEL_PATH = "results/models/bilstm_mixup_regression_hybrid_cyclic.pt"
MIXUP_VOCAB_PATH = "data/vocabs/mixup_vocab.json"
SYNTHETIC_MODEL_PATH = "results/models/bilstm_synthetic_regression.pt"
SYNTHETIC_VOCAB_PATH = "data/vocabs/synthetic_vocab.json"
IMG_DIR = "research/img/analysis"

os.makedirs(IMG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# 1. Generate Learning Curve Plot
epochs = list(range(1, 16))
train_losses = [0.0980, 0.0551, 0.0437, 0.0356, 0.0291, 0.0246, 0.0220, 0.0178, 0.0156, 0.0139, 0.0134, 0.0128, 0.0117, 0.0115, 0.0112]
val_losses = [0.0691, 0.0539, 0.0425, 0.0522, 0.0630, 0.0456, 0.0432, 0.0490, 0.0401, 0.0419, 0.0496, 0.0537, 0.0542, 0.0528, 0.0515]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train MSE Loss", marker="o", linewidth=2)
plt.plot(epochs, val_losses, label="Val MSE Loss", marker="s", linewidth=2)
plt.axvline(x=9, color="red", linestyle="--", label="Bester Checkpoint (Epoche 9)")
plt.title("Lernkurve des BiLSTM-Regressors (Synthetische Stufen)")
plt.xlabel("Epoche")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "synthetic_bilstm_learning_curve.png"), dpi=300)
plt.close()
print("Saved learning curve.")

# 2. Evaluate Models
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

nlp = spacy.blank("de")

def load_vocab_dict(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "stoi" in data:
        return data["stoi"]
    return data

class EvalDataset(Dataset):
    def __init__(self, samples, vocab_map, nlp_spacy, max_len=256):
        self.samples = samples
        self.vocab = vocab_map
        self.nlp = nlp_spacy
        self.max_len = max_len
        self.unk_id = self.vocab.get("<unk>") or self.vocab.get("<UNK>") or 1
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item = self.samples[idx]
        doc = self.nlp(item["text"])
        tokens = [t.text.lower() for t in doc if not t.is_space][:self.max_len]
        token_ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if not token_ids:
            token_ids = [0]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(item["target"], dtype=torch.float32)

def pad_collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded_seqs, targets

def evaluate_model(model_path, vocab_path, samples):
    vocab_map = load_vocab_dict(vocab_path)
    vocab_size = len(vocab_map)
    
    model = BiLSTMRegressor(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ds = EvalDataset(samples, vocab_map, nlp)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    
    preds, targets = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            x_b = x_b.to(device)
            out = model(x_b).cpu().numpy()
            preds.extend(out)
            targets.extend(y_b.numpy())
            
    return np.array(preds), np.array(targets)

# Load samples
with open(LH_WITH_STEPS_PATH, "r", encoding="utf-8") as f:
    articles = json.load(f)
    
eval_samples = []
for art_idx, art in enumerate(articles):
    ls_text = art.get("ls_text", "").strip()
    as_text = art.get("as_text", "").strip()
    steps = art.get("intermediate_steps", {})
    
    if ls_text:
        eval_samples.append({"text": ls_text, "target": 0.0, "stage": "0.00"})
    if as_text:
        eval_samples.append({"text": as_text, "target": 1.0, "stage": "1.00"})
        
    for step_str, step_text in steps.items():
        try:
            target_val = float(step_str)
            if step_text and step_text.strip():
                eval_samples.append({
                    "text": step_text.strip(),
                    "target": target_val,
                    "stage": f"{target_val:.2f}"
                })
        except ValueError:
            continue

print(f"Loaded {len(eval_samples)} evaluation samples.")
preds_mixup, targets_eval = evaluate_model(MIXUP_MODEL_PATH, MIXUP_VOCAB_PATH, eval_samples)
preds_synthetic, _ = evaluate_model(SYNTHETIC_MODEL_PATH, SYNTHETIC_VOCAB_PATH, eval_samples)

# Harmonize scales to Simplicity (1.0 = LS, 0.0 = AS)
# MixUp is already Simplicity scale (1.0 = LS, 0.0 = AS), so we do not invert it.
# Synthetic is Complexity scale (0.0 = LS, 1.0 = AS), so we invert it.
preds_synthetic = 1.0 - preds_synthetic
targets_eval = 1.0 - targets_eval

# 3. Save Comparison Boxplot Plot
df_comp = pd.DataFrame({
    "Zielstufe": [f"{t:.2f}" for t in targets_eval] * 2,
    "Vorhergesagter Score": np.concatenate([preds_mixup, preds_synthetic]),
    "Modell": ["MixUp-Modell (Variante D)"] * len(preds_mixup) + ["Synthetisches LLM-Modell"] * len(preds_synthetic)
})

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_comp, x="Zielstufe", y="Vorhergesagter Score", hue="Modell", palette="Set2")
plt.title("Vergleich der Einfachheits-Vorhersagen (1.0 = LS, 0.0 = AS)")
plt.xlabel("Vorgegebene Zielstufe (LLM)")
plt.ylabel("Modellvorhersage (BiLSTM Einfachheits-Score)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Modell")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "compare_boxplots_mixup_vs_synthetic.png"), dpi=300)
plt.close()
print("Saved boxplots comparison.")

# 4. Save Regress Plot Comparison
r_m, _ = pearsonr(preds_mixup, targets_eval)
r_s, _ = pearsonr(preds_synthetic, targets_eval)

plt.figure(figsize=(10, 6))
sns.regplot(x=targets_eval, y=preds_mixup, label=f"MixUp (r={r_m:.3f})", scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
sns.regplot(x=targets_eval, y=preds_synthetic, label=f"Synthetisch (r={r_s:.3f})", scatter_kws={"alpha": 0.3}, line_kws={"color": "blue"})
plt.plot([0, 1], [0, 1], "k--", label="Ideale Monotonie (1:1)")
plt.title("Regressionsvergleich auf den 5 Synthetik-Stufen (Einfachheit)")
plt.xlabel("Zielstufe (Ground Truth / LLM Target - Einfachheit)")
plt.ylabel("Vorhergesagter Einfachheits-Score")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "compare_regplot_mixup_vs_synthetic.png"), dpi=300)
plt.close()
print("Saved regression plot comparison.")

# 5. Save KDE Density Comparison Plot
plt.figure(figsize=(10, 5))
sns.kdeplot(targets_eval, label="Zielstufe (Ground Truth)", fill=True, alpha=0.15, color="gray", linewidth=2)
sns.kdeplot(preds_mixup, label=f"MixUp-Modell (r={r_m:.3f})", color="red", linewidth=2)
sns.kdeplot(preds_synthetic, label=f"Synthetisches LLM-Modell (r={r_s:.3f})", color="blue", linewidth=2)
plt.title("Dichtevergleich (KDE) der Vorhersagen vs. Zielstufe (Einfachheit)")
plt.xlabel("Einfachheits-Score (1.0 = LS, 0.0 = AS)")
plt.ylabel("Dichte")
plt.xlim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "compare_kde_mixup_vs_synthetic.png"), dpi=300)
plt.close()
print("Saved KDE comparison plot.")
