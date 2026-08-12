import os
import sys
import datetime
import random
import argparse
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# LOGGING & DIRECTORY SETUP
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
parser.add_argument('--lh_dataset_path', required=True)
parser.add_argument('--corpus_csv_path', required=True)
parser.add_argument('--sft_model_temp_path', required=True)
parser.add_argument('--min_sim', type=float, required=True)
parser.add_argument('--max_sim', type=float, required=True)
parser.add_argument('--max_source_len', type=int, required=True)
parser.add_argument('--max_target_len', type=int, required=True)
parser.add_argument('--model_name', required=True)
args = parser.parse_args()

LH_DATASET_PATH = args.lh_dataset_path
CORPUS_CSV_PATH = args.corpus_csv_path
SFT_MODEL_TEMP_PATH = args.sft_model_temp_path
MIN_SIM = args.min_sim
MAX_SIM = args.max_sim
MAX_SOURCE_LEN = args.max_source_len
MAX_TARGET_LEN = args.max_target_len
MODEL_NAME = args.model_name

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Nutze Device: {DEVICE}")

# ==============================================================================
# DATA LOADING & SPLITTING
# ==============================================================================
def load_filtered_corpus(csv_path=CORPUS_CSV_PATH, min_sim=MIN_SIM, max_sim=MAX_SIM):
    json_path = csv_path.replace(".csv", ".json")
    if os.path.exists(json_path):
        print(f"Lade Datensatz aus JSON: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        pairs = []
        for row in data:
            sim = row.get("semantic_similarity_8192")
            if sim is not None and min_sim <= sim <= max_sim:
                as_text = str(row.get("as_text") or "").strip()
                ls_text = str(row.get("ls_text") or "").strip()
                if as_text and ls_text:
                    pairs.append({
                        "source": row.get("source"),
                        "as_text": as_text,
                        "ls_text": ls_text
                    })
        print(f"Gesamtpaare in JSON: {len(data)}")
        print(f"Paare nach Filterung ({min_sim} <= Sim <= {max_sim}): {len(pairs)}")
        return pairs

    print(f"Lade Datensatz aus CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    filtered_df = df[
        (df["semantic_similarity_8192"] >= min_sim) & 
        (df["semantic_similarity_8192"] <= max_sim)
    ]
    print(f"Gesamtpaare in CSV: {len(df)}")
    print(f"Paare nach Filterung ({min_sim} <= Sim <= {max_sim}): {len(filtered_df)}")
    
    pairs = []
    for _, row in filtered_df.iterrows():
        as_text = str(row["as_text"]).strip()
        ls_text = str(row["ls_text"]).strip()
        if as_text and ls_text:
            pairs.append({
                "source": row["source"],
                "as_text": as_text,
                "ls_text": ls_text
            })
    return pairs

all_pairs = load_filtered_corpus(min_sim=MIN_SIM, max_sim=MAX_SIM)
print(f"Gesamtzahl geladener Artikel-Paare: {len(all_pairs)}")

random.shuffle(all_pairs)
split_idx = int(0.85 * len(all_pairs))
train_data = all_pairs[:split_idx]
val_data = all_pairs[split_idx:]
print(f"Trainingsdaten: {len(train_data)} Paare | Validierungsdaten: {len(val_data)} Paare")

# ==============================================================================
# MODEL & TOKENIZER SETUP
# ==============================================================================
print(f"Lade Tokenizer & Modell: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

if "mbart" in MODEL_NAME.lower():
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

if os.path.exists(SFT_MODEL_TEMP_PATH):
    state_dict = torch.load(SFT_MODEL_TEMP_PATH, map_location=DEVICE)
    seq2seq_model.load_state_dict(state_dict)
    print("Erfolgreich SFT-Modellgewichte von Festplatte geladen!")

# ==============================================================================
# TRANSLATION DATASET & DATALOADER
# ==============================================================================
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len=256, max_tgt_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = "Übersetze in Leichte Sprache: " + item["as_text"]
        tgt_text = item["ls_text"]
        
        inputs = self.tokenizer(
            src_text, max_length=self.max_src_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            tgt_text, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"]
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "raw_as": item["as_text"],
            "raw_ls": item["ls_text"]
        }

train_dataset = TranslationDataset(train_data, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN)
val_dataset = TranslationDataset(val_data, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(f"DataLoader erstellt: {len(train_loader)} Batches (Train), {len(val_loader)} Batches (Val).")

# ==============================================================================
# TRAINING & VALIDATION LOOPS
# ==============================================================================
def train_sft_epoch(model, dataloader, optimizer, scheduler, accumulation_steps=4):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="SFT Training")):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * accumulation_steps
        
    return total_loss / len(dataloader)

def validate_sft_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="SFT Validation"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
            total_loss += outputs.loss.item()
            
    return total_loss / len(dataloader)

NUM_EPOCHS = 20
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": []}

optimizer = AdamW(seq2seq_model.parameters(), lr=1e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=150, 
    num_training_steps=len(train_loader) * NUM_EPOCHS
)

print("Starte SFT Training mit Validation und Early Stopping...")
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoche {epoch + 1}/{NUM_EPOCHS} ---")
    sft_loss = train_sft_epoch(seq2seq_model, train_loader, optimizer, scheduler)
    val_loss = validate_sft_epoch(seq2seq_model, val_loader)
    
    history["train_loss"].append(sft_loss)
    history["val_loss"].append(val_loss)
    
    print(f"Epoche {epoch + 1} | Train Loss: {sft_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(seq2seq_model.state_dict(), SFT_MODEL_TEMP_PATH)
        print("-> Neues bestes Modell gespeichert.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early Stopping getriggert! Keine Verbesserung des Val Loss seit {patience} Epochen.")
            seq2seq_model.load_state_dict(torch.load(SFT_MODEL_TEMP_PATH))
            break

if os.path.exists(SFT_MODEL_TEMP_PATH):
    seq2seq_model.load_state_dict(torch.load(SFT_MODEL_TEMP_PATH))
    print("Die besten SFT-Gewichte wurden erfolgreich geladen!")

# Save SFT Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], marker='o', label='Train Loss')
plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], marker='s', label='Validation Loss')
plt.title("SFT Loss Kurven (Overfitting-Check)")
plt.xlabel("Epoche")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "sft_loss_curves.png"))
plt.close()

# ==============================================================================
# EVALUATION ON LEBENSHILFE DATASET
# ==============================================================================
SYNTHETIC_MODEL_PATH = "results/models/bilstm_synthetic_regression.pt"
SYNTHETIC_VOCAB_PATH = "data/vocabs/synthetic_vocab.json"
W_STYLE = 0.5
W_SEM = 0.5

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
        return self.sigmoid(out)

if not os.path.exists(LH_DATASET_PATH):
    print(f"Lebenshilfe dataset not found at {LH_DATASET_PATH}. Skipping evaluation.")
else:
    with open(LH_DATASET_PATH, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    with open(SYNTHETIC_VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        synthetic_stoi = vocab_data.get("stoi", vocab_data)

    bilstm_model = BiLSTMRegressor(len(synthetic_stoi), embed_dim=128, hidden_dim=128)
    if os.path.exists(SYNTHETIC_MODEL_PATH):
        bilstm_model.load_state_dict(torch.load(SYNTHETIC_MODEL_PATH, map_location="cpu"))
        bilstm_model.eval()
        print(f"BiLSTM Synthetic Regressor erfolgreich geladen von {SYNTHETIC_MODEL_PATH}")
    else:
        print(f"Warnung: {SYNTHETIC_MODEL_PATH} nicht gefunden.")

    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])

    def predict_simplicity_score(texts):
        scores = []
        unk_idx = synthetic_stoi.get("<unk>") or synthetic_stoi.get("<UNK>") or 1
        for text in texts:
            doc = nlp(text)
            tokens = [t.text.lower() for t in doc if not t.is_space]
            indices = [synthetic_stoi.get(t, unk_idx) for t in tokens[:150]]
            if len(indices) == 0:
                indices = [0]
            inp_tensor = torch.tensor([indices], dtype=torch.long, device="cpu")
            with torch.no_grad():
                score = bilstm_model(inp_tensor).item()
            scores.append(score)
        return np.array(scores)

    SBERT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device="cpu")

    def predict_semantic_similarity(source_texts, generated_texts):
        emb_src = sbert_model.encode(source_texts, convert_to_tensor=True)
        emb_gen = sbert_model.encode(generated_texts, convert_to_tensor=True)
        cosine_sims = util.cos_sim(emb_src, emb_gen).diagonal().cpu().numpy()
        return cosine_sims

    class CompositeRewardEvaluator:
        def __init__(self, w_style=0.5, w_sem=0.5):
            self.w_style = w_style
            self.w_sem = w_sem
            
        def compute_reward(self, source_texts, generated_texts):
            r_style = predict_simplicity_score(generated_texts)
            r_sem = predict_semantic_similarity(source_texts, generated_texts)
            r_sem_norm = np.clip((r_sem + 1.0) / 2.0, 0.0, 1.0)
            total_reward = self.w_style * r_style + self.w_sem * r_sem_norm
            return total_reward, r_style, r_sem_norm

    reward_evaluator = CompositeRewardEvaluator(w_style=W_STYLE, w_sem=W_SEM)

    def evaluate_on_lebenshilfe(model, tokenizer, lh_data, reward_evaluator, max_samples=49):
        model.eval()
        as_texts = [item["as_text"] for item in lh_data[:max_samples]]
        ls_ref_texts = [item["ls_text"] for item in lh_data[:max_samples]]
        
        batch_size = 16
        gen_texts = []
        for i in tqdm(range(0, len(as_texts), batch_size), desc="Übersetze Lebenshilfe Datensatz"):
            batch_src = as_texts[i:i+batch_size]
            prompts = ["Übersetze in Leichte Sprache: " + t for t in batch_src]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=MAX_SOURCE_LEN, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=MAX_TARGET_LEN,
                    num_beams=4,
                    repetition_penalty=2.5,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_texts.extend(decoded)
            
        tot_reward, r_style, r_sem = reward_evaluator.compute_reward(as_texts, gen_texts)
        sim_to_ref = predict_semantic_similarity(ls_ref_texts, gen_texts)
        sim_to_ref_norm = np.clip((sim_to_ref + 1.0) / 2.0, 0.0, 1.0)
        
        df_res = pd.DataFrame({
            "as_text": as_texts,
            "ls_reference": ls_ref_texts,
            "model_translation": gen_texts,
            "synthetic_simplicity_score": r_style,
            "sbert_sim_to_as": r_sem,
            "sbert_sim_to_ref": sim_to_ref_norm,
            "composite_reward": tot_reward
        })
        
        print("\n=================== EVALUIERUNGSERGEBNISSE (LEBENSHILFE) ===================")
        print(f"Ø Synthetic-Einfachheits-Score (R_style):       {r_style.mean():.4f} ± {r_style.std():.4f}")
        print(f"Ø SBERT-Ähnlichkeit zur AS-Quelle (R_sem):   {r_sem.mean():.4f} ± {r_sem.std():.4f}")
        print(f"Ø SBERT-Ähnlichkeit zu echter LS-Referenz:  {sim_to_ref_norm.mean():.4f} ± {sim_to_ref_norm.std():.4f}")
        print(f"Ø Composite Reward:                        {tot_reward.mean():.4f} ± {tot_reward.std():.4f}")
        print("========================================================================\n")
        
        return df_res

    lh_eval_df = evaluate_on_lebenshilfe(seq2seq_model, tokenizer, lh_data, reward_evaluator)

    print("\n--- QUALITATIVE STICHPROBEN (LEBENSHILFE TESTSET) ---")
    for idx, row in lh_eval_df.head(3).iterrows():
        print(f"\nArtikel #{idx+1}:")
        print(f"AS-Quelle:   {row['as_text'][:300]}...")
        print(f"LS-Referenz: {row['ls_reference'][:300]}...")
        print(f"Modell-Uebers: {row['model_translation'][:300]}...")
        print(f"Style Reward: {row['synthetic_simplicity_score']:.4f} | Sem Similarity: {row['sbert_sim_to_as']:.4f} | Sim to Ref: {row['sbert_sim_to_ref']:.4f}")

    # Plot metrics distributions
    plt.figure(figsize=(10, 6))
    metrics = [
        ("synthetic_simplicity_score", "Style Reward (R_style)", "blue"),
        ("sbert_sim_to_as", "Semantic Preservation (R_sem)", "green"),
        ("sbert_sim_to_ref", "Similarity to Reference", "orange")
    ]
    for col, label, color in metrics:
        plt.hist(lh_eval_df[col], bins=15, alpha=0.5, label=label, color=color, edgecolor="black")
    plt.xlabel("Metrik-Score [0, 1]")
    plt.ylabel("Anzahl Artikel")
    plt.title("Metrik-Verteilung auf dem Lebenshilfe-Testdatensatz (Out-of-Domain)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plot_dir, "sft_lh_metrics_distribution.png"))
    plt.close()
