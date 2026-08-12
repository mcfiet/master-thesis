import os
import sys
import datetime
import random
import json
import argparse
import copy
import gc
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
parser.add_argument('--output_dir', required=True)
parser.add_argument('--sft_model_temp_path', required=True)
parser.add_argument('--synthetic_model_path', required=True)
parser.add_argument('--synthetic_vocab_path', required=True)
parser.add_argument('--min_sim', type=float, required=True)
parser.add_argument('--max_sim', type=float, required=True)
parser.add_argument('--w_style', type=float, required=True)
parser.add_argument('--w_sem', type=float, required=True)
parser.add_argument('--max_source_len', type=int, required=True)
parser.add_argument('--max_target_len', type=int, required=True)
parser.add_argument('--model_name', required=True)
args = parser.parse_args()

LH_DATASET_PATH = args.lh_dataset_path
CORPUS_CSV_PATH = args.corpus_csv_path
OUTPUT_DIR = args.output_dir
SFT_MODEL_TEMP_PATH = args.sft_model_temp_path
SYNTHETIC_MODEL_PATH = args.synthetic_model_path
SYNTHETIC_VOCAB_PATH = args.synthetic_vocab_path
MIN_SIM = args.min_sim
MAX_SIM = args.max_sim
W_STYLE = args.w_style
W_SEM = args.w_sem
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

# ==============================================================================
# DATASET DEFINITION
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

if os.path.exists(SFT_MODEL_TEMP_PATH):
    seq2seq_model.load_state_dict(torch.load(SFT_MODEL_TEMP_PATH, map_location=DEVICE))
    print("Erfolgreich SFT-Modellgewichte von Festplatte geladen:", SFT_MODEL_TEMP_PATH)
else:
    raise FileNotFoundError(f"SFT-Modell unter {SFT_MODEL_TEMP_PATH} nicht gefunden! Bitte führe zuerst das SFT Training aus.")

# ==============================================================================
# REWARD MODELS SETUP
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
        return self.sigmoid(out)

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
print(f"Lade SBERT Modell: {SBERT_MODEL_NAME}...")
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

# ==============================================================================
# DPO TRAINING FUNCTIONS
# ==============================================================================
def generate_candidates(model, tokenizer, source_texts, num_return_sequences=2):
    model.eval()
    prompt_texts = ["Übersetze in Leichte Sprache: " + text for text in source_texts]
    inputs = tokenizer(prompt_texts, padding=True, truncation=True, max_length=MAX_SOURCE_LEN, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=MAX_TARGET_LEN,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.8,
            num_return_sequences=num_return_sequences
        )
        
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    expected_len = len(source_texts) * num_return_sequences
    if len(decoded) < expected_len:
        print(f"Warnung: Generierte Sequenzen ({len(decoded)}) weichen von Erwartung ({expected_len}) ab!")
        decoded += [""] * (expected_len - len(decoded))
        
    candidates = []
    for i in range(len(source_texts)):
        cands = decoded[i * num_return_sequences : (i + 1) * num_return_sequences]
        while len(cands) < num_return_sequences:
            cands.append("")
        candidates.append(cands)
    return candidates

def compute_dpo_loss(model, ref_model, tokenizer, src_texts, chosen_texts, rejected_texts, beta=0.3):
    model.train()
    ref_model.eval()
    
    prompts = ["Übersetze in Leichte Sprache: " + t for t in src_texts]
    inputs = tokenizer(prompts, padding=True, truncation=True, max_length=MAX_SOURCE_LEN, return_tensors="pt").to(DEVICE)
    chosen_enc = tokenizer(chosen_texts, padding=True, truncation=True, max_length=MAX_TARGET_LEN, return_tensors="pt").to(DEVICE)
    rejected_enc = tokenizer(rejected_texts, padding=True, truncation=True, max_length=MAX_TARGET_LEN, return_tensors="pt").to(DEVICE)
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            ref_chosen_loss = ref_model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                labels=chosen_enc["input_ids"]
            ).loss.item()
            
            ref_rejected_loss = ref_model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                labels=rejected_enc["input_ids"]
            ).loss.item()
            
        ref_logratios = -ref_chosen_loss + ref_rejected_loss
        
        out_chosen = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=chosen_enc["input_ids"]
        )
        pi_chosen_loss = out_chosen.loss
        
        out_rejected = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            labels=rejected_enc["input_ids"]
        )
        pi_rejected_loss = out_rejected.loss
        
        pi_logratios = -pi_chosen_loss + pi_rejected_loss
        
        logits = pi_logratios - ref_logratios
        dpo_loss = -torch.nn.functional.logsigmoid(beta * logits).mean()
        
    return dpo_loss

# ==============================================================================
# DPO TRAINING LOOP
# ==============================================================================
ref_model = copy.deepcopy(seq2seq_model).to(DEVICE)
for param in ref_model.parameters():
    param.requires_grad = False
ref_model.eval()

dpo_optimizer = AdamW(seq2seq_model.parameters(), lr=1e-6)
DPO_EPOCHS = 2
dpo_history = []

print("Starte DPO Training...")
for epoch in range(1, DPO_EPOCHS + 1):
    epoch_dpo_loss = 0.0
    epoch_style_rewards = []
    epoch_sem_rewards = []
    
    dpo_loader = DataLoader(
        TranslationDataset(train_data, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN), 
        batch_size=4, 
        shuffle=True
    )
    
    for batch in tqdm(dpo_loader, desc=f"DPO Epoch {epoch}/{DPO_EPOCHS}"):
        src_texts = batch["raw_as"]
        
        candidates = generate_candidates(seq2seq_model, tokenizer, src_texts, num_return_sequences=2)
        chosen_list = []
        rejected_list = []
        
        for src, cands in zip(src_texts, candidates):
            r1, st1, sem1 = reward_evaluator.compute_reward([src], [cands[0]])
            r2, st2, sem2 = reward_evaluator.compute_reward([src], [cands[1]])
            
            if r1[0] >= r2[0]:
                chosen_list.append(cands[0])
                rejected_list.append(cands[1])
                epoch_style_rewards.append(st1[0])
                epoch_sem_rewards.append(sem1[0])
            else:
                chosen_list.append(cands[1])
                rejected_list.append(cands[0])
                epoch_style_rewards.append(st2[0])
                epoch_sem_rewards.append(sem2[0])
                
        dpo_loss = compute_dpo_loss(seq2seq_model, ref_model, tokenizer, src_texts, chosen_list, rejected_list)
        dpo_optimizer.zero_grad()
        dpo_loss.backward()
        dpo_optimizer.step()
        
        epoch_dpo_loss += dpo_loss.item()
        
    avg_loss = epoch_dpo_loss / len(dpo_loader)
    avg_style = np.mean(epoch_style_rewards)
    avg_sem = np.mean(epoch_sem_rewards)
    
    dpo_history.append({
        "epoch": epoch,
        "loss": avg_loss,
        "style_reward": avg_style,
        "sem_reward": avg_sem
    })
    
    print(f"\n--- DPO Epoche {epoch} Ergebnisse ---")
    print(f"Ø DPO Loss:       {avg_loss:.4f}")
    print(f"Ø Style Reward:    {avg_style:.4f}")
    print(f"Ø Semantic Reward: {avg_sem:.4f}\n")

# Save DPO History Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs = [h["epoch"] for h in dpo_history]
losses = [h["loss"] for h in dpo_history]
style_rewards = [h["style_reward"] for h in dpo_history]
sem_rewards = [h["sem_reward"] for h in dpo_history]

ax1.plot(epochs, losses, marker='o', color='red', label='DPO Loss')
ax1.set_title("DPO Loss Verlauf")
ax1.set_xlabel("Epoche")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, style_rewards, marker='s', color='blue', label='Style Reward')
ax2.plot(epochs, sem_rewards, marker='^', color='green', label='Semantic Reward')
ax2.set_title("DPO Reward Verlauf")
ax2.set_xlabel("Epoche")
ax2.set_ylabel("Reward Score")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "dpo_training_history.png"))
plt.close()

# Save final model
os.makedirs(OUTPUT_DIR, exist_ok=True)
seq2seq_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Modell und Tokenizer erfolgreich gespeichert unter: {OUTPUT_DIR}")

# ==============================================================================
# EVALUATION ON LEBENSHILFE DATASET
# ==============================================================================
if not os.path.exists(LH_DATASET_PATH):
    print(f"Lebenshilfe dataset not found at {LH_DATASET_PATH}. Skipping evaluation.")
else:
    with open(LH_DATASET_PATH, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    print(f"Lebenshilfe Datensatz geladen: {len(lh_data)} Artikel-Paare.")

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

    print(lh_eval_df[["synthetic_simplicity_score", "sbert_sim_to_as", "sbert_sim_to_ref", "composite_reward"]].describe())

    print("\n--- QUALITATIVE STICHPROBEN (LEBENSHILFE TESTSET) ---")
    for idx, row in lh_eval_df.head(3).iterrows():
        print(f"\n[Artikel {idx + 1}]")
        print(f"AS-Quelle:     {row['as_text'][:130]}...")
        print(f"LS-Referenz:   {row['ls_reference'][:130]}...")
        print(f"Modell-Übers.: {row['model_translation']}")
        print(f"Scores: Style={row['synthetic_simplicity_score']:.3f} | Sim(AS)={row['sbert_sim_to_as']:.3f} | Sim(Ref)={row['sbert_sim_to_ref']:.3f}")

    # Plot metrics distributions
    plt.figure(figsize=(10, 5))
    plt.hist(lh_eval_df["synthetic_simplicity_score"], bins=15, alpha=0.6, label="Synthetic Einfachheit ($R_{style}$)", color="blue")
    plt.hist(lh_eval_df["sbert_sim_to_as"], bins=15, alpha=0.6, label="SBERT Ähnlichkeit zur AS ($R_{sem}$)", color="green")
    plt.hist(lh_eval_df["sbert_sim_to_ref"], bins=15, alpha=0.6, label="SBERT Ähnlichkeit zu LS-Referenz", color="orange")
    plt.title("Metrik-Verteilung auf dem Lebenshilfe-Testdatensatz (Out-of-Domain)")
    plt.xlabel("Score")
    plt.ylabel("Anzahl Artikel")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(plot_dir, "dpo_lh_metrics_distribution.png"))
    plt.close()
