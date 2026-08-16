import os
import sys
import datetime
import random
import argparse
import numpy as np
import pandas as pd
import json
import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
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

# ==============================================================================
# ZENTRALE KONFIGURATION & PARAMS
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--lh_dataset_path', required=True)
parser.add_argument('--corpus_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--min_sim', type=float, required=True)
parser.add_argument('--max_sim', type=float, required=True)
parser.add_argument('--max_source_len', type=int, required=True)
parser.add_argument('--max_target_len', type=int, required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=2)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--warmup_ratio', type=float, default=0.10)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--val_split', type=float, default=0.15)
parser.add_argument('--prompt_prefix', default="", help='Task prompt prefix prepended to source text')
parser.add_argument('--use_peft', '--use_lora', dest='use_peft', action='store_true', default=False, help='Use LoRA / PEFT for SFT training')
parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank r')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
parser.add_argument('--reward_model_path', default=None)
parser.add_argument('--reward_vocab_path', default=None)
parser.add_argument('--w_style', type=float, default=0.5)
parser.add_argument('--w_sem', type=float, default=0.5)
parser.add_argument('--resume_from_checkpoint', action='store_true', default=False, help='Resume training from existing output_dir checkpoint')
args = parser.parse_args()

LH_DATASET_PATH = args.lh_dataset_path
CORPUS_PATH = args.corpus_path
OUTPUT_DIR = args.output_dir
MIN_SIM = args.min_sim
MAX_SIM = args.max_sim
MAX_SOURCE_LEN = args.max_source_len
MAX_TARGET_LEN = args.max_target_len
MODEL_NAME = args.model_name
BATCH_SIZE = args.batch_size
ACCUMULATION_STEPS = args.accumulation_steps
NUM_EPOCHS = args.epochs
LR = args.lr
WARMUP_RATIO = args.warmup_ratio
PATIENCE = args.patience
SEED = args.seed
VAL_SPLIT = args.val_split
PROMPT_PREFIX = args.prompt_prefix
USE_PEFT = args.use_peft
LORA_R = args.lora_r
LORA_ALPHA = args.lora_alpha
LORA_DROPOUT = args.lora_dropout
REWARD_MODEL_PATH = args.reward_model_path
REWARD_VOCAB_PATH = args.reward_vocab_path
W_STYLE = args.w_style
W_SEM = args.w_sem

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Nutze Device: {DEVICE}")

# ==============================================================================
# DATA LOADING & SPLITTING
# ==============================================================================
def load_filtered_corpus(file_path=CORPUS_PATH, min_sim=MIN_SIM, max_sim=MAX_SIM):
    if file_path.endswith(".json"):
        print(f"Lade Datensatz aus JSON: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
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

    print(f"Lade Datensatz aus CSV: {file_path}")
    df = pd.read_csv(file_path)
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
split_idx = int((1.0 - VAL_SPLIT) * len(all_pairs))
train_data = all_pairs[:split_idx]
val_data = all_pairs[split_idx:]
print(f"Trainingsdaten: {len(train_data)} Paare | Validierungsdaten: {len(val_data)} Paare")

# ==============================================================================
# MODEL & TOKENIZER SETUP
# ==============================================================================
if getattr(args, "resume_from_checkpoint", False) and OUTPUT_DIR and os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
    print(f"Setze Training fort: Lade Modell aus lokalem Verzeichnis: {OUTPUT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=False)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(DEVICE)
else:
    print(f"Starte neues SFT-Training von Basismodell: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

if "mbart" in MODEL_NAME.lower():
    tokenizer.src_lang = "de_DE"
    tokenizer.tgt_lang = "de_DE"

if USE_PEFT:
    print(f"Konfiguriere LoRA für Seq2Seq SFT (r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT})...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        bias="none",
    )
    seq2seq_model = get_peft_model(seq2seq_model, peft_config)
    seq2seq_model.print_trainable_parameters()

# ==============================================================================
# TRANSLATION DATASET & DATALOADER
# ==============================================================================
class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len=256, max_tgt_len=256, prompt_prefix=PROMPT_PREFIX):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.prompt_prefix = prompt_prefix
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        src_text = self.prompt_prefix + item["as_text"]
        tgt_text = item["ls_text"]
        
        inputs = self.tokenizer(
            src_text, max_length=self.max_src_len, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = self.tokenizer(
            text_target=tgt_text, max_length=self.max_tgt_len, padding="max_length", truncation=True, return_tensors="pt"
        )["input_ids"]
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "raw_as": item["as_text"],
            "raw_ls": item["ls_text"]
        }

train_dataset = TranslationDataset(train_data, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, PROMPT_PREFIX)
val_dataset = TranslationDataset(val_data, tokenizer, MAX_SOURCE_LEN, MAX_TARGET_LEN, PROMPT_PREFIX)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
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

patience = PATIENCE
best_val_loss = float('inf')
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": []}

trainable_params = [p for p in seq2seq_model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps, 
    num_training_steps=total_steps
)

print("Starte SFT Training mit Validation und Early Stopping...")
for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoche {epoch + 1}/{NUM_EPOCHS} ---")
    sft_loss = train_sft_epoch(seq2seq_model, train_loader, optimizer, scheduler, ACCUMULATION_STEPS)
    val_loss = validate_sft_epoch(seq2seq_model, val_loader)
    
    history["train_loss"].append(sft_loss)
    history["val_loss"].append(val_loss)
    
    print(f"Epoche {epoch + 1} | Train Loss: {sft_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        seq2seq_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        if hasattr(seq2seq_model, "config"):
            seq2seq_model.config.save_pretrained(OUTPUT_DIR)
        elif hasattr(seq2seq_model, "base_model") and hasattr(seq2seq_model.base_model, "config"):
            seq2seq_model.base_model.config.save_pretrained(OUTPUT_DIR)
        torch.save(seq2seq_model.state_dict(), os.path.join(OUTPUT_DIR, "sft.pt"))
        print(f"-> Neues bestes Modell (HF-Format & sft.pt) unter {OUTPUT_DIR} gespeichert.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early Stopping getriggert! Keine Verbesserung des Val Loss seit {patience} Epochen.")
            break

if USE_PEFT:
    print(f"\nLade bestes LoRA-Modell aus {OUTPUT_DIR} und führe merge_and_unload() durch...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    peft_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(merged_model.state_dict(), os.path.join(OUTPUT_DIR, "sft.pt"))
    seq2seq_model = merged_model
    print(f"Erfolgreich gemergtes SFT-Modell unter {OUTPUT_DIR} gespeichert!")
elif os.path.exists(OUTPUT_DIR) and len(os.listdir(OUTPUT_DIR)) > 0:
    seq2seq_model = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(DEVICE)
    print("Das beste Modell wurde erfolgreich geladen!")

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
if not os.path.exists(LH_DATASET_PATH):
    print(f"Lebenshilfe dataset not found at {LH_DATASET_PATH}. Skipping evaluation.")
else:
    with open(LH_DATASET_PATH, "r", encoding="utf-8") as f:
        lh_data = json.load(f)

    print(f"Lebenshilfe Datensatz geladen: {len(lh_data)} Artikel-Paare.")

    # Check if reward model paths are provided to do full quantitative evaluation
    has_reward_model = False
    if REWARD_MODEL_PATH and REWARD_VOCAB_PATH and os.path.exists(REWARD_MODEL_PATH) and os.path.exists(REWARD_VOCAB_PATH):
        has_reward_model = True
        print("Reward-Modell-Pfade gefunden. Führe vollständige quantitative Evaluation durch...")
        
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

        with open(REWARD_VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            reward_stoi = vocab_data.get("stoi", vocab_data)

        bilstm_model = BiLSTMRegressor(len(reward_stoi), embed_dim=128, hidden_dim=128)
        bilstm_model.load_state_dict(torch.load(REWARD_MODEL_PATH, map_location="cpu"))
        bilstm_model.eval()
        
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
        
        def predict_simplicity_score(texts):
            scores = []
            unk_idx = reward_stoi.get("<unk>") or reward_stoi.get("<UNK>") or 1
            for text in texts:
                doc = nlp(text)
                tokens = [t.text.lower() for t in doc if not t.is_space]
                indices = [reward_stoi.get(t, unk_idx) for t in tokens[:150]]
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
    else:
        print("Kein/ungültiges Reward-Modell angegeben. Führe nur qualitative Evaluation durch (Übersetzung ohne Scores)...")

    def evaluate_on_lebenshilfe(model, tokenizer, lh_data, max_samples=49):
        model.eval()
        as_texts = [item["as_text"] for item in lh_data[:max_samples]]
        ls_ref_texts = [item["ls_text"] for item in lh_data[:max_samples]]
        
        batch_size = 16
        gen_texts = []
        for i in tqdm(range(0, len(as_texts), batch_size), desc="Übersetze Lebenshilfe Datensatz"):
            batch_src = as_texts[i:i+batch_size]
            prompts = [t for t in batch_src]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=MAX_SOURCE_LEN, return_tensors="pt").to(DEVICE)
            gen_kwargs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "max_length": MAX_TARGET_LEN,
                "num_beams": 4,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
            if hasattr(tokenizer, "lang_code_to_id") and "de_DE" in tokenizer.lang_code_to_id:
                gen_kwargs["forced_bos_token_id"] = tokenizer.lang_code_to_id["de_DE"]
            elif hasattr(model.config, "forced_bos_token_id") and model.config.forced_bos_token_id is not None:
                gen_kwargs["forced_bos_token_id"] = model.config.forced_bos_token_id

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            gen_texts.extend(decoded)
            
        if has_reward_model:
            tot_reward, r_style, r_sem = reward_evaluator.compute_reward(as_texts, gen_texts)
            sim_to_ref = predict_semantic_similarity(ls_ref_texts, gen_texts)
            sim_to_ref_norm = np.clip((sim_to_ref + 1.0) / 2.0, 0.0, 1.0)
            
            df_res = pd.DataFrame({
                "as_text": as_texts,
                "ls_reference": ls_ref_texts,
                "model_translation": gen_texts,
                "style_score": r_style,
                "sbert_sim_to_as": r_sem,
                "sbert_sim_to_ref": sim_to_ref_norm,
                "composite_reward": tot_reward
            })
            
            print("\n=================== EVALUIERUNGSERGEBNISSE (LEBENSHILFE - SFT) ===================")
            print(f"Ø Style-Einfachheits-Score (R_style):       {r_style.mean():.4f} ± {r_style.std():.4f}")
            print(f"Ø SBERT-Ähnlichkeit zur AS-Quelle (R_sem):   {r_sem.mean():.4f} ± {r_sem.std():.4f}")
            print(f"Ø SBERT-Ähnlichkeit zu echter LS-Referenz:  {sim_to_ref_norm.mean():.4f} ± {sim_to_ref_norm.std():.4f}")
            print(f"Ø Composite Reward:                        {tot_reward.mean():.4f} ± {tot_reward.std():.4f}")
            print("========================================================================\n")
            
            print(df_res[["style_score", "sbert_sim_to_as", "sbert_sim_to_ref", "composite_reward"]].describe())
            
            # Plot metrics distributions
            plt.figure(figsize=(10, 5))
            plt.hist(df_res["style_score"], bins=15, alpha=0.6, label="Style Einfachheit ($R_{style}$)", color="blue")
            plt.hist(df_res["sbert_sim_to_as"], bins=15, alpha=0.6, label="SBERT Ähnlichkeit zur AS ($R_{sem}$)", color="green")
            plt.hist(df_res["sbert_sim_to_ref"], bins=15, alpha=0.6, label="SBERT Ähnlichkeit zu LS-Referenz", color="orange")
            plt.title("SFT Metrik-Verteilung auf dem Lebenshilfe-Testdatensatz (Out-of-Domain)")
            plt.xlabel("Score")
            plt.ylabel("Anzahl Artikel")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(plot_dir, "sft_lh_metrics_distribution.png"))
            plt.close()
            
            print("\n--- QUALITATIVE STICHPROBEN (LEBENSHILFE TESTSET) ---")
            for idx, row in df_res.head(3).iterrows():
                print(f"\n[Artikel {idx + 1}]")
                print(f"AS-Quelle:     {row['as_text'][:130]}...")
                print(f"LS-Referenz:   {row['ls_reference'][:130]}...")
                print(f"Modell-Übers.: {row['model_translation']}")
                print(f"Scores: Style={row['style_score']:.3f} | Sim(AS)={row['sbert_sim_to_as']:.3f} | Sim(Ref)={row['sbert_sim_to_ref']:.3f}")
        else:
            print("\n--- QUALITATIVE STICHPROBEN (LEBENSHILFE TESTSET - OHNE SCORES) ---")
            for idx in range(min(3, len(as_texts))):
                print(f"\n[Artikel {idx + 1}]")
                print(f"AS-Quelle:     {as_texts[idx][:130]}...")
                print(f"LS-Referenz:   {ls_ref_texts[idx][:130]}...")
                print(f"Modell-Übers.: {gen_texts[idx]}")

    evaluate_on_lebenshilfe(seq2seq_model, tokenizer, lh_data)


