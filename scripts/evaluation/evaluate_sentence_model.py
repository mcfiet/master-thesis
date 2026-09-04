import torch
import torch.nn as nn
import pandas as pd
import json
import os
import spacy
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score
import textstat
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json" 
MODEL_PATH = "results/models/lstm_baseline_sim_0.80_to_0.98.pt"
VOCAB_SOURCE_CSV = "data/analysis/information_loss_analysis_cleaned.csv"
VOCAB_SIM_RANGE = (0.8, 0.98)

MAX_SEQ_LEN = 100
MIN_SENT_LEN = 3
EMBED_DIM = 128
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(self.dropout(hidden))

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

def build_original_vocab():
    print(f"Reconstructing training vocab from {VOCAB_SOURCE_CSV}...")
    df = pd.read_csv(VOCAB_SOURCE_CSV)
    
    mask = (df["semantic_similarity_8192"] >= VOCAB_SIM_RANGE[0]) & (df["semantic_similarity_8192"] <= VOCAB_SIM_RANGE[1])
    df_filtered = df[mask]
    
    ls_sentences = []
    as_sentences = []
    
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    
    for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Vocab reconstruction"):
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
                
    # Balance classes
    min_len = min(len(ls_sentences), len(as_sentences))
    random.seed(42)
    random.shuffle(ls_sentences)
    random.shuffle(as_sentences)
    ls_sentences = ls_sentences[:min_len]
    as_sentences = as_sentences[:min_len]
    
    X = ls_sentences + as_sentences
    y = [1] * len(ls_sentences) + [0] * len(as_sentences)
    
    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, _, _, _ = train_test_split(X_train_val, y_train_val, test_size=0.11, random_state=42, stratify=y_train_val)
    
    return Vocab(X_train)

def main(output_csv: str = "results/evaluation/eval_sentence_classifier.csv", output_summary: str = "results/evaluation/sentence_classifier_metrics.json"):
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
        
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    
    def predict_sentence(tokens):
        encoded = vocab.encode(tokens)[:MAX_SEQ_LEN]
        if not encoded:
            encoded = [0]
        tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
        
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
            
        # 1. Process LS sentences
        ls_doc = nlp(ls_text)
        ls_preds, ls_probs = [], []
        for sent in ls_doc.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= MIN_SENT_LEN:
                pred, prob = predict_sentence(tokens)
                ls_preds.append(pred)
                ls_probs.append(prob)
                
        # 2. Process AS sentences
        as_doc = nlp(as_text)
        as_preds, as_probs = [], []
        for sent in as_doc.sents:
            tokens = [t.text.lower() for t in sent if not t.is_space]
            if len(tokens) >= MIN_SENT_LEN:
                pred, prob = predict_sentence(tokens)
                as_preds.append(pred)
                as_probs.append(prob)
                
        # Fallback for empty results
        if not ls_preds:
            ls_preds, ls_probs = [0], [0.5]
        if not as_preds:
            as_preds, as_probs = [0], [0.5]
            
        # Aggregation
        ls_pred_maj = 1 if np.mean(ls_preds) > 0.5 else 0
        as_pred_maj = 1 if np.mean(as_preds) > 0.5 else 0
        ls_avg_prob = np.mean(ls_probs)
        as_avg_prob = np.mean(as_probs)
        
        results.append({
            "LS_ID": item.get("ls_filename", "N/A"),
            "AS_ID": item.get("as_filename", "N/A"),
            "LS_Sents_Count": len(ls_preds),
            "LS_Sents_Correct": sum(ls_preds),
            "AS_Sents_Count": len(as_preds),
            "AS_Sents_Correct": len(as_preds) - sum(as_preds),
            "LS_Pred": "Simple" if ls_pred_maj == 1 else "Normal",
            "LS_Conf": ls_avg_prob if ls_pred_maj == 1 else 1 - ls_avg_prob,
            "AS_Pred": "Simple" if as_pred_maj == 1 else "Normal",
            "AS_Conf": as_avg_prob if as_pred_maj == 1 else 1 - as_avg_prob,
            "LS_Flesch": textstat.flesch_reading_ease(ls_text),
            "AS_Flesch": textstat.flesch_reading_ease(as_text),
            "LS_Wiener": textstat.wiener_sachtextformel(ls_text, 1),
            "AS_Wiener": textstat.wiener_sachtextformel(as_text, 1),
            "Correct": (ls_pred_maj == 1 and as_pred_maj == 0),
            "LS_Sents_Preds": ls_preds,
            "AS_Sents_Preds": as_preds
        })

    df_res = pd.DataFrame(results)
    
    # Sentence level calculations
    all_ls_preds, all_as_preds = [], []
    for r in results:
        all_ls_preds.extend(r["LS_Sents_Preds"])
        all_as_preds.extend(r["AS_Sents_Preds"])
        
    y_true_sents = [1] * len(all_ls_preds) + [0] * len(all_as_preds)
    y_pred_sents = all_ls_preds + all_as_preds
    
    sent_acc = accuracy_score(y_true_sents, y_pred_sents)
    sent_bacc = balanced_accuracy_score(y_true_sents, y_pred_sents)
    
    # Article level calculations (aggregated)
    y_true_art = [1] * len(df_res) + [0] * len(df_res)
    y_pred_art = list(df_res["LS_Pred"].map({"Simple": 1, "Normal": 0})) + list(df_res["AS_Pred"].map({"Simple": 1, "Normal": 0}))
    
    art_acc = accuracy_score(y_true_art, y_pred_art)
    art_bacc = balanced_accuracy_score(y_true_art, y_pred_art)
    
    print("\n" + "="*50)
    print(" SENTENCE-LEVEL CLASSIFICATION METRICS")
    print("="*50)
    print(f"Overall Accuracy: {sent_acc*100:.2f}%")
    print(f"Balanced Accuracy: {sent_bacc*100:.2f}%")
    print(f"LS sentences correctly identified as Simple: {sum(all_ls_preds)} / {len(all_ls_preds)} ({sum(all_ls_preds)/len(all_ls_preds)*100:.2f}%)")
    print(f"AS sentences correctly identified as Normal: {len(all_as_preds) - sum(all_as_preds)} / {len(all_as_preds)} ({(len(all_as_preds) - sum(all_as_preds))/len(all_as_preds)*100:.2f}%)")
    print("\nClassification Report (Sentence-Level):")
    print(classification_report(y_true_sents, y_pred_sents, target_names=["Normal (AS)", "Simple (LS)"]))
    
    print("\n" + "="*50)
    print(" AGGREGATED ARTICLE-LEVEL METRICS (Majority Vote)")
    print("="*50)
    print(f"Overall Accuracy: {art_acc*100:.2f}%")
    print(f"Balanced Accuracy: {art_bacc*100:.2f}%")
    print(f"Perfect Pair Match: {df_res['Correct'].sum()} / {len(df_res)} ({df_res['Correct'].mean()*100:.2f}%)")
    print(f"LS articles correctly identified as Simple: {df_res[df_res['LS_Pred']=='Simple'].shape[0]} / {len(df_res)} ({df_res[df_res['LS_Pred']=='Simple'].shape[0]/len(df_res)*100:.2f}%)")
    print(f"AS articles correctly identified as Normal: {df_res[df_res['AS_Pred']=='Normal'].shape[0]} / {len(df_res)} ({df_res[df_res['AS_Pred']=='Normal'].shape[0]/len(df_res)*100:.2f}%)")
    
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
        # Drop list column before saving flat csv if desired or convert to string
        df_to_save = df_res.copy()
        if "LS_Sents_Preds" in df_to_save.columns:
            df_to_save["LS_Sents_Preds"] = df_to_save["LS_Sents_Preds"].astype(str)
        if "AS_Sents_Preds" in df_to_save.columns:
            df_to_save["AS_Sents_Preds"] = df_to_save["AS_Sents_Preds"].astype(str)
        df_to_save.to_csv(output_csv, index=False)
        print(f"\n[ERFOLG] Detail-Vorhersagen gespeichert unter: {output_csv}")

    if output_summary:
        os.makedirs(os.path.dirname(output_summary), exist_ok=True)
        summary_data = {
            "dataset_path": DATASET_PATH,
            "model_path": MODEL_PATH,
            "sentence_level": {
                "accuracy": float(sent_acc),
                "balanced_accuracy": float(sent_bacc),
                "classification_report": classification_report(y_true_sents, y_pred_sents, target_names=["Normal (AS)", "Simple (LS)"], output_dict=True)
            },
            "article_level_majority_vote": {
                "accuracy": float(art_acc),
                "balanced_accuracy": float(art_bacc),
                "perfect_pair_match_ratio": float(df_res["Correct"].mean()),
                "classification_report": classification_report(y_true_art, y_pred_art, target_names=["Normal (AS)", "Simple (LS)"], output_dict=True)
            },
            "readability": {
                "avg_ls_flesch": float(df_res["LS_Flesch"].mean()),
                "avg_as_flesch": float(df_res["AS_Flesch"].mean()),
                "flesch_gap": float(df_res["LS_Flesch"].mean() - df_res["AS_Flesch"].mean()),
                "avg_ls_wiener": float(df_res["LS_Wiener"].mean()),
                "avg_as_wiener": float(df_res["AS_Wiener"].mean())
            }
        }
        with open(output_summary, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"[ERFOLG] Metrik-Zusammenfassung gespeichert unter: {output_summary}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate sentence-level BiLSTM classifier")
    parser.add_argument("--dataset_path", default="data/lebenshilfe/lebenshilfe_dataset_clean.json" if os.path.exists("data/lebenshilfe/lebenshilfe_dataset_clean.json") else "data/lebenshilfe/lebenshilfe_dataset_no_paragraphs.json")
    parser.add_argument("--model_path", default="results/models/lstm_sentence_sim_0.80_to_0.98.pt")
    parser.add_argument("--vocab_source_csv", default="data/analysis/corpus_master.csv" if os.path.exists("data/analysis/corpus_master.csv") else "data/analysis/information_loss_analysis_cleaned.csv")
    parser.add_argument("--output_csv", default="results/evaluation/eval_sentence_classifier.csv")
    parser.add_argument("--output_summary", default="results/evaluation/sentence_classifier_metrics.json")
    args = parser.parse_args()
    
    if os.path.exists(args.dataset_path):
        DATASET_PATH = args.dataset_path
    if os.path.exists(args.model_path):
        MODEL_PATH = args.model_path
    if os.path.exists(args.vocab_source_csv):
        VOCAB_SOURCE_CSV = args.vocab_source_csv
    
    main(output_csv=args.output_csv, output_summary=args.output_summary)

