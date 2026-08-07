import os

# Set working directory to repository root
while not os.path.exists(".git"):
    parent = os.path.dirname(os.getcwd())
    if parent == os.getcwd():
        break
    os.chdir("..")
print("FastAPI working directory set to:", os.getcwd())

import json
import torch
import torch.nn as nn
import spacy
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional

# Setup FastAPI App
app = FastAPI(title="Linguistische Evaluation & Übersetzung in Leichte Sprache")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Model architecture for Regressor
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

# Load spacy
try:
    nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
except Exception:
    nlp = None

# Global Model & Vocab holders
models = {
    "mixup_regressor": None,
    "synthetic_regressor": None,
    "translation_mixup": None,
    "translation_synthetic": None
}
vocabs = {
    "mixup": None,
    "synthetic": None
}
tokenizers = {
    "mixup": None,
    "synthetic": None
}

# Default Paths (from notebooks)
PATHS = {
    "mixup_model": "results/models/07-08/bilstm_mixup_regression_hybrid_cyclic.pt",
    "mixup_vocab": "data/vocabs/mixup_vocab.json",
    "synthetic_model": "results/models/07-08/bilstm_synthetic_regression.pt",
    "synthetic_vocab": "data/vocabs/synthetic_vocab.json",
    "translation_mixup": "results/models/07-08/seq2seq_dpo_mixup_translation_model",
    "translation_synthetic": "results/models/07-08/seq2seq_dpo_synthetic_exact_translation_model",
    "translation_fallback": "facebook/mbart-large-50"
}

def load_vocab(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    if "stoi" in vocab_data:
        return vocab_data["stoi"]
    return vocab_data

# Load Regressor Models on startup
def init_regressors():
    # Mixup
    vocab_mixup = load_vocab(PATHS["mixup_vocab"])
    if vocab_mixup and os.path.exists(PATHS["mixup_model"]):
        vocabs["mixup"] = vocab_mixup
        model = BiLSTMRegressor(len(vocab_mixup))
        model.load_state_dict(torch.load(PATHS["mixup_model"], map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models["mixup_regressor"] = model
        print("MixUp Regressor loaded successfully.")
    else:
        print("MixUp Regressor loading failed (missing file/vocab).")

    # Synthetic
    vocab_syn = load_vocab(PATHS["synthetic_vocab"])
    if vocab_syn and os.path.exists(PATHS["synthetic_model"]):
        vocabs["synthetic"] = vocab_syn
        model = BiLSTMRegressor(len(vocab_syn))
        model.load_state_dict(torch.load(PATHS["synthetic_model"], map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        models["synthetic_regressor"] = model
        print("Synthetic Regressor loaded successfully.")
    else:
        print("Synthetic Regressor loading failed (missing file/vocab).")

# Lazy-load translation models to save startup memory/time
def get_translation_model(model_type: str):
    custom_path = PATHS["translation_mixup"] if model_type == "mixup" else PATHS["translation_synthetic"]
    
    if not os.path.exists(custom_path):
        raise FileNotFoundError(f"Das feingetunte Modell unter '{custom_path}' wurde nicht gefunden.")
    
    model_key = f"translation_{model_type}"
    tokenizer_key = model_type
    
    if models[model_key] is None:
        print(f"Loading translation model for {model_type} from {custom_path}...")
        tokenizers[tokenizer_key] = AutoTokenizer.from_pretrained(PATHS["translation_fallback"], use_fast=False)
        tokenizers[tokenizer_key].src_lang = "de_DE"
        tokenizers[tokenizer_key].tgt_lang = "de_DE"
        models[model_key] = AutoModelForSeq2SeqLM.from_pretrained(custom_path).to(DEVICE)
        models[model_key].eval()
        
    return models[model_key], tokenizers[tokenizer_key], custom_path

# Helper prediction functions
def predict_simplicity(text: str, model_type: str) -> float:
    model = models["mixup_regressor"] if model_type == "mixup" else models["synthetic_regressor"]
    stoi = vocabs["mixup"] if model_type == "mixup" else vocabs["synthetic"]
    
    if model is None or stoi is None or nlp is None:
        return 0.0
        
    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if not t.is_space]
    indices = [stoi.get(t, stoi.get("<unk>", stoi.get("<UNK>", 1))) for t in tokens[:150]]
    if len(indices) == 0:
        indices = [0]
    
    inp_tensor = torch.tensor([indices], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        score = model(inp_tensor).item()
        
    # Both models now output simplicity directly (1.0 = LS, 0.0 = AS)
    return score

# Startup event
@app.on_event("startup")
def startup_event():
    init_regressors()

# API Schemas
class EvaluateRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    model_type: str  # "mixup" or "synthetic"

@app.get("/api/status")
def get_status():
    return {
        "device": str(DEVICE),
        "mixup_regressor": {
            "loaded": models["mixup_regressor"] is not None,
            "model_path": PATHS["mixup_model"],
            "vocab_path": PATHS["mixup_vocab"],
            "exists": os.path.exists(PATHS["mixup_model"]) and os.path.exists(PATHS["mixup_vocab"])
        },
        "synthetic_regressor": {
            "loaded": models["synthetic_regressor"] is not None,
            "model_path": PATHS["synthetic_model"],
            "vocab_path": PATHS["synthetic_vocab"],
            "exists": os.path.exists(PATHS["synthetic_model"]) and os.path.exists(PATHS["synthetic_vocab"])
        },
        "translation_paths": {
            "mixup": PATHS["translation_mixup"],
            "mixup_exists": os.path.exists(PATHS["translation_mixup"]),
            "synthetic": PATHS["translation_synthetic"],
            "synthetic_exists": os.path.exists(PATHS["translation_synthetic"])
        }
    }

@app.post("/api/evaluate")
def evaluate_text(req: EvaluateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    mixup_score = predict_simplicity(req.text, "mixup")
    synthetic_score = predict_simplicity(req.text, "synthetic")
    
    # Compute some basic stats
    tokens = [t.text for t in nlp(req.text)] if nlp else req.text.split()
    sentences = list(nlp(req.text).sents) if nlp else [req.text]
    
    return {
        "mixup_score": mixup_score,
        "synthetic_score": synthetic_score,
        "stats": {
            "token_count": len(tokens),
            "sentence_count": max(1, len(sentences)),
            "avg_sentence_length": round(len(tokens) / max(1, len(sentences)), 1)
        }
    }

@app.post("/api/translate")
def translate_text(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if req.model_type not in ["mixup", "synthetic"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'mixup' or 'synthetic'.")
        
    try:
        model, tokenizer, load_path = get_translation_model(req.model_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load translation model: {str(e)}")
        
    # Run translation
    prompt = "Übersetze in Leichte Sprache: " + req.text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    
    # Get target language token ID for mBART
    forced_bos_token_id = tokenizer.lang_code_to_id["de_DE"]
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=4,
            length_penalty=1.0,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=forced_bos_token_id
        )
        
    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Strip the prompt prefix if it was echoed by the model
    prefix = "Übersetze in Leichte Sprache: "
    if translated_text.startswith(prefix):
        translated_text = translated_text[len(prefix):]
    
    # Strip echoed source text if present
    cleaned_req_text = req.text.strip()
    if translated_text.startswith(cleaned_req_text):
        translated_text = translated_text[len(cleaned_req_text):].strip()
    elif translated_text.startswith(cleaned_req_text[:50]):  # fallback for partially truncated echoes
        # Find where the echoed part ends and the translation begins
        # Often the translation starts with a question like 'Was' or a new simple sentence
        # Let's see if we can find a sensible boundary or just strip up to the length of the source text if it's mostly similar
        pass
    
    # Calculate simplicity before and after
    source_mixup = predict_simplicity(req.text, "mixup")
    source_synthetic = predict_simplicity(req.text, "synthetic")
    target_mixup = predict_simplicity(translated_text, "mixup")
    target_synthetic = predict_simplicity(translated_text, "synthetic")
    
    return {
        "translation": translated_text,
        "model_used": load_path,
        "source_simplicity": {
            "mixup": source_mixup,
            "synthetic": source_synthetic
        },
        "target_simplicity": {
            "mixup": target_mixup,
            "synthetic": target_synthetic
        }
    }

# HTML page serving
@app.get("/", response_class=HTMLResponse)
def get_index():
    index_path = "templates/index.html"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html is missing. Please create templates/index.html.</h3>"
