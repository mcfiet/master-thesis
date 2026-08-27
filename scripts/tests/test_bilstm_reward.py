#!/usr/bin/env python3
import json
import os
import torch
import spacy
import numpy as np
from scripts.evaluation.evaluate_dpo_beta_experiment import BiLSTMRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reward_model_path = "results/models/bilstm_mixup_regression.pt"
reward_vocab_path = "data/vocabs/mixup_vocab.json"

print("1. Checking file existence:")
print(f"  - Vocab path exists: {os.path.exists(reward_vocab_path)}")
print(f"  - Model path exists: {os.path.exists(reward_model_path)}")

if os.path.exists(reward_vocab_path) and os.path.exists(reward_model_path):
    with open(reward_vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        stoi = vocab_data.get("stoi", vocab_data)
    unk_idx = stoi.get("<unk>", 1)
    print(f"  - Vocab size: {len(stoi)}")

    model = BiLSTMRegressor(vocab_size=len(stoi), embed_dim=128, hidden_dim=128).to(device)
    state = torch.load(reward_model_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    print("  - Model weights loaded successfully!")

    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
    except Exception:
        nlp = spacy.blank("de")

    test_sentences = [
        "Das ist ein einfacher Satz in Leichter Sprache.",
        "Auf Grund des § 4 i.V. m. § 47d der Gemeindeordnung wird folgende Satzung erlassen.",
        "In der Stadt Kiel gibt es eine Werkstatt für Menschen mit Behinderung."
    ]

    batch_indices = []
    max_l = 0
    for s in test_sentences:
        doc = nlp(s)
        tokens = [t.text.lower() for t in doc if not t.is_space]
        indices = [stoi.get(t, unk_idx) for t in tokens[:256]]
        if len(indices) == 0:
            indices = [0]
        max_l = max(max_l, len(indices))
        batch_indices.append(indices)

    padded = np.zeros((len(batch_indices), max(1, max_l)), dtype=np.int64)
    for i, idxs in enumerate(batch_indices):
        padded[i, :len(idxs)] = idxs

    tensor_x = torch.tensor(padded, dtype=torch.long, device=device)
    with torch.no_grad():
        preds = model(tensor_x).squeeze(-1).cpu().numpy()

    print("\n2. Prediction on test sentences:")
    for s, p in zip(test_sentences, preds):
        print(f"  Score: {p:.4f} | Text: {s}")
else:
    print("ERROR: One or both files missing.")
