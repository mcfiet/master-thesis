#!/usr/bin/env python3
"""
=============================================================================
PPO Training for Seq2Seq Models (facebook/mbart-large-50)
=============================================================================
This script performs Proximal Policy Optimization (PPO) starting from a
pre-trained Supervised Fine-Tuned (SFT) Seq2Seq model (e.g. mBART-50).

Features:
  - Autoregressive Online Rollout Generation with Seq2Seq Decoder
  - Zero-VRAM Reference Model evaluation via PEFT adapter disabling (model.disable_adapter())
  - Custom Token-Level Seq2Seq Value Head attached to Decoder Hidden States
  - Real-time Composite Reward Evaluator:
      R(x, y) = w_style * R_style(y) + w_sem * R_sem(x, y)
      using the 500-Token BiLSTM MixUp Regressor & SBERT/Jina Embeddings
  - PPO Clipped Surrogate Loss with Generalized Advantage Estimation (GAE)
  - Comprehensive Metric Tracking (Reward, Style, Sem Sim, KL, Policy/Value Loss)
  - Checkpointing & Automatic Training Curve Plot Generation
=============================================================================
"""

import os
import sys
import json
import random
import datetime
import argparse
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    MBart50TokenizerFast,
    get_linear_schedule_with_warmup,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger(object):
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==============================================================================
# BILSTM REGRESSOR DEFINITION (500-Token Compatible)
# ==============================================================================
class BiLSTMRegressor(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super(BiLSTMRegressor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        out = self.fc(self.dropout(hidden))
        return self.sigmoid(out)


# ==============================================================================
# COMPOSITE REWARD EVALUATOR
# ==============================================================================
class CompositeRewardEvaluator:
    def __init__(
        self,
        reward_model_path: str,
        reward_vocab_path: str,
        sbert_model_name: str = "jinaai/jina-embeddings-v2-base-de",
        w_style: float = 0.5,
        w_sem: float = 0.5,
        max_seq_len: int = 500,
        device: torch.device = torch.device("cpu"),
    ):
        self.w_style = w_style
        self.w_sem = w_sem
        self.max_seq_len = max_seq_len
        self.device = device

        print(f"Loading Reward Model vocabulary from: {reward_vocab_path}")
        with open(reward_vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            self.stoi = vocab_data.get("stoi", vocab_data)
        self.unk_idx = self.stoi.get("<unk>") or self.stoi.get("<UNK>") or 1

        print(f"Loading BiLSTM Regressor weights from: {reward_model_path}")
        self.bilstm_model = BiLSTMRegressor(
            vocab_size=len(self.stoi),
            embed_dim=128,
            hidden_dim=128,
        ).to(self.device)

        raw_state = torch.load(reward_model_path, map_location=self.device)
        if isinstance(raw_state, dict):
            if "model_state_dict" in raw_state:
                raw_state = raw_state["model_state_dict"]
            elif "state_dict" in raw_state:
                raw_state = raw_state["state_dict"]
        self.bilstm_model.load_state_dict(raw_state)
        self.bilstm_model.eval()

        try:
            self.nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer", "parser"])
        except Exception:
            self.nlp = spacy.blank("de")
            self.nlp.add_pipe("sentencizer")

        print(f"Loading SBERT model for semantic evaluation: {sbert_model_name}")
        self.sbert_model = SentenceTransformer(sbert_model_name, trust_remote_code=True, device=str(self.device))
        if hasattr(self.sbert_model, "max_seq_length"):
            self.sbert_model.max_seq_length = max(self.max_seq_len, 256)

    def predict_simplicity_scores(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        batch_indices = []
        max_l = 0
        docs = list(self.nlp.pipe(texts, batch_size=len(texts))) if len(texts) > 1 else [self.nlp(str(texts[0] or ""))]
        for doc in docs:
            tokens = [t.text.lower() for t in doc if not t.is_space]
            indices = [self.stoi.get(t, self.unk_idx) for t in tokens[: self.max_seq_len]]
            if len(indices) == 0:
                indices = [0]
            batch_indices.append(indices)
            if len(indices) > max_l:
                max_l = len(indices)

        padded = np.zeros((len(batch_indices), max(max_l, 1)), dtype=np.int64)
        for i, idxs in enumerate(batch_indices):
            padded[i, : len(idxs)] = idxs

        inp_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            scores = self.bilstm_model(inp_tensor).squeeze(-1).cpu().numpy()
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

    def compute_composite_rewards(
        self,
        source_texts: List[str],
        generated_texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        clean_gens = [t.strip() for t in generated_texts]
        clean_srcs = [t.strip() for t in source_texts]

        style_scores = self.predict_simplicity_scores(clean_gens)

        eff_len = getattr(self.sbert_model, "max_seq_length", self.max_seq_len)
        sbert_bs = 2 if eff_len > 4096 else (4 if eff_len > 1024 else (8 if eff_len > 512 else 16))

        with torch.inference_mode():
            src_emb = self.sbert_model.encode(clean_srcs, convert_to_tensor=True, batch_size=min(sbert_bs, len(clean_srcs)), show_progress_bar=False)
            gen_emb = self.sbert_model.encode(clean_gens, convert_to_tensor=True, batch_size=min(sbert_bs, len(clean_gens)), show_progress_bar=False)
            sem_cos = util.cos_sim(src_emb, gen_emb)
            sem_scores = torch.diagonal(sem_cos).clamp(0.0, 1.0).cpu().numpy()

        composite_rewards = (self.w_style * style_scores) + (self.w_sem * sem_scores)
        return composite_rewards, style_scores, sem_scores


# ==============================================================================
# DATASET FOR PROMPT SAMPLING
# ==============================================================================
class Seq2SeqPromptDataset(Dataset):
    def __init__(self, data_path: str, min_sim: float = 0.70, max_sim: float = 0.98, max_samples: Optional[int] = None):
        self.samples: List[str] = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Corpus file not found: {data_path}")

        print(f"Loading source prompts from: {data_path}")
        if data_path.endswith(".json"):
            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "data" in data:
                    data = data["data"]
                for item in data:
                    sim = (
                        item.get("semantic_similarity_8192")
                        or item.get("semantic_similarity")
                        or item.get("similarity")
                        or item.get("sim")
                        or item.get("cosine_similarity")
                    )
                    if sim is not None:
                        try:
                            sim_val = float(sim)
                            if not (min_sim <= sim_val <= max_sim):
                                continue
                        except (ValueError, TypeError):
                            pass
                    src = str(
                        item.get("as_text")
                        or item.get("as")
                        or item.get("source_text")
                        or item.get("source")
                        or item.get("prompt")
                        or item.get("text_as")
                        or ""
                    ).strip()
                    if src and len(src.split()) >= 10:
                        self.samples.append(src)
        elif data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        src = str(
                            item.get("as_text")
                            or item.get("as")
                            or item.get("prompt")
                            or item.get("source_text")
                            or item.get("source")
                            or ""
                        ).strip()
                        if src and len(src.split()) >= 10:
                            self.samples.append(src)
        elif data_path.endswith(".csv"):
            df = pd.read_csv(data_path)
            sim_col = next((c for c in ["semantic_similarity_8192", "semantic_similarity", "similarity", "sim", "cosine_similarity"] if c in df.columns), None)
            as_col = next((c for c in ["as_text", "as", "source_text", "source", "text_as", "prompt"] if c in df.columns), None)
            if as_col:
                if sim_col:
                    df = df[(df[sim_col] >= min_sim) & (df[sim_col] <= max_sim)]
                self.samples = [str(s).strip() for s in df[as_col].dropna().tolist() if len(str(s).split()) >= 10]

        if max_samples and len(self.samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        print(f"Loaded {len(self.samples)} valid source prompts for Seq2Seq PPO training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> str:
        return self.samples[idx]


# ==============================================================================
# VALUE HEAD FOR SEQ2SEQ (mBART)
# ==============================================================================
class Seq2SeqWithValueHead(nn.Module):
    """
    Wraps AutoModelForSeq2SeqLM with a dedicated token-level Value Head
    on the Decoder hidden states.
    """
    def __init__(self, base_model: nn.Module, hidden_size: int, dtype: Optional[torch.dtype] = None):
        super(Seq2SeqWithValueHead, self).__init__()
        self.pretrained_model = base_model
        if dtype is None:
            param = next(base_model.parameters(), None)
            dtype = param.dtype if param is not None else torch.float32
        self.v_head = nn.Linear(hidden_size, 1, bias=False, dtype=dtype)
        nn.init.normal_(self.v_head.weight, mean=0.0, std=1.0 / (hidden_size ** 0.5))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        return_values: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        values = None
        if return_values:
            # Last decoder hidden states
            dec_hidden = outputs.decoder_hidden_states[-1]
            values = self.v_head(dec_hidden.to(self.v_head.weight.dtype)).squeeze(-1)  # (batch_size, dec_seq_len)
        return logits, values


# ==============================================================================
# GAE & PPO ADVANTAGE CALCULATION
# ==============================================================================
def compute_gae(
    rewards: List[torch.Tensor],
    values: List[torch.Tensor],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    advantages = []
    returns = []
    for r, v in zip(rewards, values):
        seq_len = len(r)
        adv = torch.zeros(seq_len, dtype=torch.float32, device=r.device)
        r_f = r.float()
        v_f = v.float()
        last_gae = 0.0
        for t in reversed(range(seq_len)):
            next_v = v_f[t + 1] if t + 1 < seq_len else 0.0
            delta = r_f[t] + gamma * next_v - v_f[t]
            last_gae = delta + gamma * lam * last_gae
            adv[t] = last_gae
        ret = adv + v_f
        advantages.append(adv)
        returns.append(ret)
    return advantages, returns


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PPO Training for Seq2Seq Models (mBART-50).")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus JSON/CSV for source prompts")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to pre-trained SFT mBART model checkpoint")
    parser.add_argument("--base_model_name", type=str, default="facebook/mbart-large-50")
    parser.add_argument("--reward_model_path", type=str, default="results/models/bilstm_mixup_regression.pt")
    parser.add_argument("--reward_vocab_path", type=str, default="data/vocabs/mixup_vocab.json")
    parser.add_argument("--sbert_model_name", type=str, default="jinaai/jina-embeddings-v2-base-de")
    parser.add_argument("--output_dir", type=str, default="results/models/ppo/seq2seq")
    parser.add_argument("--log_dir", type=str, default="results/logs/experiments/ppo/seq2seq")
    parser.add_argument("--plot_dir", type=str, default="results/plots/experiments/ppo/seq2seq")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of outer PPO epochs")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="Inner PPO update epochs per rollout")
    parser.add_argument("--batch_size", type=int, default=4, help="Rollout batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO SGD mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Actor policy learning rate")
    parser.add_argument("--vf_lr", type=float, default=3e-5, help="Value head learning rate")
    parser.add_argument("--kl_beta", type=float, default=0.05, help="KL divergence penalty coefficient")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping parameter epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss coefficient")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--w_style", type=float, default=0.5)
    parser.add_argument("--w_sem", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_peft", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    # Logging Setup
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"train_ppo_seq2seq_{timestamp}.log")
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print("=" * 80)
    print(f"Starting Seq2Seq (mBART-50) PPO Training at {timestamp}")
    print(f"Base Model: {args.base_model_name}")
    print(f"SFT Checkpoint: {args.sft_model_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"KL Beta: {args.kl_beta} | Clip Eps: {args.clip_eps} | Style/Sem Weights: {args.w_style}/{args.w_sem}")
    print("=" * 80)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Tokenizer
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, src_lang="de_DE", tgt_lang="de_DE")

    # 2. Load Base Model and LoRA
    print(f"Loading Base Seq2Seq LM from {args.base_model_name}...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Robust SFT model integration
    if os.path.exists(os.path.join(args.sft_model_path, "adapter_config.json")):
        print(f"Loading and merging SFT LoRA adapter from {args.sft_model_path}...")
        base_lm = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=torch_dtype)
        sft_peft = PeftModel.from_pretrained(base_lm, args.sft_model_path)
        base_lm = sft_peft.merge_and_unload()
        print("[ERFOLG] SFT-Adapter erfolgreich in Basisgewichte integriert!")
    elif os.path.exists(os.path.join(args.sft_model_path, "model.safetensors")) or os.path.exists(os.path.join(args.sft_model_path, "pytorch_model.bin")):
        print(f"Loading standalone SFT model from {args.sft_model_path}...")
        base_lm = AutoModelForSeq2SeqLM.from_pretrained(args.sft_model_path, torch_dtype=torch_dtype)
    else:
        print(f"Loading base model {args.base_model_name}...")
        base_lm = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_name, torch_dtype=torch_dtype)

    if args.use_peft:
        print("Initializing LoRA adapters for Seq2Seq PPO Policy...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
        )
        base_lm = get_peft_model(base_lm, peft_config)

    hidden_size = getattr(base_lm.config, "d_model", 1024)
    model = Seq2SeqWithValueHead(base_lm, hidden_size=hidden_size, dtype=torch_dtype).to(device)

    # 3. Load Reward Evaluator
    print("Initializing Composite Reward Evaluator...")
    reward_evaluator = CompositeRewardEvaluator(
        reward_model_path=args.reward_model_path,
        reward_vocab_path=args.reward_vocab_path,
        sbert_model_name=args.sbert_model_name,
        w_style=args.w_style,
        w_sem=args.w_sem,
        device=device,
    )

    # 4. Dataset & DataLoader
    dataset = Seq2SeqPromptDataset(
        data_path=args.corpus_path,
        min_sim=0.70,
        max_sim=0.98,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 5. Optimizers
    policy_params = [p for n, p in model.pretrained_model.named_parameters() if p.requires_grad]
    vf_params = list(model.v_head.parameters())
    optimizer = AdamW([
        {"params": policy_params, "lr": args.lr},
        {"params": vf_params, "lr": args.vf_lr},
    ])

    total_steps = len(dataloader) * args.epochs * args.ppo_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    # History Tracking
    history = {
        "step": [],
        "mean_reward": [],
        "style_score": [],
        "sem_score": [],
        "kl_divergence": [],
        "policy_loss": [],
        "value_loss": [],
    }

    # ==========================================================================
    # PPO TRAINING LOOP
    # ==========================================================================
    global_step = 0
    de_lang_id = tokenizer.lang_code_to_id.get("de_DE", tokenizer.eos_token_id) if hasattr(tokenizer, "lang_code_to_id") else None

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        epoch_rewards = []
        epoch_styles = []
        epoch_sems = []
        epoch_kls = []
        epoch_ploss = []
        epoch_vloss = []

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_prompts in progress_bar:
            global_step += 1
            model.eval()

            # Encode Source Prompts
            prompt_enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=args.max_source_len,
                return_tensors="pt",
            ).to(device)

            src_input_ids = prompt_enc["input_ids"]
            src_attention_mask = prompt_enc["attention_mask"]

            # ------------------------------------------------------------------
            # 1. Rollout Generation
            # ------------------------------------------------------------------
            with torch.no_grad():
                gen_kwargs = {
                    "input_ids": src_input_ids,
                    "attention_mask": src_attention_mask,
                    "max_length": args.max_target_len,
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                }

                gen_outputs = model.pretrained_model.generate(**gen_kwargs)

            # Decode generated target strings
            generated_texts = [
                tokenizer.decode(g, skip_special_tokens=True).strip()
                for g in gen_outputs
            ]

            # ------------------------------------------------------------------
            # 2. Reward & KL Evaluation
            # ------------------------------------------------------------------
            comp_rewards, style_scores, sem_scores = reward_evaluator.compute_composite_rewards(
                batch_prompts,
                generated_texts
            )

            # Create decoder inputs (shifted right)
            decoder_input_ids = gen_outputs[:, :-1]
            target_ids = gen_outputs[:, 1:]
            dec_mask = (target_ids != tokenizer.pad_token_id)

            # Forward pass with policy and reference model
            with torch.no_grad():
                policy_logits, policy_values = model(
                    input_ids=src_input_ids,
                    attention_mask=src_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    return_values=True,
                )
                policy_logps_all = F.log_softmax(policy_logits.float(), dim=-1)
                old_logps = torch.gather(policy_logps_all, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

                if hasattr(model.pretrained_model, "disable_adapter"):
                    with model.pretrained_model.disable_adapter():
                        ref_logits, _ = model(
                            input_ids=src_input_ids,
                            attention_mask=src_attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            return_values=False,
                        )
                else:
                    ref_logits = policy_logits
                ref_logps_all = F.log_softmax(ref_logits.float(), dim=-1)
                ref_logps = torch.gather(ref_logps_all, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

            step_rewards = []
            kl_divs = []
            action_values = []

            for i in range(len(batch_prompts)):
                mask_i = dec_mask[i]
                seq_old_lp = old_logps[i, mask_i]
                seq_ref_lp = ref_logps[i, mask_i]
                seq_v = policy_values[i, mask_i].float()

                seq_kl = seq_old_lp - seq_ref_lp
                mean_kl = seq_kl.mean().item() if len(seq_kl) > 0 else 0.0
                kl_divs.append(mean_kl)

                seq_r = -args.kl_beta * seq_kl
                if len(seq_r) > 0:
                    seq_r[-1] += float(comp_rewards[i])
                step_rewards.append(seq_r)
                action_values.append(seq_v)

            # ------------------------------------------------------------------
            # 3. GAE & Returns
            # ------------------------------------------------------------------
            advantages, returns = compute_gae(
                step_rewards,
                action_values,
                gamma=args.gamma,
                lam=args.lam,
            )

            all_adv = torch.cat([a for a in advantages if len(a) > 0]) if any(len(a) > 0 for a in advantages) else torch.tensor([0.0], device=device)
            adv_mean, adv_std = all_adv.mean(), all_adv.std() + 1e-8
            norm_advantages = [(a - adv_mean) / adv_std for a in advantages]

            # ------------------------------------------------------------------
            # 4. PPO Optimization Epochs
            # ------------------------------------------------------------------
            model.train()
            for ppo_epoch in range(args.ppo_epochs):
                for i in range(0, len(batch_prompts), args.mini_batch_size):
                    mb_indices = list(range(i, min(i + args.mini_batch_size, len(batch_prompts))))
                    mb_src_ids = src_input_ids[mb_indices]
                    mb_src_mask = src_attention_mask[mb_indices]
                    mb_dec_ids = decoder_input_ids[mb_indices]
                    mb_target_ids = target_ids[mb_indices]
                    mb_dec_mask = dec_mask[mb_indices]

                    cur_logits, cur_values = model(
                        input_ids=mb_src_ids,
                        attention_mask=mb_src_mask,
                        decoder_input_ids=mb_dec_ids,
                        return_values=True,
                    )
                    cur_logps_all = F.log_softmax(cur_logits.float(), dim=-1)
                    cur_logps = torch.gather(cur_logps_all, dim=-1, index=mb_target_ids.unsqueeze(-1)).squeeze(-1)

                    total_p_loss = 0.0
                    total_v_loss = 0.0
                    total_entropy = 0.0
                    count_tokens = 0

                    for mb_i, orig_i in enumerate(mb_indices):
                        mask_i = mb_dec_mask[mb_i]
                        if not mask_i.any():
                            continue
                        act_v = cur_values[mb_i, mask_i].float()
                        act_lp = cur_logps[mb_i, mask_i]
                        act_old_lp = old_logps[orig_i, mask_i].detach()
                        act_adv = norm_advantages[orig_i].detach()
                        act_ret = returns[orig_i].detach()

                        ratio = torch.exp(act_lp - act_old_lp)
                        surr1 = ratio * act_adv
                        surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * act_adv
                        p_loss = -torch.min(surr1, surr2).mean()

                        v_loss = 0.5 * F.mse_loss(act_v, act_ret)
                        act_entropy = -(torch.exp(act_lp) * act_lp).mean()

                        total_p_loss += p_loss
                        total_v_loss += v_loss
                        total_entropy += act_entropy
                        count_tokens += 1

                    if count_tokens > 0:
                        loss = (total_p_loss / count_tokens) + (args.vf_coef * total_v_loss / count_tokens) - (args.entropy_coef * total_entropy / count_tokens)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()

                        epoch_ploss.append((total_p_loss / count_tokens).item())
                        epoch_vloss.append((total_v_loss / count_tokens).item())

            epoch_rewards.extend(comp_rewards.tolist())
            epoch_styles.extend(style_scores.tolist())
            epoch_sems.extend(sem_scores.tolist())
            epoch_kls.extend(kl_divs)

            progress_bar.set_postfix({
                "Rew": f"{np.mean(comp_rewards):.3f}",
                "Style": f"{np.mean(style_scores):.3f}",
                "Sem": f"{np.mean(sem_scores):.3f}",
                "KL": f"{np.mean(kl_divs):.3f}",
            })

        mean_r = np.mean(epoch_rewards)
        mean_s = np.mean(epoch_styles)
        mean_sem = np.mean(epoch_sems)
        mean_kl = np.mean(epoch_kls)
        mean_pl = np.mean(epoch_ploss) if epoch_ploss else 0.0
        mean_vl = np.mean(epoch_vloss) if epoch_vloss else 0.0

        history["step"].append(epoch)
        history["mean_reward"].append(mean_r)
        history["style_score"].append(mean_s)
        history["sem_score"].append(mean_sem)
        history["kl_divergence"].append(mean_kl)
        history["policy_loss"].append(mean_pl)
        history["value_loss"].append(mean_vl)

        print(f"\n[Epoch {epoch} Summary] Mean Reward: {mean_r:.4f} | Style: {mean_s:.4f} | Sem: {mean_sem:.4f} | KL: {mean_kl:.4f} | Policy Loss: {mean_pl:.4f} | Value Loss: {mean_vl:.4f}")

    # Save Checkpoint & Tokenizer
    print(f"\nSaving merged standalone Seq2Seq PPO model and Value Head to: {args.output_dir}")
    try:
        merged_m = model.pretrained_model.merge_and_unload()
        merged_m.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Fallback saving PEFT model: {e}")
        model.pretrained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.v_head.state_dict(), os.path.join(args.output_dir, "value_head.pt"))

    history_file = os.path.join(args.output_dir, "ppo_seq2seq_training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Plot Curves
    plot_path = os.path.join(args.plot_dir, "ppo_seq2seq_training_curves.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["step"], history["mean_reward"], marker="o", color="#1f77b4", label="Composite Reward")
    axes[0, 0].set_title("Mean Composite Reward (Seq2Seq)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)

    axes[0, 1].plot(history["step"], history["style_score"], marker="s", color="#2ca02c", label="BiLSTM Style")
    axes[0, 1].plot(history["step"], history["sem_score"], marker="^", color="#ff7f0e", label="SBERT Semantic")
    axes[0, 1].set_title("Style Simplicity vs. Semantic Similarity")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)

    axes[1, 0].plot(history["step"], history["kl_divergence"], marker="d", color="#d62728", label="KL Divergence")
    axes[1, 0].set_title("Policy KL Divergence from SFT Reference")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("KL")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    axes[1, 1].plot(history["step"], history["policy_loss"], marker="x", color="#9467bd", label="Policy Loss")
    axes[1, 1].plot(history["step"], history["value_loss"], marker="v", color="#8c564b", label="Value Loss")
    axes[1, 1].set_title("PPO Loss Curves (Seq2Seq)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"PPO Seq2Seq training curves saved to: {plot_path}")
    print("=" * 80)
    print("Seq2Seq PPO Training Completed Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
