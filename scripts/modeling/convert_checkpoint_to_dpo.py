#!/usr/bin/env python3
"""
Convert an in-progress DPO checkpoint file (.checkpoint.jsonl) into final train and eval JSONL datasets.
"""
import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Convert DPO checkpoint to train/eval dataset.")
    parser.add_argument("--checkpoint_file", type=str, required=True, help="Path to *.checkpoint.jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to target output .jsonl")
    parser.add_argument("--val_split_ratio", type=float, default=0.15, help="Validation split ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    pairs = []
    seen = set()
    with open(args.checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("type") == "pair" and "data" in rec:
                    pair = rec["data"]
                elif "prompt" in rec and "chosen" in rec and "rejected" in rec:
                    pair = rec
                else:
                    continue

                key = pair.get("prompt") or pair.get("as_text")
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                pairs.append(pair)
            except Exception:
                continue

    print(f"Loaded {len(pairs)} valid pairs from {args.checkpoint_file}")
    if not pairs:
        print("No valid pairs found!")
        return

    random.seed(args.seed)
    random.shuffle(pairs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if args.val_split_ratio > 0 and len(pairs) > 10:
        split_idx = int((1.0 - args.val_split_ratio) * len(pairs))
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        base, ext = os.path.splitext(args.output_file)
        eval_file = f"{base}_eval{ext}"

        with open(args.output_file, "w", encoding="utf-8") as f:
            for p in train_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(train_pairs)} train pairs -> {args.output_file}")

        with open(eval_file, "w", encoding="utf-8") as f:
            for p in val_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved {len(val_pairs)} eval pairs -> {eval_file}")
    else:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Saved all {len(pairs)} pairs -> {args.output_file}")


if __name__ == "__main__":
    main()
