#!/usr/bin/env python3
"""
Merge sharded DPO JSONL files into a single unified dataset and create train/validation splits.
"""
import argparse
import glob
import json
import os
import random


def main():
    parser = argparse.ArgumentParser(description="Merge DPO shard files into unified dataset.")
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="data/dpo/dpo_preference_pairs_decoder_only_shard_*.jsonl",
        help="Glob pattern matching shard JSONL files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/dpo/dpo_preference_pairs_decoder_only.jsonl",
        help="Target unified JSONL file.",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.15,
        help="Validation split ratio (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_pattern))
    if not files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return

    print(f"Found {len(files)} shard files: {files}")
    all_pairs = []
    seen_prompts = set()

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    prompt_key = record.get("prompt") or record.get("as_text")
                    if prompt_key and prompt_key in seen_prompts:
                        continue
                    if prompt_key:
                        seen_prompts.add(prompt_key)
                    all_pairs.append(record)
                except Exception as e:
                    print(f"Skipping malformed line in {fp}: {e}")

    print(f"Total unique pairs collected: {len(all_pairs)}")
    random.seed(args.seed)
    random.shuffle(all_pairs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    if args.val_split_ratio > 0 and len(all_pairs) > 10:
        split_idx = int((1.0 - args.val_split_ratio) * len(all_pairs))
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        base, ext = os.path.splitext(args.output_file)
        eval_file = f"{base}_eval{ext}"

        with open(args.output_file, "w", encoding="utf-8") as f:
            for p in train_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(train_pairs)} train pairs -> {args.output_file}")

        with open(eval_file, "w", encoding="utf-8") as f:
            for p in val_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_pairs)} eval pairs -> {eval_file}")
    else:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Wrote all {len(all_pairs)} pairs -> {args.output_file}")


if __name__ == "__main__":
    main()
