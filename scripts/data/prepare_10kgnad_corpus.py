#!/usr/bin/env python3
"""
=============================================================================
10kGNAD Alltagssprache Corpus Preprocessor (500 Tokens)
=============================================================================
Downloads/loads the 10kGNAD (10,000 German News Articles Dataset) and segments
articles into clean, coherent Alltagssprache passages aligned with a target
token length of up to 500 tokens.

Features:
  - Normalized deduplication (exact & near-duplicates, typography agnostic)
  - Prefix deduplication (catches repetitive editorial templates/teasers)
  - Boilerplate & disclaimer filtering (removes quizzes, OTS headers, legal notes)
  - Token bounds: 50 to 500 tokens per passage

Output JSON format:
[
  {
    "as_text": "...",
    "as_tokens": 342,
    "source": "10kgnad_panorama",
    "category": "Panorama",
    "article_id": 0
  },
  ...
]
=============================================================================
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import spacy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger("Prepare10kGNAD")

DEFAULT_URL = "https://raw.githubusercontent.com/tblock/10kGNAD/master/articles.csv"

# Known noise & boilerplate patterns in 10kGNAD / Der Standard
BOILERPLATE_PATTERNS = [
    re.compile(r"wochenquiz", re.IGNORECASE),
    re.compile(r"wir wünschen viel spaß beim mitmachen", re.IGNORECASE),
    re.compile(r"was sie über diese woche wissen sollten", re.IGNORECASE),
    re.compile(r"der volltext dieses .* artikels steht aus rechtlichen", re.IGNORECASE),
    re.compile(r"apa[\s\-_/]*ots", re.IGNORECASE),
    re.compile(r"otsmeldung von", re.IGNORECASE),
    re.compile(r"welche bücher befinden sich aktuell auf ihrer leseliste", re.IGNORECASE),
    re.compile(r"gemeindeergebnisse mandate und mögliche koalitionen", re.IGNORECASE),
]


def load_spacy_model():
    """Loads SpaCy German pipeline or falls back to blank model with sentencizer."""
    try:
        nlp = spacy.load("de_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
    except Exception:
        logger.warning("Spacy model 'de_core_news_sm' not found, using blank('de') with sentencizer.")
        nlp = spacy.blank("de")
        nlp.add_pipe("sentencizer")
    return nlp


def load_raw_articles(source_url: str, local_csv: Optional[str] = None) -> pd.DataFrame:
    """Loads 10kGNAD dataframe from local file or GitHub raw URL."""
    if local_csv and os.path.isfile(local_csv):
        logger.info(f"Loading local 10kGNAD CSV from: {local_csv}")
        df = pd.read_csv(local_csv, sep=";", header=None, names=["label", "text"], on_bad_lines="skip", quotechar='"')
    else:
        logger.info(f"Downloading 10kGNAD dataset from: {source_url}")
        req = urllib.request.Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            content = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(content), sep=";", header=None, names=["label", "text"], on_bad_lines="skip", quotechar='"')

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} articles across categories: {dict(df['label'].value_counts())}")
    return df


def normalize_text_for_dedup(text: str) -> str:
    """Normalizes text for robust near-duplicate matching regardless of typography."""
    text = str(text).lower()
    # Normalize dashes, quotes, and whitespace
    text = re.sub(r"[\s\-_–—]+", " ", text)
    text = re.sub(r'["\'„“»«`]+', "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_boilerplate(text: str) -> bool:
    """Checks if text matches known boilerplate or noise patterns."""
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def chunk_article_text(
    nlp,
    article_text: str,
    category: str,
    article_id: int,
    seen_full_hashes: Set[str],
    seen_prefix_hashes: Set[str],
    min_tokens: int = 50,
    max_tokens: int = 500,
) -> Tuple[List[Dict], int, int]:
    """
    Splits an article into coherent paragraph/sentence chunks bounded by token limits
    with built-in deduplication and boilerplate rejection.
    """
    doc = nlp(str(article_text).strip())
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    raw_chunks = []
    current_sentences = []
    current_token_count = 0

    for sent in sentences:
        sent_tokens = len(sent.split())

        # If adding this sentence exceeds max_tokens and we already meet min_tokens
        if current_token_count + sent_tokens > max_tokens and current_token_count >= min_tokens:
            chunk_text = " ".join(current_sentences).strip()
            raw_chunks.append(chunk_text)
            current_sentences = [sent]
            current_token_count = sent_tokens
        else:
            current_sentences.append(sent)
            current_token_count += sent_tokens

    # Remainder chunk
    if current_token_count >= min_tokens:
        chunk_text = " ".join(current_sentences).strip()
        raw_chunks.append(chunk_text)

    # Process and filter chunks
    valid_chunks = []
    dropped_dupes = 0
    dropped_boilerplate = 0

    for chunk in raw_chunks:
        # Check boilerplate
        if is_boilerplate(chunk):
            dropped_boilerplate += 1
            continue

        norm_full = normalize_text_for_dedup(chunk)
        norm_prefix = norm_full[:80]

        # Check full text duplication
        if norm_full in seen_full_hashes:
            dropped_dupes += 1
            continue

        # Check prefix template duplication
        if norm_prefix in seen_prefix_hashes and len(norm_prefix) >= 40:
            dropped_dupes += 1
            continue

        # Mark as seen
        seen_full_hashes.add(norm_full)
        if len(norm_prefix) >= 40:
            seen_prefix_hashes.add(norm_prefix)

        token_len = len(chunk.split())
        valid_chunks.append({
            "as_text": chunk,
            "as_tokens": token_len,
            "source": f"10kgnad_{category.lower()}",
            "category": category,
            "article_id": article_id,
        })

    return valid_chunks, dropped_dupes, dropped_boilerplate


def main():
    parser = argparse.ArgumentParser(description="Prepare 500-token Alltagssprache corpus from 10kGNAD.")
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/temperature_ladder_500/corpus_10kgnad_len500_as.json",
        help="Path for output JSON corpus.",
    )
    parser.add_argument(
        "--source_url",
        type=str,
        default=DEFAULT_URL,
        help="Download URL for 10kGNAD articles.csv.",
    )
    parser.add_argument(
        "--local_csv",
        type=str,
        default=None,
        help="Optional local path to 10kGNAD articles.csv.",
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=50,
        help="Minimum token length per chunk (default: 50).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Maximum token length per chunk (default: 500).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on total chunks to generate (e.g. 1500).",
    )
    args = parser.parse_args()

    nlp = load_spacy_model()
    df = load_raw_articles(source_url=args.source_url, local_csv=args.local_csv)

    seen_full_hashes: Set[str] = set()
    seen_prefix_hashes: Set[str] = set()

    all_chunks = []
    total_dropped_dupes = 0
    total_dropped_boilerplate = 0

    for idx, row in df.iterrows():
        cat = str(row.get("label") or "news").strip()
        text = str(row.get("text") or "").strip()
        chunks, d_dupes, d_bp = chunk_article_text(
            nlp=nlp,
            article_text=text,
            category=cat,
            article_id=idx,
            seen_full_hashes=seen_full_hashes,
            seen_prefix_hashes=seen_prefix_hashes,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
        )
        all_chunks.extend(chunks)
        total_dropped_dupes += d_dupes
        total_dropped_boilerplate += d_bp

        if args.max_samples and len(all_chunks) >= args.max_samples:
            all_chunks = all_chunks[: args.max_samples]
            break

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"=== Corpus Preparation Completed ===")
    logger.info(f"Total Valid Clean Chunks: {len(all_chunks)}")
    logger.info(f"Dropped Duplicate Chunks: {total_dropped_dupes}")
    logger.info(f"Dropped Boilerplate/Noise Chunks: {total_dropped_boilerplate}")
    logger.info(f"Saved to: {args.output_file}")

    if len(all_chunks) > 0:
        token_lengths = [c["as_tokens"] for c in all_chunks]
        logger.info(
            f"Token Length Stats -> Min: {min(token_lengths)} | Max: {max(token_lengths)} | "
            f"Avg: {sum(token_lengths)/len(token_lengths):.1f}"
        )


if __name__ == "__main__":
    main()
