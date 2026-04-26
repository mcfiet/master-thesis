import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os

def count_tokens(text):
    """Simple whitespace-based token counting."""
    if not text:
        return 0
    clean_text = re.sub(r'<[^>]+>', '', text)
    tokens = clean_text.split()
    return len(tokens)

def fetch_with_retry(url, max_retries=3):
    """Fetches a URL with simple retry logic."""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=20, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"Attempt {i+1} failed for {url}: {e}")
            time.sleep(2)
    return None

def extract_mdr_content(soup):
    """Extracts clean article content from an MDR page."""
    # 1. Target the main content area to avoid nav/header/footer noise
    content_area = soup.select_one('main, article, .article-content')
    if not content_area:
        content_area = soup # Fallback
        
    # 2. Selection: Get Headline, Lead, and Paragraphs
    # In MDR LS, paragraphs are often in div.paragraph
    # In MDR AS, paragraphs are often in div.paragraph or p.text
    candidates = content_area.select('h1, .article-header__lead, .media-item__caption, div.paragraph, p.text')
    
    content_parts = []
    for el in candidates:
        # Avoid duplicate text if tags are nested
        if any(parent in candidates for parent in el.parents):
            continue
            
        text = el.get_text(separator=" ", strip=True)
        if text:
            # 3. Targeted Boilerplate removal (within elements)
            # Skip elements that are clearly not content
            lower_text = text.lower()
            skip_signals = [
                "hier können sie", 
                "schwerer sprache lesen", 
                "bildrechte:", 
                "nachrichten in leichter sprache",
                "neuer bereich",
                "neuer abschnitt",
                "hauptinhalt"
            ]
            if any(sig in lower_text for sig in skip_signals):
                continue
                
            content_parts.append(text)
    
    full_text = " ".join(content_parts)

    # 4. Clean up whitespace and special characters
    full_text = full_text.replace("•", " • ")
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return full_text

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "mdr_aligned_urls.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    total_ls_tokens = 0
    total_as_tokens = 0
    skipped_ratio = 0
    skipped_short = 0
    
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_response:
            # Use content (bytes) for better encoding detection
            ls_soup = BeautifulSoup(ls_response.content, 'html.parser')
            ls_text = extract_mdr_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(0.5)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_response:
            as_soup = BeautifulSoup(as_response.content, 'html.parser')
            as_text = extract_mdr_content(as_soup)
            as_tokens = count_tokens(as_text)
            
        if ls_text and as_text:
            # --- FILTERING LOGIC ---
            # 1. Length check
            if ls_tokens < 20 or as_tokens < 20:
                print(f"  -> Skipped: Text too short (LS: {ls_tokens}, AS: {as_tokens})")
                skipped_short += 1
                continue
                
            # 2. Ratio check (e.g., skip live tickers where AS is 10x longer)
            ratio = as_tokens / ls_tokens if ls_tokens > 0 else 0
            if ratio > 5.0 or ratio < 0.2:
                print(f"  -> Skipped: Extreme length ratio ({ratio:.2f})")
                skipped_ratio += 1
                continue

            aligned_pairs.append({
                "ls_url": ls_url,
                "as_url": as_url,
                "ls_text": ls_text,
                "as_text": as_text,
                "ls_tokens": ls_tokens,
                "as_tokens": as_tokens
            })
            total_ls_tokens += ls_tokens
            total_as_tokens += as_tokens
        
        time.sleep(0.5)

    results = {
        "summary": {
            "total_pairs_attempted": len(data['pairs']),
            "aligned_pairs_count": len(aligned_pairs),
            "skipped_short": skipped_short,
            "skipped_ratio": skipped_ratio,
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / len(aligned_pairs) if aligned_pairs else 0,
            "average_as_tokens": total_as_tokens / len(aligned_pairs) if aligned_pairs else 0
        },
        "pairs": aligned_pairs
    }

    output_file = os.path.join("results", "corpus", "mdr_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
