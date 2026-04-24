import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
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

def fetch_with_retry(url, max_retries=5):
    """Fetches a URL with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_brandeins_content(soup):
    """Extracts AS and LS content using structural heuristics and color indicators."""
    as_text_parts = []
    ls_text_parts = []
    
    # 1. Capture Title and Kicker
    title = soup.select_one('.title-headline')
    if title:
        ls_text_parts.append(title.get_text(strip=True))
    
    # 2. Capture Summary (Intro)
    intro = soup.select_one('.title-text.text-big')
    if intro:
        ls_text_parts.append(intro.get_text(strip=True))

    # 3. Process Textblocks
    # Heuristic: First <p> in a textblock is AS, subsequent <p> are LS.
    textblocks = soup.select('section.textblock')
    for block in textblocks:
        paragraphs = block.find_all('p')
        if not paragraphs:
            continue
            
        current_block_as = []
        current_block_ls = []
        
        has_color_indicator = False
        for p in paragraphs:
            # Check for red colors in style attribute
            style = p.get('style', '').lower()
            if any(c in style for c in ['#fa4600', '#ff4948', '#ff0000', 'color: red']):
                current_block_ls.append(p.get_text(strip=True))
                has_color_indicator = True
            elif has_color_indicator:
                # If we already found LS in this block, assume subsequent are LS unless color changes
                current_block_ls.append(p.get_text(strip=True))
            else:
                # No color yet, might be AS
                current_block_as.append(p.get_text(strip=True))
        
        if not has_color_indicator:
            # Fallback to Structural Heuristic: First P is AS, rest is LS
            if len(paragraphs) > 1:
                as_text_parts.append(paragraphs[0].get_text(strip=True))
                for p in paragraphs[1:]:
                    ls_text_parts.append(p.get_text(strip=True))
            else:
                # If only one paragraph, check if it looks like AS (shortened with (...))
                txt = paragraphs[0].get_text(strip=True)
                if '(…)' in txt or len(txt) > 300:
                    as_text_parts.append(txt)
                else:
                    ls_text_parts.append(txt)
        else:
            as_text_parts.extend(current_block_as)
            ls_text_parts.extend(current_block_ls)
            
    return " ".join(as_text_parts), " ".join(ls_text_parts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "brandeins_aligned_urls.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    total_ls_tokens = 0
    total_as_tokens = 0
    
    for i, pair in enumerate(data['pairs']):
        url = pair['url']
        
        print(f"Processing item {i+1}/{len(data['pairs'])}: {url}")
        
        # Fetch content once
        response = fetch_with_retry(url)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            as_text, ls_text = extract_brandeins_content(soup)
            
            ls_tokens = count_tokens(ls_text)
            as_tokens = count_tokens(as_text)
            
            if ls_text and as_text:
                aligned_pairs.append({
                    "url": url,
                    "ls_text": ls_text,
                    "as_text": as_text,
                    "ls_tokens": ls_tokens,
                    "as_tokens": as_tokens
                })
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
        
        time.sleep(2)

    results = {
        "summary": {
            "total_items_attempted": len(data['pairs']),
            "aligned_pairs_count": len(aligned_pairs),
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / len(aligned_pairs) if aligned_pairs else 0,
            "average_as_tokens": total_as_tokens / len(aligned_pairs) if aligned_pairs else 0
        },
        "pairs": aligned_pairs
    }

    output_file = os.path.join("results", "corpus", "brandeins_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
