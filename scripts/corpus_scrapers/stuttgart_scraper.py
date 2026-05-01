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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    for i in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if i == max_retries - 1:
                print(f"Error fetching {url}: {e}")
                return None
            time.sleep(2 ** i)
    return None

def extract_stuttgart_content(soup):
    """Extracts clean article content from a stuttgart.de page."""
    main_content = soup.find('main')
    if not main_content:
        return ""

    # Remove non-content elements
    for el in main_content.select('.SP-Intro__tools, .SP-Linklist, nav, footer, script, style'):
        el.decompose()

    texts = []
    # Focus on paragraphs, list items, and headings
    content_tags = main_content.find_all(['p', 'li', 'h1', 'h2', 'h3'])
    for tag in content_tags:
        text = tag.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
            
    return " ".join(texts)

def main():
    aligned_urls_path = os.path.join("results", "aligned_urls", "stuttgart_aligned_urls.json")
    output_file = os.path.join("results", "corpus", "stuttgart_articles.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    processed_urls = set()
    
    # Load existing progress if available
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                aligned_pairs = existing_data.get('pairs', [])
                processed_urls = {p['ls_url'] for p in aligned_pairs}
                print(f"Resuming from {len(aligned_pairs)} already processed pairs.")
        except:
            pass

    total_ls_tokens = sum(p['ls_tokens'] for p in aligned_pairs)
    total_as_tokens = sum(p['as_tokens'] for p in aligned_pairs)

    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        if ls_url in processed_urls:
            continue

        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        ls_res = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_res:
            ls_soup = BeautifulSoup(ls_res.text, 'html.parser')
            ls_text = extract_stuttgart_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(1)
        
        as_res = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_res:
            as_soup = BeautifulSoup(as_res.text, 'html.parser')
            as_text = extract_stuttgart_content(as_soup)
            as_tokens = count_tokens(as_text)
            
        if ls_text and as_text:
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
            
            # Save progress periodically
            results = {
                "summary": {
                    "total_pairs_attempted": len(data['pairs']),
                    "aligned_pairs_count": len(aligned_pairs),
                    "total_ls_tokens": total_ls_tokens,
                    "total_as_tokens": total_as_tokens,
                    "average_ls_tokens": total_ls_tokens / len(aligned_pairs),
                    "average_as_tokens": total_as_tokens / len(aligned_pairs)
                },
                "pairs": aligned_pairs
            }
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        time.sleep(1)

    print(f"\nFinal results saved to {output_file}")

if __name__ == "__main__":
    main()
