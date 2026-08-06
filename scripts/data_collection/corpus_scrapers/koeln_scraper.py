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

def fetch_with_retry(url, max_retries=3):
    """Fetches a URL with exponential backoff to handle rate limits/connection drops."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=2, # 2s, 4s, 8s
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

def extract_koeln_content(soup):
    """Extracts clean article content from a Stadt Koeln page."""
    main_content = soup.find('main', id='inhalt')
    
    if not main_content:
        return ""

    # Remove boilerplate elements
    boilerplate_selectors = [
        '#bewertung', 
        '#produktinfocenter', 
        '#download_info',
        '.rs_skip',
        '.feedback-form',
        '#wm-ipp-base' # Wayback Machine banner
    ]
    for selector in boilerplate_selectors:
        for element in main_content.select(selector):
            element.decompose()
    
    texts = []
    # Find all potential content tags
    content_tags = main_content.find_all(['p', 'li', 'h2', 'h3'])
    
    for tag in content_tags:
        # Avoid nested duplication: skip if any ancestor is also a content tag
        if tag.find_parent(['p', 'li', 'h2', 'h3']):
            continue
            
        text = tag.get_text(separator=" ", strip=True)
        
        # Skip standard repetitive phrases or short noise
        skip_phrases = [
            "Alltags-Sprache lesen",
            "Informationen in Leichter Sprache",
            "Diese Seite in Leichter Sprache anzeigen",
            "[Vorlesen lassen]",
            "Ihre E-Mail-Adresse",
            "email confirmation",
            "War dieser Artikel hilfreich für Sie?",
            "Falls Ihnen der Artikel nicht weiter geholfen hat",
            "Das würde uns helfen, unsere Qualitätsstandards zu verbessern",
            "Ihre Meinung ist uns wichtig",
            "Ihr Name",
            "Kommentar"
        ]
        if any(phrase in text for phrase in skip_phrases):
            continue
            
        if text:
            # Avoid adding the exact same block twice (common on Stadt Koeln pages)
            if text not in texts:
                texts.append(text)
                
    return " ".join(texts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "koeln_aligned_urls.json")
    output_file = os.path.join("results", "corpus", "koeln_articles.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load existing progress if available
    aligned_pairs = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                aligned_pairs = existing_data.get('pairs', [])
                print(f"Resuming from {len(aligned_pairs)} already processed pairs.")
        except:
            pass

    total_ls_tokens = sum(p['ls_tokens'] for p in aligned_pairs)
    total_as_tokens = sum(p['as_tokens'] for p in aligned_pairs)
    
    processed_urls = {p['ls_url'] for p in aligned_pairs}

    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        if ls_url in processed_urls:
            continue

        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_response:
            # Use apparent_encoding to better guess the actual charset
            ls_response.encoding = ls_response.apparent_encoding
            ls_soup = BeautifulSoup(ls_response.text, 'html.parser')
            ls_text = extract_koeln_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(1)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_response:
            as_response.encoding = as_response.apparent_encoding
            as_soup = BeautifulSoup(as_response.text, 'html.parser')
            as_text = extract_koeln_content(as_soup)
            as_tokens = count_tokens(as_text)
            
        if ls_text and as_text:
            if as_tokens < 100 and ls_tokens > 300:
                print(f"  [WARNING] Very short AS text ({as_tokens} tokens) compared to LS text ({ls_tokens} tokens). Potential Hub-Page.")

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
            
            # Save progress after each pair
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
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        time.sleep(1)

    print(f"\nFinal results saved to {output_file}")

if __name__ == "__main__":
    main()
