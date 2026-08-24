import requests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import cleaner
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

def clean_text(text):
    """Removes editorial boilerplate and cleaning whitespace."""
    if not text:
        return ""
    
    # 1. Remove standard intro sentence
    text = text.replace("Die Leichte Sprache nimmt den Inhalt ernst, aber nicht schwer. Das kann erhellend sein.", "")
    
    # 2. Remove "Hier die Übersetzung..." and the following sentence (introductory context)
    # These usually end with a period and a year or just a period.
    text = re.sub(r"Hier die Übersetzung.*?(\d{4}\.|\.)", "", text)
    
    # 3. Remove Author/Credits
    # Matches "Text: Holger Fröhlich", "Text: ...", etc.
    text = re.sub(r"Text:\s+.*?(?=[A-ZÄÖÜ][a-z]|Hier|$)", "", text)
    # Fallback for remaining author fragments
    text = text.replace("öhlich", "") 
    
    return " ".join(text.split())

def extract_brandeins_content(soup):
    """Extracts AS and LS content strictly based on color indicators."""
    as_text_parts = []
    ls_text_parts = []
    
    # 1. Capture Title (usually LS)
    title = soup.select_one('.title-headline')
    if title:
        ls_text_parts.append(clean_text(title.get_text(strip=True)))
    
    # 2. Capture Summary (Intro)
    intro = soup.select_one('.title-text.text-big')
    if intro:
        # Intros often contain boilerplate, cleaning is essential
        ls_text_parts.append(clean_text(intro.get_text(strip=True)))

    # 3. Process Textblocks
    # We iterate through each paragraph and check for red color indicators
    # in the tag itself OR any nested children (spans, etc.)
    textblocks = soup.select('section.textblock')
    red_indicators = ['#fa4600', '#ff4948', '#ff0000', 'color: red', 'color:#ff0000']
    
    for block in textblocks:
        paragraphs = block.find_all('p')
        for p in paragraphs:
            p_html = str(p).lower()
            p_text = clean_text(p.get_text(strip=True))
            
            if not p_text:
                continue

            # Check for color in the paragraph's HTML (deep inspection)
            if any(c in p_html for c in red_indicators):
                ls_text_parts.append(p_text)
            else:
                as_text_parts.append(p_text)
            
    return " ".join(as_text_parts), " ".join(ls_text_parts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("data", "corpus", "1_aligned_urls", "brandeins_aligned_urls.json")
    if not os.path.exists(aligned_urls_path):
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

    output_file = os.path.join("data", "corpus", "2_raw_scraped", "brandeins_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
