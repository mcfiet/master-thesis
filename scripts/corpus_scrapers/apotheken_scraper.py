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
    """Fetches a URL with exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_au_content(soup):
    """Extracts clean article content."""
    texts = []
    # Narrow down to the actual article text area
    article_part = soup.select_one('.article-body') or soup.select_one('article') or soup.select_one('main')
    
    if article_part:
        # p.text is the main content paragraph class in AU
        content_tags = article_part.find_all(['p', 'h2', 'h3', 'li'])
        for tag in content_tags:
            # Skip navigation, TOC, or the "hier finden Sie" footer
            if any(cls in tag.get('class', []) for cls in ['article-chapter', 'nav', 'footer']):
                continue
            if "Hier finden Sie" in tag.get_text() and tag.find('a'):
                continue
            if "Achtung : Dieser Link führt" in tag.get_text():
                continue
                
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
    return " ".join(texts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "apotheken_aligned_urls.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    total_ls_tokens = 0
    total_as_tokens = 0
    
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_response:
            ls_soup = BeautifulSoup(ls_response.text, 'html.parser')
            ls_text = extract_au_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(1)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_response:
            as_soup = BeautifulSoup(as_response.text, 'html.parser')
            as_text = extract_au_content(as_soup)
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
        
        time.sleep(1)

    results = {
        "summary": {
            "total_pairs_attempted": len(data['pairs']),
            "aligned_pairs_count": len(aligned_pairs),
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / len(aligned_pairs) if aligned_pairs else 0,
            "average_as_tokens": total_as_tokens / len(aligned_pairs) if aligned_pairs else 0
        },
        "pairs": aligned_pairs
    }

    output_file = os.path.join("results", "corpus", "apotheken_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
