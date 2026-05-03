import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_hannover_content(soup):
    """Extracts clean article content from a Hannover.de page."""
    # Main content container
    content_area = soup.find('article', class_='content-detail')
    if not content_area:
        content_area = soup.find('div', class_='ezlandingpage-field')
    
    if not content_area:
        # Fallback to main or body-like content
        content_area = soup.find('main') or soup.find('div', id='content')
    
    if not content_area:
        return ""

    # Clone the area to avoid modifying the original soup
    import copy
    area = copy.copy(content_area)

    # Remove boilerplate elements
    boilerplate_selectors = [
        'script', 'style', 'nav', 'footer', 'header',
        '.action-toolbar', '.breadcrumb', '.social-share',
        '.copyright', '.more-items', '.breadcrumb__item'
    ]
    for selector in boilerplate_selectors:
        for element in area.select(selector):
            element.decompose()
    
    texts = []
    # Relevant tags for content
    content_tags = area.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li'])
    
    for tag in content_tags:
        # Avoid duplicate text from nested tags
        if tag.find_parent(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            continue
            
        text = tag.get_text(separator=" ", strip=True)
        
        # Skip UI elements or repetitive noise
        skip_phrases = [
            "Leichte Sprache",
            "Gebärdensprache",
            "Seite teilen",
            "Vorlesen",
            "E-Mail",
            "Drucken",
            "Diese Nachricht in Leichter Sprache ist vom"
        ]
        if any(phrase == text or phrase in text for phrase in skip_phrases if len(text) < 60):
            continue
            
        if text:
            if text not in texts:
                texts.append(text)
                
    return "\n".join(texts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "hannover_aligned_urls.json")
    output_file = os.path.join("results", "corpus", "hannover_articles.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    
    # Check if output already exists to resume
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            aligned_pairs = existing_data.get('pairs', [])
            print(f"Resuming from {len(aligned_pairs)} existing pairs.")

    processed_ls_urls = {p['ls_url'] for p in aligned_pairs}
    
    total_ls_tokens = sum(p['ls_tokens'] for p in aligned_pairs)
    total_as_tokens = sum(p['as_tokens'] for p in aligned_pairs)
    
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        if ls_url in processed_ls_urls:
            continue
            
        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_response:
            ls_soup = BeautifulSoup(ls_response.text, 'html.parser')
            ls_text = extract_hannover_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_response:
            as_soup = BeautifulSoup(as_response.text, 'html.parser')
            as_text = extract_hannover_content(as_soup)
            as_tokens = count_tokens(as_text)
            
        # Quality filter
        if ls_text and as_text and ls_tokens > 40:
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
            
            # Save progress every 10 pairs
            if len(aligned_pairs) % 10 == 0:
                results = {
                    "summary": {
                        "total_pairs_attempted": i + 1,
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
        
        time.sleep(0.2) # Hannover server seems fast and robust

    # Final save
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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nFinal results saved to {output_file}")

if __name__ == "__main__":
    main()
