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

def extract_wiesbaden_content(soup):
    """Extracts clean article content from a Wiesbaden.de page."""
    # Main content container
    article = soup.find('article', id='SP-Content')
    if not article:
        article = soup.find('div', class_='SP-Content__body')
    
    if not article:
        return ""

    # Clone the article to avoid modifying the original soup
    import copy
    content_area = copy.copy(article)

    # Remove boilerplate elements
    boilerplate_selectors = [
        '.SP-DataProtection',
        '.SP-PrivacyBarrier',
        '.SP-Bookmark',
        '.SP-Path',
        '.SP-Content__header--article nav',
        '.SP-Link--simple-language',
        'footer',
        '.SP-MoreLikeThis',
        '.SP-Content__footer',
        '.SP-OffCanvas__sidebar',
        '.SP-Navigation',
        'script',
        'style'
    ]
    for selector in boilerplate_selectors:
        for element in content_area.select(selector):
            element.decompose()
    
    texts = []
    # Relevant tags for content: headlines, paragraphs, list items
    content_tags = content_area.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li'])
    
    for tag in content_tags:
        # Avoid duplicate text from nested tags (e.g. <li><p>...</p></li>)
        if tag.find_parent(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
            continue
            
        text = tag.get_text(separator=" ", strip=True)
        
        # Clean noise
        text = text.replace("(Öffnet in einem neuen Tab)", "").strip()
        text = text.replace("(Öffnet in einem neuen Fenster)", "").strip()
        
        # Skip UI elements or repetitive noise
        skip_phrases = [
            "Leichte Sprache",
            "Gebärdensprache",
            "Seite teilen",
            "Zum Anfang springen",
            "War dieser Artikel hilfreich?",
            "Alle Dienstleistungen",
            "Veranstaltungskalender",
            "Suche öffnen",
            "Menü öffnen",
            "Terminanfrage",
            "Abholung",
            "Routenplaner öffnen",
            "Zum Fahrplan",
            "Hinweise zum ÖPNV"
        ]
        if any(phrase == text or phrase in text for phrase in skip_phrases if len(text) < 50):
            continue
            
        if text:
            if text not in texts:
                texts.append(text)
                
    return "\n".join(texts)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("results", "aligned_urls", "wiesbaden_aligned_urls.json")
    output_file = os.path.join("results", "corpus", "wiesbaden_articles.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    
    # We will NOT resume from existing file if we want to re-scrape with better logic,
    # but for efficiency let's just keep it as is and the user can decide to delete the file.
    # To re-scrape everything, we could just delete the output file.
    
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
            ls_response.encoding = ls_response.apparent_encoding
            ls_soup = BeautifulSoup(ls_response.text, 'html.parser')
            ls_text = extract_wiesbaden_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(1)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        as_tokens = 0
        if as_response:
            as_response.encoding = as_response.apparent_encoding
            as_soup = BeautifulSoup(as_response.text, 'html.parser')
            as_text = extract_wiesbaden_content(as_soup)
            as_tokens = count_tokens(as_text)
            
        # Quality filter: LS text should be long enough and not just a fragment
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
            
            # Save progress
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
        
        time.sleep(0.5)

    print(f"\nFinal results saved to {output_file}")


if __name__ == "__main__":
    main()
