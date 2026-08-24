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
import hashlib

def count_tokens(text):
    """Simple whitespace-based token counting."""
    if not text:
        return 0
    # Text is already cleaned from HTML tags in extraction
    tokens = text.split()
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
    """Removes artifacts, navigation phrases and cuts off contact info."""
    if not text:
        return ""
    
    # Remove technical artifacts and file errors
    text = re.sub(r'\(Diese Datei existiert leider nicht mehr\.\)', '', text)
    text = re.sub(r'mutex/[a-z]+/mutex', '', text)
    
    # Remove navigation phrases
    nav_phrases = [
        r'Hier kommen Sie zum.*',
        r'Hier erfahren Sie mehr über.*',
        r'Hier erfahren Sie mehr zu.*',
        r'Dort können Sie.*ausleihen\.',
        r'Mehr darüber erfahren Sie hier',
        r'Hier finden Sie uns: Wegbeschreibung',
        r'Ansprech-Partner:',
        r'Ansprech-Partnerin:',
        r'Ihre Ansprechpartnerin:',
        r'Ihr Ansprechpartner:',
        r'Ansprechpartnerin:',
        r'Ansprechpartner:',
        r'Kontakt:',
        r'Adresse:',
        r'Telefon:',
        r'E-Mail:',
        r'Telefax:',
        r'Fax:',
        r'Hier finden Sie unsere Falt-Blätter.*',
        r'Hier finden Sie unsere Kooperationspartner.*',
    ]
    
    # Cut off text at contact signal words (case-insensitive)
    signal_words = [
        r'\bAnsprech-Partner',
        r'\bIhre Ansprechpartner',
        r'\bIhr Ansprechpartner',
        r'\bKontakt zur Lebenshilfe',
        r'\bKontakt Lebenshilfe',
        r'\bAdresse:',
        r'\bTelefon \d+',
        r'\bE-Mail central',
        r'\bE-Mail für alle',
        r'\bTermine finden Sie hier',
        r'\bSprechen Sie uns an!'
    ]
    
    for signal in signal_words:
        match = re.search(signal, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break

    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return cleaner.clean_text(text, source="main_taunus")

def extract_lmt_content(soup):
    """Extracts clean article content from a Lebenshilfe Main-Taunus page."""
    # Main content is typically in div.inhalt
    content_div = soup.find('div', class_='inhalt')
    
    texts = []
    if content_div:
        # Get paragraphs, list items, and headings
        found_tags = content_div.find_all(['p', 'li', 'h1', 'h2', 'h3'])
        
        for tag in found_tags:
            # Avoid double extraction of nested tags
            if any(parent in found_tags for parent in tag.parents):
                continue
                
            if tag.find_parent('nav') or tag.find_parent(id='sidebar'):
                continue
                
            text = tag.get_text(separator=" ", strip=True)
            # Remove non-breaking spaces
            text = text.replace('\xa0', ' ')
            if text:
                texts.append(text)
                
    full_text = " ".join(texts)
    return cleaner.clean_text(clean_text, source="main_taunus")(full_text)

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("data", "corpus", "1_aligned_urls", "main_taunus_aligned_urls.json")
    if not os.path.exists(aligned_urls_path):
        aligned_urls_path = os.path.join("results", "aligned_urls", "main_taunus_aligned_urls.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    seen_contents = set()
    total_ls_tokens = 0
    total_as_tokens = 0
    
    placeholder_patterns = [
        "Bald steht hier ein Text",
        "bitte ich Sie um etwas Geduld",
        "In Kürze finden Sie hier"
    ]
    
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        if ls_response:
            # Use content (bytes) to let BeautifulSoup detect encoding
            ls_soup = BeautifulSoup(ls_response.content, 'html.parser')
            ls_text = extract_lmt_content(ls_soup)
        
        # Check for placeholders
        if any(pattern in ls_text for pattern in placeholder_patterns):
            print(f"  Skipping pair: Placeholder detected in LS text.")
            continue

        time.sleep(1)
        
        # Fetch AS content
        as_response = fetch_with_retry(as_url)
        as_text = ""
        if as_response:
            as_soup = BeautifulSoup(as_response.content, 'html.parser')
            as_text = extract_lmt_content(as_soup)
            
        if ls_text and as_text:
            ls_tokens = count_tokens(ls_text)
            as_tokens = count_tokens(as_text)
            
            # Short text filter
            if ls_tokens < 20 or as_tokens < 20:
                print(f"  Skipping pair: Text too short (LS: {ls_tokens}, AS: {as_tokens})")
                continue
            
            # De-duplication
            content_hash = hashlib.md5((ls_text + as_text).encode('utf-8')).hexdigest()
            if content_hash in seen_contents:
                print(f"  Skipping pair: Duplicate content detected.")
                continue
                
            seen_contents.add(content_hash)
            
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
            print(f"  Added pair. Tokens: LS={ls_tokens}, AS={as_tokens}")
        
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

    output_file = os.path.join("data", "corpus", "2_raw_scraped", "main_taunus_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
