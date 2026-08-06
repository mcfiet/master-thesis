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
    # 1. Decompose unwanted tags globally
    for tag in soup(['figcaption', 'figure', 'script', 'style']):
        tag.decompose()

    # 2. Decompose TOC: ul elements where more than 50% of links start with #
    for ul in soup.find_all('ul'):
        links = ul.find_all('a')
        if links:
            hash_links = [a for a in links if a.get('href', '').startswith('#')]
            if len(hash_links) / len(links) > 0.5:
                ul.decompose()

    # 3. Decompose Summary Boxes
    summary_phrases = ["Kurz zusammengefasst", "Kurz erklärt", "Das Wichtigste zu"]
    for header in soup.find_all(['h2', 'h3']):
        if header.parent is None:
            continue
        header_text = header.get_text(strip=True)
        if any(phrase in header_text for phrase in summary_phrases) or "kurz erklärt" in header_text.lower():
            parent = header.parent
            if parent and parent.name in ['div', 'section']:
                parent.decompose()
            else:
                # Decompose the header and subsequent siblings until the next h2
                current = header
                while current:
                    next_sibling = current.next_sibling
                    if current.name == 'h2' and current != header:
                        break
                    if hasattr(current, 'decompose'):
                        current.decompose()
                    current = next_sibling

    # 4. Identify and decompose elements with specific classes
    unwanted_classes = ['copyright', 'picture-copyright', 'image-copyright', 'teaser', 'related-articles', 'toc', 'jump-links']
    for cls in unwanted_classes:
        for element in soup.find_all(class_=re.compile(cls, re.I)):
            element.decompose()

    texts = []
    # Preference for .article-body to avoid footer/sidebar
    article_part = soup.select_one('.article-body')
    if not article_part:
        article_part = soup.select_one('article') or soup.select_one('main')
    
    if article_part:
        content_tags = article_part.find_all(['p', 'h2', 'h3', 'li'])
        for tag in content_tags:
            if tag.parent is None:
                continue
                
            # Skip navigation or known footer elements
            if any(cls in tag.get('class', []) for cls in ['article-chapter', 'nav', 'footer', 'toc']):
                continue
            
            text = tag.get_text(separator=" ", strip=True)
            if not text:
                continue

            # 5. Content Filters
            # Skip if text contains copyright symbol
            if '©' in text:
                continue
                
            # Strict Boilerplate Removal
            boilerplate_phrases = [
                "Jetzt kostenlos anmelden",
                "Apotheken Umschau BASIS",
                "Dauerhaft kostenlos",
                "Hier kostenlos registrieren",
                "Das könnte Sie auch interessieren",
                "Mehr zum Thema",
                "Lesen Sie auch",
                "Passend zum Thema"
            ]
            if any(phrase in text for phrase in boilerplate_phrases):
                continue

            # Skip image captions/descriptions
            if text.startswith(("Das Bild zeigt", "Die Grafik zeigt")):
                continue
                
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
    
    # Process ALL pairs
    total_pairs = len(data['pairs'])
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_url = pair['as_url']
        
        print(f"Processing pair {i+1}/{total_pairs}: {ls_url}")
        
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
