import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os
from urllib.parse import urljoin, urlparse, urlunparse

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

    # Remove known boilerplate containers
    boilerplate_selectors = [
        '.SP-Intro__tools', 
        '.SP-LinkList', 
        '.SP-ContentFooter',
        '.SP-Share',
        '.SP-SocialMedia',
        '.SP-JumboButton__container',
        '.SP-LastUpdatedTimestamp',
        'nav', 
        'footer', 
        'script', 
        'style',
        'aside'
    ]
    for selector in boilerplate_selectors:
        for el in main_content.select(selector):
            el.decompose()

    texts = []
    # Focus on paragraphs, list items, and headings
    content_tags = main_content.find_all(['p', 'li', 'h1', 'h2', 'h3'])

    # Phrases to filter out or ignore
    skip_phrases = [
        "Seite teilen",
        "Das könnte Sie auch interessieren",
        "Übersetzt und geprüft vom",
        "Büro für Leichte Sprache",
        "Öffnet in einem neuen Tab",
        "PDF -Datei",
        "Stand:"
    ]

    for tag in content_tags:
        text = tag.get_text(separator=" ", strip=True)
        if text:
            # Check if text block should be skipped entirely
            if any(text.startswith(p) for p in ["Das könnte Sie auch interessieren", "Seite teilen", "Stand:"]):
                continue

            # Remove specific unwanted substrings
            for phrase in skip_phrases:
                text = text.replace(phrase, "").strip()

            # Basic cleanup of remaining text
            text = re.sub(r'\s+', ' ', text).strip()

            if text and len(text) > 2: # Avoid tiny fragments
                texts.append(text)

    return " ".join(texts)


def get_stuttgart_ls_articles(base_url):
    """Extracts LS article links from the stuttgart.de overview page."""
    print(f"Crawling overview page: {base_url}")
    response = fetch_with_retry(base_url)
    if not response:
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_links = []
    
    # Links are in SP-LinkList items and contain the text "(Leichte Sprache)"
    # The text is often nested in spans like <span class="SP-Link__title">
    links = soup.find_all('a', class_='SP-Link')
    for link in links:
        text = link.get_text(separator=" ", strip=True)
        if '(Leichte Sprache)' in text:
            href = link.get('href')
            if href:
                # Handle cases where href is just "?sp:out=easy" or similar
                if href.startswith('?'):
                    full_url = urljoin(base_url, href)
                else:
                    full_url = urljoin("https://www.stuttgart.de", href)
                ls_links.append(full_url)
    
    return list(set(ls_links))

def derive_as_url(ls_url):
    """Derives the AS URL by removing the ?sp:out=easy query parameter."""
    parsed = urlparse(ls_url)
    query = parsed.query
    # Remove sp:out=easy parameter
    params = [p for p in query.split("&") if p and "sp%3Aout=easy" not in p and "sp:out=easy" not in p]
    new_query = "&".join(params)
    
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))

def main():
    start_url = "https://www.stuttgart.de/leichte-sprache-index"
    
    ls_urls = get_stuttgart_ls_articles(start_url)
    print(f"Found {len(ls_urls)} potential LS articles.")
    
    aligned_pairs = []
    total_ls_tokens = 0
    total_as_tokens = 0
    pair_count = 0
    
    # Limit for testing if needed, but here we go for all
    for ls_url in ls_urls:
        as_url = derive_as_url(ls_url)
        print(f"Processing: {ls_url} -> {as_url}")
        
        # Verify and get token counts for metadata
        ls_res = fetch_with_retry(ls_url)
        as_res = fetch_with_retry(as_url)
        
        if ls_res and as_res:
            ls_soup = BeautifulSoup(ls_res.text, 'html.parser')
            as_soup = BeautifulSoup(as_res.text, 'html.parser')
            
            ls_text = extract_stuttgart_content(ls_soup)
            as_text = extract_stuttgart_content(as_soup)
            
            ls_tokens = count_tokens(ls_text)
            as_tokens = count_tokens(as_text)
            
            if ls_tokens > 0 and as_tokens > 0:
                pair_count += 1
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
                aligned_pairs.append({
                    "ls_url": ls_url,
                    "as_url": as_url,
                    "ls_tokens": ls_tokens,
                    "as_tokens": as_tokens
                })
                print(f"  [SUCCESS] Pair {pair_count} aligned.")
            else:
                print(f"  [WARNING] Empty content extracted for {ls_url}")
        else:
            print(f"  [ERROR] Could not fetch one of the URLs.")
            
        time.sleep(1)

    results = {
        "summary": {
            "total_ls_articles_scanned": len(ls_urls),
            "aligned_pairs_count": pair_count,
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / pair_count if pair_count > 0 else 0,
            "average_as_tokens": total_as_tokens / pair_count if pair_count > 0 else 0
        },
        "pairs": aligned_pairs
    }

    output_file = "results/aligned_urls/stuttgart_aligned_urls.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
