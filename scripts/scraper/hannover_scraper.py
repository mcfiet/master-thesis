import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

def count_tokens(text):
    """Simple whitespace-based token counting."""
    if not text:
        return 0
    clean_text = re.sub(r'<[^>]+>', '', text)
    tokens = clean_text.split()
    return len(tokens)

def fetch_with_retry(url, max_retries=3):
    """Fetches a URL with retry logic."""
    session = requests.Session()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                return response
            elif response.status_code in [429, 500, 502, 503, 504]:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            time.sleep(2 ** attempt)
    return None

def extract_content(soup):
    """Extracts text content from a page."""
    # Common content selectors for hannover.de
    content_area = soup.find('article', class_='content-detail')
    if not content_area:
        content_area = soup.find('div', class_='ezlandingpage-field')
    
    if not content_area:
        # Fallback to main or body-like content
        content_area = soup.find('main') or soup.find('div', id='content')
        
    if not content_area:
        return ""
    
    # Remove unwanted elements
    for unwanted in content_area.find_all(['script', 'style', 'nav', 'footer', 'header']):
        unwanted.decompose()
        
    # Extract text with preserved line breaks for better readability
    text = content_area.get_text(separator='\n')
    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

def crawl_hannover():
    base_url = "https://www.hannover.de/Leichte-Sprache"
    visited = set()
    queue = [base_url]
    aligned_pairs = []
    
    output_file = "results/aligned_urls/hannover_aligned_urls.json"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Starting crawl from {base_url}")
    
    while queue:
        url = queue.pop(0)
        # Strip fragment
        url = url.split('#')[0]
        
        if url in visited:
            continue
        visited.add(url)
        
        # Skip common non-HTML extensions
        if url.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.docx', '.xlsx')):
            continue

        print(f"Crawling: {url} (Queue size: {len(queue)}, Pairs found: {len(aligned_pairs)})")
        response = fetch_with_retry(url)
        if not response:
            continue
            
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            continue

        try:
            soup = BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"  Error parsing {url}: {e}")
            continue
        
        # 1. Extract canonical link to check if it's an article
        canonical = soup.find('link', rel='canonical')
        if canonical and 'href' in canonical.attrs:
            as_url = canonical['href'].split('#')[0]
            # If canonical is NOT a Leichte-Sprache URL, it's an article with an AS counterpart
            if "hannover.de" in as_url and "/Leichte-Sprache" not in as_url:
                # Avoid duplicates in pairs
                if not any(p['ls_url'] == url for p in aligned_pairs):
                    print(f"  Found article: {url} -> {as_url}")
                    aligned_pairs.append({
                        "ls_url": url,
                        "as_url": as_url
                    })
                    # Save progress occasionally
                    if len(aligned_pairs) % 10 == 0:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump({"pairs": aligned_pairs}, f, ensure_ascii=False, indent=4)
        
        # 2. Find new LS links
        for a in soup.find_all('a', href=True):
            href = a['href']
            # Normalize URL
            full_url = urljoin(url, href).split('#')[0]
            # Filter for LS links on the same domain
            parsed = urlparse(full_url)
            if parsed.netloc == "www.hannover.de" and parsed.path.startswith("/Leichte-Sprache"):
                # Exclude service pages
                exclude = ['/AGB', '/Impressum', '/Kontakt', '/Datenschutz', '/Barrierefreiheit', '/api/', '/Wichtiger-Hinweis']
                if not any(ex in parsed.path for ex in exclude) and full_url not in visited:
                    queue.append(full_url)
                    
    # Final save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"pairs": aligned_pairs}, f, ensure_ascii=False, indent=4)
    
    print(f"Crawl complete. Found {len(aligned_pairs)} pairs.")

if __name__ == "__main__":
    crawl_hannover()
