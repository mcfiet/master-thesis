import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin

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
    return session.get(url, headers=headers, timeout=30)

def get_ls_articles_from_sitemap(sitemap_url):
    """Extracts LS article links from the Wayback Machine sitemap."""
    print(f"Crawling sitemap: {sitemap_url}")
    try:
        response = fetch_with_retry(sitemap_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_urls = []
    
    # Strategy: Find links with /ls/ in the href
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/ls/' in href:
            # Filter out administrative pages
            exclude = ['/ls/inhalt/', '/ls/kontakt/', '/ls/impressum', '/ls/test-m-0-']
            if any(ex in href for ex in exclude):
                continue
            
            full_url = urljoin("https://web.archive.org", href)
            ls_urls.append(full_url)
            
    return list(set(ls_urls))

def extract_lmt_content(soup):
    """Extracts clean article content from a Lebenshilfe Main-Taunus page."""
    # Main content is typically in div.inhalt
    content_div = soup.find('div', class_='inhalt')
    
    texts = []
    if content_div:
        # Get paragraphs, list items, and headings
        for tag in content_div.find_all(['p', 'li', 'h1', 'h2', 'h3']):
            # Skip navigation or UI text if it somehow gets in
            if tag.find_parent('nav') or tag.find_parent(id='sidebar'):
                continue
                
            text = tag.get_text(separator=" ", strip=True)
            # Remove non-breaking spaces and other artifacts
            text = text.replace('\xa0', ' ')
            if text:
                texts.append(text)
                
    return " ".join(texts)

def extract_as_link_and_content(ls_url):
    """Finds the AS link and extracts content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        ls_text = extract_lmt_content(soup)
        ls_tokens = count_tokens(ls_text)

        as_link = None
        # Strategy: title="Auf Alltags-Sprache umstellen"
        as_tag = soup.find('a', title='Auf Alltags-Sprache umstellen')
        if as_tag and as_tag.get('href'):
            as_link = urljoin("https://web.archive.org", as_tag['href'])
            
        return as_link, ls_tokens, ls_text
    except Exception as e:
        print(f"Error LS {ls_url}: {e}")
        return None, 0, ""

def get_as_article_data(as_url):
    """Fetches AS content."""
    try:
        response = fetch_with_retry(as_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        as_text = extract_lmt_content(soup)
        return count_tokens(as_text), as_text
    except Exception as e:
        print(f"Error AS {as_url}: {e}")
        return 0, ""

def main():
    start_url = "https://web.archive.org/web/20200926190423/https://www.lebenshilfe-main-taunus.de/ls/inhalt/"
    
    ls_urls = get_ls_articles_from_sitemap(start_url)
    print(f"Found {len(ls_urls)} potential LS articles.")
    
    aligned_pairs = []
    pair_count = 0
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    
    print(f"Starting alignment check...")

    for ls_url in ls_urls:
        as_link, ls_tokens, _ = extract_as_link_and_content(ls_url)
        if as_link:
            as_tokens, _ = get_as_article_data(as_link)
            if as_tokens > 0:
                pair_count += 1
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
                aligned_pairs.append({
                    "ls_url": ls_url, "as_url": as_link,
                    "ls_tokens": ls_tokens, "as_tokens": as_tokens
                })
                print(f"Match found: {pair_count} - {ls_url}")
            else:
                unaligned_count += 1
                print(f"AS links found but no content could be extracted: {ls_url}")
        else:
            unaligned_count += 1
            print(f"No AS link found for: {ls_url}")
        
        # Respectful delay for Wayback Machine
        time.sleep(5)

    results = {
        "summary": {
            "total_ls_articles_scanned": len(ls_urls),
            "aligned_pairs_count": pair_count,
            "unaligned_articles_count": unaligned_count,
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / pair_count if pair_count > 0 else 0,
            "average_as_tokens": total_as_tokens / pair_count if pair_count > 0 else 0
        },
        "pairs": aligned_pairs
    }

    output_file = "main_taunus_aligned_urls.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print("\n--- Scraping Report ---")
    print(f"Total LS articles scanned: {len(ls_urls)}")
    print(f"Aligned pairs found: {pair_count}")
    print(f"Unaligned articles: {unaligned_count}")
    print(f"Total LS tokens: {total_ls_tokens}")
    print(f"Total AS tokens: {total_as_tokens}")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
