import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import json
import re
import time
import os
from urllib.parse import urljoin

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
    return session.get(url, headers=headers, timeout=20)

def get_ls_articles_from_sitemap(sitemap_url):
    """Extracts all LS article links from the sitemap overview."""
    print(f"Crawling sitemap: {sitemap_url}")
    try:
        response = fetch_with_retry(sitemap_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_urls = []
    
    # Exclude list
    exclude = ['barrierefreiheit', 'seiten-uebersicht', 'gebaerden-sprache', 
               'kontakt', 'benutzer-hinweise', 'impressum', 'datenschutz', 
               'material-archiv', 'fuer-lehrkraefte']

    for a in soup.find_all('a', href=True):
        href = a['href']
        # Remove hash anchors
        if '#' in href:
            href = href.split('#')[0]
            
        # Standardize and check for LS links, excluding root /es/
        is_ls = href.startswith('/es/') or href == '/es'
        is_root = href.rstrip('/') == '/es'
        
        if is_ls and not is_root and not any(ex in href for ex in exclude):
            ls_urls.append(urljoin("https://www.sozialpolitik.com", href))
            
    return list(set(ls_urls))

def extract_sp_content(soup):
    """Extracts clean article content from a sozialpolitik.com page."""
    # Main content is in <main>
    main_content = soup.find('main')
    
    if not main_content:
        return ""

    # Remove boilerplate elements
    for boilerplate in main_content.select('.quiz-container, .quiz-wrapper, .download-area, .box-grey, aside, .sidebar-right, .info-box'):
        boilerplate.decompose()
    
    texts = []
    # Get paragraphs, list items, and headings
    for tag in main_content.find_all(['p', 'li', 'h1', 'h2', 'h3']):
        # Skip language switchers
        if 'header-navigation-point' in (tag.get('class') or []) or tag.find_parent(class_='header-navigation-point'):
            continue
        if 'underline easy' in (tag.get('class') or []) or tag.find_parent(class_='underline'):
            continue
        
        text = tag.get_text(separator=" ", strip=True)
        if text:
            texts.append(text)
                
    # Join with newlines to preserve structure
    content = "\n".join(texts)
    # Clean up multiple newlines
    content = re.sub(r'\n+', '\n', content).strip()
    return content

def extract_as_link_and_content(ls_url):
    """Finds the AS link and extracts content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        ls_text = extract_sp_content(soup)
        ls_tokens = count_tokens(ls_text)

        as_link = None
        # Strategy: class="underline easy" containing "Standardsprache"
        for a in soup.find_all('a', class_='underline easy', href=True):
            if "standardsprache" in a.get_text().lower():
                as_link = urljoin("https://www.sozialpolitik.com", a['href'])
                break
                
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
        as_text = extract_sp_content(soup)
        return count_tokens(as_text), as_text
    except Exception as e:
        print(f"Error AS {as_url}: {e}")
        return 0, ""

def main():
    start_url = "https://www.sozialpolitik.com/es/seiten-uebersicht"
    
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
        
        time.sleep(0.5)

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

    output_file = os.path.join("results", "aligned_urls", "sozialpolitik_aligned_urls.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
