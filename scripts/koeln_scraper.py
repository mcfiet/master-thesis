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

def fetch_with_retry(url, max_retries=3):
    """Fetches a URL with exponential backoff to handle rate limits/connection drops."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        read=max_retries,
        connect=max_retries,
        backoff_factor=2, # 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    return session.get(url, headers=headers, timeout=30)

def get_koeln_ls_articles(base_url):
    """Extracts LS article links from the Stadt Koeln overview page (via Wayback Machine)."""
    print(f"Crawling overview page: {base_url}")
    try:
        response = fetch_with_retry(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching overview: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    ls_links = []
    # Find lists with class textteaserliste
    lists = soup.find_all('ul', class_='textteaserliste')
    for ul in lists:
        links = ul.find_all('a', class_='linkintern')
        for link in links:
            href = link.get('href')
            if href:
                full_url = urljoin("https://web.archive.org", href)
                ls_links.append(full_url)
    
    return list(set(ls_links))

def extract_koeln_content(soup):
    """Extracts clean article content from a Stadt Koeln page."""
    main_content = soup.find('main', id='inhalt')
    
    texts = []
    if main_content:
        content_tags = main_content.find_all(['p', 'li', 'h2', 'h3'])
        for tag in content_tags:
            if "Alltags-Sprache lesen" in tag.get_text():
                continue
            if tag.find_parent(id='wm-ipp-base'):
                continue
                
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
                
    return " ".join(texts)

def extract_as_link_and_content(ls_url):
    """Finds the AS link within an LS article and extracts content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching LS article {ls_url}: {e}")
        return None, 0, ""

    soup = BeautifulSoup(response.text, 'html.parser')
    
    ls_text = extract_koeln_content(soup)
    ls_tokens = count_tokens(ls_text)

    as_link = None
    for a in soup.find_all('a', href=True):
        if "alltags-sprache lesen" in a.get_text().lower():
            href = a['href']
            as_link = urljoin("https://web.archive.org", href)
            break
            
    return as_link, ls_tokens, ls_text

def get_as_article_data(as_url):
    """Fetches the AS article, extracts content and counts tokens."""
    if not as_url:
        return 0, ""
    try:
        response = fetch_with_retry(as_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching AS article {as_url}: {e}")
        return 0, ""

    soup = BeautifulSoup(response.text, 'html.parser')
    as_text = extract_koeln_content(soup)
    return count_tokens(as_text), as_text

def main():
    start_url = "https://web.archive.org/web/20220804230818/https://www.stadt-koeln.de/leben-in-koeln/soziales/informationen-leichter-sprache"
    
    ls_urls = get_koeln_ls_articles(start_url)
    print(f"Found {len(ls_urls)} articles on overview page.")
    
    aligned_pairs = []
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    pair_count = 0
    
    print(f"Found {len(ls_urls)} potential LS articles. Starting alignment check...")

    for ls_url in ls_urls:
        as_link, ls_tokens, _ = extract_as_link_and_content(ls_url)
        
        if as_link:
            as_tokens, _ = get_as_article_data(as_link)
            if as_tokens > 0:
                pair_count += 1
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
                aligned_pairs.append({
                    "ls_url": ls_url,
                    "as_url": as_link,
                    "ls_tokens": ls_tokens,
                    "as_tokens": as_tokens
                })
                print(f"Match found: {pair_count} - {ls_url} (1 AS links)")
            else:
                unaligned_count += 1
                print(f"AS links found but no content could be extracted: {ls_url}")
        else:
            unaligned_count += 1
            print(f"No AS link found for: {ls_url}")
        
        # Increased delay significantly to avoid Wayback Machine bans
        time.sleep(4)

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

    output_file = "koeln_aligned_urls.json"
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
