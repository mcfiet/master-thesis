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

def get_categories(base_url):
    """Extracts category links."""
    try:
        response = fetch_with_retry(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.select('a.stretched-link')
        categories = [urljoin("https://www.apotheken-umschau.de", l['href']) for l in links if '/einfache-sprache/' in l['href'] and l['href'] != '/einfache-sprache/']
        return list(set(categories))
    except Exception as e:
        print(f"Error categories: {e}")
        return []

def get_articles_from_category(cat_url):
    """Extracts article links."""
    try:
        response = fetch_with_retry(cat_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if href.endswith('.html') and '/einfache-sprache/' in href:
                articles.append(urljoin("https://www.apotheken-umschau.de", href))
        return list(set(articles))
    except Exception as e:
        print(f"Error articles in {cat_url}: {e}")
        return []

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

def extract_as_link_and_content(ls_url):
    """Finds the AS link and content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        ls_text = extract_au_content(soup)
        ls_tokens = count_tokens(ls_text)

        as_link = None
        # Robust strategy for AU: 
        # 1. Look for a link in a paragraph containing "noch mehr über ... lesen"
        # 2. Look for title="hier" or title containing "Informationen" in the article footer
        for a in soup.find_all('a', href=True):
            title = a.get('title', '').lower()
            href = a['href']
            # AS links are outside /einfache-sprache/
            if '/einfache-sprache/' not in href and href.endswith('.html'):
                if "hier" in title or "informationen" in title:
                    as_link = urljoin("https://www.apotheken-umschau.de", href)
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
        as_text = extract_au_content(soup)
        return count_tokens(as_text), as_text
    except Exception as e:
        return 0, ""

def main():
    start_url = "https://www.apotheken-umschau.de/einfache-sprache/"
    categories = get_categories(start_url)
    print(f"Found {len(categories)} categories.")
    
    ls_urls = []
    for cat in categories:
        ls_urls.extend(get_articles_from_category(cat))
    ls_urls = list(set(ls_urls))
    print(f"Found {len(ls_urls)} potential LS articles.")
    
    aligned_pairs = []
    pair_count = 0
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    
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

    output_file = "apotheken_aligned_urls.json"
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
