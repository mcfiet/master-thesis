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

def get_hamburg_categories(base_url):
    """Extracts category links from the Hamburg.de LS overview page."""
    print(f"Crawling overview page: {base_url}")
    try:
        response = fetch_with_retry(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching overview: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    categories = []
    
    # Valid categories usually have exactly 3 slashes, e.g., /barrierefrei/leichte-sprache/politik
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/barrierefrei/leichte-sprache/') and href.count('/') == 3:
            # Avoid self-references or purely generic links if any
            if href != '/barrierefrei/leichte-sprache/':
                categories.append(urljoin("https://www.hamburg.de", href))
    
    return list(set(categories))

def get_articles_from_category(cat_url):
    """Extracts article links from a category page."""
    print(f"Scanning category: {cat_url}")
    try:
        response = fetch_with_retry(cat_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching category {cat_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    articles = []
    # Article links often have class km1-teaser__heading-link
    links = soup.select('a.km1-teaser__heading-link')
    for link in links:
        href = link.get('href')
        if href:
            articles.append(urljoin("https://www.hamburg.de", href))
            
    # Fallback: search for any links that look like LS articles
    if not articles:
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/leichte-sprache/' in href and '-' in href and href.endswith(tuple(str(i) for i in range(10))):
                articles.append(urljoin("https://www.hamburg.de", href))
                
    return list(set(articles))

def extract_hamburg_content(soup):
    """Extracts clean article content from a Hamburg.de page and filters boilerplate."""
    # Check for 404 error pages
    if "Seite nicht gefunden" in soup.get_text() or "Fehler 404" in soup.get_text():
        return ""
        
    content_areas = soup.select('.km1-richtext')
    if not content_areas:
        content_areas = soup.select('.km1-article')
    
    texts = []
    
    # Boilerplate patterns to exclude
    boilerplate_patterns = [
        r"Bitte beachten Sie, dass dieser Inhalt automatisch übersetzt wurde",
        r"Ein Computer hat diesen Text in Leichte Sprache übertragen",
        r"Der Text ist nicht durch Menschen mit Behinderungen geprüft worden",
        r"Sie können hier dazu mehr lesen",
        r"Büro für Leichte Sprache Köln",
        r"Kirsten Scholz hat den Text",
        r"Dirk Stauber, Sandra Mambrini und Wolfgang Klein haben den Text",
        r"\[zorn\d+\]"
    ]

    if content_areas:
        for area in content_areas:
            content_tags = area.find_all(['p', 'h2', 'h3', 'li'])
            for tag in content_tags:
                # Skip language bar or navigation
                if tag.find_parent(class_='km1-language-bar') or tag.find_parent(class_='km1-navigation'):
                    continue
                
                text = tag.get_text(separator=" ", strip=True)
                
                # Filter out boilerplate
                is_boilerplate = False
                for pattern in boilerplate_patterns:
                    if re.search(pattern, text):
                        is_boilerplate = True
                        break
                
                if text and not is_boilerplate:
                    texts.append(text)
                
    return " ".join(texts)

def extract_as_link_and_content(ls_url):
    """Finds the AS link and extracts content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check for machine-translated disclaimer in the raw text first
        if "Ein Computer hat diesen Text in Leichte Sprache übertragen" in soup.get_text():
            print(f"Skipping machine-translated article: {ls_url}")
            return None, 0, ""

        ls_text = extract_hamburg_content(soup)
        ls_tokens = count_tokens(ls_text)

        as_link = None
        # Strategy 1: Look in known language bar containers
        lang_bar = soup.select_one('.km1-language-bar') or \
                   soup.select_one('.language-bar') or \
                   soup.select_one('.km1-language-bar__btn-wrapper')
        
        if lang_bar:
            for a in lang_bar.find_all('a', href=True):
                text = a.get_text().lower()
                title = a.get('title', '').lower()
                if "originaltext" in text or "alltagssprache" in text or "originaltext" in title:
                    as_link = urljoin("https://www.hamburg.de", a['href'])
                    break
        
        # Strategy 2: Look for specific 'original-text' class which is unlikely to be used for unrelated links
        if not as_link:
            original_link = soup.select_one('a.original-text') or \
                            soup.select_one('a.original-language') or \
                            soup.find('a', class_='original-text')
            if original_link and original_link.get('href'):
                as_link = urljoin("https://www.hamburg.de", original_link['href'])
            
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
        as_text = extract_hamburg_content(soup)
        return count_tokens(as_text), as_text
    except Exception as e:
        return 0, ""

def main():
    start_url = "https://www.hamburg.de/barrierefrei/leichte-sprache"
    
    categories = get_hamburg_categories(start_url)
    print(f"Found {len(categories)} categories.")
    
    all_ls_urls = []
    for cat in categories:
        articles = get_articles_from_category(cat)
        all_ls_urls.extend(articles)
        time.sleep(0.5)
        
    ls_urls = list(set(all_ls_urls))
    print(f"Found {len(ls_urls)} total potential LS articles.")
    
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

    import os
    output_file = os.path.join("results", "aligned_urls", "hamburg_aligned_urls.json")
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
