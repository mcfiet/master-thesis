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

def get_ls_articles_from_search(start_url):
    """Extracts all LS article links from the search pages."""
    current_url = start_url
    ls_urls = []
    
    while current_url:
        print(f"Scanning search page: {current_url}")
        try:
            response = fetch_with_retry(current_url)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching search page {current_url}: {e}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Strategy: ul.searchresult > li.teaser
        search_results = soup.select('ul.searchresult li.teaser')
        if not search_results:
            # Fallback: just links with DE/LS in searchresult
            res_list = soup.select_one('ul.searchresult')
            if res_list:
                search_results = res_list.find_all('a', href=True)
            else:
                # Last fallback: any DE/LS links in the page content
                search_results = soup.find_all('a', href=re.compile(r'DE/LS/'))

        for item in search_results:
            a = item if item.name == 'a' else item.find('a', href=True)
            if a and a.get('href'):
                href = a['href']
                if 'DE/LS/' in href:
                    ls_urls.append(urljoin("https://www.behindertenbeauftragter.de", href))
        
        # Pagination: a.forward.button
        next_link = soup.select_one('a.forward.button')
        if next_link and next_link.get('href'):
            current_url = urljoin("https://www.behindertenbeauftragter.de", next_link['href'])
            # Avoid infinite loops
            if current_url == start_url:
                current_url = None
        else:
            current_url = None
            
        time.sleep(0.5)
        
    return list(set(ls_urls))

def extract_bb_content(soup):
    """Extracts clean article content from the Behindertenbeauftragter page."""
    # Main content is in <div id="content">
    content_div = soup.find('div', id='content')
    if not content_div:
        content_div = soup.find('main')
        
    texts = []
    if content_div:
        # Avoid navigation, language switcher, etc.
        # Often these have specific classes or are in <nav> or <header>
        for tag in content_div.find_all(['p', 'h1', 'h2', 'h3', 'li']):
            # Skip if it's in a language switcher or other meta areas
            if any(cls in (tag.get('class') or []) for cls in ['c-language-switch__li', 'c-mobile-nav__link']):
                continue
            if tag.find_parent(class_='c-language-switch'):
                continue
                
            text = tag.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)
                
    return " ".join(texts)

def extract_as_link_and_content(ls_url):
    """Finds the AS link and extracts content."""
    try:
        response = fetch_with_retry(ls_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        ls_text = extract_bb_content(soup)
        ls_tokens = count_tokens(ls_text)
        
        as_link = None
        # Strategy: .c-language-switch__l--as
        as_tag = soup.select_one('.c-language-switch__l--as')
        if as_tag and as_tag.get('href'):
            as_link = urljoin("https://www.behindertenbeauftragter.de", as_tag['href'])
        else:
            # Fallback: search for any link with "Alltagssprache" text
            for a in soup.find_all('a', href=True):
                if "alltagssprache" in a.get_text().lower():
                    as_link = urljoin("https://www.behindertenbeauftragter.de", a['href'])
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
        as_text = extract_bb_content(soup)
        return count_tokens(as_text), as_text
    except Exception as e:
        print(f"Error AS {as_url}: {e}")
        return 0, ""

def main():
    start_url = "https://www.behindertenbeauftragter.de/SiteGlobals/Forms/Suche/Expertensuche_Formular.html?documentLanguage_str=de_ls"
    
    ls_urls = get_ls_articles_from_search(start_url)
    print(f"Found {len(ls_urls)} potential LS articles.")
    
    aligned_pairs = []
    pair_count = 0
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    
    print(f"Starting alignment check...")

    for ls_url in ls_urls:
        # Filter out obvious non-article links if any
        if ls_url.endswith('.pdf'):
            unaligned_count += 1
            # print(f"Skipping PDF: {ls_url}")
            continue

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

    output_file = "behindertenbeauftragter_aligned_urls.json"
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
