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

def get_brandeins_articles(overview_url):
    """Extracts article links from the brand eins LS overview page."""
    print(f"Crawling overview page: {overview_url}")
    try:
        response = fetch_with_retry(overview_url)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching overview: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_links = []
    
    # Strategy: Find links containing /magazine/ and are within article blocks
    # Based on investigation, they are often in span.like-h1 or similar
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/magazine/' in href and not any(x in href for x in ['/products/', '/sign_in']):
            # Ensure it's a full URL (Wayback machine sometimes has relative ones)
            full_url = urljoin("https://web.archive.org", href)
            ls_links.append(full_url)
    
    return list(set(ls_links))

def extract_brandeins_content(soup):
    """Extracts AS and LS content using structural heuristics and color indicators."""
    as_text_parts = []
    ls_text_parts = []
    
    # 1. Capture Title and Kicker
    title = soup.select_one('.title-headline')
    if title:
        ls_text_parts.append(title.get_text(strip=True))
    
    # 2. Capture Summary (Intro)
    intro = soup.select_one('.title-text.text-big')
    if intro:
        ls_text_parts.append(intro.get_text(strip=True))

    # 3. Process Textblocks
    # Heuristic: First <p> in a textblock is AS, subsequent <p> are LS.
    # Exception: If only one <p>, it might be part of an AS-only or LS-only block.
    # But usually Brand Eins LS articles pair them.
    textblocks = soup.select('section.textblock')
    for block in textblocks:
        paragraphs = block.find_all('p')
        if not paragraphs:
            continue
            
        # Color strategy (from strategy document)
        # Even if not visible in some snapshots, we check for it
        current_block_as = []
        current_block_ls = []
        
        has_color_indicator = False
        for p in paragraphs:
            # Check for red colors in style attribute
            style = p.get('style', '').lower()
            if any(c in style for c in ['#fa4600', '#ff4948', '#ff0000', 'color: red']):
                current_block_ls.append(p.get_text(strip=True))
                has_color_indicator = True
            elif has_color_indicator:
                # If we already found LS in this block, assume subsequent are LS unless color changes
                current_block_ls.append(p.get_text(strip=True))
            else:
                # No color yet, might be AS
                current_block_as.append(p.get_text(strip=True))
        
        if not has_color_indicator:
            # Fallback to Structural Heuristic: First P is AS, rest is LS
            if len(paragraphs) > 1:
                as_text_parts.append(paragraphs[0].get_text(strip=True))
                for p in paragraphs[1:]:
                    ls_text_parts.append(p.get_text(strip=True))
            else:
                # If only one paragraph, check if it looks like AS (shortened with (...))
                txt = paragraphs[0].get_text(strip=True)
                if '(…)' in txt or len(txt) > 300:
                    as_text_parts.append(txt)
                else:
                    ls_text_parts.append(txt)
        else:
            as_text_parts.extend(current_block_as)
            ls_text_parts.extend(current_block_ls)
            
    return " ".join(as_text_parts), " ".join(ls_text_parts)

def main():
    # Using a 2024 snapshot of the LS overview
    start_url = "https://web.archive.org/web/20240401000000/https://www.brandeins.de/themen/rubriken/leichte-sprache"
    
    ls_urls = get_brandeins_articles(start_url)
    print(f"Found {len(ls_urls)} potential articles.")
    
    aligned_pairs = []
    pair_count = 0
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    
    print(f"Starting extraction...")

    for url in ls_urls:
        try:
            response = fetch_with_retry(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            as_text, ls_text = extract_brandeins_content(soup)
            
            ls_tokens = count_tokens(ls_text)
            as_tokens = count_tokens(as_text)
            
            if ls_tokens > 0 and as_tokens > 0:
                pair_count += 1
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
                aligned_pairs.append({
                    "url": url,
                    "ls_tokens": ls_tokens,
                    "as_tokens": as_tokens
                })
                print(f"Match found: {pair_count} - {url}")
            else:
                unaligned_count += 1
                # print(f"Could not separate AS/LS for: {url}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
            unaligned_count += 1
        
        # Respectful delay
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

    output_file = "brandeins_aligned_urls.json"
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
