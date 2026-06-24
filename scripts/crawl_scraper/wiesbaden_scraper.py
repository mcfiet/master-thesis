import requests
from bs4 import BeautifulSoup
import json
import re
import time
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
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

def get_sitemap_urls(index_url):
    """Extracts sub-sitemap URLs from the main index."""
    print(f"Fetching sitemap index: {index_url}")
    response = fetch_with_retry(index_url)
    if not response:
        return []
    
    sitemaps = re.findall(r'<loc>(https?://[^<]+)</loc>', response.text)
    return sitemaps

def get_urls_from_sitemap(sitemap_url):
    """Extracts German base URLs from a sub-sitemap."""
    print(f"Processing sitemap: {sitemap_url}")
    response = fetch_with_retry(sitemap_url)
    if not response:
        return []
    
    # Use regex for speed as sitemaps are huge
    all_urls = re.findall(r'<loc>(https?://[^<]+)</loc>', response.text)
    
    urls = []
    # Filter for German base URLs
    lang_patterns = ['/en/', '/ar/', '/bg/', '/fr/', '/el/', '/pl/', '/ro/', '/ru/', '/es/', '/tr/', '/uk/', '/it/']
    
    for url in all_urls:
        if not any(pattern in url for pattern in lang_patterns):
            urls.append(url)
    
    return urls

def check_for_ls_toggle(url):
    """Checks if a page has the Leichte Sprache toggle."""
    response = fetch_with_retry(url)
    if not response:
        return None
    
    # Quick check in text before parsing DOM to save time
    if "simple-language" not in response.text and "easylanguage=" not in response.text:
        return None
        
    soup = BeautifulSoup(response.text, 'html.parser')
    # Look for class SP-Link--simple-language
    toggle = soup.find('a', class_='SP-Link--simple-language')
    if not toggle:
        # Fallback: search for any link with easylanguage parameter
        toggle = soup.find('a', href=re.compile(r'sp:easylanguage=1|sp%3Aeasylanguage=1'))
    
    if toggle:
        ls_url = urljoin(url, toggle['href'])
        return {
            "as_url": url,
            "ls_url": ls_url
        }
    return None

def main():
    sitemap_index = "https://www.wiesbaden.de/sitemap.xml"
    output_file = "results/aligned_urls/wiesbaden_aligned_urls.json"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    sub_sitemaps = get_sitemap_urls(sitemap_index)
    
    all_as_urls = []
    for sitemap in sub_sitemaps:
        urls = get_urls_from_sitemap(sitemap)
        all_as_urls.extend(urls)
    
    # Remove duplicates
    all_as_urls = list(set(all_as_urls))
    print(f"Found {len(all_as_urls)} German URLs to check.")
    
    aligned_pairs = []
    
    # Limit to 50 for testing if you want, but here we go for all.
    # Actually, for the final run we want all, but let's add progress tracking.
    
    print("Starting concurrent checks for Leichte Sprache toggles...")
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(check_for_ls_toggle, url): url for url in all_as_urls}
        
        count = 0
        for future in as_completed(future_to_url):
            count += 1
            result = future.result()
            if result:
                aligned_pairs.append(result)
                print(f"[{count}/{len(all_as_urls)}] Found: {result['as_url']}")
            
            if count % 100 == 0:
                print(f"Progress: {count}/{len(all_as_urls)} URLs checked. Pairs found: {len(aligned_pairs)}")
                # Save partial results
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "summary": {
                            "total_urls_checked": count,
                            "aligned_pairs_count": len(aligned_pairs)
                        },
                        "pairs": aligned_pairs
                    }, f, ensure_ascii=False, indent=4)

    output_data = {
        "summary": {
            "total_urls_checked": len(all_as_urls),
            "aligned_pairs_count": len(aligned_pairs)
        },
        "pairs": aligned_pairs
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    
    print(f"Scraping complete. Found {len(aligned_pairs)} pairs. Saved to {output_file}")

if __name__ == "__main__":
    main()
