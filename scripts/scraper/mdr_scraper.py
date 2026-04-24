import requests
from bs4 import BeautifulSoup
import json
import re
import time
import os

def count_tokens(text):
    """Simple whitespace-based token counting."""
    if not text:
        return 0
    # Remove HTML tags if any (though we usually pass clean text)
    clean_text = re.sub(r'<[^>]+>', '', text)
    tokens = clean_text.split()
    return len(tokens)

def get_mdr_ls_articles(base_url):
    """Extracts LS article links from the MDR overview page."""
    print(f"Crawling overview page: {base_url}")
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching overview: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_links = []
    
    # Strategy from research: div.box.cssArticle or div.box.cssInfoTeaser
    # Links in h4 a.headline
    articles = soup.select('div.box.cssArticle, div.box.cssInfoTeaser')
    for article in articles:
        link_tag = article.select_one('h4 a.headline')
        if link_tag and link_tag.get('href'):
            url = link_tag['href']
            if not url.startswith('http'):
                url = "https://www.mdr.de" + url
            ls_links.append(url)
    
    return list(set(ls_links)) # Remove duplicates

def extract_as_link_and_content(ls_url):
    """Finds the AS link within an LS article and extracts content for token counting."""
    try:
        response = requests.get(ls_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching LS article {ls_url}: {e}")
        return None, 0, ""

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 1. Extract LS content and count tokens
    # Typically MDR articles have content in div.paragraph or similar
    ls_text_elements = soup.select('div.paragraph, p.text')
    ls_text = " ".join([el.get_text() for el in ls_text_elements])
    ls_tokens = count_tokens(ls_text)

    # 2. Find AS link
    # Strategy: conHeadline with text "Hier können Sie diese Nachricht auch in schwerer Sprache lesen:"
    as_link = None
    headlines = soup.select('.conHeadline')
    for hl in headlines:
        if "schwerer sprache lesen" in hl.get_text().lower():
            # The link is usually in a following or nested element, often in a teaser box
            parent_box = hl.find_parent('div', class_='box')
            if parent_box:
                link_tag = parent_box.select_one('a')
                if link_tag and link_tag.get('href'):
                    as_link = link_tag['href']
                    break
            
            # Fallback: check adjacent links
            link_tag = hl.find_next('a')
            if link_tag and link_tag.get('href'):
                as_link = link_tag['href']
                break

    if as_link and not as_link.startswith('http'):
        as_link = "https://www.mdr.de" + as_link
        
    return as_link, ls_tokens, ls_text

def get_as_token_count(as_url):
    """Fetches the AS article and counts its tokens."""
    if not as_url:
        return 0
    try:
        response = requests.get(as_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching AS article {as_url}: {e}")
        return 0

    soup = BeautifulSoup(response.text, 'html.parser')
    as_text_elements = soup.select('div.paragraph, p.text')
    as_text = " ".join([el.get_text() for el in as_text_elements])
    return count_tokens(as_text)

def get_archive_state_links(archive_index_url):
    """Extracts the links to the state-specific archive pages."""
    print(f"Crawling archive index: {archive_index_url}")
    try:
        response = requests.get(archive_index_url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching archive index: {e}")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    state_links = []
    for a in soup.select('a'):
        href = a.get('href')
        if href and 'rueckblick-buendelgruppe' in href:
            if not href.startswith('http'):
                href = "https://www.mdr.de" + href
            state_links.append(href)
    return list(set(state_links))

def main():
    start_url = "https://www.mdr.de/nachrichten-leicht/nachrichten-in-leichter-sprache-114.html"
    archive_url = "https://www.mdr.de/nachrichten-leicht/rueckblick/index.html"
    
    # 1. Get current news
    ls_urls = get_mdr_ls_articles(start_url)
    print(f"Found {len(ls_urls)} articles on main page.")
    
    # 2. Get archive state links
    state_archive_links = get_archive_state_links(archive_url)
    
    # 3. Get articles from each state archive
    for state_link in state_archive_links:
        archive_ls_urls = get_mdr_ls_articles(state_link)
        print(f"Found {len(archive_ls_urls)} articles in {state_link.split('/')[-1]}")
        ls_urls.extend(archive_ls_urls)
        
    # Deduplicate
    ls_urls = list(set(ls_urls))
    
    aligned_pairs = []
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    pair_count = 0
    
    print(f"Found {len(ls_urls)} potential LS articles. Starting alignment check...")

    for ls_url in ls_urls:
        as_url, ls_tokens, _ = extract_as_link_and_content(ls_url)
        
        if as_url:
            as_tokens = get_as_token_count(as_url)
            if as_tokens > 0:
                pair_count += 1
                total_ls_tokens += ls_tokens
                total_as_tokens += as_tokens
                aligned_pairs.append({
                    "ls_url": ls_url,
                    "as_url": as_url,
                    "ls_tokens": ls_tokens,
                    "as_tokens": as_tokens
                })
                print(f"Match found: {pair_count} - {ls_url}")
            else:
                unaligned_count += 1
                print(f"AS article found but no content: {as_url}")
        else:
            unaligned_count += 1
            print(f"No AS link found for: {ls_url}")
        
        # Respectful scraping
        time.sleep(0.5)

    # Results
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

    # Save to file
    output_file = "mdr_aligned_urls.json"
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
