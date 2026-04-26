import requests
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

def get_taz_ls_articles(base_url):
    """Extracts LS article links from the taz overview page."""
    print(f"Crawling overview page: {base_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(base_url, timeout=10, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching overview: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.select('a.teaser-link')
    
    easy_urls = []
    for link in links:
        href = link.get('href')
        if href and 'Leichte-Sprache' in href:
            full_url = urljoin("https://taz.de/", href)
            if ";" in full_url:
                full_url = full_url[:full_url.find(";")]
            easy_urls.append(full_url)
    
    return list(set(easy_urls))

def extract_taz_content(soup):
    """Extracts clean article content."""
    # Remove donation boilerplate and other non-article elements before extracting
    for unwanted in soup.select('.tzi-bottom-container, .tziBottom, .social-media-title'):
        unwanted.decompose()

    texts = []
    
    # Try finding elements by specific classes first
    content_tags = soup.select('p.bodytext, p.article, h2.bodytext, h3.bodytext')
    
    if not content_tags:
        # Fallback to the structure inside articleBody
        article_body = soup.select_one('article[itemprop="articleBody"]')
        if article_body:
            content_tags = article_body.find_all(["p", "h2", "h3", "h6", "ul"])
            
    if content_tags:
        for tag in content_tags:
            if tag.find_parent('nav') or tag.find_parent('footer'):
                continue
            text = tag.get_text(separator=" ", strip=True)
            
            # Filter standard boilerplate phrases
            if "──────────────────" in text or "Hinweis:" in text:
                continue
            if "können Sie den Text herunterladen" in text or text == "Hier":
                continue
            if "Dieser Text ist Werbung für die Zeitung taz" in text:
                continue
            if "Als Genossenschaft gehören wir unseren Leser:innen" in text:
                continue
            if "Die Infos in diesem leichten Text kommen aus" in text:
                continue
                
            if text:
                texts.append(text)
        
    return " ".join(texts)

def extract_as_links_and_content(ls_url):
    """Finds AS links within a taz LS article and extracts content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(ls_url, timeout=10, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching LS article {ls_url}: {e}")
        return [], 0, ""

    soup = BeautifulSoup(response.text, 'html.parser')
    ls_text = extract_taz_content(soup)
    ls_tokens = count_tokens(ls_text)

    as_links = []
    # Strategy 1: Look for links with "schweren Text"
    for a in soup.find_all('a', href=True):
        txt = a.get_text().lower()
        if "schweren" in txt and "text" in txt:
            # specifically check for singular to avoid overview pages if possible, 
            # though we will filter URLs next.
            as_links.append(urljoin("https://taz.de/", a['href']))
            
    # Strategy 2: Look for <em> tags with links
    if not as_links:
        for em in soup.find_all('em'):
            for a in em.find_all('a', href=True):
                as_links.append(urljoin("https://taz.de/", a['href']))
    
    valid_as_links = []
    for link in set(as_links):
        if 'taz.de' in link and 'Leichte-Sprache' not in link:
            # Ignore tag pages (!t...) and overview pages (!p...)
            if '/!t' in link or '/!p' in link:
                continue
            valid_as_links.append(link)
        
    return valid_as_links, ls_tokens, ls_text

def get_as_article_data(as_url):
    """Fetches the AS article, extracts content and counts tokens."""
    if not as_url:
        return 0, ""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(as_url, timeout=10, headers=headers)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching AS article {as_url}: {e}")
        return 0, ""

    soup = BeautifulSoup(response.text, 'html.parser')
    as_text = extract_taz_content(soup)
    return count_tokens(as_text), as_text

def main():
    start_url = "https://taz.de/Politik/Deutschland/Leichte-Sprache/!p5097/"
    ls_urls = get_taz_ls_articles(start_url)
    print(f"Found {len(ls_urls)} articles on overview page.")
    
    aligned_pairs = []
    unaligned_count = 0
    total_ls_tokens = 0
    total_as_tokens = 0
    pair_count = 0
    
    print(f"Found {len(ls_urls)} potential LS articles. Starting alignment check...")

    for ls_url in ls_urls:
        as_links, ls_tokens, _ = extract_as_links_and_content(ls_url)
        
        if as_links:
            current_pair_as_tokens = 0
            valid_as_links = []
            for as_url in as_links:
                as_tokens, _ = get_as_article_data(as_url)
                if as_tokens > 0:
                    current_pair_as_tokens += as_tokens
                    valid_as_links.append(as_url)
                    total_as_tokens += as_tokens
            
            if valid_as_links:
                pair_count += 1
                total_ls_tokens += ls_tokens
                aligned_pairs.append({
                    "ls_url": ls_url,
                    "as_urls": valid_as_links,
                    "ls_tokens": ls_tokens,
                    "as_tokens_total": current_pair_as_tokens
                })
                print(f"Match found: {pair_count} - {ls_url} ({len(valid_as_links)} AS links)")
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
            "average_as_tokens_per_ls_article": total_as_tokens / pair_count if pair_count > 0 else 0
        },
        "pairs": aligned_pairs
    }

    output_file = "taz_aligned_urls.json"
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
