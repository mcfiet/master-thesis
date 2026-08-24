import requests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
import cleaner
from bs4 import BeautifulSoup
import json
import re
import time
import os

def count_tokens(text):
    """Simple whitespace-based token counting."""
    if not text:
        return 0
    clean_text = re.sub(r'<[^>]+>', '', text)
    tokens = clean_text.split()
    return len(tokens)

def fetch_with_retry(url, max_retries=3):
    """Fetches a URL with simple retry logic."""
    for i in range(max_retries):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"Attempt {i+1} failed for {url}: {e}")
            time.sleep(2)
    return None

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

def main():
    # Path to the aligned URLs
    aligned_urls_path = os.path.join("data", "corpus", "1_aligned_urls", "taz_aligned_urls.json")
    if not os.path.exists(aligned_urls_path):
        aligned_urls_path = os.path.join("results", "aligned_urls", "taz_aligned_urls.json")
    
    if not os.path.exists(aligned_urls_path):
        print(f"Aligned URLs file not found at {aligned_urls_path}")
        return

    with open(aligned_urls_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aligned_pairs = []
    total_ls_tokens = 0
    total_as_tokens = 0
    
    for i, pair in enumerate(data['pairs']):
        ls_url = pair['ls_url']
        as_urls = pair['as_urls'] # This is a list for Taz
        
        print(f"Processing pair {i+1}/{len(data['pairs'])}: {ls_url}")
        
        # Fetch LS content
        ls_response = fetch_with_retry(ls_url)
        ls_text = ""
        ls_tokens = 0
        if ls_response:
            ls_soup = BeautifulSoup(ls_response.text, 'html.parser')
            ls_text = extract_taz_content(ls_soup)
            ls_tokens = count_tokens(ls_text)
        
        time.sleep(1)
        
        # Fetch AS content for all URLs in the list
        valid_as_texts = []
        current_pair_as_tokens = 0
        for as_url in as_urls:
            as_response = fetch_with_retry(as_url)
            if as_response:
                as_soup = BeautifulSoup(as_response.text, 'html.parser')
                as_text = extract_taz_content(as_soup)
                if as_text:
                    valid_as_texts.append(as_text)
                    current_pair_as_tokens += count_tokens(as_text)
            time.sleep(1)
            
        if ls_text and valid_as_texts:
            aligned_pairs.append({
                "ls_url": ls_url,
                "as_urls": as_urls,
                "ls_text": ls_text,
                "as_texts": valid_as_texts,
                "ls_tokens": ls_tokens,
                "as_tokens_total": current_pair_as_tokens
            })
            total_ls_tokens += ls_tokens
            total_as_tokens += current_pair_as_tokens
        
        time.sleep(1)

    results = {
        "summary": {
            "total_pairs_attempted": len(data['pairs']),
            "aligned_pairs_count": len(aligned_pairs),
            "total_ls_tokens": total_ls_tokens,
            "total_as_tokens": total_as_tokens,
            "average_ls_tokens": total_ls_tokens / len(aligned_pairs) if aligned_pairs else 0,
            "average_as_tokens_per_ls_article": total_as_tokens / len(aligned_pairs) if aligned_pairs else 0
        },
        "pairs": aligned_pairs
    }

    # Ensure output directory exists
    os.makedirs(os.path.join("results", "corpus"), exist_ok=True)
    output_file = os.path.join("data", "corpus", "2_raw_scraped", "taz_articles.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()