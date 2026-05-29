import json
import os
import pandas as pd
from tqdm import tqdm

# Configuration
ANALYSIS_CSV = "results/information_loss_analysis.csv"
SOURCE_DIR = "results/corpus"
OUTPUT_DIR = "results/corpus_cleaned"
SIM_MIN = 0.60
SIM_MAX = 0.99
MIN_LS_TOKENS = 10

def clean_corpus():
    # 1. Load Analysis Results
    if not os.path.exists(ANALYSIS_CSV):
        print(f"Error: {ANALYSIS_CSV} not found. Run analysis first.")
        return
    
    df = pd.read_csv(ANALYSIS_CSV)
    print(f"Total pairs in analysis: {len(df)}")

    # 2. Apply Filters
    # Filter by Similarity
    mask_sim = (df['semantic_similarity_512'] >= SIM_MIN) & (df['semantic_similarity_512'] <= SIM_MAX)
    
    # Filter by Length (LS)
    mask_length = df['ls_tokens'] > MIN_LS_TOKENS
    
    # Filter by Placeholder (Lorem Ipsum)
    mask_lorem = ~df['ls_text'].str.contains("Lorem ipsum", case=False, na=False)

    df_clean = df[mask_sim & mask_length & mask_lorem]
    
    print(f"Filtered pairs: {len(df_clean)} ({len(df_clean)/len(df)*100:.2f}% retained)")
    
    # Create a mapping of (source, as_url) to include
    # We use (source, as_url, ls_url) as a unique key
    keep_keys = set(zip(df_clean['source'], df_clean['as_url'], df_clean['ls_url']))

    # 3. Process Original Files
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_kept = 0
    
    for filename in os.listdir(SOURCE_DIR):
        if not filename.endswith("_articles.json"):
            continue
            
        source = filename.replace("_articles.json", "")
        input_path = os.path.join(SOURCE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pairs = data.get("pairs", [])
        cleaned_pairs = []
        
        for p in pairs:
            # Match pair from original file with the "keep" list from our analysis
            # Note: URLs might be NaN in CSV, which zip handles differently than original dicts
            key = (source, p.get('as_url'), p.get('ls_url'))
            
            # Handle potential NaN issues in comparison
            # We match by looking if the pair's attributes are in our filtered set
            # A safer way is to check if the urls match (handling None/NaN)
            
            match_found = False
            # Optimization: only check if source matches
            relevant_subset = df_clean[df_clean['source'] == source]
            
            # Since URLs can be tricky (whitespace, etc.), we trust the index if possible,
            # but here we just check against our key set.
            # We need to make sure NaNs in CSV match None in JSON.
            
            # Simple check:
            if key in keep_keys:
                cleaned_pairs.append(p)
                match_found = True
            else:
                # Handle NaN case: pandas read_csv turns empty as_url into NaN
                # We normalize the key for comparison
                as_u = p.get('as_url') if p.get('as_url') else "nan"
                ls_u = p.get('ls_url') if p.get('ls_url') else "nan"
                norm_key = (source, str(as_u).strip(), str(ls_u).strip())
                
                # Check normalized key
                # This is a bit slow but safe for this corpus size
                # Better: Use the dataframe directly
                pass

        # Use dataframe filtering for speed and reliability with NaNs
        source_df = df_clean[df_clean['source'] == source]
        # To avoid URL matching issues, we can also use the text hashes or indices if we had them.
        # But for now, let's just use the dataframe rows to rebuild the JSON.
        
        final_pairs = []
        for _, row in source_df.iterrows():
            final_pairs.append({
                "as_url": None if str(row['as_url']) == 'nan' else row['as_url'],
                "ls_url": None if str(row['ls_url']) == 'nan' else row['ls_url'],
                "as_text": row['as_text'], # Note: CSV might have truncated text? No, our analysis script saved preview but we should take full if possible.
                # Actually, our analysis script saved full text (or at least 1000 chars preview in the dict but the CSV has full text if not truncated by pandas)
                # Wait, the analysis script wrote the CSV. Let's check if it truncated.
            })
            
        # RE-DESIGN: The best way is to iterate over original JSON and check against filtered CSV IDs
        # Let's use the indices from the CSV.
        
        # New approach: Use the cleaned dataframe to write new files
        # We need to ensure we have the FULL text. My previous script saved the full text to the CSV.
        
        new_data = {
            "source": source,
            "count": len(source_df),
            "pairs": []
        }
        
        # We need to get the original texts because CSV might have truncated something (though it shouldn't)
        # and we want to preserve the exact original structure.
        
        # Actually, let's just use the data from the CSV, but I need to make sure I didn't truncate as_text/ls_text in the analysis script.
        # Looking back at analyze_corpus(): 'as_text': as_text[:1000] -> IT WAS TRUNCATED!
        
        # So I MUST read from the original JSONs.
        
        cleaned_json_pairs = []
        # Create a set of (source, as_text_start, ls_text_start) to match
        # Using a prefix of the text is safer than URLs which can be null.
        match_set = set()
        for _, row in source_df.iterrows():
            match_set.add((row['source'], str(row['as_text'])[:100], str(row['ls_text'])[:100]))

        for p in pairs:
            m_key = (source, str(p.get('as_text'))[:100], str(p.get('ls_text'))[:100])
            if m_key in match_set:
                cleaned_json_pairs.append(p)
                total_kept += 1

        new_data["pairs"] = cleaned_json_pairs
        new_data["count"] = len(cleaned_json_pairs)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            
    print(f"Cleanup complete. Total pairs kept: {total_kept}")
    print(f"Cleaned files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    clean_corpus()
