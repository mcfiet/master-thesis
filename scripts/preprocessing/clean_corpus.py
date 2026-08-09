import json
import os
import pandas as pd
from tqdm import tqdm

# Configuration
ANALYSIS_CSV = "data/analysis/information_loss_analysis_cleaned.csv"
SOURCE_DIR = "data/corpus/2_raw_scraped"
OUTPUT_DIR = "data/corpus/3_filtered_similarity"
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
    # Filter by Similarity - Now using the better Jina 8192 metric
    mask_sim = (df['semantic_similarity_8192'] >= SIM_MIN) & (df['semantic_similarity_8192'] <= SIM_MAX)
    
    # Filter by Length (LS)
    mask_length = df['ls_tokens'] >= MIN_LS_TOKENS
    
    # Filter by Placeholder (Lorem Ipsum)
    mask_lorem = ~df['ls_text'].str.contains("Lorem ipsum", case=False, na=False)

    df_clean = df[mask_sim & mask_length & mask_lorem]
    
    print(f"Filtered pairs: {len(df_clean)} ({len(df_clean)/len(df)*100:.2f}% retained)")
    
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
        
        # Use dataframe filtering for reliability
        source_df = df_clean[df_clean['source'] == source]
        
        # Create a set of (source, ls_text_start) to match
        # LS text is usually unique enough per pair within a source
        match_set = set()
        for _, row in source_df.iterrows():
            match_set.add((row['source'], str(row['ls_text'])[:200]))

        cleaned_json_pairs = []
        for p in pairs:
            # Check LS text prefix
            m_key = (source, str(p.get('ls_text'))[:200])
            if m_key in match_set:
                cleaned_json_pairs.append(p)
                total_kept += 1

        new_data = {
            "source": source,
            "count": len(cleaned_json_pairs),
            "pairs": cleaned_json_pairs
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)
            
    print(f"Cleanup complete. Total pairs kept: {total_kept}")
    print(f"Cleaned files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    clean_corpus()
