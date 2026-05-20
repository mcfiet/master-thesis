import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_CSV = "results/information_loss_analysis.csv"
OUTPUT_DIR = "research/img/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_visualizations(plots_to_run):
    df = pd.read_csv(INPUT_CSV)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    if 'jina_context' in plots_to_run or 'all' in plots_to_run:
        print("Generating Jina Context Comparison...")
        # 1. Jina Context Comparison (128 vs 512 vs 8192)
        sim_cols = ['semantic_similarity_128', 'semantic_similarity_512', 'semantic_similarity_8192']
        # Check if 8192 exists and has data, otherwise just plot 128 and 512
        valid_cols = [c for c in sim_cols if c in df.columns and not df[c].isna().all()]
        mean_sims = df.groupby('source')[valid_cols].mean().reset_index()
        
        # Rename columns for plot
        rename_dict = {'semantic_similarity_128': '128 Tokens', 
                       'semantic_similarity_512': '512 Tokens', 
                       'semantic_similarity_8192': '8192 Tokens'}
        mean_sims = mean_sims.rename(columns=rename_dict)
        
        sim_df = mean_sims.melt(id_vars='source', var_name='Context Limit', value_name='Similarity')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=sim_df, x='source', y='Similarity', hue='Context Limit')
        plt.xticks(rotation=45, ha='right')
        plt.title('Semantische Ähnlichkeit: Einfluss der Kontextlänge (Jina Model)')
        plt.ylabel('Cosine Similarity')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'jina_context_comparison.png'))
        plt.close()

    if 'semantic_sim_8192' in plots_to_run or 'all' in plots_to_run:
        # 1.5 Semantic Similarity by Source (Jina 8192) - Boxplot
        if 'semantic_similarity_8192' in df.columns and not df['semantic_similarity_8192'].isna().all():
            print("Generating Semantic Similarity by Source (Jina 8192)...")
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=df, x='source', y='semantic_similarity_8192', color='#4C72B0') 
            plt.xticks(rotation=45, ha='right')
            plt.title('Semantische Ähnlichkeit nach Quelle (Jina 8192 Tokens)')
            plt.ylabel('Cosine Similarity (SBERT)')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'semantic_similarity_8192.png'))
            plt.close()

    if 'bidirectional_ner' in plots_to_run or 'all' in plots_to_run:
        print("Generating Bidirectional NER Comparison...")
        # 2. Bidirectional NER Comparison
        ner_cols = ['ner_recall_as_ls', 'ner_recall_ls_as']
        mean_ner = df.groupby('source')[ner_cols].mean().reset_index()
        mean_ner.columns = ['Quelle', 'AS -> LS (Faktenerhalt)', 'LS -> AS (Faktentreue)']
        ner_df = mean_ner.melt(id_vars='Quelle', var_name='Richtung', value_name='Recall')

        plt.figure(figsize=(12, 6))
        sns.barplot(data=ner_df, x='Quelle', y='Recall', hue='Richtung')
        plt.xticks(rotation=45, ha='right')
        plt.title('Faktenerhalt vs. Faktentreue (Bidirektionales NER)')
        plt.ylabel('Recall Rate')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'bidirectional_ner_comparison.png'))
        plt.close()

    if 'token_ratio_sim' in plots_to_run or 'all' in plots_to_run:
        print("Generating Token Ratio vs Semantic Similarity...")
        # 3. Token Ratio vs Semantic Similarity
        plt.figure(figsize=(10, 6))
        sns.regplot(data=df, x='token_ratio', y='semantic_similarity_512', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
        plt.title('Korrelation: Token-Verhältnis vs. Semantische Ähnlichkeit')
        plt.xlabel('Token Ratio (LS / AS)')
        plt.ylabel('Semantische Ähnlichkeit (Jina 512)')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'token_ratio_vs_similarity_scatter.png'))
        plt.close()

    if 'sim_hist' in plots_to_run or 'all' in plots_to_run:
        print("Generating Semantic Similarity Histogram...")
        # 4. Semantic Similarity Histogram (Filtering)
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='semantic_similarity_512', bins=50, kde=True)
        plt.axvline(0.6, color='red', linestyle='--', label='Untere Grenze (0.6)')
        plt.axvline(0.98, color='green', linestyle='--', label='Obere Grenze (0.98)')
        plt.title('Verteilung der Semantischen Ähnlichkeit (inkl. Filtergrenzen)')
        plt.xlabel('Semantische Ähnlichkeit (Jina 512)')
        plt.ylabel('Anzahl Artikelpaare')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'similarity_distribution_hist.png'))
        plt.close()

    if 'sent_len' in plots_to_run or 'all' in plots_to_run:
        print("Generating Sentence Length Comparison...")
        # 5. Sentence Length Comparison
        sent_len_data = {
            'Quelle': df['source'].tolist() * 2,
            'Sprache': ['Alltagssprache'] * len(df) + ['Leichte Sprache'] * len(df),
            'Satzlänge': df['as_avg_sent_len'].tolist() + df['ls_avg_sent_len'].tolist()
        }
        sent_df = pd.DataFrame(sent_len_data)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=sent_df, x='Quelle', y='Satzlänge', hue='Sprache')
        plt.xticks(rotation=45, ha='right')
        plt.title('Durchschnittliche Satzlänge nach Quelle')
        plt.ylabel('Tokens pro Satz')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sentence_length_comparison_bar.png'))
        plt.close()

    if 'pos_dist' in plots_to_run or 'all' in plots_to_run:
        print("Generating POS Distribution Comparison...")
        # 6. POS Distribution Comparison
        pos_cols_as = ['as_adj_ratio', 'as_noun_ratio', 'as_verb_ratio', 'as_conj_ratio']
        pos_cols_ls = ['ls_adj_ratio', 'ls_noun_ratio', 'ls_verb_ratio', 'ls_conj_ratio']
        
        mean_pos = df[pos_cols_as + pos_cols_ls].mean()
        pos_data = {
            'Wortart': ['Adjektive', 'Nomen', 'Verben', 'Konjunktionen'],
            'Alltagssprache': [mean_pos['as_adj_ratio'], mean_pos['as_noun_ratio'], mean_pos['as_verb_ratio'], mean_pos['as_conj_ratio']],
            'Leichte Sprache': [mean_pos['ls_adj_ratio'], mean_pos['ls_noun_ratio'], mean_pos['ls_verb_ratio'], mean_pos['ls_conj_ratio']]
        }
        pos_df = pd.DataFrame(pos_data).melt(id_vars='Wortart', var_name='Sprache', value_name='Anteil')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=pos_df, x='Wortart', y='Anteil', hue='Sprache')
        plt.title('Durchschnittliche Wortarten-Verteilung (Gesamtes Korpus)')
        plt.ylabel('Anteil an allen Tokens')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'pos_distribution_bar.png'))
        plt.close()

    if 'article_len_dist' in plots_to_run or 'all' in plots_to_run:
        print("Generating Article Length Distribution (Histogram)...")
        # 7. Article Length Distribution (Histogram)
        plt.figure(figsize=(12, 6))
        # Log scale might be better as lengths can vary wildly
        # sns.histplot(data=df, x='as_tokens', label='Alltagssprache', kde=True, color='blue', alpha=0.5)
        # sns.histplot(data=df, x='ls_tokens', label='Leichte Sprache', kde=True, color='green', alpha=0.5)
        
        # Melting for easier plotting with hue
        length_df = df[['as_tokens', 'ls_tokens']].melt(var_name='Sprache', value_name='Token-Anzahl')
        length_df['Sprache'] = length_df['Sprache'].map({'as_tokens': 'Alltagssprache', 'ls_tokens': 'Leichte Sprache'})
        
        ax = sns.histplot(data=length_df, x='Token-Anzahl', hue='Sprache', kde=True, bins=50, element="step", common_norm=False)
        
        # Add SBERT limits for context
        plt.axvline(128, color='red', linestyle='--', label='SBERT Limit (128)')
        plt.axvline(512, color='orange', linestyle='--', label='SBERT Limit (512)')
        
        plt.title('Verteilung der Artikellängen (Token-Anzahl)')
        plt.xlabel('Anzahl Tokens')
        plt.ylabel('Anzahl Artikel')
        
        # Merge legends to include both histplot labels and SBERT limit lines
        h_lines, l_lines = ax.get_legend_handles_labels()
        if ax.legend_:
            try:
                h_sb = ax.legend_.legend_handles
            except AttributeError:
                h_sb = ax.legend_.legendHandles
            l_sb = [t.get_text() for t in ax.legend_.texts]
            ax.legend(h_sb + h_lines, l_sb + l_lines)
        else:
            plt.legend()
            
        plt.xlim(0, 3000) # Limit x-axis for better visibility of the bulk of articles
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'article_length_distribution.png'))
        plt.close()

    print(f"Visualizations updated and saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for dataset analysis.")
    parser.add_argument(
        '--plots', 
        nargs='+', 
        default=['all'],
        choices=[
            'all', 
            'jina_context', 
            'semantic_sim_8192', 
            'bidirectional_ner', 
            'token_ratio_sim', 
            'sim_hist', 
            'sent_len', 
            'pos_dist',
            'article_len_dist'
        ],
        help="List of plots to generate. 'all' generates everything."
    )
    args = parser.parse_args()
    
    create_visualizations(args.plots)
