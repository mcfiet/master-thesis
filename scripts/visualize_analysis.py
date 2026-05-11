import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_CSV = "results/information_loss_analysis.csv"
OUTPUT_DIR = "research/img/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_visualizations():
    df = pd.read_csv(INPUT_CSV)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Token Ratio vs. NER Recall
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='token_ratio', y='ner_recall', hue='source', alpha=0.5)
    plt.title('Token Ratio vs. NER Recall (per Article Pair)')
    plt.xlabel('Token Ratio (LS / AS)')
    plt.ylabel('NER Recall (Fakten-Erhalt)')
    plt.axvline(1.0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'token_ratio_vs_ner_recall.png'))
    plt.close()
    
    # 2. Semantic Similarity by Source
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='source', y='semantic_similarity')
    plt.xticks(rotation=45)
    plt.title('Semantische Ähnlichkeit nach Quelle')
    plt.ylabel('Cosine Similarity (SBERT)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'semantic_similarity_by_source.png'))
    plt.close()
    
    # 3. POS Ratio Changes (Mean)
    pos_cols_as = ['as_adj_ratio', 'as_noun_ratio', 'as_verb_ratio', 'as_conj_ratio']
    pos_cols_ls = ['ls_adj_ratio', 'ls_noun_ratio', 'ls_verb_ratio', 'ls_conj_ratio']
    
    mean_pos = df[pos_cols_as + pos_cols_ls].mean()
    pos_data = {
        'Metric': ['Adjectives', 'Nouns', 'Verbs', 'Conjunctions'],
        'AS': [mean_pos['as_adj_ratio'], mean_pos['as_noun_ratio'], mean_pos['as_verb_ratio'], mean_pos['as_conj_ratio']],
        'LS': [mean_pos['ls_adj_ratio'], mean_pos['ls_noun_ratio'], mean_pos['ls_verb_ratio'], mean_pos['ls_conj_ratio']]
    }
    pos_df = pd.DataFrame(pos_data).melt(id_vars='Metric', var_name='Language', value_name='Ratio')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=pos_df, x='Metric', y='Ratio', hue='Language')
    plt.title('Durchschnittliche Wortarten-Verteilung (AS vs. LS)')
    plt.ylabel('Anteil am Gesamttext')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pos_distribution_comparison.png'))
    plt.close()

    # 5. Sentence Length Comparison
    if 'as_avg_sent_len' in df.columns:
        plt.figure(figsize=(10, 6))
        sent_len_data = {
            'Language': ['AS', 'LS'],
            'Avg Sentence Length': [df['as_avg_sent_len'].mean(), df['ls_avg_sent_len'].mean()]
        }
        sent_df = pd.DataFrame(sent_len_data)
        sns.barplot(data=sent_df, x='Language', y='Avg Sentence Length')
        plt.title('Durchschnittliche Satzlänge (Tokens)')
        plt.ylabel('Tokens pro Satz')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sentence_length_comparison.png'))
        plt.close()

if __name__ == "__main__":
    create_visualizations()
