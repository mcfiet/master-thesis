import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def visualize_ttr(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    
    # Drop rows with None/NaN metrics
    df = df.dropna(subset=['as_mattr', 'ls_mattr', 'as_ttr', 'ls_ttr'])
    
    # Create output dir if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. MATTR Comparison (Längenneutral) ---
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting (melt for seaborn)
    df_melted = df.melt(id_vars=['source'], value_vars=['as_mattr', 'ls_mattr'], 
                        var_name='Type', value_name='MATTR')
    df_melted['Type'] = df_melted['Type'].map({'as_mattr': 'Standard (AS)', 'ls_mattr': 'Leicht (LS)'})
    
    sns.boxplot(data=df_melted, x='source', y='MATTR', hue='Type')
    plt.xticks(rotation=45, ha='right')
    plt.title('Lexikalische Vielfalt (MATTR, Window=50) nach Quelle')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttr_mattr_comparison.png'), dpi=300)
    plt.close()
    
    # --- 2. Scatterplot: TTR vs Length (Längeneffekt visualisieren) ---
    plt.figure(figsize=(10, 7))
    
    # Scatter for AS
    sns.regplot(data=df, x='as_tokens', y='as_ttr', scatter_kws={'alpha':0.3}, 
                label='Standard (AS)', color='blue', x_ci=None)
    # Scatter for LS
    sns.regplot(data=df, x='ls_tokens', y='ls_ttr', scatter_kws={'alpha':0.3}, 
                label='Leicht (LS)', color='orange', x_ci=None)
    
    plt.xscale('log') # Log scale since token counts vary wildly
    plt.xlabel('Anzahl Tokens (log-Skala)')
    plt.ylabel('Type-Token-Ratio (TTR)')
    plt.title('Abhängigkeit der TTR von der Textlänge (Lemmatisiert)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttr_vs_length_scatter.png'), dpi=300)
    plt.close()

    # --- 3. Summary Statistics ---
    summary = df.groupby('source').agg({
        'as_mattr': 'mean',
        'ls_mattr': 'mean',
        'as_ttr': 'mean',
        'ls_ttr': 'mean'
    }).reset_index()
    
    print("\nDurchschnittliche MATTR (lexikalische Vielfalt) pro Quelle:")
    print(summary[['source', 'as_mattr', 'ls_mattr']])
    
    overall_as = df['as_mattr'].mean()
    overall_ls = df['ls_mattr'].mean()
    print(f"\nGesamtdurchschnitt MATTR:")
    print(f"AS: {overall_as:.3f}")
    print(f"LS: {overall_ls:.3f}")
    print(f"Reduktion: {((overall_as - overall_ls) / overall_as * 100):.1f}%")

if __name__ == "__main__":
    CSV_PATH = "results/ttr_analysis.csv"
    OUTPUT_DIR = "research/img/analysis"
    visualize_ttr(CSV_PATH, OUTPUT_DIR)
