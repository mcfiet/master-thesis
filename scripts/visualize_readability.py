import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    input_file = "results/readability_analysis.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run measure_readability.py first.")
        return

    df = pd.read_csv(input_file)
    output_dir = "research/img/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setting the aesthetic style of the plots
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    print("Generating plots...")

    # Define the metrics and their characteristics
    metrics = [
        ('wiener', 'Wiener Sachtextformel', 'Required Grade Level', 'Lower is Easier'),
        ('flesch', 'Flesch Reading Ease', 'FRE Score', 'Higher is Easier'),
        ('lix', 'LIX Index', 'LIX Score', 'Lower is Easier')
    ]
    
    for key, title, ylabel, subtitle in metrics:
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        cols = [f'ls_{key}', f'as_{key}']
        plot_df = pd.melt(df[cols], var_name='Type', value_name='Value')
        plot_df['Type'] = plot_df['Type'].map({
            f'ls_{key}': 'Easy Language (LS)', 
            f'as_{key}': 'Standard Language (AS)'
        })
        
        # Create boxplot with violin overlay for better distribution visualization
        sns.violinplot(x='Type', y='Value', data=plot_df, inner="quart", palette='pastel')
        sns.stripplot(x='Type', y='Value', data=plot_df, color='black', alpha=0.05, size=2)
        
        plt.title(f'{title}\n({subtitle})')
        plt.ylabel(ylabel)
        plt.xlabel('')
        
        filename = f'readability_{key}_comparison.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - Saved {filename}")

    # 4. By-Source Analysis for Wiener Sachtextformel
    plt.figure(figsize=(14, 8))
    source_wstf = pd.melt(df, id_vars=['source'], value_vars=['ls_wiener', 'as_wiener'], 
                         var_name='Type', value_name='Grade')
    source_wstf['Type'] = source_wstf['Type'].map({'ls_wiener': 'LS', 'as_wiener': 'AS'})
    
    sns.boxplot(x='source', y='Grade', hue='Type', data=source_wstf)
    plt.title('Wiener Sachtextformel by Source')
    plt.xticks(rotation=45)
    plt.ylabel('Required Grade Level')
    plt.savefig(os.path.join(output_dir, 'readability_wiener_by_source.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  - Saved readability_wiener_by_source.png")

    # Summary Statistics
    summary = df[['ls_wiener', 'as_wiener', 'ls_flesch', 'as_flesch', 'ls_lix', 'as_lix']].describe()
    
    # Calculate means by source
    source_means = df.groupby('source')[['ls_wiener', 'as_wiener', 'ls_flesch', 'as_flesch']].mean()
    
    print("\nReadability Statistics Summary:")
    print(summary.loc[['mean', '50%', 'std']])
    
    # Save stats to markdown with better formatting and embedded images
    stats_file = "research/readability_summary.md"
    with open(stats_file, "w") as f:
        f.write("# Readability Analysis Summary\n\n")
        f.write("This report compares the readability of Standard Language (AS) and Easy Language (LS) texts in the final corpus.\n\n")
        
        f.write("## Key Findings\n")
        as_wstf_mean = summary.loc['mean', 'as_wiener']
        ls_wstf_mean = summary.loc['mean', 'ls_wiener']
        diff_wstf = as_wstf_mean - ls_wstf_mean
        
        f.write(f"- **Difficulty Reduction (WSTF):** On average, LS texts are **{diff_wstf:.2f} grade levels easier** than their AS counterparts according to the Wiener Sachtextformel.\n")
        f.write(f"- **Standard Language Average:** Grade level {as_wstf_mean:.2f}\n")
        f.write(f"- **Easy Language Average:** Grade level {ls_wstf_mean:.2f}\n\n")

        f.write("## Visualizations\n\n")
        f.write("### Wiener Sachtextformel Comparison\n")
        f.write("![Wiener Sachtextformel](img/analysis/readability_wiener_comparison.png)\n\n")
        
        f.write("### Flesch Reading Ease Comparison\n")
        f.write("![Flesch Reading Ease](img/analysis/readability_flesch_comparison.png)\n\n")

        f.write("### LIX Index Comparison\n")
        f.write("![LIX Index](img/analysis/readability_lix_comparison.png)\n\n")

        f.write("### Analysis by Source\n")
        f.write("![Wiener by Source](img/analysis/readability_wiener_by_source.png)\n\n")

        f.write("## Detailed Statistics\n\n")
        f.write("### Descriptive Statistics\n\n")
        # Creating a proper markdown table
        f.write("| Metric | " + " | ".join(summary.columns) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(summary.columns)) + "|\n")
        for idx, row in summary.iterrows():
            f.write(f"| {idx} | " + " | ".join([f"{val:.2f}" for val in row]) + " |\n")
        
        f.write("\n\n### Means by Source\n\n")
        f.write("| Source | " + " | ".join(source_means.columns) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(source_means.columns)) + "|\n")
        for idx, row in source_means.iterrows():
            f.write(f"| {idx} | " + " | ".join([f"{val:.2f}" for val in row]) + " |\n")

    print(f"\nSummary report updated and saved to {stats_file}")

if __name__ == "__main__":
    main()
