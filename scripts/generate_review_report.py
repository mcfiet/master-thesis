import pandas as pd
import os

# Paths
INPUT_CSV = "results/information_loss_analysis.csv"
OUTPUT_MD = "results/outlier_review.md"

def generate_report():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Define Outliers
    low_sim = df[df['semantic_similarity_512'] < 0.6].sort_values(by='semantic_similarity_512')
    high_sim = df[df['semantic_similarity_512'] > 0.98].sort_values(by='semantic_similarity_512', ascending=False)

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write("# Outlier Review Report\n\n")
        f.write(f"Dieses Dokument enthält Artikelpaare mit extremer semantischer Ähnlichkeit (< 0.6 oder > 0.98).\n")
        f.write(f"Ziel: Manuelle Prüfung auf Alignment-Fehler oder identische Texte.\n\n")

        f.write(f"## Statistik\n")
        f.write(f"- Niedrige Ähnlichkeit (< 0.6): {len(low_sim)} Artikel\n")
        f.write(f"- Hohe Ähnlichkeit (> 0.98): {len(high_sim)} Artikel\n\n")

        f.write("---")

        f.write(f"\n## 1. Niedrige Ähnlichkeit (< 0.6)\n")
        f.write("Häufige Ursachen: Alignment-Fehler, extreme Kürzung (nur Teaser), völlig andere Themen.\n\n")

        for i, row in low_sim.iterrows():
            f.write(f"### ID: {i} | Source: {row['source']} | Sim: {row['semantic_similarity_512']:.4f}\n")
            f.write(f"- **AS URL:** {row['as_url']}\n")
            f.write(f"- **LS URL:** {row['ls_url']}\n")
            f.write(f"- **Token Ratio:** {row['token_ratio']:.4f}\n\n")
            
            f.write("#### Alltagssprache (AS)\n")
            f.write(f"```text\n{row['as_text']}\n```\n\n")
            
            f.write("#### Leichte Sprache (LS)\n")
            f.write(f"```text\n{row['ls_text']}\n```\n\n")
            f.write("---\n\n")

        f.write(f"\n## 2. Hohe Ähnlichkeit (> 0.98)\n")
        f.write("Häufige Ursachen: Identische Texte (keine Übersetzung), nur Menü-Strukturen, Link-Listen.\n\n")

        for i, row in high_sim.iterrows():
            f.write(f"### ID: {i} | Source: {row['source']} | Sim: {row['semantic_similarity_512']:.4f}\n")
            f.write(f"- **AS URL:** {row['as_url']}\n")
            f.write(f"- **LS URL:** {row['ls_url']}\n")
            f.write(f"- **Token Ratio:** {row['token_ratio']:.4f}\n\n")
            
            f.write("#### Alltagssprache (AS)\n")
            f.write(f"```text\n{row['as_text']}\n```\n\n")
            
            f.write("#### Leichte Sprache (LS)\n")
            f.write(f"```text\n{row['ls_text']}\n```\n\n")
            f.write("---\n\n")

    print(f"Report erfolgreich generiert: {OUTPUT_MD}")

if __name__ == "__main__":
    generate_report()
