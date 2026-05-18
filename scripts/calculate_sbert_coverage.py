import pandas as pd

# Konfiguration
INPUT_CSV = "results/information_loss_analysis.csv"
MAX_SBERT_LENGTH = 512

def calculate_coverage_stats():
    try:
        # CSV-Datei einlesen
        df = pd.read_csv(INPUT_CSV)
        total = len(df)
        
        print(f"=== SBERT Coverage Analyse (Limit: {MAX_SBERT_LENGTH} Tokens) ===")
        print(f"Gesamtanzahl Artikelpaare: {total}\n")
        
        for col, label in [('as_tokens', 'Alltagssprache (AS)'), ('ls_tokens', 'Leichte Sprache (LS)')]:
            full_coverage = sum(df[col] <= MAX_SBERT_LENGTH)
            truncated = total - full_coverage
            
            percent_full = (full_coverage / total) * 100
            percent_trunc = (truncated / total) * 100
            
            print(f"{label}:")
            print(f"  Vollständig erfasst: {full_coverage:4d} Artikel ({percent_full:5.1f}%)")
            print(f"  Abgeschnitten:       {truncated:4d} Artikel ({percent_trunc:5.1f}%)")
            
            # Zusätzliche Info: Wie viel fehlt im Durchschnitt bei den Abgeschnittenen?
            truncated_df = df[df[col] > MAX_SBERT_LENGTH]
            if not truncated_df.empty:
                avg_total = truncated_df[col].mean()
                print(f"  Ø Länge bei Kürzung: {avg_total:5.1f} Tokens")
            print("-" * 45)

    except FileNotFoundError:
        print(f"Fehler: Die Datei {INPUT_CSV} wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    calculate_coverage_stats()
