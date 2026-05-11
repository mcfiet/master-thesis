import pandas as pd

# Konfiguration
INPUT_CSV = "results/information_loss_analysis.csv"

def print_information_loss_statistics():
    try:
        # CSV-Datei einlesen
        df = pd.read_csv(INPUT_CSV)
        
        print("=== Information Loss: Token-Statistik ===")
        print(f"Gesamtanzahl analysierter Artikelpaare: {len(df)}\n")
        
        # describe() berechnet automatisch count, mean, std, min, 25%, 50%, 75%, max
        stats = df[['as_tokens', 'ls_tokens']].describe()
        
        # Umbenennen der Spalten für schönere Ausgabe
        stats.columns = ['Alltagssprache (AS)', 'Leichte Sprache (LS)']
        
        # Ausgabe formatieren (nur 2 Nachkommastellen)
        print(stats.round(2).to_string())
        
        print("\n--- Interpretation ---")
        print("Median (50%): Genau die Hälfte aller Artikel ist kürzer, die andere Hälfte länger als dieser Wert.")
        print("Max: Der längste Artikel im Korpus.")
        print("Achtung: SBERT verarbeitet standardmäßig nur die ersten 128 Tokens. Liegt der Median deutlich darüber, wird ein Großteil der Texte bei der semantischen Ähnlichkeitsmessung ignoriert.")
        
    except FileNotFoundError:
        print(f"Fehler: Die Datei {INPUT_CSV} wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    print_information_loss_statistics()
