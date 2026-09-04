import os
import sys
import json
import re
import argparse
from typing import List, Dict, Any, Tuple
import pandas as pd

SYNTHETIC_BASE_TEXTS = [
    {
        "id": "text_01_arztbesuch",
        "title": "Der Besuch beim Arzt",
        "domain": "Gesundheit",
        "sentences": [
            ("Ein Mensch geht zu dem Arzt.", "Ein Mensch geht zu dem Arzt.", "Ein Mensch wird von dem Arzt untersucht.", "Ein Mensch geht des Arztes wegen.", "Eine Person geht zu dem Arzt."),
            ("Der Arzt untersucht den Patienten genau.", "Der Arzt untersuchte den Patienten genau.", "Der Patient wird von dem Arzt genau untersucht.", "Der Arzt untersucht den Patienten des Hauses.", "Der Arzt prüft den Patienten genau."),
            ("Der Patient hat starke Schmerzen im Rücken.", "Der Patient hätte starke Schmerzen im Rücken.", "Starke Schmerzen im Rücken werden von dem Patienten ertragen.", "Der Patient hat starke Schmerzen des Rückens.", "Der Patient hat große Schmerzen im Rücken."),
            ("Die Assistentin misst den Blutdruck sofort.", "Die Assistentin messe den Blutdruck sofort.", "Der Blutdruck wird von der Assistentin sofort gemessen.", "Die Assistentin des Chefs misst den Blutdruck.", "Die Helferin misst den Blutdruck sofort."),
            ("Der Arzt erklärt die richtige Behandlung.", "Der Arzt erklärte die richtige Behandlung.", "Die richtige Behandlung wird von dem Arzt erklärt.", "Der Arzt erklärt die Behandlung des Patienten.", "Der Arzt schildert die richtige Behandlung."),
            ("Die Apotheke liefert die neuen Medikamente.", "Die Apotheke lieferte die neuen Medikamente.", "Die neuen Medikamente werden von der Apotheke geliefert.", "Die Apotheke des Ortes liefert die Medikamente.", "Die Apotheke bringt die neuen Medikamente."),
            ("Der Patient nimmt die Tabletten mit Wasser.", "Der Patient nähme die Tabletten mit Wasser.", "Die Tabletten werden von dem Patienten mit Wasser genommen.", "Der Patient nimmt die Tabletten des Arztes.", "Der Patient schluckt die Tabletten mit Wasser."),
            ("Wegen dem schnellen Handeln hilft die Medizin.", "Wegen dem schnellen Handeln hülfe die Medizin.", "Schnelle Hilfe wird von der Medizin geleistet.", "Wegen des schnellen Handelns hilft die Medizin.", "Wegen dem raschen Handeln hilft die Medizin."),
            ("Der Mann kann bald wieder schmerzfrei arbeiten.", "Der Mann könnte bald wieder schmerzfrei arbeiten.", "Die Arbeit wird von dem Mann bald wieder aufgenommen.", "Der Mann kann wegen des Erfolgs schmerzfrei arbeiten.", "Der Mann vermag bald wieder schmerzfrei arbeiten."),
            ("Das Büro von dem Arzt ist modern.", "Das Büro von dem Arzt wäre modern.", "Das moderne Büro wird von dem Arzt betrieben.", "Das Büro des Arztes ist modern.", "Der Raum von dem Arzt ist modern.")
        ]
    },
    {
        "id": "text_02_waehlen_politik",
        "title": "Wählen in Deutschland",
        "domain": "Politik",
        "sentences": [
            ("Die Bürger wählen das neue Parlament.", "Die Bürger wählten das neue Parlament.", "Das neue Parlament wird von den Bürgern gewählt.", "Die Bürger des Landes wählen das Parlament.", "Die Menschen wählen das neue Parlament."),
            ("Die Stadt schickt den Brief mit der Wahlbenachrichtigung.", "Die Stadt schickte den Brief mit der Wahlbenachrichtigung.", "Der Brief wird von der Stadt verschickt.", "Die Stadt schickt den Brief des Wahlamtes.", "Die Stadt sendet den Brief mit der Benachrichtigung."),
            ("Jeder Wähler hat genau eine Stimme.", "Jeder Wähler hätte genau eine Stimme.", "Eine Stimme wird von jedem Wähler abgegeben.", "Jeder Wähler hat das Recht der Stimme.", "Jeder Wähler besitzt genau eine Stimme."),
            ("Der Helfer prüft den Ausweis am Eingang.", "Der Helfer prüfte den Ausweis am Eingang.", "Der Ausweis wird von dem Helfer am Eingang geprüft.", "Der Helfer prüft den Ausweis des Bürgers.", "Der Helfer kontrolliert den Ausweis am Eingang."),
            ("Der Wähler setzt das Kreuz auf dem Stimmzettel.", "Der Wähler setzte das Kreuz auf dem Stimmzettel.", "Das Kreuz wird von dem Wähler gesetzt.", "Der Wähler setzt das Kreuz des Zettels.", "Der Wähler macht das Kreuz auf dem Stimmzettel."),
            ("Die Wahlhelfer zählen alle Stimmen am Abend.", "Die Wahlhelfer zählten alle Stimmen am Abend.", "Alle Stimmen werden von den Wahlhelfern gezählt.", "Die Wahlhelfer des Bezirks zählen alle Stimmen.", "Die Wahlhelfer erfassen alle Stimmen am Abend."),
            ("Das Wahlergebnis bestimmt die Zukunft von dem Land.", "Das Wahlergebnis bestimmte die Zukunft von dem Land.", "Die Zukunft wird von dem Wahlergebnis bestimmt.", "Das Wahlergebnis bestimmt die Zukunft des Landes.", "Das Resultat bestimmt die Zukunft von dem Land."),
            ("Alle Parteien müssen die Regeln genau beachten.", "Alle Parteien müssten die Regeln genau beachten.", "Die Regeln werden von allen Parteien beachtet.", "Alle Parteien müssen die Regeln des Gesetzes beachten.", "Alle Gruppen müssen die Regeln genau beachten."),
            ("Wegen dem fairen Ablauf vertrauen die Menschen.", "Wegen dem fairen Ablauf vertrauten die Menschen.", "Großes Vertrauen wird von den Menschen gezeigt.", "Wegen des fairen Ablaufs vertrauen die Menschen.", "Wegen dem ehrlichen Ablauf vertrauen die Menschen."),
            ("Die Wahl gibt den Menschen viel Kraft.", "Die Wahl gäbe den Menschen viel Kraft.", "Viel Kraft wird von der Wahl gegeben.", "Die Wahl gibt die Kraft des Volkes.", "Die Wahl schenkt den Menschen viel Kraft.")
        ]
    },
    {
        "id": "text_03_arbeit_werkstatt",
        "title": "Die Arbeit in der Werkstatt",
        "domain": "Arbeit",
        "sentences": [
            ("Viele Menschen arbeiten in der Werkstatt.", "Viele Menschen arbeiteten in der Werkstatt.", "Gute Arbeit wird von vielen Menschen geleistet.", "Viele Menschen arbeiten wegen des Lohns.", "Viele Personen arbeiten in der Werkstatt."),
            ("Der Gruppenleiter erklärt die tägliche Aufgabe.", "Der Gruppenleiter erklärte die tägliche Aufgabe.", "Die tägliche Aufgabe wird von dem Gruppenleiter erklärt.", "Der Gruppenleiter der Gruppe erklärt die Aufgabe.", "Der Leiter schildert die tägliche Aufgabe."),
            ("Die Mitarbeiter bauen schöne Möbel aus Holz.", "Die Mitarbeiter bauten schöne Möbel aus Holz.", "Schöne Möbel aus Holz werden von den Mitarbeitern gebaut.", "Die Mitarbeiter bauen die Möbel des Kunden.", "Die Mitarbeiter fertigen schöne Möbel aus Holz."),
            ("Der Handwerker schleift das Brett sehr sauber.", "Der Handwerker schliffe das Brett sehr sauber.", "Das Brett wird von dem Handwerker sehr sauber geschliffen.", "Der Handwerker schleift das Brett des Tisches.", "Der Handwerker glättet das Brett sehr sauber."),
            ("Die Werkstatt liefert die Ware an Kunden.", "Die Werkstatt lieferte die Ware an Kunden.", "Die Ware wird von der Werkstatt an Kunden geliefert.", "Die Werkstatt liefert die Ware des Betriebs.", "Der Betrieb bringt die Ware an Kunden."),
            ("Jeder Arbeiter hat feste Pausen am Tag.", "Jeder Arbeiter hätte feste Pausen am Tag.", "Feste Pausen werden von jedem Arbeiter eingehalten.", "Jeder Arbeiter hat das Recht des Ausruhens.", "Jeder Arbeiter besitzt feste Pausen am Tag."),
            ("Wegen dem guten Schutz passiert kein Unfall.", "Wegen dem guten Schutz passierte kein Unfall.", "Guter Schutz wird von allen getragen.", "Wegen des guten Schutzes passiert kein Unfall.", "Wegen der guten Vorsicht passiert kein Unfall."),
            ("Das Werkzeug von dem Meister ist neu.", "Das Werkzeug von dem Meister wäre neu.", "Neues Werkzeug wird von dem Meister benutzt.", "Das Werkzeug des Meisters ist neu.", "Das Gerät von dem Chef ist neu."),
            ("Die Kollegen helfen einander bei schwerer Arbeit.", "Die Kollegen hülfen einander bei schwerer Arbeit.", "Hilfe wird von den Kollegen geleistet.", "Die Kollegen helfen während des Tages.", "Die Mitarbeiter stützen einander bei schwerer Arbeit."),
            ("Die gemeinsame Arbeit macht allen große Freude.", "Die gemeinsame Arbeit machte allen große Freude.", "Große Freude wird von allen empfunden.", "Die Arbeit bringt die Freude des Teams.", "Die gemeinsame Arbeit bringt allen großen Spaß.")
        ]
    },
    {
        "id": "text_04_bus_und_bahn",
        "title": "Fahren mit Bus und Bahn",
        "domain": "Mobilität",
        "sentences": [
            ("Der Fahrgast kauft eine Fahrkarte am Schalter.", "Der Fahrgast kaufte eine Fahrkarte am Schalter.", "Eine Fahrkarte wird von dem Fahrgast am Schalter gekauft.", "Der Fahrgast kauft die Fahrkarte des Verbunds.", "Der Reisende kauft ein Ticket am Schalter."),
            ("Der Automat zeigt den passenden Preis an.", "Der Automat zeigte den passenden Preis an.", "Der passende Preis wird von dem Automaten angezeigt.", "Der Automat zeigt den Preis des Tickets an.", "Das Gerät weist den passenden Preis aus."),
            ("Der Fahrer steuert den Bus durch die Stadt.", "Der Fahrer steuerte den Bus durch die Stadt.", "Der Bus wird von dem Fahrer durch die Stadt gesteuert.", "Der Fahrer steuert den Bus der Linie.", "Der Fahrer lenkt den Bus durch die Stadt."),
            ("Die Bahn hält an jeder Haltestelle an.", "Die Bahn hielte an jeder Haltestelle an.", "Jeder Halt wird von der Bahn bedient.", "Die Bahn hält an der Haltestelle des Ortes.", "Der Zug stoppt an jeder Station an."),
            ("Die Fahrgäste steigen zügig in den Zug.", "Die Fahrgäste stiegen zügig in den Zug.", "Der Einstieg wird von den Fahrgästen zügig gemacht.", "Die Fahrgäste steigen in den Zug der Bahn.", "Die Reisenden steigen rasch in den Zug."),
            ("Der Kontrolleur prüft die Fahrkarten im Abteil.", "Der Kontrolleur prüfte die Fahrkarten im Abteil.", "Die Fahrkarten werden von dem Kontrolleur geprüft.", "Der Kontrolleur prüft die Karten des Zuges.", "Der Prüfer kontrolliert die Tickets im Abteil."),
            ("Ein Schild informiert die Menschen über Verspätung.", "Ein Schild informierte die Menschen über Verspätung.", "Die Menschen werden von dem Schild informiert.", "Ein Schild informiert wegen des Schadens.", "Ein Aushang unterrichtet die Personen über Verspätung."),
            ("Wegen dem schlechten Wetter fährt der Bus langsam.", "Wegen dem schlechten Wetter führe der Bus langsam.", "Langsames Fahren wird von dem Fahrer gewählt.", "Wegen des schlechten Wetters fährt der Bus langsam.", "Wegen dem eisigen Wetter fährt der Bus langsam."),
            ("Der Ausweis von der Begleitperson gilt überall.", "Der Ausweis von der Begleitperson gälte überall.", "Der Ausweis wird von der Begleitperson vorgezeigt.", "Der Ausweis der Begleitperson gilt überall.", "Das Dokument von der Begleitperson gilt überall."),
            ("Alle Menschen erreichen ihr Ziel sicher.", "Alle Menschen erreichten ihr Ziel sicher.", "Das Ziel wird von allen Menschen sicher erreicht.", "Alle Menschen erreichen das Ziel des Weges.", "Alle Personen kommen an ihrem Ziel sicher an.")
        ]
    },
    {
        "id": "text_05_wohnen_alltag",
        "title": "Selbstbestimmt Wohnen",
        "domain": "Wohnen",
        "sentences": [
            ("Frau Müller bewohnt eine eigene Wohnung.", "Frau Müller bewohnte eine eigene Wohnung.", "Eine eigene Wohnung wird von Frau Müller bewohnt.", "Frau Müller bewohnt die Wohnung des Hauses.", "Frau Müller nutzt ein eigenes Heim."),
            ("Der Vermieter übergibt den Schlüssel für das Haus.", "Der Vermieter übergab den Schlüssel für das Haus.", "Der Schlüssel wird von dem Vermieter übergeben.", "Der Vermieter übergibt den Schlüssel des Hauses.", "Der Eigentümer reicht den Schlüssel für das Haus."),
            ("Die Bewohnerin putzt die Küche jede Woche.", "Die Bewohnerin putzte die Küche jede Woche.", "Die Küche wird von der Bewohnerin geputzt.", "Die Bewohnerin putzt die Küche des Heims.", "Die Mieterin säubert die Küche jede Woche."),
            ("Die Assistentin unterstützt Frau Müller beim Kochen.", "Die Assistentin unterstützte Frau Müller beim Kochen.", "Frau Müller wird von der Assistentin unterstützt.", "Die Assistentin hilft während des Kochens.", "Die Betreuerin stützt Frau Müller beim Kochen."),
            ("Der Nachbar bringt die Post aus dem Kasten.", "Der Nachbar brachte die Post aus dem Kasten.", "Die Post wird von dem Nachbarn gebracht.", "Der Nachbar bringt die Post des Tages.", "Der Anwohner holt die Briefe aus dem Kasten."),
            ("Frau Müller bezahlt die Miete pünktlich.", "Frau Müller bezahlte die Miete pünktlich.", "Die Miete wird von Frau Müller pünktlich bezahlt.", "Frau Müller bezahlt die Miete des Monats.", "Frau Müller überweist die Miete pünktlich."),
            ("Wegen dem Aufzug kann sie barrierefrei wohnen.", "Wegen dem Aufzug könnte sie barrierefrei wohnen.", "Die Wohnung wird von ihr barrierefrei genutzt.", "Wegen des Aufzugs kann sie barrierefrei wohnen.", "Dank dem Fahrstuhl kann sie barrierefrei wohnen."),
            ("Der Garten von dem Haus bietet viel Ruhe.", "Der Garten von dem Haus böte viel Ruhe.", "Viel Ruhe wird von dem Garten geboten.", "Der Garten des Hauses bietet viel Ruhe.", "Die Grünfläche von dem Gebäude schenkt viel Ruhe."),
            ("Freunde besuchen die Wohnung am Wochenende.", "Freunde besuchten die Wohnung am Wochenende.", "Die Wohnung wird von Freunden besucht.", "Freunde besuchen die Wohnung der Frau.", "Bekannte kommen in die Wohnung am Wochenende."),
            ("Ein eigenes Zuhause gibt große Sicherheit.", "Ein eigenes Zuhause gäbe große Sicherheit.", "Große Sicherheit wird von dem Zuhause gegeben.", "Das Heim gibt die Sicherheit des Lebens.", "Ein eigenes Quartier stiftet große Sicherheit.")
        ]
    }
]

def count_tokens(text: str) -> int:
    return len(re.findall(r"\w+", text))

def count_vowels(text: str) -> int:
    vowels = "aeiouyäöüAEIOUYÄÖÜ"
    return sum(1 for c in text if c in vowels)

def generate_benchmark():
    records = []
    for item in SYNTHETIC_BASE_TEXTS:
        t_id = item["id"]
        title = item["title"]
        domain = item["domain"]
        s_tuples = item["sentences"]
        
        base_sentences = [t[0] for t in s_tuples]
        base_text = " ".join(base_sentences)
        
        subj_sentences = [t[1] for t in s_tuples]
        subj_text = " ".join(subj_sentences)
        
        pass_sentences = [t[2] for t in s_tuples]
        pass_text = " ".join(pass_sentences)
        
        gen_sentences = [t[3] for t in s_tuples]
        gen_text = " ".join(gen_sentences)
        
        comb_sentences = []
        for orig, subj, pas, gen, syn in s_tuples:
            c = pas.replace("wird von", "würde von").replace("werden von", "würden von")
            c = c.replace("von dem Haus", "des Hauses").replace("von dem Arzt", "des Arztes")
            comb_sentences.append(c)
        comb_text = " ".join(comb_sentences)
        
        ctrl_sentences = [t[4] for t in s_tuples]
        ctrl_text = " ".join(ctrl_sentences)
        
        variants = {
            "base_ls": base_text,
            "subjunctive_100": subj_text,
            "passive_100": pass_text,
            "genitive_100": gen_text,
            "combined_all": comb_text,
            "control_synonyms": ctrl_text,
        }
        
        base_tok = count_tokens(base_text)
        base_syl = count_vowels(base_text)
        
        for v_name, v_text in variants.items():
            tok = count_tokens(v_text)
            syl = count_vowels(v_text)
            records.append({
                "text_id": t_id,
                "title": title,
                "domain": domain,
                "variant": v_name,
                "text": v_text,
                "token_count": tok,
                "syllable_count": syl,
                "token_delta_vs_base": tok - base_tok,
                "syllable_delta_vs_base": syl - base_syl,
                "is_base": (v_name == "base_ls")
            })
    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_json", default="data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.json")
    parser.add_argument("--output_csv", default="data/experiments/rule_sensitivity/synthetic_rule_benchmark_256.csv")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    data = generate_benchmark()
    print(f"Generiert: {len(data)} synthetische Textvarianten.")
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Gespeichert: {args.output_json}")
    
    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)
    print(f"Gespeichert: {args.output_csv}")

if __name__ == "__main__":
    main()
