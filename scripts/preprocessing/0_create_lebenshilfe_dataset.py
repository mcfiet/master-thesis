import os
import json
import re
from docx import Document
from striprtf.striprtf import rtf_to_text
from odf import text, teletype
from odf.opendocument import load

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_rtf(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()
    return rtf_to_text(content)

def extract_text_from_odt(file_path):
    textdoc = load(file_path)
    all_paras = textdoc.getElementsByType(text.P)
    return "\n".join([teletype.extractText(p) for p in all_paras])

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.rtf':
            return extract_text_from_rtf(file_path)
        elif ext == '.odt':
            return extract_text_from_odt(file_path)
        elif ext == '.doc':
            # Skip for now as it needs antiword
            return ""
        else:
            return ""
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")
        return ""

base_dir = 'data/lebenshilfe/texts_lebenshilfe'
ls_dir = os.path.join(base_dir, 'ls')
as_dir = os.path.join(base_dir, 'as')

def normalize(name):
    name = os.path.splitext(name)[0]
    name = re.sub(r'ILS\d*', '', name)
    name = re.sub(r'AD\d*', '', name)
    name = re.sub(r'DS\d*', '', name)
    name = re.sub(r'Prüfer', '', name)
    name = re.sub(r'geprüft', '', name)
    name = re.sub(r'testgelesen', '', name)
    name = re.sub(r'prüfen', '', name)
    name = re.sub(r'Leichte Sprache', '', name, flags=re.IGNORECASE)
    name = re.sub(r'leichteSprache', '', name, flags=re.IGNORECASE)
    name = re.sub(r'ILS', '', name)
    name = name.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'^\d+\s*', '', name)
    name = ' '.join(name.split())
    return name.lower()

ls_files = [f for f in os.listdir(ls_dir) if os.path.isfile(os.path.join(ls_dir, f))]
as_files = [f for f in os.listdir(as_dir) if os.path.isfile(os.path.join(as_dir, f))]

as_normalized = {normalize(f): f for f in as_files}

manual_matches = [
    ("ILS_CAU_Geologiemuseum AD002 Prüfer Mail.docx", "20241106-PM-Aktionstag-CAU und StK-an PS_neu.docx"),
    ("ILS StK Evaluationsbericht LAP intern AD002 Prüfer.docx", "20241108-ILS StK Evaluationsbericht LAP-mit Anmerkungen.docx"),
    ("ILS LB MmB Positionspapier Arbeit 004.docx", "Positionspapier Originaltext.docx"),
    ("ILS07 Einwilligung KS DS001 Prüfer.docx", "07 BwH KS-Einwilligung-2024-10-16-19-16-06.rtf"),
    ("ILS08 BwH Entbindung Schweigepflicht AD002 prüfen.docx", "08 BwH Schweigepflichtentbindung - allgemein.odt"),
    ("ILS09 BwH Einverständniserklärung AD001 prüfen.docx", "09 BwH Einverständnis Bericht § 160 StGB (Brief der Klient_in).odt"),
    ("ILS06 BwH Info KSKS AD001 prüfen.docx", "06 BwH KSKS - Handreichung-2024-10-16-16-47-39.docx"),
    ("ILS_IBA_Fischbeker_Reethen_AD002.docx", "IBA Hamburg_Fischbeker Reethen_Web_Leichte Sprache.docx"),
    ("ILS FRAGEN Podium - Parlamentarischer Abend - AD001.docx", "FRAGEN Podium - Parlamentarischer Abend - Stand 1. August .docx"),
    ("ILS KIWA_Bedarfsumfrage AD002 Prüfer.docx", "Bedarfsumfrage KIWA.docx"),
    ("ILS Pinneberg Satzung Behindertenbeirat 001 AD geprüft.docx", "Satzung der Stadt Pinneberg für den Behindertenbeirat (Endfassung nach Ratsbeschluss ink. Ergänzungsvorschlägen).docx"),
    ("ILS_Impressum ZuMiNET LS.docx", "Impressum von ZuMiNET.docx"),
    ("ILS Texte MiPi Tablet-Führerschein.docx", "Anleitung zum MiPi Tablet.docx"),
    ("ILS Elternbrief Kita AD002 Prüfer.docx", "Brieftext roh.docx"),
    ("ILS GVOBl-leichte-sprache AD002_geprüft.docx", "GVOBl-leichte-sprache_Entwurf.docx"),
    ("ILS03 GH Infoblatt Beschuldigte AD001 geprüft.docx", "03 GH Informationsblatt für Beschuldigte.docx"),
    ("ILS04 GH Infoblatt Geschädigte AD001 geprüft.docx", "04 GH Informationsblatt für Geschädigte.docx"),
    ("ILS05 GH Infoblatt TOA AD001 geprüft.docx", "05 GH TOA Merkblatt.docx"),
    ("ILS LHW Plön BV Hinweisgeber AD002 testgelesen.docx", "20240228 überarbeitete Version BV Hinweisgeberschutz.docx"),
    ("ILS StK_PI_Fonds_Barrierefreiheit AD002.docx", "20240730- Entwurf PI Fonds für Barrierefreiheit.odt"),
    ("ILS Hausordungung geschlossener Männervollzug HL AD002 Prüfer.docx", "Hausordungung geschlossener Männervollzug.docx"),
    ("ILS Hausordnung SL E-Vollzug AD001 Prüfer.docx", "Hausordnung Haus 11 E-Vollzug.docx"),
    ("ILS IIB MD Textteil 1 003 geprüft.docx", "IIB MD Vorlage 1 231117_zu_übersetzender_Teil_1.docx"),
    ("ILS IIB MD Textteil2 002.docx", "240129_Übersetzung_Teil2.docx"),
    ("ILS Hausordungung geschlossener Frauenvollzug HL AD001.docx", "Hausordnung Frauenvollzug Stand 25.09.24.doc"),
    ("ILS Infoschreiben MItarbeiter 003.docx", "Infoschreiben an Mitarbeiter Vorlage.docx"),
    ("ILS Hausordnung der JAA Moltsfelde DS002 Prüfer.docx", "Hausordnung der JAA Moltsfelde NEU 12.07.2024.docx"),
    ("ILS 2025-01 Bericht 15 Jahre UN-BRK 002AD geprüft.docx", "2024-11 Bericht 15 Jahre UN-BRK.docx"),
    ("ILS IIB_MD-Stendal_Textteil_5 001 geprüft.docx", "240507_Übersetzung Teil 5_Forschung.docx"),
    ("ILS Hausordungung offener Vollzug HL AD002 Prüfer.docx", "Hausordnung Offener Vollzug Stand 09.02.2023.docx"),
    ("ILS Hausordnung MLF DS002 Prüfer.docx", "Hausordnung MLF NEU.doc"),
    ("ILS_Verkündungsportal_LeichteSprache_AD_004.docx", "240705_LeichteSprache_verkündungsportal_AA.docx"),
    ("ILS Anlage Resozialisierung Hausordungung HL AD001.docx", "Anlage zur Hausordnung.docx"),
    ("01 ILS_Flyer_psychosoziale_Prozessbegleitung_hoch AD001 Prüfer.docx", "Psychosoziale Begleitung ohne LGBezirk (wg Formate).docx"),
]

dataset = []
matched_ls = []
matched_as = []

for ls_f, as_f in manual_matches:
    if ls_f in ls_files and as_f in as_files:
        ls_text = extract_text(os.path.join(ls_dir, ls_f))
        as_text = extract_text(os.path.join(as_dir, as_f))
        if ls_text.strip() and as_text.strip():
            dataset.append({
                "source": "lebenshilfe",
                "ls_filename": ls_f,
                "as_filename": as_f,
                "ls_text": ls_text,
                "as_text": as_text
            })
            matched_ls.append(ls_f)
            matched_as.append(as_f)

for ls_f in ls_files:
    if ls_f in matched_ls: continue
    norm_ls = normalize(ls_f)
    if norm_ls in as_normalized:
        as_f = as_normalized[norm_ls]
        if as_f in matched_as: continue
        ls_text = extract_text(os.path.join(ls_dir, ls_f))
        as_text = extract_text(os.path.join(as_dir, as_f))
        if ls_text.strip() and as_text.strip():
            dataset.append({
                "source": "lebenshilfe",
                "ls_filename": ls_f,
                "as_filename": as_f,
                "ls_text": ls_text,
                "as_text": as_text
            })

output_file = 'data/lebenshilfe/lebenshilfe_dataset.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset saved to {output_file} with {len(dataset)} articles.")
