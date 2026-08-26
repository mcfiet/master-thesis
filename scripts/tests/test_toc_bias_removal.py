#!/usr/bin/env python3
"""
scripts/tests/test_toc_bias_removal.py

Unit- und Regressionstests für die TOC-Bereinigung und die Erhaltung des
Frage-Antwort-Prinzips im Korpus für Leichte Sprache.
"""

import os
import sys
import json
import glob
import re
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_collection')))
import cleaner

class TestTOCBiasRemoval(unittest.TestCase):

    def test_single_question_unchanged(self):
        """Testet, dass eine legitime einzelne Überschriftenfrage vor einer Antwort erhalten bleibt."""
        text = "Was ist Leichte Sprache? Leichte Sprache ist eine einfache Form von Deutsch."
        cleaned = cleaner.remove_toc_question_streaks(text)
        self.assertEqual(cleaned, text)

    def test_no_question_unchanged(self):
        """Testet, dass Texte ohne Fragen am Anfang unverändert bleiben."""
        text = "Seit dem 4. Mai gibt es neue Parkzonen. Die Parkzonen sind in Hannover."
        cleaned = cleaner.remove_toc_question_streaks(text)
        self.assertEqual(cleaned, text)

    def test_five_leading_questions_strips_four(self):
        """Testet, dass bei 5 Fragen am Textanfang die ersten 4 (TOC) entfernt werden und die 5. bleibt."""
        text = (
            "Was ist Weitsichtigkeit? "
            "Was sind die Ursachen von Weitsichtigkeit? "
            "Woran können Sie eine Weitsichtigkeit erkennen? "
            "Was können Sie gegen Weitsichtigkeit tun? "
            "Was ist Weitsichtigkeit? "
            "Weitsichtigkeit ist eine Sehschwäche. "
            "Bei einer Weitsichtigkeit kann die betroffene Person nicht mehr gut sehen."
        )
        expected = (
            "Was ist Weitsichtigkeit? "
            "Weitsichtigkeit ist eine Sehschwäche. "
            "Bei einer Weitsichtigkeit kann die betroffene Person nicht mehr gut sehen."
        )
        cleaned = cleaner.remove_toc_question_streaks(text)
        self.assertEqual(cleaned, expected)

    def test_subsequent_qa_sections_preserved(self):
        """Testet, dass spätere Zwischenüberschriften im Fließtext erhalten bleiben."""
        text = (
            "Was ist Baldrian? "
            "Wie wirkt Baldrian? "
            "Was ist Baldrian? "
            "Baldrian ist eine Heilpflanze. "
            "Wie wirkt Baldrian? "
            "Baldrian hilft beim Schlafen."
        )
        expected = (
            "Was ist Baldrian? "
            "Baldrian ist eine Heilpflanze. "
            "Wie wirkt Baldrian? "
            "Baldrian hilft beim Schlafen."
        )
        cleaned = cleaner.remove_toc_question_streaks(text)
        self.assertEqual(cleaned, expected)

    def test_all_corpus_files_valid_and_non_empty(self):
        """Überprüft alle 11 bereinigten Korpus-Dateien auf Validität und Satzintegrität."""
        clean_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'corpus', '4_normalized_clean'))
        files = sorted(glob.glob(os.path.join(clean_dir, '*_articles.json')))
        self.assertGreaterEqual(len(files), 11, "Es müssen mindestens 11 Korpus-Quellen vorhanden sein.")

        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pairs = data.get('pairs', [])
            self.assertGreater(len(pairs), 0, f"Korpus-Datei {fpath} darf nicht leer sein.")
            for p in pairs:
                ls_text = p.get('ls_text', '')
                as_text = p.get('as_text', '')
                self.assertGreater(len(ls_text.strip()), 15, f"LS-Text in {fpath} zu kurz.")
                self.assertGreater(len(as_text.strip()), 15, f"AS-Text in {fpath} zu kurz.")


if __name__ == '__main__':
    unittest.main()
