# Corpus Statistics Overview

Diese Datei wird automatisch durch das Skript `scripts/evaluation/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).
Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.

| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apotheken | 161 | 128789 | 202033 | 251521 | 444865 | 16792 | 14872 | 6979 | 21715 | 0.054 | 0.107 | 7.7 | 13.6 |
| behindertenbeauftragter | 58 | 18871 | 25131 | 37405 | 55197 | 1967 | 1271 | 2749 | 4296 | 0.146 | 0.171 | 9.6 | 19.8 |
| brandeins | 32 | 5886 | 5904 | 9528 | 12565 | 202 | 340 | 1112 | 2505 | 0.189 | 0.424 | 29.1 | 17.4 |
| hamburg | 54 | 32663 | 31827 | 58984 | 69561 | 3546 | 2026 | 3482 | 7135 | 0.107 | 0.224 | 9.2 | 15.7 |
| hannover | 768 | 440237 | 386331 | 839915 | 831816 | 55346 | 25287 | 7650 | 18572 | 0.017 | 0.048 | 8.0 | 15.3 |
| koeln | 40 | 32010 | 21358 | 59347 | 45232 | 3470 | 1353 | 2558 | 3769 | 0.080 | 0.176 | 9.2 | 15.8 |
| mdr | 221 | 55724 | 110798 | 102860 | 234277 | 6257 | 9948 | 6268 | 16695 | 0.112 | 0.151 | 8.9 | 11.1 |
| sozialpolitik | 15 | 5201 | 12443 | 9724 | 25596 | 554 | 802 | 1137 | 3417 | 0.219 | 0.275 | 9.4 | 15.5 |
| stuttgart | 42 | 23727 | 50700 | 45424 | 119827 | 2339 | 2961 | 2698 | 4611 | 0.114 | 0.091 | 10.1 | 17.1 |
| taz | 7 | 3631 | 6694 | 6415 | 12833 | 411 | 386 | 856 | 2256 | 0.236 | 0.337 | 8.8 | 17.3 |
| wiesbaden | 40 | 7029 | 9837 | 13610 | 22731 | 803 | 523 | 1461 | 2982 | 0.208 | 0.303 | 8.8 | 18.8 |
| **TOTAL** | **1438** | **753768** | **863056** | **1434733** | **1874500** | **91687** | **59769** | **21772** | **53910** | **0.029** | **0.062** | **8.2** | **14.4** |
