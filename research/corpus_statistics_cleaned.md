# Corpus Statistics Overview

Diese Datei wird automatisch durch das Skript `scripts/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).
Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.

| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apotheken | 157 | 120058 | 201234 | 234608 | 443722 | 15439 | 14808 | 6784 | 21496 | 0.057 | 0.107 | 7.8 | 13.6 |
| behindertenbeauftragter | 51 | 19548 | 20691 | 38724 | 44725 | 2102 | 1087 | 2807 | 4347 | 0.144 | 0.210 | 9.3 | 19.0 |
| brandeins | 18 | 3720 | 3413 | 6044 | 7216 | 116 | 188 | 855 | 1625 | 0.230 | 0.476 | 32.1 | 18.2 |
| hamburg | 56 | 34002 | 31002 | 61204 | 66977 | 3658 | 1978 | 3530 | 7324 | 0.104 | 0.236 | 9.3 | 15.7 |
| hannover | 796 | 453031 | 399497 | 861967 | 858086 | 56870 | 25555 | 7693 | 19141 | 0.017 | 0.048 | 8.0 | 15.6 |
| koeln | 38 | 30114 | 20169 | 55985 | 43275 | 3286 | 1253 | 2606 | 3899 | 0.087 | 0.193 | 9.2 | 16.1 |
| main_taunus | 34 | 5618 | 5446 | 9974 | 11037 | 657 | 297 | 1030 | 1556 | 0.183 | 0.286 | 8.6 | 18.3 |
| mdr | 227 | 51786 | 80849 | 94976 | 168440 | 5399 | 5494 | 4720 | 14169 | 0.091 | 0.175 | 9.6 | 14.7 |
| sozialpolitik | 14 | 4950 | 11495 | 9249 | 23573 | 528 | 739 | 1116 | 3269 | 0.225 | 0.284 | 9.4 | 15.6 |
| stuttgart | 39 | 22479 | 39557 | 42894 | 91331 | 2228 | 2326 | 2630 | 4512 | 0.117 | 0.114 | 10.1 | 17.0 |
| taz | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wiesbaden | 41 | 7138 | 10127 | 13808 | 23332 | 820 | 531 | 1461 | 3040 | 0.205 | 0.300 | 8.7 | 19.1 |
| **TOTAL** | **1471** | **752444** | **823480** | **1429433** | **1781714** | **91103** | **54256** | **20626** | **51929** | **0.027** | **0.063** | **8.3** | **15.2** |
