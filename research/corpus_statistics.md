# Corpus Statistics Overview

Diese Datei wird automatisch durch das Skript `scripts/evaluation/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).
Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.

| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apotheken | 161 | 123505 | 205063 | 241325 | 451812 | 15857 | 15068 | 6979 | 21915 | 0.057 | 0.107 | 7.8 | 13.6 |
| behindertenbeauftragter | 60 | 21312 | 28092 | 42027 | 61618 | 2278 | 1467 | 2985 | 4763 | 0.140 | 0.170 | 9.4 | 19.1 |
| brandeins | 32 | 6014 | 5920 | 9716 | 12626 | 185 | 343 | 1134 | 2523 | 0.189 | 0.426 | 32.5 | 17.3 |
| hamburg | 57 | 34124 | 33688 | 61455 | 73137 | 3673 | 2137 | 3536 | 7549 | 0.104 | 0.224 | 9.3 | 15.8 |
| hannover | 808 | 458621 | 405321 | 872291 | 871830 | 57582 | 25967 | 7770 | 19529 | 0.017 | 0.048 | 8.0 | 15.6 |
| koeln | 39 | 30776 | 20831 | 57152 | 44442 | 3361 | 1328 | 2629 | 3962 | 0.085 | 0.190 | 9.2 | 15.7 |
| main_taunus | 36 | 5882 | 5605 | 10450 | 11374 | 691 | 304 | 1069 | 1602 | 0.182 | 0.286 | 8.5 | 18.4 |
| mdr | 235 | 53869 | 83487 | 98780 | 173765 | 5633 | 5656 | 4897 | 14643 | 0.091 | 0.175 | 9.6 | 14.8 |
| sozialpolitik | 15 | 5201 | 12443 | 9724 | 25596 | 554 | 802 | 1137 | 3417 | 0.219 | 0.275 | 9.4 | 15.5 |
| stuttgart | 42 | 23653 | 46060 | 45202 | 106629 | 2342 | 2727 | 2676 | 4557 | 0.113 | 0.099 | 10.1 | 16.9 |
| taz | 7 | 3631 | 6694 | 6415 | 12833 | 411 | 386 | 856 | 2256 | 0.236 | 0.337 | 8.8 | 17.3 |
| wiesbaden | 41 | 7138 | 10127 | 13808 | 23332 | 820 | 531 | 1461 | 3040 | 0.205 | 0.300 | 8.7 | 19.1 |
| **TOTAL** | **1533** | **773726** | **863331** | **1468345** | **1868994** | **93387** | **56716** | **21272** | **54057** | **0.027** | **0.063** | **8.3** | **15.2** |
