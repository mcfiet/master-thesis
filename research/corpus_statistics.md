# Corpus Statistics Overview

Diese Datei wird automatisch durch das Skript `scripts/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).
Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.

| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apotheken | 153 | 116725 | 197796 | 228018 | 436269 | 15009 | 14579 | 6657 | 21243 | 0.057 | 0.107 | 7.8 | 13.6 |
| behindertenbeauftragter | 52 | 20278 | 21371 | 39976 | 45898 | 2192 | 1167 | 2889 | 4448 | 0.142 | 0.208 | 9.3 | 18.3 |
| brandeins | 18 | 3720 | 3413 | 6044 | 7216 | 116 | 188 | 855 | 1625 | 0.230 | 0.476 | 32.1 | 18.2 |
| hamburg | 55 | 32965 | 31515 | 59380 | 68643 | 3562 | 1976 | 3448 | 7129 | 0.105 | 0.226 | 9.3 | 15.9 |
| hannover | 788 | 445653 | 387505 | 848117 | 831912 | 55902 | 24897 | 7649 | 18955 | 0.017 | 0.049 | 8.0 | 15.6 |
| koeln | 38 | 30114 | 20169 | 55985 | 43275 | 3286 | 1253 | 2606 | 3899 | 0.087 | 0.193 | 9.2 | 16.1 |
| main_taunus | 34 | 5618 | 5446 | 9974 | 11037 | 657 | 297 | 1030 | 1556 | 0.183 | 0.286 | 8.6 | 18.3 |
| mdr | 227 | 51840 | 80326 | 95189 | 167299 | 5408 | 5468 | 4733 | 14180 | 0.091 | 0.177 | 9.6 | 14.7 |
| sozialpolitik | 15 | 5201 | 12443 | 9724 | 25596 | 554 | 802 | 1137 | 3417 | 0.219 | 0.275 | 9.4 | 15.5 |
| stuttgart | 41 | 23610 | 46017 | 45099 | 106526 | 2339 | 2724 | 2665 | 4549 | 0.113 | 0.099 | 10.1 | 16.9 |
| taz | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| wiesbaden | 39 | 6920 | 9850 | 13412 | 22739 | 786 | 514 | 1461 | 2971 | 0.211 | 0.302 | 8.8 | 19.2 |
| **TOTAL** | **1460** | **742644** | **815851** | **1410918** | **1766410** | **89811** | **53865** | **20498** | **51716** | **0.028** | **0.063** | **8.3** | **15.1** |
