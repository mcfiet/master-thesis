# Corpus Statistics Overview

Diese Datei wird automatisch durch das Skript `scripts/evaluation/corpus_stats.py` generiert. Sie enthält die zusammenfassenden Statistiken für alle Quellen im Korpus, aufgeschlüsselt nach Leichter Sprache (LS) und Alltagssprache (AS).
Die Spalte 'Tokens' verwendet nun `tiktoken` (cl100k_base), was der tatsächlichen Token-Anzahl für LLMs (z.B. GPT-4) entspricht.

| Source | Pairs | Words (LS) | Words (AS) | Tokens (LS) | Tokens (AS) | Sentences (LS) | Sentences (AS) | Vocab (LS) | Vocab (AS) | TTR (LS) | TTR (AS) | W/S (LS) | W/S (AS) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| apotheken | 153 | 105301 | 178374 | 209005 | 390983 | 13725 | 12994 | 6622 | 20089 | 0.063 | 0.113 | 7.7 | 13.7 |
| behindertenbeauftragter | 48 | 17623 | 17834 | 34879 | 38491 | 1875 | 971 | 2633 | 3938 | 0.149 | 0.221 | 9.4 | 18.4 |
| brandeins | 19 | 3624 | 3614 | 5953 | 7635 | 419 | 196 | 822 | 1708 | 0.227 | 0.473 | 8.6 | 18.4 |
| hamburg | 45 | 19968 | 21258 | 35895 | 45687 | 2137 | 1262 | 2649 | 5666 | 0.133 | 0.267 | 9.3 | 16.8 |
| hannover | 219 | 117696 | 112844 | 206790 | 238084 | 14926 | 7466 | 6993 | 17286 | 0.059 | 0.153 | 7.9 | 15.1 |
| koeln | 38 | 28708 | 20708 | 52916 | 43580 | 3150 | 1325 | 2391 | 3680 | 0.083 | 0.178 | 9.1 | 15.6 |
| mdr | 219 | 55042 | 109325 | 101610 | 231294 | 6174 | 9835 | 6200 | 16477 | 0.113 | 0.151 | 8.9 | 11.1 |
| sozialpolitik | 13 | 4765 | 10549 | 8744 | 21437 | 508 | 682 | 1088 | 3066 | 0.228 | 0.291 | 9.4 | 15.5 |
| stuttgart | 35 | 20768 | 32337 | 39515 | 74516 | 2093 | 2048 | 2542 | 4113 | 0.122 | 0.127 | 9.9 | 15.8 |
| taz | 4 | 1891 | 4353 | 3378 | 8300 | 233 | 268 | 578 | 1563 | 0.306 | 0.359 | 8.1 | 16.2 |
| wiesbaden | 37 | 6681 | 9147 | 12839 | 20373 | 779 | 486 | 1449 | 2862 | 0.217 | 0.313 | 8.6 | 18.8 |
| **TOTAL** | **830** | **382067** | **520343** | **711524** | **1120380** | **46019** | **37533** | **20037** | **49857** | **0.052** | **0.096** | **8.3** | **13.9** |
