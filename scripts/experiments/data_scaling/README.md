# Data Scaling & Learning Curve Experiment

Dieses Experiment untersucht systematisch das Skalierungsverhalten des **BiLSTM MixUp Simplicity Regressors** entlang zweier Achsen:

1. **Achse 1: Synthetischer MixUp-Multiplikator (`mixtures_per_pair`)**
   - Werte: $M \in \{2, 5, 10, 20, 40, 80\}$ bei 100% der Basis-Trainingsartikel.
   - Frage: *Bringt eine dichtere kontinuierliche Abtastung der Mischstufen $\lambda \in [0, 1]$ noch signifikanten Mehrwert oder setzt eine Sättigung ein?*

2. **Achse 2: Reale Artikelpaare im Training (`train_fraction` / `num_train_article_pairs`)**
   - Werte: $F \in \{10\%, 25\%, 50\%, 75\%, 100\%\}$ bei festem $M=20$.
   - Frage: *Wie stark profitiert das Modell von zusätzlichen realen Basis-Texten (Vokabular-/Domänenerweiterung)?*

---

## Dateistruktur

```
scripts/
├── experiments/
│   └── data_scaling/
│       ├── train_mixup_scaling.py       # Parametrisierter PyTorch-Trainings- & Testrunner
│       └── evaluate_all_scaling.py      # Aggregiert JSON-Metriken in scaling_summary.csv
└── sbatch/
    └── experiments/
        └── data_scaling/
            ├── 1_scaling_mixtures_grid.sh   # SLURM Grid für mixtures_per_pair
            ├── 2_scaling_articles_grid.sh   # SLURM Grid für train_fraction
            ├── 3_evaluate_scaling.sh        # SLURM Job für Aggregation
            └── run_all_data_scaling.sh      # Master Runner mit Abhängigkeitskette (afterok)

notebooks/
└── research/
    └── metric/
        └── 5_mixup_data_scaling_analysis.ipynb  # Interaktive Auswertung & Plotting der Lernkurven
```

---

## Ausführung auf dem HPC-Cluster

```bash
bash scripts/sbatch/experiments/data_scaling/run_all_data_scaling.sh
```
