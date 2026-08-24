# Experiment: Synthetischer Regressor

Dieses Experiment untersucht die Generierung von kontinuierlichen Komplexitäts-Zwischenstufen ($0.25, 0.50, 0.75$) mittels LLM-Prompting und das Training eines darauf basierenden BiLSTM-Regressors als Reward-Modell für DPO im Vergleich zum datengetriebenen MixUp-Ansatz.

## Enthaltene Skripte:
* `generate_synthetic_steps.py`: Generiert Zwischenstufen zwischen Alltagssprache ($0.0$) und Leichter Sprache ($1.0$) über eine OpenAI-kompatible LLM-API.
* `regression_train_synthetic.py`: Trainiert ein BiLSTM-Regressionsmodell auf den synthetischen Textstufen.

## SBATCH-Ausführungsskripte:
Befinden sich unter `scripts/sbatch/experiments/synthetic_regressor/`.
