#!/bin/bash
#SBATCH --job-name=run_master_thesis_all
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
# =============================================================================
# Master Thesis Complete Orchestrator: Pipeline & All Experiments
# =============================================================================
# Startet die vollständige End-to-End Pipeline sowie sämtliche 17 experimentelle
# Ablationen und Benchmarks mit optimierter Slurm-Job-Abhängigkeitskette
# (--dependency=afterok).
#
# Standardmäßig startet das Skript ab Schritt 02 (Content-Extraktion), da Schritt 01
# (URL-Alignment) typischerweise bereits vorliegt. 
# Mit START_STEP=1 kann die Pipeline inkl. Schritt 01 gestartet werden:
#   START_STEP=1 bash scripts/sbatch/run_all_pipeline_and_experiments.sh
# =============================================================================

set -e

START_STEP=${START_STEP:-2}
if [ "$1" == "--from-step-1" ] || [ "$1" == "-1" ]; then
    START_STEP=1
fi

echo "========================================================================"
echo " MASTER THESIS - GESAMT-ORCHESTRIERUNG (Pipeline + Alle Experimente)"
echo " Startzeit: $(date)"
echo " Start-Schritt Pipeline: Schritt $START_STEP"
echo "========================================================================"

# -----------------------------------------------------------------------------
# 0. Verzeichnisstruktur sicherstellen
# -----------------------------------------------------------------------------
mkdir -p results/logs/run_pipeline
mkdir -p results/logs/experiments/{benchmark,context_length_ablation,data_scaling,decoder_only,dpo_beta_sweep,factuality_metric,glossary,length_bias,loss_aggregation,metric_weights,ppo,rnn_baseline,rule_adherence,sft_scaling,synthetic_regressor,textcomplexityde,token_length}
mkdir -p results/plots/run_pipeline
mkdir -p results/plots/experiments/{benchmark,context_length_ablation,decoder_only,dpo_beta_sweep,factuality_metric,glossary,length_bias,loss_aggregation,metric_weights,ppo,rnn_baseline,rule_adherence,sft_scaling,synthetic_regressor,textcomplexityde,token_length}
mkdir -p data/{corpus,lebenshilfe,analysis,vocabs,synthetic,dpo,metric_weights_exp,evaluation_sets}
mkdir -p data/corpus/{1_aligned_urls,2_raw_scraped,3_content_extracted,4_normalized_clean}
mkdir -p results/models/{sft,dpo,decoder_only,ppo,loss_aggregation_exp,experiments}
mkdir -p results/models/decoder_only/{sft,dpo,ppo}
mkdir -p results/models/ppo/seq2seq
mkdir -p results/models/loss_aggregation_exp/{dpo_sum,dpo_mean}
mkdir -p results/models/experiments/synthetic_regressor
mkdir -p results/evaluation

# =============================================================================
# TEIL 1: MASTER-PIPELINE (13 SCHRITTE)
# =============================================================================
echo ""
echo ">>> [1/4] Reiche Haupt-Pipeline ein..."

if [ "$START_STEP" -eq 1 ]; then
    JOB01=$(sbatch --parsable scripts/sbatch/run_pipeline/01_crawl_url_alignment.sh)
    echo "  [01] URL-Alignment: Job ID $JOB01"
    JOB02=$(sbatch --parsable --dependency=afterok:$JOB01 scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh)
    echo "  [02] Content-Extraktion: Job ID $JOB02"
else
    echo "  [01] URL-Alignment: Übersprungen (bereits vorhanden)"
    JOB02=$(sbatch --parsable scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh)
    echo "  [02] Content-Extraktion: Job ID $JOB02"
fi

JOB03=$(sbatch --parsable scripts/sbatch/run_pipeline/03_create_lebenshilfe_dataset.sh)
echo "  [03] Lebenshilfe einlesen: Job ID $JOB03"

JOB04=$(sbatch --parsable --dependency=afterok:$JOB03 scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh)
echo "  [04] Lebenshilfe bereinigen: Job ID $JOB04"

JOB05=$(sbatch --parsable --dependency=afterok:$JOB02:$JOB04 scripts/sbatch/run_pipeline/05_build_corpus_master.sh)
echo "  [05] Corpus Master erstellen: Job ID $JOB05"

JOB06=$(sbatch --parsable scripts/sbatch/run_pipeline/06_prepare_10kgnad_dpo_corpus.sh)
echo "  [06] 10kGNAD DPO-Korpus vorbereiten: Job ID $JOB06"

JOB07=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/run_pipeline/07_train_sentence_classifier.sh)
echo "  [07] BiLSTM Satz-Klassifikator: Job ID $JOB07"

JOB08=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/run_pipeline/08_train_article_classifier.sh)
echo "  [08] BiLSTM Artikel-Klassifikator: Job ID $JOB08"

JOB09=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/run_pipeline/09_train_mixup_regressor.sh)
echo "  [09] BiLSTM MixUp-Regressor: Job ID $JOB09"

JOB10=$(sbatch --parsable --dependency=afterok:$JOB05:$JOB09 scripts/sbatch/run_pipeline/10_train_sft.sh)
echo "  [10] mBART-50 SFT Training: Job ID $JOB10"

JOB11=$(sbatch --parsable --dependency=afterok:$JOB06:$JOB09:$JOB10 scripts/sbatch/run_pipeline/11_generate_dpo_dataset.sh)
echo "  [11] DPO-Paare generieren (Self-Play): Job ID $JOB11"

JOB12=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/run_pipeline/12_train_dpo.sh)
echo "  [12] mBART-50 LoRA DPO Training: Job ID $JOB12"

JOB13=$(sbatch --parsable --dependency=afterok:$JOB04:$JOB10:$JOB12 scripts/sbatch/run_pipeline/13_evaluate_pipeline.sh)
echo "  [13] Finale Pipeline-Evaluierung: Job ID $JOB13"

# =============================================================================
# TEIL 2: DATEN-, METRIK- & ANALYSE-EXPERIMENTE
# =============================================================================
echo ""
echo ">>> [2/4] Reiche Daten-, Metrik- & Korpus-Experimente ein..."

# 2.1 Factuality Benchmark (unabhängig)
EXP_FACT=$(sbatch --parsable scripts/sbatch/experiments/factuality_metric/1_run_factuality_metric_experiment.sh)
echo "  [Exp 01] Factuality & Hallucination Benchmark: Job ID $EXP_FACT"

# 2.2 Glossar-Extraktion (benötigt Scraped Raw / Normalized Data)
EXP_GLOSS_1=$(sbatch --parsable --dependency=afterok:$JOB02 scripts/sbatch/experiments/glossary/1_build_glossary.sh)
EXP_GLOSS_2=$(sbatch --parsable --dependency=afterok:$EXP_GLOSS_1 scripts/sbatch/experiments/glossary/2_enrich_glossary.sh)
echo "  [Exp 02] Glossar-Extraktion & Enrichment: Job ID $EXP_GLOSS_1 -> $EXP_GLOSS_2"

# 2.3 Jina Kontextlängen-Ablation (benötigt Corpus Master)
EXP_CTX=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/context_length_ablation/1_run_context_length_ablation.sh)
echo "  [Exp 03] Jina Kontextlängen-Ablation: Job ID $EXP_CTX"

# 2.4 Length Bias & Shortcut Analyse (benötigt Lebenshilfe, Corpus Master & Artikel-Klassifikator)
EXP_LENBIAS=$(sbatch --parsable --dependency=afterok:$JOB04:$JOB05:$JOB08 scripts/sbatch/experiments/length_bias/1_check_length_bias.sh)
echo "  [Exp 04] Length-Bias & Shortcut Analyse: Job ID $EXP_LENBIAS"

# 2.5 TextComplexityDE Validierung (benötigt MixUp Regressor)
EXP_TCDE=$(sbatch --parsable --dependency=afterok:$JOB09 scripts/sbatch/experiments/textcomplexityde/1_evaluate_textcomplexityde.sh)
echo "  [Exp 05] TextComplexityDE Validierung: Job ID $EXP_TCDE"

# 2.6 RNN Baseline Modell (benötigt Corpus Master; Eval benötigt Lebenshilfe & MixUp-Modell)
EXP_RNN_TR=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/rnn_baseline/1_train_rnn_baseline.sh)
EXP_RNN_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$JOB09:$EXP_RNN_TR scripts/sbatch/experiments/rnn_baseline/2_evaluate_rnn_baseline.sh)
echo "  [Exp 06] RNN Baseline Regressor: Training $EXP_RNN_TR -> Eval $EXP_RNN_EV"

# 2.7 SFT Data Scaling Grid (benötigt Corpus Master & MixUp Regressor)
EXP_SFTSCAL_TR=$(sbatch --parsable --dependency=afterok:$JOB05:$JOB09 scripts/sbatch/experiments/sft_scaling/1_train_sft_scaling_grid.sh)
EXP_SFTSCAL_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$EXP_SFTSCAL_TR scripts/sbatch/experiments/sft_scaling/2_evaluate_sft_scaling.sh)
echo "  [Exp 07] SFT Data Scaling Grid: Training $EXP_SFTSCAL_TR -> Eval $EXP_SFTSCAL_EV"

# 2.8 MixUp Data Scaling Grid (benötigt Corpus Master & MixUp Vocab)
EXP_DATASCAL_M=$(sbatch --parsable --dependency=afterok:$JOB05:$JOB09 scripts/sbatch/experiments/data_scaling/1_scaling_mixtures_grid.sh)
EXP_DATASCAL_A=$(sbatch --parsable --dependency=afterok:$JOB05:$JOB09 scripts/sbatch/experiments/data_scaling/2_scaling_articles_grid.sh)
EXP_DATASCAL_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$EXP_DATASCAL_M:$EXP_DATASCAL_A scripts/sbatch/experiments/data_scaling/3_evaluate_scaling.sh)
echo "  [Exp 08] MixUp Data Scaling: Mixtures $EXP_DATASCAL_M, Articles $EXP_DATASCAL_A -> Eval $EXP_DATASCAL_EV"

# 2.9 Synthetischer Regressor Pipeline (7 Schritte)
EXP_SYN_1=$(sbatch --parsable --dependency=afterok:$JOB04 scripts/sbatch/experiments/synthetic_regressor/1_generate_synthetic_steps_lh.sh)
EXP_SYN_2=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/synthetic_regressor/2_generate_synthetic_steps_corpus.sh)
EXP_SYN_3=$(sbatch --parsable --dependency=afterok:$EXP_SYN_1:$EXP_SYN_2 scripts/sbatch/experiments/synthetic_regressor/3_train_synthetic_regressor.sh)
EXP_SYN_4=$(sbatch --parsable --dependency=afterok:$EXP_SYN_3:$JOB05 scripts/sbatch/experiments/synthetic_regressor/4_train_sft_synthetic.sh)
EXP_SYN_5=$(sbatch --parsable --dependency=afterok:$EXP_SYN_3:$EXP_SYN_4:$JOB06 scripts/sbatch/experiments/synthetic_regressor/5_generate_dpo_synthetic.sh)
EXP_SYN_6=$(sbatch --parsable --dependency=afterok:$EXP_SYN_4:$EXP_SYN_5 scripts/sbatch/experiments/synthetic_regressor/6_train_dpo_synthetic.sh)
EXP_SYN_7=$(sbatch --parsable --dependency=afterok:$JOB04:$EXP_SYN_3:$EXP_SYN_6 scripts/sbatch/experiments/synthetic_regressor/7_evaluate_synthetic_experiments.sh)
echo "  [Exp 09] Synthetischer Regressor Pipeline: 7 Jobs eingereiht (Final Eval: Job ID $EXP_SYN_7)"

# 2.10 Token Length Experiment (256, 512, 1024)
EXP_TL_M256=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/token_length/1_train_metric_256.sh)
EXP_TL_M512=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/token_length/1_train_metric_512.sh)
EXP_TL_M1024=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/token_length/1_train_metric_1024.sh)

EXP_TL_S256=$(sbatch --parsable --dependency=afterok:$EXP_TL_M256:$JOB05 scripts/sbatch/experiments/token_length/2_train_sft_256.sh)
EXP_TL_S512=$(sbatch --parsable --dependency=afterok:$EXP_TL_M512:$JOB05 scripts/sbatch/experiments/token_length/2_train_sft_512.sh)
EXP_TL_S1024=$(sbatch --parsable --dependency=afterok:$EXP_TL_M1024:$JOB05 scripts/sbatch/experiments/token_length/2_train_sft_1024.sh)

EXP_TL_G256=$(sbatch --parsable --dependency=afterok:$EXP_TL_S256:$JOB06 scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_256.sh)
EXP_TL_G512=$(sbatch --parsable --dependency=afterok:$EXP_TL_S512:$JOB06 scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_512.sh)
EXP_TL_G1024=$(sbatch --parsable --dependency=afterok:$EXP_TL_S1024:$JOB06 scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_1024.sh)

EXP_TL_D256=$(sbatch --parsable --dependency=afterok:$EXP_TL_G256 scripts/sbatch/experiments/token_length/4_train_dpo_256.sh)
EXP_TL_D512=$(sbatch --parsable --dependency=afterok:$EXP_TL_G512 scripts/sbatch/experiments/token_length/4_train_dpo_512.sh)
EXP_TL_D1024=$(sbatch --parsable --dependency=afterok:$EXP_TL_G1024 scripts/sbatch/experiments/token_length/4_train_dpo_1024.sh)

EXP_TL_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$EXP_TL_D256:$EXP_TL_D512:$EXP_TL_D1024 scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh)
echo "  [Exp 10] Token Length Ablation (256/512/1024): 13 Jobs eingereiht (Final Eval: Job ID $EXP_TL_EV)"

# 2.11 Rule Adherence Messung (benötigt Token-Length Evaluation & Corpus Master)
EXP_RULE=$(sbatch --parsable --dependency=afterok:$JOB05:$EXP_TL_EV scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh)
echo "  [Exp 11] Regel-Adhärenz Evaluation: Job ID $EXP_RULE"

# =============================================================================
# TEIL 3: MODELING- & ALIGNMENT-EXPERIMENTE
# =============================================================================
echo ""
echo ">>> [3/4] Reiche Modell- & Alignment-Experimente ein..."

# 3.1 Decoder-Only Pipeline (Qwen 2.5 1.5B)
DEC_SFT=$(sbatch --parsable --dependency=afterok:$JOB05 scripts/sbatch/experiments/decoder_only/1_train_sft_decoder_only.sh)
DEC_GEN=$(sbatch --parsable --dependency=afterok:$DEC_SFT:$JOB06:$JOB09 scripts/sbatch/experiments/decoder_only/2_generate_dpo_pairs_decoder_only.sh)
DEC_DPO=$(sbatch --parsable --dependency=afterok:$DEC_GEN:$DEC_SFT scripts/sbatch/experiments/decoder_only/3_train_dpo_decoder_only.sh)
DEC_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$DEC_SFT:$DEC_DPO scripts/sbatch/experiments/decoder_only/4_evaluate_decoder_only.sh)
echo "  [Exp 12] Decoder-Only (Qwen 2.5): SFT $DEC_SFT -> DPO Gen $DEC_GEN -> DPO $DEC_DPO -> Eval $DEC_EV"

# 3.2 Metric Weights DPO Experiment (0.5/0.5, 0.7/0.3, 1.0/0.0)
MW_G05=$(sbatch --parsable --dependency=afterok:$JOB06:$JOB09:$JOB10 scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w05_w05.sh)
MW_G07=$(sbatch --parsable --dependency=afterok:$JOB06:$JOB09:$JOB10 scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w07_w03.sh)
MW_G10=$(sbatch --parsable --dependency=afterok:$JOB06:$JOB09:$JOB10 scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w10_w00.sh)

MW_D05=$(sbatch --parsable --dependency=afterok:$MW_G05 scripts/sbatch/experiments/metric_weights/2_train_dpo_w05_w05.sh)
MW_D07=$(sbatch --parsable --dependency=afterok:$MW_G07 scripts/sbatch/experiments/metric_weights/2_train_dpo_w07_w03.sh)
MW_D10=$(sbatch --parsable --dependency=afterok:$MW_G10 scripts/sbatch/experiments/metric_weights/2_train_dpo_w10_w00.sh)

MW_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$MW_D05:$MW_D07:$MW_D10 scripts/sbatch/experiments/metric_weights/3_run_full_evaluation.sh)
echo "  [Exp 13] Metric Weighting Experiment: 7 Jobs eingereiht (Final Eval: Job ID $MW_EV)"

# 3.3 Loss Aggregation DPO Experiment (Sum vs. Mean)
LA_SUM=$(sbatch --parsable --dependency=afterok:$MW_G05:$JOB10 scripts/sbatch/experiments/loss_aggregation/1_train_dpo_sum.sh)
LA_MEAN=$(sbatch --parsable --dependency=afterok:$MW_G05:$JOB10 scripts/sbatch/experiments/loss_aggregation/1_train_dpo_mean.sh)
LA_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$LA_SUM:$LA_MEAN scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh)
echo "  [Exp 14] Loss Aggregation (Sum vs. Mean): Sum $LA_SUM, Mean $LA_MEAN -> Eval $LA_EV"

# 3.4 DPO Beta Sweep (0.01, 0.05, 0.10, 0.20, 0.50)
BETA_001=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_001.sh)
BETA_005=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_005.sh)
BETA_010=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_010.sh)
BETA_020=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_020.sh)
BETA_050=$(sbatch --parsable --dependency=afterok:$JOB10:$JOB11 scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_050.sh)
BETA_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$BETA_001:$BETA_005:$BETA_010:$BETA_020:$BETA_050 scripts/sbatch/experiments/dpo_beta_sweep/2_run_full_evaluation.sh)
echo "  [Exp 15] DPO Beta Sweep (5 Modelle): 5 Trainingsjobs -> Eval $BETA_EV"

# 3.5 PPO Alignment (Seq2Seq & Decoder-Only)
PPO_DEC=$(sbatch --parsable --dependency=afterok:$DEC_SFT:$JOB09 scripts/sbatch/experiments/ppo/1_train_ppo_decoder_only.sh)
PPO_SEQ=$(sbatch --parsable --dependency=afterok:$JOB09:$JOB10 scripts/sbatch/experiments/ppo/1_train_ppo_seq2seq.sh)
PPO_EV=$(sbatch --parsable --dependency=afterok:$JOB04:$PPO_DEC:$PPO_SEQ scripts/sbatch/experiments/ppo/2_evaluate_all_ppo.sh)
echo "  [Exp 16] PPO Alignment: Decoder $PPO_DEC, Seq2Seq $PPO_SEQ -> Eval $PPO_EV"

# =============================================================================
# TEIL 4: GRAND MASTER 5-WEGE / 7-WEGE BENCHMARK
# =============================================================================
echo ""
echo ">>> [4/4] Reiche Grand Master Benchmark ein..."

# 4.1 Master 5-Wege Benchmark (benötigt Lebenshilfe, Regressor, mBART SFT/DPO & Qwen SFT/DPO)
EXP_BENCH=$(sbatch --parsable --dependency=afterok:$JOB04:$JOB09:$JOB10:$JOB12:$DEC_SFT:$DEC_DPO scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh)
echo "  [Exp 17] Master 5-Wege Benchmark: Job ID $EXP_BENCH"

echo ""
echo "========================================================================"
echo " Alle Pipeline-Schritte & 17 Experimente erfolgreich eingereiht!"
echo "========================================================================"
echo " Slurm Queue überwachen:   squeue -u \$USER"
echo " Live-Logs Hauptpipeline:  tail -f results/logs/run_pipeline/*.out"
echo " Live-Logs Experimente:    tail -f results/logs/experiments/*/*.out"
echo "========================================================================"
