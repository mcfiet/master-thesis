#!/bin/bash
#SBATCH --job-name=run_master_thesis_all
#SBATCH --partition=research
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -e
# =============================================================================
# Master Thesis Complete Orchestrator: Pipeline & All Experiments
# =============================================================================
# Startet die vollständige End-to-End Pipeline sowie sämtliche 20 experimentelle
# Ablationen und Benchmarks mit optimierter Slurm-Job-Abhängigkeitskette
# (--dependency=afterok).
#
# Standardmäßig startet das Skript ab Schritt 02 (Content-Extraktion), da Schritt 01
# (URL-Alignment) typischerweise bereits vorliegt. 
# Mit START_STEP=1 kann die Pipeline inkl. Schritt 01 gestartet werden:
#   START_STEP=1 bash scripts/sbatch/run_all_pipeline_and_experiments.sh
# =============================================================================


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
mkdir -p results/logs/experiments/{benchmark,classifier_length,classifier_stability,context_length_ablation,data_scaling,decoder_only,dpo_beta_sweep,factuality_metric,length_bias,merlin,metric_weights,mixup_variants,ppo,regressor_length,rnn_baseline,rule_adherence,sft_scaling,similarity_threshold,synthetic_regressor,textcomplexityde,token_length}
mkdir -p results/plots/run_pipeline
mkdir -p results/plots/experiments/{benchmark,classifier_length,classifier_stability,context_length_ablation,decoder_only,dpo_beta_sweep,factuality_metric,length_bias,merlin,metric_weights,mixup_variants,ppo,regressor_length,rnn_baseline,rule_adherence,sft_scaling,similarity_threshold,synthetic_regressor,textcomplexityde,token_length}
mkdir -p data/{corpus,lebenshilfe,analysis,vocabs,synthetic,dpo,metric_weights_exp,evaluation_sets,classifier_length_exp,regressor_length_exp,mixup_variants,token_length_exp,expert_eval}
mkdir -p data/corpus/{1_aligned_urls,2_raw_scraped,3_content_extracted,4_normalized_clean}
mkdir -p results/models/{sft,dpo,decoder_only,ppo,classifier_length_exp,regressor_length_exp,mixup_variants,token_length_exp,experiments}
mkdir -p results/models/decoder_only/{sft,dpo,ppo}
mkdir -p results/models/ppo/seq2seq
mkdir -p results/models/experiments/synthetic_regressor
mkdir -p results/evaluation/classifier_stability
mkdir -p results/experiments/similarity_threshold/plots
mkdir -p results/expert_eval
mkdir -p results/evaluation

# -----------------------------------------------------------------------------
# Hilfsfunktion: Job einreichen mit dynamischen Abhängigkeiten
# -----------------------------------------------------------------------------
submit_job() {
    local script="$1"
    shift
    local deps=()
    for d in "$@"; do
        if [ -n "$d" ]; then
            deps+=("$d")
        fi
    done
    if [ ${#deps[@]} -gt 0 ]; then
        local IFS=":"
        sbatch --parsable --dependency=afterok:"${deps[*]}" "$script"
    else
        sbatch --parsable "$script"
    fi
}

# =============================================================================
# TEIL 1: MASTER-PIPELINE (13 SCHRITTE)
# =============================================================================
echo ""
echo ">>> [1/4] Reiche Haupt-Pipeline ein..."

JOB01=""
if [ "$START_STEP" -le 1 ]; then
    JOB01=$(submit_job scripts/sbatch/run_pipeline/01_crawl_url_alignment.sh)
    echo "  [01] URL-Alignment: Job ID $JOB01"
else
    echo "  [01] URL-Alignment: Übersprungen"
fi

JOB02=""
if [ "$START_STEP" -le 2 ]; then
    JOB02=$(submit_job scripts/sbatch/run_pipeline/02_crawl_content_extraction.sh $JOB01)
    echo "  [02] Content-Extraktion: Job ID $JOB02"
else
    echo "  [02] Content-Extraktion: Übersprungen"
fi

JOB03=""
if [ "$START_STEP" -le 3 ]; then
    JOB03=$(submit_job scripts/sbatch/run_pipeline/03_create_lebenshilfe_dataset.sh)
    echo "  [03] Lebenshilfe einlesen: Job ID $JOB03"
else
    echo "  [03] Lebenshilfe einlesen: Übersprungen"
fi

JOB04=""
if [ "$START_STEP" -le 4 ]; then
    JOB04=$(submit_job scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh $JOB03)
    echo "  [04] Lebenshilfe bereinigen: Job ID $JOB04"
else
    echo "  [04] Lebenshilfe bereinigen: Übersprungen"
fi

JOB05=""
if [ "$START_STEP" -le 5 ]; then
    JOB05=$(submit_job scripts/sbatch/run_pipeline/05_build_corpus_master.sh $JOB02 $JOB04)
    echo "  [05] Corpus Master erstellen: Job ID $JOB05"
else
    echo "  [05] Corpus Master erstellen: Übersprungen"
fi

JOB06=""
if [ "$START_STEP" -le 6 ]; then
    JOB06=$(submit_job scripts/sbatch/run_pipeline/06_prepare_10kgnad_dpo_corpus.sh)
    echo "  [06] 10kGNAD DPO-Korpus vorbereiten: Job ID $JOB06"
else
    echo "  [06] 10kGNAD DPO-Korpus vorbereiten: Übersprungen"
fi

JOB07=""
if [ "$START_STEP" -le 7 ]; then
    JOB07=$(submit_job scripts/sbatch/run_pipeline/07_train_sentence_classifier.sh $JOB05)
    echo "  [07] BiLSTM Satz-Klassifikator: Job ID $JOB07"
fi

JOB08=""
if [ "$START_STEP" -le 8 ]; then
    JOB08=$(submit_job scripts/sbatch/run_pipeline/08_train_article_classifier.sh $JOB05)
    echo "  [08] BiLSTM Artikel-Klassifikator: Job ID $JOB08"
fi

JOB09=""
if [ "$START_STEP" -le 9 ]; then
    JOB09=$(submit_job scripts/sbatch/run_pipeline/09_train_mixup_regressor.sh $JOB05)
    echo "  [09] BiLSTM MixUp-Regressor: Job ID $JOB09"
fi

JOB10=""
if [ "$START_STEP" -le 10 ]; then
    JOB10=$(submit_job scripts/sbatch/run_pipeline/10_train_sft.sh $JOB05 $JOB09)
    echo "  [10] mBART-50 SFT Training: Job ID $JOB10"
fi

JOB11=""
JOB11B=""
if [ "$START_STEP" -le 11 ]; then
    JOB11=$(submit_job scripts/sbatch/run_pipeline/11_generate_dpo_dataset.sh $JOB06 $JOB09 $JOB10)
    echo "  [11] DPO-Paare generieren (4 Shards): Job ID $JOB11"
    JOB11B=$(submit_job scripts/sbatch/run_pipeline/11b_merge_dpo_dataset.sh $JOB11)
    echo "  [11b] DPO-Shards zusammenführen: Job ID $JOB11B"
fi

JOB12=""
if [ "$START_STEP" -le 12 ]; then
    JOB12=$(submit_job scripts/sbatch/run_pipeline/12_train_dpo.sh $JOB10 $JOB11B)
    echo "  [12] mBART-50 LoRA DPO Training: Job ID $JOB12"
fi

JOB13=""
if [ "$START_STEP" -le 13 ]; then
    JOB13=$(submit_job scripts/sbatch/run_pipeline/13_evaluate_pipeline.sh $JOB04 $JOB10 $JOB12)
    echo "  [13] Finale Pipeline-Evaluierung: Job ID $JOB13"
fi

# =============================================================================
# TEIL 2: DATEN-, METRIK- & ANALYSE-EXPERIMENTE
# =============================================================================
echo ""
echo ">>> [2/4] Reiche Daten-, Metrik- & Korpus-Experimente ein..."

# 2.1 Factuality Benchmark (unabhängig)
EXP_FACT=$(sbatch --parsable scripts/sbatch/experiments/metric/factuality_metric/1_run_factuality_metric_experiment.sh)
echo "  [Exp 01] Factuality & Hallucination Benchmark: Job ID $EXP_FACT"

# 2.2 Klassifikator-Stabilitäts- & Kapazitäts-Experiment (Multi-Seed & Epochen)
EXP_CLF_STAB_TR=$(submit_job scripts/sbatch/experiments/metric/classifier_stability/1_train_stability_experiment.sh $JOB04 $JOB05)
EXP_CLF_STAB_EV=$(submit_job scripts/sbatch/experiments/metric/classifier_stability/2_evaluate_stability_experiment.sh $EXP_CLF_STAB_TR)
echo "  [Exp 02] Klassifikator-Stabilität & Kapazität: Training $EXP_CLF_STAB_TR -> Eval $EXP_CLF_STAB_EV"

# 2.3 Klassifikator-Sequenzlängen-Ablation (256, 512, 1024 Tokens)
EXP_CLFLEN_256=$(submit_job scripts/sbatch/experiments/metric/classifier_length/1_train_article_classifier_256.sh $JOB04 $JOB05)
EXP_CLFLEN_512=$(submit_job scripts/sbatch/experiments/metric/classifier_length/1_train_article_classifier_512.sh $JOB04 $JOB05)
EXP_CLFLEN_1024=$(submit_job scripts/sbatch/experiments/metric/classifier_length/1_train_article_classifier_1024.sh $JOB04 $JOB05)
EXP_CLFLEN_EV=$(submit_job scripts/sbatch/experiments/metric/classifier_length/2_evaluate_classifiers.sh $JOB04 $JOB07 $EXP_CLFLEN_256 $EXP_CLFLEN_512 $EXP_CLFLEN_1024)
echo "  [Exp 03] Klassifikator-Längenablation (256/512/1024): 3 Trainingsjobs -> Eval $EXP_CLFLEN_EV"

# 2.4 Regressor-Sequenzlängen-Ablation (256, 512, 1024 Tokens)
EXP_REGLEN_256=$(submit_job scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_256.sh $JOB05)
EXP_REGLEN_512=$(submit_job scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_512.sh $JOB05)
EXP_REGLEN_1024=$(submit_job scripts/sbatch/experiments/metric/regressor_length/1_train_mixup_regressor_1024.sh $JOB05)
EXP_REGLEN_EV=$(submit_job scripts/sbatch/experiments/metric/regressor_length/2_evaluate_regressors.sh $JOB04 $EXP_REGLEN_256 $EXP_REGLEN_512 $EXP_REGLEN_1024)
echo "  [Exp 04] Regressor-Längenablation (256/512/1024): 3 Trainingsjobs -> Eval $EXP_REGLEN_EV"

# 2.5 MixUp-Modell-Varianten (Statisch, Dynamisch, Hybrid, Hybrid Cyclic)
EXP_MXV_STAT=$(submit_job scripts/sbatch/experiments/metric/mixup_variants/1_train_mixup_static.sh $JOB05)
EXP_MXV_DYN=$(submit_job scripts/sbatch/experiments/metric/mixup_variants/1_train_mixup_dynamic.sh $JOB05)
EXP_MXV_HYB=$(submit_job scripts/sbatch/experiments/metric/mixup_variants/1_train_mixup_hybrid.sh $JOB05)
EXP_MXV_CYC=$(submit_job scripts/sbatch/experiments/metric/mixup_variants/1_train_mixup_hybrid_cyclic.sh $JOB05)
EXP_MXV_EV=$(submit_job scripts/sbatch/experiments/metric/mixup_variants/2_evaluate_mixup_variants.sh $JOB04 $JOB05 $EXP_MXV_STAT $EXP_MXV_DYN $EXP_MXV_HYB $EXP_MXV_CYC)
echo "  [Exp 05] MixUp-Modell-Varianten (4 Strategien): 4 Trainingsjobs -> Eval $EXP_MXV_EV"

# 2.6 Ähnlichkeitsschwellen-Ablation (0.60 vs. 0.70 vs. 0.80)
EXP_SIM_M060=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/1_train_mixup_060.sh $JOB04 $JOB05)
EXP_SIM_M070=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/1_train_mixup_070.sh $JOB04 $JOB05)
EXP_SIM_M080=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/1_train_mixup_080.sh $JOB04 $JOB05)
EXP_SIM_S060=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/2_train_sft_060.sh $JOB04 $JOB05 $JOB09)
EXP_SIM_S070=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/2_train_sft_070.sh $JOB04 $JOB05 $JOB09)
EXP_SIM_S080=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/2_train_sft_080.sh $JOB04 $JOB05 $JOB09)
EXP_SIM_EV=$(submit_job scripts/sbatch/experiments/metric/similarity_threshold/3_evaluate_all.sh $JOB05 $EXP_SIM_M060 $EXP_SIM_M070 $EXP_SIM_M080 $EXP_SIM_S060 $EXP_SIM_S070 $EXP_SIM_S080)
echo "  [Exp 06] Ähnlichkeitsschwellen (0.60/0.70/0.80): 6 Trainingsjobs -> Eval $EXP_SIM_EV"

# 2.7 Jina Kontextlängen-Ablation (benötigt Corpus Master)
EXP_CTX=$(submit_job scripts/sbatch/experiments/metric/context_length_ablation/1_run_context_length_ablation.sh $JOB05)
echo "  [Exp 07] Jina Kontextlängen-Ablation: Job ID $EXP_CTX"

# 2.8 Length Bias & Shortcut Analyse (benötigt Lebenshilfe, Corpus Master & Artikel-Klassifikator)
EXP_LENBIAS=$(submit_job scripts/sbatch/experiments/metric/length_bias/1_check_length_bias.sh $JOB04 $JOB05 $JOB08)
echo "  [Exp 08] Length-Bias & Shortcut Analyse: Job ID $EXP_LENBIAS"

# 2.9 TextComplexityDE Validierung (benötigt MixUp Regressor)
EXP_TCDE=$(submit_job scripts/sbatch/experiments/metric/textcomplexityde/1_evaluate_textcomplexityde.sh $JOB09)
echo "  [Exp 09a] TextComplexityDE Validierung: Job ID $EXP_TCDE"

# 2.9b MERLIN CEFR Validierung (benötigt MixUp Regressor)
EXP_MERLIN=$(submit_job scripts/sbatch/experiments/metric/merlin/1_evaluate_merlin.sh $JOB09)
echo "  [Exp 09b] MERLIN CEFR Validierung (Dokumentebene): Job ID $EXP_MERLIN"

# 2.10 RNN Baseline Modell (benötigt Corpus Master; Eval benötigt Lebenshilfe & MixUp-Modell)
EXP_RNN_TR=$(submit_job scripts/sbatch/experiments/metric/rnn_baseline/1_train_rnn_baseline.sh $JOB05)
EXP_RNN_EV=$(submit_job scripts/sbatch/experiments/metric/rnn_baseline/2_evaluate_rnn_baseline.sh $JOB04 $JOB09 $EXP_RNN_TR)
echo "  [Exp 10] RNN Baseline Regressor: Training $EXP_RNN_TR -> Eval $EXP_RNN_EV"

# 2.11 SFT Data Scaling Grid (benötigt Corpus Master & MixUp Regressor)
EXP_SFTSCAL_TR=$(submit_job scripts/sbatch/experiments/sft_scaling/1_train_sft_scaling_grid.sh $JOB05 $JOB09)
EXP_SFTSCAL_EV=$(submit_job scripts/sbatch/experiments/sft_scaling/2_evaluate_sft_scaling.sh $JOB04 $EXP_SFTSCAL_TR)
echo "  [Exp 11] SFT Data Scaling Grid: Training $EXP_SFTSCAL_TR -> Eval $EXP_SFTSCAL_EV"

# 2.12 MixUp Data Scaling Grid (benötigt Corpus Master & MixUp Vocab)
EXP_DATASCAL_M=$(submit_job scripts/sbatch/experiments/metric/data_scaling/1_scaling_mixtures_grid.sh $JOB05 $JOB09)
EXP_DATASCAL_A=$(submit_job scripts/sbatch/experiments/metric/data_scaling/2_scaling_articles_grid.sh $JOB05 $JOB09)
EXP_DATASCAL_EV=$(submit_job scripts/sbatch/experiments/metric/data_scaling/3_evaluate_scaling.sh $JOB04 $EXP_DATASCAL_M $EXP_DATASCAL_A)
echo "  [Exp 12] MixUp Data Scaling: Mixtures $EXP_DATASCAL_M, Articles $EXP_DATASCAL_A -> Eval $EXP_DATASCAL_EV"

# 2.13 Synthetischer Regressor Pipeline (LLM-Stufen & Regressor-Training / Eval)
EXP_SYN_1=$(submit_job scripts/sbatch/experiments/metric/synthetic_regressor/1_generate_synthetic_steps_lh.sh $JOB04)
EXP_SYN_2=$(submit_job scripts/sbatch/experiments/metric/synthetic_regressor/2_generate_synthetic_steps_corpus.sh $JOB05)
EXP_SYN_3=$(submit_job scripts/sbatch/experiments/metric/synthetic_regressor/3_train_synthetic_regressor.sh $EXP_SYN_1 $EXP_SYN_2)
EXP_SYN_4=$(submit_job scripts/sbatch/experiments/metric/synthetic_regressor/4_evaluate_synthetic_experiments.sh $JOB04 $JOB09 $EXP_SYN_3)
EXP_SYN_5=$(submit_job scripts/sbatch/experiments/metric/synthetic_regressor/5_evaluate_synthetic_kde.sh $JOB04 $JOB09 $EXP_SYN_3)
echo "  [Exp 13] Synthetischer Regressor Pipeline: 5 Jobs eingereiht (Final Eval: Job ID $EXP_SYN_4, $EXP_SYN_5)"

# 2.14 Token Length Experiment (256, 512, 1024)
EXP_TL_M256=$(submit_job scripts/sbatch/experiments/token_length/1_train_metric_256.sh $JOB05)
EXP_TL_M512=$(submit_job scripts/sbatch/experiments/token_length/1_train_metric_512.sh $JOB05)
EXP_TL_M1024=$(submit_job scripts/sbatch/experiments/token_length/1_train_metric_1024.sh $JOB05)

EXP_TL_S256=$(submit_job scripts/sbatch/experiments/token_length/2_train_sft_256.sh $EXP_TL_M256 $JOB05)
EXP_TL_S512=$(submit_job scripts/sbatch/experiments/token_length/2_train_sft_512.sh $EXP_TL_M512 $JOB05)
EXP_TL_S1024=$(submit_job scripts/sbatch/experiments/token_length/2_train_sft_1024.sh $EXP_TL_M1024 $JOB05)

EXP_TL_G256=$(submit_job scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_256.sh $EXP_TL_S256 $JOB06)
EXP_TL_G512=$(submit_job scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_512.sh $EXP_TL_S512 $JOB06)
EXP_TL_G1024=$(submit_job scripts/sbatch/experiments/token_length/3_generate_dpo_pairs_1024.sh $EXP_TL_S1024 $JOB06)

EXP_TL_MRG256=$(submit_job scripts/sbatch/experiments/token_length/3b_merge_dpo_pairs_256.sh $EXP_TL_G256)
EXP_TL_MRG512=$(submit_job scripts/sbatch/experiments/token_length/3b_merge_dpo_pairs_512.sh $EXP_TL_G512)
EXP_TL_MRG1024=$(submit_job scripts/sbatch/experiments/token_length/3b_merge_dpo_pairs_1024.sh $EXP_TL_G1024)

EXP_TL_D256=$(submit_job scripts/sbatch/experiments/token_length/4_train_dpo_256.sh $EXP_TL_MRG256)
EXP_TL_D512=$(submit_job scripts/sbatch/experiments/token_length/4_train_dpo_512.sh $EXP_TL_MRG512)
EXP_TL_D1024=$(submit_job scripts/sbatch/experiments/token_length/4_train_dpo_1024.sh $EXP_TL_MRG1024)

EXP_TL_EV=$(submit_job scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh $JOB04 $EXP_TL_D256 $EXP_TL_D512 $EXP_TL_D1024)
echo "  [Exp 14] Token Length End-to-End Ablation (256/512/1024): 16 Jobs eingereiht (Final Eval: Job ID $EXP_TL_EV)"

# 2.15 Rule Adherence Messung (benötigt Token-Length Evaluation & Corpus Master)
EXP_RULE=$(submit_job scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh $JOB05 $EXP_TL_EV)
echo "  [Exp 15] Regel-Adhärenz Evaluation: Job ID $EXP_RULE"

# =============================================================================
# TEIL 3: MODELING- & ALIGNMENT-EXPERIMENTE
# =============================================================================
echo ""
echo ">>> [3/4] Reiche Modell- & Alignment-Experimente ein..."

# 3.1 Decoder-Only Pipeline (Qwen 2.5 1.5B)
DEC_SFT=$(submit_job scripts/sbatch/experiments/decoder_only/1_train_sft_decoder_only.sh $JOB05)
DEC_GEN=$(submit_job scripts/sbatch/experiments/decoder_only/2_generate_dpo_pairs_decoder_only_array.sh $DEC_SFT $JOB06 $JOB09)
DEC_MRG=$(submit_job scripts/sbatch/experiments/decoder_only/2b_merge_dpo_pairs_decoder_only.sh $DEC_GEN)
DEC_DPO=$(submit_job scripts/sbatch/experiments/decoder_only/3_train_dpo_decoder_only.sh $DEC_MRG $DEC_SFT)
DEC_EV=$(submit_job scripts/sbatch/experiments/decoder_only/4_evaluate_decoder_only.sh $JOB04 $DEC_SFT $DEC_DPO)
echo "  [Exp 16] Decoder-Only (Qwen 2.5): SFT $DEC_SFT -> DPO Gen $DEC_GEN -> Merge $DEC_MRG -> DPO $DEC_DPO -> Eval $DEC_EV"

# 3.2 Metric Weights DPO Experiment (0.5/0.5, 0.7/0.3, 1.0/0.0)
MW_G05=$(submit_job scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w05_w05.sh $JOB06 $JOB09 $JOB10)
MW_G07=$(submit_job scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w07_w03.sh $JOB06 $JOB09 $JOB10)
MW_G10=$(submit_job scripts/sbatch/experiments/metric_weights/1_generate_dpo_pairs_w10_w00.sh $JOB06 $JOB09 $JOB10)

MW_MRG05=$(submit_job scripts/sbatch/experiments/metric_weights/1b_merge_dpo_pairs_w05_w05.sh $MW_G05)
MW_MRG07=$(submit_job scripts/sbatch/experiments/metric_weights/1b_merge_dpo_pairs_w07_w03.sh $MW_G07)
MW_MRG10=$(submit_job scripts/sbatch/experiments/metric_weights/1b_merge_dpo_pairs_w10_w00.sh $MW_G10)

MW_D05=$(submit_job scripts/sbatch/experiments/metric_weights/2_train_dpo_w05_w05.sh $MW_MRG05)
MW_D07=$(submit_job scripts/sbatch/experiments/metric_weights/2_train_dpo_w07_w03.sh $MW_MRG07)
MW_D10=$(submit_job scripts/sbatch/experiments/metric_weights/2_train_dpo_w10_w00.sh $MW_MRG10)

MW_EV=$(submit_job scripts/sbatch/experiments/metric_weights/3_run_full_evaluation.sh $JOB04 $MW_D05 $MW_D07 $MW_D10)
echo "  [Exp 17] Metric Weighting Experiment: 10 Jobs eingereiht (Final Eval: Job ID $MW_EV)"

# 3.3 DPO Beta Sweep (0.01, 0.05, 0.10, 0.20, 0.50)
BETA_001=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_001.sh $JOB10 $JOB11B)
BETA_005=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_005.sh $JOB10 $JOB11B)
BETA_010=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_010.sh $JOB10 $JOB11B)
BETA_020=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_020.sh $JOB10 $JOB11B)
BETA_050=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/1_train_dpo_beta_050.sh $JOB10 $JOB11B)
BETA_EV=$(submit_job scripts/sbatch/experiments/dpo_beta_sweep/2_run_full_evaluation.sh $JOB04 $BETA_001 $BETA_005 $BETA_010 $BETA_020 $BETA_050)
echo "  [Exp 18] DPO Beta Sweep (5 Modelle): 5 Trainingsjobs -> Eval $BETA_EV"

# 3.4 PPO Alignment (Seq2Seq & Decoder-Only)
PPO_DEC=$(submit_job scripts/sbatch/experiments/ppo/1_train_ppo_decoder_only.sh $DEC_SFT $JOB09)
PPO_SEQ=$(submit_job scripts/sbatch/experiments/ppo/1_train_ppo_seq2seq.sh $JOB09 $JOB10)
PPO_EV=$(submit_job scripts/sbatch/experiments/ppo/2_evaluate_all_ppo.sh $JOB04 $PPO_DEC $PPO_SEQ)
echo "  [Exp 19] PPO Alignment: Decoder $PPO_DEC, Seq2Seq $PPO_SEQ -> Eval $PPO_EV"

# =============================================================================
# TEIL 4: GRAND MASTER 5-WEGE / 7-WEGE BENCHMARK
# =============================================================================
echo ""
echo ">>> [4/4] Reiche Grand Master Benchmark ein..."

# 4.1 Master 5-Wege Benchmark (benötigt Lebenshilfe, Regressor, mBART SFT/DPO & Qwen SFT/DPO)
EXP_BENCH=$(submit_job scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh $JOB04 $JOB09 $JOB10 $JOB12 $DEC_SFT $DEC_DPO)
echo "  [Exp 20] Master 5-Wege Benchmark: Job ID $EXP_BENCH"

# 4.2 Experten-Evaluationspool (50 Items, 10 Nicht-Lebenshilfe Domänen, verblindet)
EXP_EXPERT=$(submit_job scripts/sbatch/experiments/expert_eval/1_build_expert_dataset.sh $JOB04 $JOB05 $JOB09 $EXP_TL_D1024)
echo "  [Experten-Eval] 50-Item Blindstudien-Pool: Job ID $EXP_EXPERT"

echo ""
echo "========================================================================"
echo " Alle Pipeline-Schritte & 20 Experimente erfolgreich eingereiht!"
echo "========================================================================"
echo " Slurm Queue überwachen:   squeue -u \$USER"
echo " Live-Logs Hauptpipeline:  tail -f results/logs/run_pipeline/*.out"
echo " Live-Logs Experimente:    tail -f results/logs/experiments/*/*.out"
echo "========================================================================"
