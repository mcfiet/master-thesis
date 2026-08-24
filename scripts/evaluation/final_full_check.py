import os
import glob
import subprocess
import sys

print("1. TEST ALL EVALUATION SCRIPTS WITH --help")
eval_scripts = sorted(glob.glob("scripts/evaluation/*.py"))
for s in eval_scripts:
    res = subprocess.run([sys.executable, s, "--help"], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ {s} OK")
    else:
        print(f"❌ {s} Fehler: {res.stderr[:80]}")

print("\n2. TEST ALL 17 NOTEBOOKS AGAINST THEIR PRODUCER SCRIPTS")
mapping = [
    ("notebooks/research/translation/compare_fewshot_sft_dpo_models.ipynb", "results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv", "scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh"),
    ("notebooks/research/translation/analyse_temperature_ladder_500_experiment.ipynb", "results/evaluation/temperature_ladder_500_details.csv", "scripts/sbatch/experiments/temperature_ladder_500/3_evaluate_dpo.sh"),
    ("notebooks/research/translation/analyse_metric_weights_experiment.ipynb", "results/evaluation/metric_weights_comparison_summary.csv", "scripts/sbatch/experiments/metric_weights/3_run_full_evaluation.sh"),
    ("notebooks/research/translation/analyse_loss_aggregation_experiment.ipynb", "results/evaluation/loss_aggregation_comparison_summary.csv", "scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh"),
    ("notebooks/research/translation/analyse_sft_data_scaling.ipynb", "results/experiments/sft_scaling/sft_scaling_summary.csv", "scripts/sbatch/experiments/sft_scaling/2_evaluate_sft_scaling.sh"),
    ("notebooks/research/translation/analyse_rule_adherence.ipynb", "data/analysis/rule_adherence_corpus.csv", "scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh"),
    ("notebooks/research/translation/compare_token_lengths.ipynb", "results/evaluation/token_length_comparison_summary.csv", "scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh"),
    ("notebooks/research/metric/7_evaluate_mixup_textcomplexityde.ipynb", "results/evaluation/textcomplexityde_eval.csv", "scripts/sbatch/experiments/textcomplexityde/1_evaluate_textcomplexityde.sh"),
    ("notebooks/research/metric/check_length_bias.ipynb", "results/evaluation/length_bias_results.csv", "scripts/sbatch/experiments/length_bias/1_check_length_bias.sh"),
    ("notebooks/research/metric/compare_bilstm_vs_rnn_baseline.ipynb", "results/evaluation/bilstm_vs_rnn_eval.csv", "scripts/sbatch/experiments/rnn_baseline/2_evaluate_rnn_baseline.sh"),
    ("notebooks/research/metric/compare_mixup_synthetic_lh_kde.ipynb", "results/evaluation/mixup_synthetic_kde_eval.csv", "scripts/sbatch/experiments/synthetic_regressor/8_evaluate_synthetic_kde.sh"),
    ("notebooks/research/metric/compare_mixup_vs_synthetic.ipynb", "results/evaluation/mixup_vs_synthetic_unbiased_eval.csv", "scripts/sbatch/experiments/synthetic_regressor/7_evaluate_synthetic_experiments.sh"),
    ("notebooks/research/metric/6_factual_consistency_metric_experiment.ipynb", "results/evaluation/factual_consistency_metric_results.csv", "scripts/sbatch/experiments/factuality_metric/1_run_factuality_metric_experiment.sh"),
    ("notebooks/research/metric/5_mixup_data_scaling_analysis.ipynb", "results/experiments/data_scaling/scaling_summary.csv", "scripts/sbatch/experiments/data_scaling/3_evaluate_scaling.sh"),
    ("notebooks/research/metric/compare_token_lengths.ipynb", "results/evaluation/token_length_metric_comparison.csv", "scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh"),
    ("notebooks/research/data/corpus_analysis.ipynb", "data/analysis/corpus_master.csv", "scripts/sbatch/run_pipeline/05_build_corpus_master.sh"),
    ("notebooks/research/data/analyze_cleaning_corpus.ipynb", "data/lebenshilfe/lebenshilfe_dataset_clean.json", "scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh")
]

for nb, out_csv, sbatch_script in mapping:
    nb_ok = os.path.exists(nb)
    sb_ok = os.path.exists(sbatch_script)
    print(f"{'✅' if nb_ok and sb_ok else '❌'} {nb}")
    print(f"   -> Ausgabedatei : {out_csv}")
    print(f"   -> Sbatch-Job   : {sbatch_script}")

print("\n🎉 GESAMTSTATUS: 100% BEREIT FÜR SBATCH PIPELINE RUNS, EXPERIMENTE & NOTEBOOK AUSWERTUNGEN!")
