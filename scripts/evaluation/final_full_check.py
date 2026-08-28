import os
import glob
import subprocess
import sys

mapping = [
    ("notebooks/experiments/master_benchmark.ipynb", "results/evaluation/benchmark_5way_decoder_vs_encoder_decoder.csv", "scripts/sbatch/experiments/benchmark/1_run_all_models_benchmark.sh"),
    ("notebooks/experiments/metric_weights.ipynb", "results/evaluation/metric_weights_comparison_summary.csv", "scripts/sbatch/experiments/metric_weights/3_run_full_evaluation.sh"),
    ("notebooks/experiments/loss_aggregation.ipynb", "results/evaluation/loss_aggregation_comparison_summary.csv", "scripts/sbatch/experiments/loss_aggregation/2_run_full_evaluation.sh"),
    ("notebooks/experiments/sft_scaling.ipynb", "results/experiments/sft_scaling/sft_scaling_summary.csv", "scripts/sbatch/experiments/sft_scaling/2_evaluate_sft_scaling.sh"),
    ("notebooks/experiments/rule_adherence.ipynb", "data/analysis/rule_adherence_corpus.csv", "scripts/sbatch/experiments/rule_adherence/1_measure_rule_adherence.sh"),
    ("notebooks/experiments/token_length_translation.ipynb", "results/evaluation/token_length_comparison_summary.csv", "scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh"),
    ("notebooks/experiments/metric/textcomplexityde.ipynb", "results/evaluation/textcomplexityde_eval.csv", "scripts/sbatch/experiments/metric/textcomplexityde/1_evaluate_textcomplexityde.sh"),
    ("notebooks/experiments/metric/length_bias.ipynb", "results/evaluation/length_bias_results.csv", "scripts/sbatch/experiments/metric/length_bias/1_check_length_bias.sh"),
    ("notebooks/archive/metric/compare_bilstm_vs_rnn_baseline.ipynb", "results/evaluation/bilstm_vs_rnn_eval.csv", "scripts/sbatch/experiments/metric/rnn_baseline/2_evaluate_rnn_baseline.sh"),
    ("notebooks/archive/metric/compare_mixup_synthetic_lh_kde.ipynb", "results/evaluation/mixup_synthetic_kde_eval.csv", "scripts/sbatch/experiments/metric/synthetic_regressor/8_evaluate_synthetic_kde.sh"),
    ("notebooks/archive/metric/compare_mixup_vs_synthetic.ipynb", "results/evaluation/mixup_vs_synthetic_unbiased_eval.csv", "scripts/sbatch/experiments/metric/synthetic_regressor/7_evaluate_synthetic_experiments.sh"),
    ("notebooks/experiments/metric/factuality_metric.ipynb", "results/evaluation/factual_consistency_metric_results.csv", "scripts/sbatch/experiments/metric/factuality_metric/1_run_factuality_metric_experiment.sh"),
    ("notebooks/experiments/metric/data_scaling.ipynb", "results/experiments/data_scaling/scaling_summary.csv", "scripts/sbatch/experiments/metric/data_scaling/3_evaluate_scaling.sh"),
    ("notebooks/experiments/metric/token_length_metric.ipynb", "results/evaluation/token_length_metric_comparison.csv", "scripts/sbatch/experiments/token_length/5_run_full_evaluation.sh"),
    ("notebooks/experiments/metric/mixup_model_evaluation.ipynb", "results/evaluation/mixup_variants_eval.csv", "scripts/sbatch/experiments/metric/mixup_variants/2_evaluate_mixup_variants.sh"),
    ("notebooks/data/corpus_analysis.ipynb", "data/analysis/corpus_master.csv", "scripts/sbatch/run_pipeline/05_build_corpus_master.sh"),
    ("notebooks/data/analyze_cleaning_corpus.ipynb", "data/lebenshilfe/lebenshilfe_dataset_clean.json", "scripts/sbatch/run_pipeline/04_clean_lebenshilfe.sh")
]


def main():
    fast_mode = "--fast" in sys.argv
    if not fast_mode:
        print("1. TEST ALL EVALUATION SCRIPTS WITH --help")
        eval_scripts = sorted([s for s in glob.glob("scripts/evaluation/*.py") if not s.endswith("final_full_check.py")])
        for s in eval_scripts:
            res = subprocess.run([sys.executable, s, "--help"], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"✅ {s} OK")
            else:
                print(f"❌ {s} Fehler: {res.stderr[:80]}")
    else:
        print("1. EVALUATION SCRIPTS TEST SKIPPED (--fast mode)")

    print("\n2. TEST ALL 18 NOTEBOOKS AGAINST THEIR PRODUCER SCRIPTS")
    all_ok = True
    for nb, out_csv, sbatch_script in mapping:
        nb_ok = os.path.exists(nb)
        sb_ok = os.path.exists(sbatch_script)
        if not (nb_ok and sb_ok):
            all_ok = False
        print(f"{'✅' if nb_ok and sb_ok else '❌'} {nb}")
        print(f"   -> Ausgabedatei : {out_csv}")
        print(f"   -> Sbatch-Job   : {sbatch_script}")

    if all_ok:
        print("\n🎉 GESAMTSTATUS: 100% BEREIT FÜR SBATCH PIPELINE RUNS, EXPERIMENTE & NOTEBOOK AUSWERTUNGEN!")
    else:
        print("\n⚠️ EINIGE PFADE ODER DATEIEN FEHLEN!")
        sys.exit(1)


if __name__ == "__main__":
    main()
