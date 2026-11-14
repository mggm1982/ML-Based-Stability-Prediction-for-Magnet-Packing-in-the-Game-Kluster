"""Runner for neural network baseline and paired comparison.

Usage:
    python scripts/run_nn.py [--data data/kluster_dataset_v2.csv] [--repeats 20] [--jobs -1]
"""
import sys
import pathlib
import time
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from kluster import nnbase as nb
from kluster.features import featurize

RESULTS_DIR = ROOT / "results"
DEFAULT_DATA = "data/kluster_dataset_v2.csv"


def main():
    """Execute Multi-Layer Perceptron (MLP 32, 16) neural network baseline benchmark.

    Trains the neural network on randomized physical parameters and compares per-bin
    screening skip rates and wall-clock training times against Histogram Gradient Boosting
    and Logistic Regression. Saves results to results/nn_baseline.csv (forming paper Table T5).
    """
    parser = argparse.ArgumentParser(description="Neural network Mondrian CRC baseline runner.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to dataset CSV (default: data/kluster_dataset_v2.csv)")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repeated splits (default: 20)")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs (default: -1)")
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = pathlib.Path(args.data)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print("[1/4] Loading and featurizing dataset...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    X = featurize(df)
    y = (df['label'] == 'contact').astype(int).values
    print(f"      Loaded {len(df)} candidates across {df['game_id'].nunique()} games in {time.time() - t0:.1f} s.")

    print(f"[2/4] Running {args.repeats} repeated splits across 3 models, 3 bins, 3 alphas...")
    t1 = time.time()
    df_baseline, df_paired = nb.run_repeated_nn(
        df, X, y,
        alphas=nb.DEFAULT_ALPHAS,
        n_repeats=args.repeats,
        n_jobs=args.jobs,
    )
    print(f"      Completed {args.repeats} repeats in {time.time() - t1:.1f} s.")

    print("[3/4] Writing output files to results/...")
    file_baseline = RESULTS_DIR / "nn_baseline.csv"
    file_paired = RESULTS_DIR / "nn_paired.csv"

    df_baseline.to_csv(file_baseline, index=False)
    df_paired.to_csv(file_paired, index=False)
    print(f"      Saved {file_baseline}")
    print(f"      Saved {file_paired}")

    print("\n[4/4] Summary Highlights:")
    print("\n--- Baseline Screening Performance (alpha = 0.05) ---")
    sub_base = df_baseline[df_baseline['alpha'] == 0.05]
    cols_base = ['model', 'alpha', 'bin', 'wrong_skip_bin', 'wrong_skip_pooled', 'mean_skip_rate', 'std_err', 'train_time_s']
    print(sub_base[cols_base].to_string(index=False))

    print("\n--- Paired Difference in Skip Rate: Neural network vs Gradient boosting (+parameters) ---")
    cols_pair = ['bin', 'alpha', 'nn_skip_mean', 'hgb_skip_mean', 'mean_diff', 'diff_lo', 'diff_hi']
    print(df_paired[cols_pair].to_string(index=False))

    print(f"\nDone in {time.time() - t_start:.1f} s.")


if __name__ == "__main__":
    main()

