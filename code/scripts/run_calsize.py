"""Runner for calibration-set size scaling experiment on kluster_dataset_v2.csv.

Usage:
    python scripts/run_calsize.py [--repeats 50] [--jobs -1] [--data data/kluster_dataset_v2.csv]
"""
import sys
import pathlib
import time
import argparse

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from kluster import calsize as cs
from kluster.features import featurize

RESULTS_DIR = ROOT / "results"
DEFAULT_DATA = "data/kluster_dataset_v2.csv"
DEFAULT_OUT = "results/calsize.csv"


def main():
    """Execute calibration sample size study evaluating n_cal in {20, 40, 80, 160, 320} games.

    Evaluates finite-sample risk slack (gap below alpha vs theoretical 1/(n_cal+1) bound)
    and documents abstention behavior for small calibration sets at tight risk targets.
    Saves results to results/calsize.csv (forming paper Table T4).
    """
    parser = argparse.ArgumentParser(description="Calibration-set size scaling runner.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to v2 dataset CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Path to output results CSV")
    parser.add_argument("--repeats", type=int, default=50, help="Number of repeated splits (default: 50)")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs (default: -1)")
    parser.add_argument("--full-tuning", action="store_true", help="Use full hyperparameter grid instead of fixed params")
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = pathlib.Path(args.data)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    print(f"[1/4] Loading and featurizing dataset from {csv_path}...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    X = featurize(df)
    y = (df.label == 'contact').astype(int).values
    print(f"      Loaded {len(df)} candidates across {df.game_id.nunique()} games in {time.time() - t0:.1f} s.")

    print(f"[2/4] Running {args.repeats} repeated splits across n_cal in {cs.CAL_SIZES} and 4 methods...")
    t1 = time.time()
    df_summary = cs.run_repeated_calsize(
        df, X, y,
        cal_sizes=cs.CAL_SIZES,
        alphas=cs.DEFAULT_ALPHAS,
        n_repeats=args.repeats,
        n_test_games=225,
        train_fraction=0.50,
        n_jobs=args.jobs,
        fast=not args.full_tuning,
    )
    print(f"      Completed {args.repeats} repeats across {len(df_summary)} configurations in {time.time() - t1:.1f} s.")

    out_path = pathlib.Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[3/4] Writing output to {out_path}...")
    df_summary.to_csv(out_path, index=False)
    print(f"      Saved {out_path} ({len(df_summary)} rows).")

    print("\n[4/4] Summary Highlights (alpha = 0.05, bin = 5-8):")
    sub = df_summary[(df_summary['alpha'] == 0.05) & (df_summary['bin'] == '5-8')]
    disp_cols = ['n_cal', 'method', 'bin', 'wrong_skip_bin', 'wrong_skip_pooled', 'skip_rate', 'std_err', 'n_bin_cal_mean', 'slack_theory', 'frac_exceed']
    print(sub[disp_cols].to_string(index=False))

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f} s.")


if __name__ == "__main__":
    main()

