"""Mondrian (size-conditional) Conformal Risk Control runner.

Compares Pooled vs Size-conditional CRC screening across board-size bins:
  - 2-4: 2 <= n_total <= 4
  - 5-8: 5 <= n_total <= 8
  - 9-14: 9 <= n_total <= 14

Usage:
    python scripts/run_mondrian.py [--data data/kluster_dataset_1000.csv] [--repeats 20] [--jobs -1] [--full-tuning]
"""
import sys, pathlib, time, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from kluster import mondrian as md
from kluster.features import featurize

RESULTS_DIR = ROOT / "results"
DEFAULT_DATA = "data/kluster_dataset_1000.csv"


def main():
    """Execute Mondrian (size-conditional) Conformal Risk Control across repeated test splits.

    Loads the dataset, extracts features, runs Mondrian calibration across board-size bins
    (2-4, 5-8, 9-14) for specified risk tolerances alpha, and saves results to:
      - results/mondrian_risk_check.csv (empirical wrong-skip rates by bin, forming paper Table T1)
      - results/mondrian_paired.csv (paired ML vs Combined rule differences, forming paper Table T2)
      - results/mondrian_splits.csv (raw per-split metrics)
    """
    parser = argparse.ArgumentParser(description="Mondrian (size-conditional) CRC screening runner.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to dataset CSV (default: data/kluster_dataset_1000.csv)")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repeated splits (default: 20)")
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

    print(f"[1/4] Loading dataset from {csv_path} and featurizing...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    X = featurize(df)
    y = (df.label == 'contact').astype(int).values
    print(f"      Loaded {len(df)} candidates across {df.game_id.nunique()} games in {time.time() - t0:.1f}s.")

    print(f"[2/4] Running {args.repeats} repeated splits (train/val/cal/test: 50/15/20/15)...")
    t1 = time.time()
    df_summary, df_raw, df_paired, df_risk_check = md.run_repeated_mondrian(
        df, X, y,
        alphas=md.DEFAULT_ALPHAS,
        n_repeats=args.repeats,
        n_jobs=args.jobs,
        fast=not args.full_tuning,
        return_all=True,
    )
    print(f"      Completed {args.repeats} splits across 8 models, 3 bins, 3 alphas in {time.time() - t1:.1f}s.")

    print("[3/4] Generating comparison between Pooled and Size-conditional calibration...")
    df_pooled = df_summary[df_summary['strategy'] == 'Pooled'].copy().reset_index(drop=True)
    df_cond = df_summary[df_summary['strategy'] == 'Size-conditional'].copy().reset_index(drop=True)

    # Build side-by-side comparison table
    df_comp = pd.merge(
        df_pooled[['bin', 'model', 'alpha', 'n_cal_games', 'skipped_fraction_mean', 'skip_lo', 'skip_hi',
                   'wrong_skip_rate_mean', 'wrong_skip_lo', 'wrong_skip_hi', 'fraction_exceeding_alpha']],
        df_cond[['bin', 'model', 'alpha', 'n_cal_games', 'skipped_fraction_mean', 'skip_lo', 'skip_hi',
                 'wrong_skip_rate_mean', 'wrong_skip_lo', 'wrong_skip_hi', 'fraction_exceeding_alpha']],
        on=['bin', 'model', 'alpha'],
        suffixes=('_pooled', '_cond')
    )
    df_comp['skip_diff_pct'] = ((df_comp['skipped_fraction_mean_cond'] - df_comp['skipped_fraction_mean_pooled']) * 100.0).round(2)
    df_comp['wrong_diff'] = (df_comp['wrong_skip_rate_mean_cond'] - df_comp['wrong_skip_rate_mean_pooled']).round(4)

    print("[4/4] Saving results to results/mondrian_*.csv...")
    df_summary.to_csv(RESULTS_DIR / "mondrian_summary.csv", index=False)
    df_pooled.to_csv(RESULTS_DIR / "mondrian_pooled.csv", index=False)
    df_cond.to_csv(RESULTS_DIR / "mondrian_conditional.csv", index=False)
    df_comp.to_csv(RESULTS_DIR / "mondrian_comparison.csv", index=False)
    df_paired.to_csv(RESULTS_DIR / "mondrian_paired.csv", index=False)
    df_risk_check.to_csv(RESULTS_DIR / "mondrian_risk_check.csv", index=False)
    df_raw.to_csv(RESULTS_DIR / "mondrian_splits.csv", index=False)

    print(f"      Saved {RESULTS_DIR / 'mondrian_summary.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_pooled.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_conditional.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_comparison.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_paired.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_risk_check.csv'}")
    print(f"      Saved {RESULTS_DIR / 'mondrian_splits.csv'}")

    # Display clean paired comparison highlights
    print("\n--- Paired Comparison: Best ML vs Combined Rule (Size-conditional) ---")
    paired_cols = ['bin', 'alpha', 'best_ml_model', 'best_ml_skip_mean', 'combined_skip_mean', 'mean_diff', 'diff_lo', 'diff_hi']
    print(df_paired[paired_cols].to_string(index=False))

    # Display clean risk check highlights
    print("\n--- Risk Check Highlights (Size-conditional, alpha = 0.05) ---")
    sub_rc = df_risk_check[df_risk_check['alpha'] == 0.05]
    display_rc_cols = ['bin', 'model', 'alpha', 'mean_wrong_skip_rate', 'mean_minus_alpha', 'std_err', 'fraction_exceeding_alpha']
    print(sub_rc[display_rc_cols].to_string(index=False))

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f}s.")


if __name__ == "__main__":
    main()

