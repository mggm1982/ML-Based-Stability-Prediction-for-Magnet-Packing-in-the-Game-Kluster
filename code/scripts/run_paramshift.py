"""Runner for Phase 5 parameter-shift experiments (Experiments A, B, C).

Usage:
    python scripts/run_paramshift.py [--data-old data/kluster_dataset.csv] [--data-v2 data/kluster_dataset_v2.csv] [--repeats 20] [--jobs -1]
"""
import sys, pathlib, time, argparse
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from kluster import paramshift as ps
from kluster.features import featurize

RESULTS_DIR = ROOT / "results"
DEFAULT_OLD = "data/kluster_dataset.csv"
DEFAULT_V2 = "data/kluster_dataset_v2.csv"


def main():
    """Execute parameter-shift experiments A, B, and C under physical friction and force variation.

    Evaluates conformal screening robustness across ratio groups (q < 0.8, 0.8 <= q <= 1.25, q > 1.25):
      - Experiment A: Fixed-parameter training evaluated under shifted testing parameters.
      - Experiment B: Randomized-parameter training with parameter features (PHYS_PARAMS).
      - Experiment C: Parameter-aware critical-distance analytical heuristic.
    Saves outputs to results/paramshift_A.csv, results/paramshift_B.csv, and results/paramshift_C.csv
    (forming paper Table T3).
    """
    parser = argparse.ArgumentParser(description="Parameter-shift conformal screening runner.")
    parser.add_argument("--data-old", type=str, default=DEFAULT_OLD, help="Path to old default-parameters dataset CSV")
    parser.add_argument("--data-v2", type=str, default=DEFAULT_V2, help="Path to v2 randomized-parameters dataset CSV")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repeated splits (default: 20)")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs (default: -1)")
    parser.add_argument("--full-tuning", action="store_true", help="Use full hyperparameter grid instead of fixed params")
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path_old = pathlib.Path(args.data_old)
    if not path_old.is_absolute():
        path_old = ROOT / path_old
    if not path_old.exists():
        raise FileNotFoundError(f"Old dataset not found at {path_old}")

    path_v2 = pathlib.Path(args.data_v2)
    if not path_v2.is_absolute():
        path_v2 = ROOT / path_v2
    if not path_v2.exists():
        raise FileNotFoundError(f"New dataset not found at {path_v2}")

    print(f"[1/4] Loading and featurizing datasets...")
    t0 = time.time()
    df_old = pd.read_csv(path_old)
    X_old = featurize(df_old)
    y_old = (df_old.label == 'contact').astype(int).values
    print(f"      Loaded old dataset: {len(df_old)} rows, {df_old.game_id.nunique()} games.")

    df_v2 = pd.read_csv(path_v2)
    X_v2 = featurize(df_v2)
    y_v2 = (df_v2.label == 'contact').astype(int).values
    print(f"      Loaded v2 dataset: {len(df_v2)} rows, {df_v2.game_id.nunique()} games.")
    print(f"      Data preparation completed in {time.time() - t0:.1f} s.")

    print(f"[2/4] Running {args.repeats} repeated splits for Experiments A, B, and C...")
    t1 = time.time()
    df_A, df_B, df_C, raw_A, raw_B, raw_C = ps.run_repeated_paramshift(
        df_old, X_old, y_old,
        df_v2, X_v2, y_v2,
        alphas=ps.DEFAULT_ALPHAS,
        n_repeats=args.repeats,
        n_jobs=args.jobs,
        fast=not args.full_tuning,
    )
    print(f"      All experiments completed in {time.time() - t1:.1f} s.")

    print(f"[3/4] Writing output files to results/paramshift_*.csv...")
    file_A = RESULTS_DIR / "paramshift_A.csv"
    file_B = RESULTS_DIR / "paramshift_B.csv"
    file_C = RESULTS_DIR / "paramshift_C.csv"

    df_A.to_csv(file_A, index=False)
    df_B.to_csv(file_B, index=False)
    df_C.to_csv(file_C, index=False)

    print(f"      Saved {file_A}")
    print(f"      Saved {file_B}")
    print(f"      Saved {file_C}")

    print(f"\n[4/4] Summary Highlights (alpha = 0.05):")
    print(f"\n--- Experiment A: Fixed to Random (Trained on Old, Tested on V2 Groups) ---")
    sub_A = df_A[df_A['alpha'] == 0.05]
    print(sub_A[['group', 'model', 'alpha', 'mean_wrong_skip_rate', 'std_err', 'mean_skip_rate', 'n_games']].to_string(index=False))

    print(f"\n--- Experiment B: Random Trained (Trained & Tested on V2 with PHYS_PARAMS) ---")
    sub_B = df_B[df_B['alpha'] == 0.05]
    print(sub_B[['group', 'model', 'alpha', 'mean_wrong_skip_rate', 'std_err', 'mean_skip_rate', 'n_games']].to_string(index=False))

    print(f"\n--- Experiment C: Parameter-Aware Critical-Distance Rule ---")
    sub_C = df_C[df_C['alpha'] == 0.05]
    print(sub_C[['group', 'model', 'alpha', 'mean_wrong_skip_rate', 'std_err', 'mean_skip_rate', 'n_games']].to_string(index=False))

    total_time = time.time() - t_start
    print(f"\nDone in {total_time:.1f} s.")


if __name__ == "__main__":
    main()

