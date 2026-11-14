"""Unit tests for calibration-set size scaling analysis.

Run:  python tests/test_calsize.py   (all lines must print PASS)
"""
import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from kluster import calsize as cs
from kluster import mondrian as md
from kluster.features import featurize


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


ok = True

# -------------------------------------------------------------------------
# 1. Theoretical Slack on Toy Case: slack equals 1 / (n + 1)
# -------------------------------------------------------------------------
for n in [10, 20, 40, 80, 160, 320]:
    expected_slack = 1.0 / (n + 1.0)
    calc_slack = cs.theoretical_slack(n)
    ok &= check(f"theoretical slack equals 1/(n+1) on toy case n={n}",
                abs(calc_slack - expected_slack) < 1e-12)

# Check abstention threshold: when n_bin + 1 < 1 / alpha (i.e. alpha < 1 / (n_bin + 1))
toy_y = np.array([0, 0, 1, 0, 1])
toy_g = np.array([1, 2, 2, 3, 3])  # 3 games
toy_scores = np.array([0.1, 0.4, 0.8, 0.2, 0.9])
# For 3 games, 1 / (3 + 1) = 0.25. If alpha = 0.10, alpha < 0.25 -> must abstain (tau = inf)
c_abstain = md.calibrate_crc_bin(toy_scores, toy_y, toy_g, alpha=0.10)
ok &= check("abstention flags feasible=False when n_bin + 1 < 1/alpha", not c_abstain['feasible'])
ok &= check("abstention returns tau=inf when n_bin + 1 < 1/alpha", np.isinf(c_abstain['tau']))

eval_abstain = md.evaluate_bin_test(toy_scores, toy_y, toy_g, tau=c_abstain['tau'])
ok &= check("evaluating abstention yields skip_rate 0.0", eval_abstain['fraction_skipped'] == 0.0)
ok &= check("evaluating abstention yields wrong_skip_rate 0.0", eval_abstain['wrong_skip_rate'] == 0.0)

# -------------------------------------------------------------------------
# 2. Subsample of 150 Games: Split, Disjointness, and Calibration Size
# -------------------------------------------------------------------------
csv_v2 = ROOT / "data" / "kluster_dataset_v2.csv"
if not csv_v2.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_v2}")

df_all_v2 = pd.read_csv(csv_v2)
unique_150_games = df_all_v2['game_id'].unique()[:150]
df_150 = df_all_v2[df_all_v2['game_id'].isin(unique_150_games)].copy().reset_index(drop=True)

ok &= check("subsample has exactly 150 unique games", df_150['game_id'].nunique() == 150)

# Check split disjointness: train (75), pool (52), test (23) for 150 games
tr_g, pool_g, te_g = cs.split_games_calsize(df_150, seed=7, n_test=23, train_fraction=0.50)
ok &= check("train, pool, and test partition 150 games",
            len(tr_g) + len(pool_g) + len(te_g) == 150)
ok &= check("train and test games strictly disjoint", len(set(tr_g) & set(te_g)) == 0)
ok &= check("train and pool games strictly disjoint", len(set(tr_g) & set(pool_g)) == 0)
ok &= check("pool and test games strictly disjoint", len(set(pool_g) & set(te_g)) == 0)

# Check draw_calibration_games: size equals requested n_cal
for n_cal_req in [20, 40]:
    cal_g = cs.draw_calibration_games(pool_g, n_cal_req, seed=42)
    ok &= check(f"calibration set size equals the requested n_cal ({n_cal_req})",
                len(cal_g) == n_cal_req)
    ok &= check(f"no game shared between calibration and test for n_cal={n_cal_req}",
                len(set(cal_g) & set(te_g)) == 0)

# Verify disjointness between calibration and test holds across 20 distinct seeds
all_disjoint_20 = True
all_sizes_correct = True
for s in range(20):
    t_tr, t_pool, t_te = cs.split_games_calsize(df_150, seed=s, n_test=23, train_fraction=0.50)
    for n_cal_req in [20, 40]:
        t_cal = cs.draw_calibration_games(t_pool, n_cal_req, seed=s + 100)
        if len(set(t_cal) & set(t_te)) > 0:
            all_disjoint_20 = False
        if len(t_cal) != n_cal_req:
            all_sizes_correct = False

ok &= check("no game shared between calibration and test across 20 seeds", all_disjoint_20)
ok &= check("calibration set size equals requested n_cal across 20 seeds", all_sizes_correct)

# -------------------------------------------------------------------------
# 3. Whole Subsample End-to-End Run in Under 60 s
# -------------------------------------------------------------------------
t_start = time.time()

# Featurize 150-game subsample
X_150 = featurize(df_150)
y_150 = (df_150.label == 'contact').astype(int).values

# Run 2 repeats across cal_sizes [20, 40], alphas [0.05, 0.10], 4 methods, 3 bins
df_subsample_res = cs.run_repeated_calsize(
    df_150, X_150, y_150,
    cal_sizes=[20, 40],
    alphas=[0.05, 0.10],
    n_repeats=2,
    n_test_games=23,
    train_fraction=0.50,
    n_jobs=1,
    fast=True,
)
elapsed_time = time.time() - t_start

ok &= check(f"the whole subsample run finishes in under 60 s ({elapsed_time:.2f} s)",
            elapsed_time < 60.0)

# Verify output structure and contents
expected_columns = [
    'n_cal', 'method', 'alpha', 'bin', 'wrong_skip_bin', 'wrong_skip_pooled',
    'skip_rate', 'std_err', 'n_bin_cal_mean', 'slack_theory', 'frac_exceed', 'n_repeats'
]
ok &= check("subsample result has exact required columns",
            list(df_subsample_res.columns) == expected_columns)

expected_rows = 2 * 4 * 2 * 3  # 2 cal_sizes * 4 methods * 2 alphas * 3 bins = 48 rows
ok &= check(f"subsample result has expected 48 rows ({len(df_subsample_res)})",
            len(df_subsample_res) == expected_rows)

ok &= check("skip_rate in [0, 1]",
            (df_subsample_res['skip_rate'] >= 0.0).all() and (df_subsample_res['skip_rate'] <= 1.0).all())
ok &= check("wrong_skip_bin in [0, 1]",
            (df_subsample_res['wrong_skip_bin'] >= 0.0).all() and (df_subsample_res['wrong_skip_bin'] <= 1.0).all())
ok &= check("wrong_skip_pooled in [0, 1]",
            (df_subsample_res['wrong_skip_pooled'] >= 0.0).all() and (df_subsample_res['wrong_skip_pooled'] <= 1.0).all())
ok &= check("frac_exceed in [0, 1]",
            (df_subsample_res['frac_exceed'] >= 0.0).all() and (df_subsample_res['frac_exceed'] <= 1.0).all())
ok &= check("n_bin_cal_mean <= n_cal",
            (df_subsample_res['n_bin_cal_mean'] <= df_subsample_res['n_cal']).all())
ok &= check("slack_theory strictly positive",
            (df_subsample_res['slack_theory'] > 0.0).all() and (df_subsample_res['slack_theory'] <= 1.0).all())
ok &= check("all 4 methods evaluated",
            set(df_subsample_res['method'].unique()) == set(cs.METHODS))
ok &= check("all 3 Mondrian bins present",
            set(df_subsample_res['bin'].unique()) == set(md.BIN_NAMES))

if not ok:
    print("SOME TESTS FAILED")
    sys.exit(1)

print("ALL CALSIZE TESTS PASSED")

