"""Unit tests for neural network baseline and paired difference evaluation.

Run:  python tests/test_nn.py   (all lines must print PASS)
"""
import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from kluster import nnbase as nb
from kluster import mondrian as md
from kluster.features import featurize


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


ok = True

# -------------------------------------------------------------------------
# Load 150-game subsample from data/kluster_dataset_v2.csv
# -------------------------------------------------------------------------
csv_v2 = ROOT / "data" / "kluster_dataset_v2.csv"
if not csv_v2.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_v2}")

df_all = pd.read_csv(csv_v2)
unique_150_games = df_all['game_id'].unique()[:150]
df_150 = df_all[df_all['game_id'].isin(unique_150_games)].copy().reset_index(drop=True)

ok &= check("subsample has exactly 150 unique games", df_150['game_id'].nunique() == 150)

# Featurize 150-game subsample
X_150 = featurize(df_150)
y_150 = (df_150['label'] == 'contact').astype(int).values

# -------------------------------------------------------------------------
# 1. No game shared between calibration and test
# -------------------------------------------------------------------------
all_disjoint = True
for s in range(20):
    tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df_150, seed=s)
    if len(set(cal_g) & set(te_g)) > 0:
        all_disjoint = False
        break

ok &= check("no game shared between calibration and test across splits", all_disjoint)

# -------------------------------------------------------------------------
# 2. Network's predicted probabilities lie in [0, 1]
# -------------------------------------------------------------------------
tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df_150, seed=42)
ok_label = (df_150['label'] != 'unresolved').values
in_tr = df_150['game_id'].isin(tr_g).values & ok_label
in_te = df_150['game_id'].isin(te_g).values & ok_label

models = nb.build_models(seed=42)
nn_pipeline = models[nb.NN_MODEL]
nn_pipeline.fit(X_150.loc[in_tr, nb.FEATURE_COLS], y_150[in_tr])

probs_te = nn_pipeline.predict_proba(X_150.loc[in_te, nb.FEATURE_COLS])[:, 1]
probs_all = nn_pipeline.predict_proba(X_150.loc[ok_label, nb.FEATURE_COLS])[:, 1]

probs_in_range = (
    (probs_te >= 0.0).all() and (probs_te <= 1.0).all() and
    (probs_all >= 0.0).all() and (probs_all <= 1.0).all() and
    np.isfinite(probs_te).all() and np.isfinite(probs_all).all()
)
ok &= check("the network's predicted probabilities lie in [0, 1]", probs_in_range)

# -------------------------------------------------------------------------
# 3. Subsample run finishes in under 90 s
# -------------------------------------------------------------------------
t_start = time.time()

df_baseline, df_paired, raw_df = nb.run_repeated_nn(
    df_150, X_150, y_150,
    alphas=[0.02, 0.05, 0.10],
    n_repeats=2,
    n_jobs=1,
    return_raw=True,
)
elapsed_time = time.time() - t_start

ok &= check(f"the subsample run finishes in under 90 s ({elapsed_time:.2f} s)", elapsed_time < 90.0)

# Verify summary structure and contents
expected_base_cols = {'model', 'alpha', 'bin', 'wrong_skip_bin', 'wrong_skip_pooled', 'mean_skip_rate', 'std_err', 'train_time_s', 'n_repeats'}
ok &= check("baseline output contains all required columns", expected_base_cols.issubset(df_baseline.columns))
ok &= check("baseline evaluates all 3 models", set(df_baseline['model'].unique()) == set(nb.MODELS))
ok &= check("baseline contains all 3 size bins", set(df_baseline['bin'].unique()) == set(md.BIN_NAMES))

expected_pair_cols = {'bin', 'alpha', 'nn_skip_mean', 'hgb_skip_mean', 'mean_diff', 'diff_lo', 'diff_hi'}
ok &= check("paired output contains all required columns", expected_pair_cols.issubset(df_paired.columns))
ok &= check("paired percentiles ordered: diff_lo <= diff_hi", (df_paired['diff_lo'] <= df_paired['diff_hi']).all())

# -------------------------------------------------------------------------
# 4. Paired difference of a model with itself is exactly 0
# -------------------------------------------------------------------------
raw_records = raw_df.to_dict('records')

df_self_nn = nb.compute_paired_diff(raw_records, model_a=nb.NN_MODEL, model_b=nb.NN_MODEL)
df_self_hgb = nb.compute_paired_diff(raw_records, model_a=nb.HGB_MODEL, model_b=nb.HGB_MODEL)

self_nn_exact_zero = (
    (df_self_nn['mean_diff'] == 0.0).all() and
    (df_self_nn['diff_lo'] == 0.0).all() and
    (df_self_nn['diff_hi'] == 0.0).all()
)
self_hgb_exact_zero = (
    (df_self_hgb['mean_diff'] == 0.0).all() and
    (df_self_hgb['diff_lo'] == 0.0).all() and
    (df_self_hgb['diff_hi'] == 0.0).all()
)

ok &= check("the paired difference of a model with itself is exactly 0 (Neural network)", self_nn_exact_zero)
ok &= check("the paired difference of a model with itself is exactly 0 (Gradient boosting)", self_hgb_exact_zero)

print("ALL NN TESTS PASSED" if ok else "SOME NN TESTS FAILED")
sys.exit(0 if ok else 1)

