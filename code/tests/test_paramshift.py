"""Unit tests for parameter-shift experiments and Mondrian Conformal Risk Control screening.
Run:  python tests/test_paramshift.py   (all lines must print PASS)
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from kluster import paramshift as ps
from kluster import simulator as sm
from kluster.features import featurize, GEOM, PHYS, PHYS_PARAMS
from kluster.mondrian import split_games_mondrian, BINS


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


ok = True

# -------------------------------------------------------------------------
# 1. Hand-calculated toy example for wrong-skip rate
# -------------------------------------------------------------------------
# Setup synthetic test games with known labels and scores:
# Game 101: 3 candidates: 2 safe (y=0, scores 0.8, 0.2), 1 contact (y=1, score 0.9)
# Game 102: 2 candidates: 1 safe (y=0, score 0.5), 1 contact (y=1, score 0.1)
# Game 103: 1 candidate:  0 safe (y=1, score 0.85)
toy_scores = np.array([0.8, 0.2, 0.9, 0.5, 0.1, 0.85])
toy_y = np.array([0, 0, 1, 0, 1, 1])
toy_g = np.array([101, 101, 101, 102, 102, 103])
tau = 0.4

# At tau = 0.4, candidates with score >= 0.4 are skipped:
# Row 0 (G101): score 0.8 >= 0.4, y=0 -> WRONG SKIP
# Row 1 (G101): score 0.2 < 0.4,  y=0 -> safe, simulated (not skipped)
# Row 2 (G101): score 0.9 >= 0.4, y=1 -> contact skipped (correct)
# Row 3 (G102): score 0.5 >= 0.4, y=0 -> WRONG SKIP
# Row 4 (G102): score 0.1 < 0.4,  y=1 -> contact simulated
# Row 5 (G103): score 0.85 >= 0.4, y=1 -> contact skipped (correct)

# Hand calculation of per-game losses L_g:
# Game 101: 2 safe candidates, 1 wrong skip -> L_101 = 1 / 2 = 0.5
# Game 102: 1 safe candidate,  1 wrong skip -> L_102 = 1 / 1 = 1.0
# Game 103: 0 safe candidates               -> L_103 = 0.0
# Expected mean wrong-skip rate = (0.5 + 1.0 + 0.0) / 3 = 1.5 / 3 = 0.50
# Expected fraction skipped = 4 / 6 = 2 / 3 ~ 0.66667
toy_is_skipped = toy_scores >= tau
toy_losses = ps.per_game_losses_group(toy_is_skipped, toy_y, toy_g)

ok &= check("toy per-game losses match hand calculation [0.5, 1.0, 0.0]",
            np.allclose(toy_losses, [0.5, 1.0, 0.0], atol=1e-7))

toy_eval = ps.evaluate_paramshift_group(toy_is_skipped, toy_y, toy_g)
ok &= check("toy mean wrong-skip rate matches hand calculation (0.50)",
            abs(toy_eval['wrong_skip_rate'] - 0.50) < 1e-7)
ok &= check("toy fraction skipped matches hand calculation (4/6)",
            abs(toy_eval['fraction_skipped'] - (4.0 / 6.0)) < 1e-7)

# Monotonicity with tau
l_low = ps.per_game_losses_group(toy_scores >= 0.15, toy_y, toy_g)
l_mid = ps.per_game_losses_group(toy_scores >= 0.40, toy_y, toy_g)
l_high = ps.per_game_losses_group(toy_scores >= 0.85, toy_y, toy_g)
l_inf = ps.per_game_losses_group(toy_scores >= np.inf, toy_y, toy_g)
ok &= check("toy per-game losses decrease monotonically with tau",
            (np.mean(l_low) >= np.mean(l_mid)) and (np.mean(l_mid) >= np.mean(l_high)) and (np.mean(l_high) >= np.mean(l_inf)))

# -------------------------------------------------------------------------
# 2. Subsample of 150 games: Splits and Disjointness
# -------------------------------------------------------------------------
# Load 150 games from v2 dataset
csv_v2 = ROOT / "data" / "kluster_dataset_v2.csv"
if not csv_v2.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_v2}")

df_all_v2 = pd.read_csv(csv_v2)
unique_150_games = df_all_v2['game_id'].unique()[:150]
df_150 = df_all_v2[df_all_v2['game_id'].isin(unique_150_games)].copy().reset_index(drop=True)

ok &= check("subsample has exactly 150 unique games", df_150['game_id'].nunique() == 150)

# Check game-level split disjointness on the 150 games: 50 / 15 / 20 / 15
tr_g, va_g, cal_g, te_g = split_games_mondrian(df_150, seed=7)
ok &= check("split sizes 75 / 23 / 30 / 22 for 150 games sum to 150",
            (len(tr_g), len(va_g), len(cal_g), len(te_g)) == (75, 23, 30, 22) and
            (len(tr_g) + len(va_g) + len(cal_g) + len(te_g) == 150))

# Strict disjointness: calibration and test games never overlap
ok &= check("no game shared between calibration and test", len(set(cal_g) & set(te_g)) == 0)
ok &= check("no game shared between calibration and train", len(set(cal_g) & set(tr_g)) == 0)
ok &= check("no game shared between calibration and validation", len(set(cal_g) & set(va_g)) == 0)
ok &= check("no game shared between test and train", len(set(te_g) & set(tr_g)) == 0)
ok &= check("no game shared between test and validation", len(set(te_g) & set(va_g)) == 0)

# Disjointness holds across 20 distinct seeds
all_disjoint = True
for s in range(20):
    t_tr, t_va, t_cal, t_te = split_games_mondrian(df_150, seed=s)
    if len(set(t_cal) & set(t_te)) > 0:
        all_disjoint = False
        break
ok &= check("no game shared between calibration and test across 20 seeds", all_disjoint)

# -------------------------------------------------------------------------
# 3. Three q groups partition all test rows
# -------------------------------------------------------------------------
X_150 = featurize(df_150)
ok_label = (df_150['label'] != 'unresolved').values
test_mask = df_150['game_id'].isin(te_g).values & ok_label
df_test = df_150[test_mask].reset_index(drop=True)
X_test = X_150.loc[test_mask].reset_index(drop=True)

# Compute q = rc_mm / default_rc for all test rows
q_test = X_test['rc_mm'].values / ps.DEFAULT_RC_MM
q_masks = ps.get_q_groups_mask(q_test)

m_low = q_masks[ps.GROUP_LOW]
m_mid = q_masks[ps.GROUP_MID]
m_high = q_masks[ps.GROUP_HIGH]

# Mutual exclusivity
ok &= check("q groups are pairwise disjoint: low & mid empty", (m_low & m_mid).sum() == 0)
ok &= check("q groups are pairwise disjoint: mid & high empty", (m_mid & m_high).sum() == 0)
ok &= check("q groups are pairwise disjoint: low & high empty", (m_low & m_high).sum() == 0)

# Exhaustiveness: covers all test rows
ok &= check("q groups partition all test rows (no row unassigned)", (m_low | m_mid | m_high).sum() == len(q_test))
ok &= check("each test row belongs to strictly one q group",
            ((m_low.astype(int) + m_mid.astype(int) + m_high.astype(int)) == 1).all())

# q group naming matches
test_q_vals = [0.75, 0.80, 1.00, 1.25, 1.30]
expected_names = [ps.GROUP_LOW, ps.GROUP_MID, ps.GROUP_MID, ps.GROUP_MID, ps.GROUP_HIGH]
ok &= check("get_q_group_name classifies boundary values correctly",
            [ps.get_q_group_name(v) for v in test_q_vals] == expected_names)

# -------------------------------------------------------------------------
# 4. Parameter-aware critical distance rule and feature sets
# -------------------------------------------------------------------------
ok &= check("PHYS_PARAMS feature set includes rc_mm and d_opp_over_rc",
            'rc_mm' in PHYS_PARAMS and 'd_opp_over_rc' in PHYS_PARAMS)
ok &= check("d_opp_over_rc is normalized distance (d_opp / rc)",
            np.allclose(X_150['d_opp_over_rc'].values,
                        (X_150['d_opp'].values * 1000.0) / X_150['rc_mm'].values, atol=1e-5))

# Parameter-aware rule score is -d_opp_over_rc
scores_mock = ps.compute_scores_paramshift(df_150.head(10), X_150.head(10), include_param_rule=True)
ok &= check("Parameter-aware critical-distance rule present when requested",
            ps.RC_PARAM_RULE in scores_mock)
ok &= check("Parameter-aware rule score equals -d_opp_over_rc",
            np.allclose(scores_mock[ps.RC_PARAM_RULE], -X_150.head(10)['d_opp_over_rc'].values))

# -------------------------------------------------------------------------
# 5. Mini execution of Experiments A, B, C on 150-game subsample
# -------------------------------------------------------------------------
# Load 50 games of old dataset for Experiment A mini-run
csv_old = ROOT / "data" / "kluster_dataset.csv"
if not csv_old.exists():
    raise FileNotFoundError(f"Dataset not found at {csv_old}")

df_all_old = pd.read_csv(csv_old)
unique_50_old = df_all_old['game_id'].unique()[:50]
df_50_old = df_all_old[df_all_old['game_id'].isin(unique_50_old)].copy().reset_index(drop=True)
X_50_old = featurize(df_50_old)
y_50_old = (df_50_old['label'] == 'contact').astype(int).values

y_150 = (df_150['label'] == 'contact').astype(int).values

# Mini Experiment A (1 split)
rec_A = ps.run_single_paramshift_A(0, df_50_old, X_50_old, y_50_old, df_150, X_150, y_150, alphas=[0.05], fast=True)
df_res_A = ps.aggregate_experiment_results(rec_A)

expected_cols = {'group', 'model', 'alpha', 'mean_wrong_skip_rate', 'std_err', 'mean_skip_rate', 'n_games', 'n_repeats'}
ok &= check("Experiment A summary contains required columns", expected_cols.issubset(df_res_A.columns))
ok &= check("Experiment A contains the 3 q groups", set(df_res_A['group'].unique()) == set(ps.Q_GROUPS))
ok &= check("Experiment A evaluates all 8 models/rules", len(df_res_A['model'].unique()) == 8)
ok &= check("Experiment A skip rates in [0, 1]",
            ((df_res_A['mean_skip_rate'] >= 0.0) & (df_res_A['mean_skip_rate'] <= 1.0)).all())
ok &= check("Experiment A wrong-skip rates in [0, 1]",
            ((df_res_A['mean_wrong_skip_rate'] >= 0.0) & (df_res_A['mean_wrong_skip_rate'] <= 1.0)).all())

# Mini Experiment B and C (1 split)
rec_B, rec_C = ps.run_single_paramshift_B_C(0, df_150, X_150, y_150, alphas=[0.05], fast=True)
df_res_B = ps.aggregate_experiment_results(rec_B)
df_res_C = ps.aggregate_experiment_results(rec_C)

ok &= check("Experiment B summary contains required columns", expected_cols.issubset(df_res_B.columns))
ok &= check("Experiment B includes overall and 3 q groups", set(df_res_B['group'].unique()) == set(ps.ALL_EVAL_GROUPS))
ok &= check("Experiment B includes 10 models (4 rules + 6 ML models including params)",
            len(df_res_B['model'].unique()) == 10)
ok &= check("Experiment B has Logistic (+parameters) and Gradient boosting (+parameters)",
            'Logistic (+parameters)' in df_res_B['model'].values and 'Gradient boosting (+parameters)' in df_res_B['model'].values)

ok &= check("Experiment C summary contains required columns", expected_cols.issubset(df_res_C.columns))
ok &= check("Experiment C evaluates Parameter-aware critical-distance rule",
            list(df_res_C['model'].unique()) == [ps.RC_PARAM_RULE])
ok &= check("Experiment C includes overall and 3 q groups", set(df_res_C['group'].unique()) == set(ps.ALL_EVAL_GROUPS))

print("ALL PARAMSHIFT TESTS PASSED" if ok else "SOME PARAMSHIFT TESTS FAILED")
sys.exit(0 if ok else 1)
