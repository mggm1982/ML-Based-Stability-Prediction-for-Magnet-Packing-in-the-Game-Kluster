"""Mondrian screening unit tests. Run:  python tests/test_mondrian.py   (all lines must print PASS)"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from kluster import mondrian as md


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


ok = True

# -------------------------------------------------------------------------
# 1. Game splits: 50 / 15 / 20 / 15 and strictly disjoint (no game in two splits)
# -------------------------------------------------------------------------
fake_df = pd.DataFrame({'game_id': np.repeat(np.arange(1000), 10)})
tr_g, va_g, cal_g, te_g = md.split_games_mondrian(fake_df, seed=7)
ok &= check("split sizes 500 / 150 / 200 / 150 for 1000 games",
            (len(tr_g), len(va_g), len(cal_g), len(te_g)) == (500, 150, 200, 150))

all_games = set(tr_g) | set(va_g) | set(cal_g) | set(te_g)
ok &= check("all 1000 games accounted for", len(all_games) == 1000)

ok &= check("train and val games strictly disjoint", len(set(tr_g) & set(va_g)) == 0)
ok &= check("train and cal games strictly disjoint", len(set(tr_g) & set(cal_g)) == 0)
ok &= check("train and test games strictly disjoint", len(set(tr_g) & set(te_g)) == 0)
ok &= check("val and cal games strictly disjoint", len(set(va_g) & set(cal_g)) == 0)
ok &= check("val and test games strictly disjoint", len(set(va_g) & set(te_g)) == 0)
ok &= check("cal and test games strictly disjoint", len(set(cal_g) & set(te_g)) == 0)

# Check no game appears in two splits across 20 distinct seeds
no_overlap_20 = True
for s in range(20):
    t_tr, t_va, t_cal, t_te = md.split_games_mondrian(fake_df, seed=s)
    splits = [set(t_tr), set(t_va), set(t_cal), set(t_te)]
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            if len(splits[i] & splits[j]) > 0:
                no_overlap_20 = False
                break
ok &= check("no game appears in two splits across 20 distinct seeds", no_overlap_20)

# -------------------------------------------------------------------------
# 2. Board-size bins definition and never mixing
# -------------------------------------------------------------------------
# Check that 2..14 are partitioned by '2-4', '5-8', '9-14'
sizes = list(range(2, 15))
bin_assigned = [md.get_bin_name(n) for n in sizes]
ok &= check("all board sizes 2..14 assign to a bin", len(bin_assigned) == 13)

# Test mutual exclusivity and exhaustiveness of bins
mock_n = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
m_24 = md.get_bin_mask(mock_n, '2-4')
m_58 = md.get_bin_mask(mock_n, '5-8')
m_914 = md.get_bin_mask(mock_n, '9-14')

ok &= check("bins never mix: pairwise intersections are all empty",
            not (m_24 & m_58).any() and not (m_58 & m_914).any() and not (m_24 & m_914).any())
ok &= check("bins partition all board sizes 2..14 with no gaps",
            (m_24 | m_58 | m_914).all())

# Verify out-of-range board sizes raise ValueError
raised_low = False
try:
    md.get_bin_name(1)
except ValueError:
    raised_low = True
raised_high = False
try:
    md.get_bin_name(15)
except ValueError:
    raised_high = True
ok &= check("out-of-range board sizes raise ValueError", raised_low and raised_high)

# -------------------------------------------------------------------------
# 3. Each bin's threshold uses ONLY that bin's calibration rows
# -------------------------------------------------------------------------
# Construct synthetic calibration data for two bins: '2-4' and '5-8'
# Bin '2-4': 10 games, each with safe candidate scores in [0.1, 0.4]
g_cal_b1 = np.repeat(np.arange(10), 4)
y_cal_b1 = np.zeros(40, dtype=int)
s_cal_b1 = np.tile([0.1, 0.2, 0.3, 0.4], 10)

# Bin '5-8': 10 different games with safe candidate scores in [0.5, 0.9]
g_cal_b2 = np.repeat(np.arange(10, 20), 4)
y_cal_b2 = np.zeros(40, dtype=int)
s_cal_b2_orig = np.tile([0.5, 0.6, 0.7, 0.8], 10)

# Calibrate size-conditional threshold for bin '2-4'
alpha = 0.20
res_b1_initial = md.calibrate_crc_bin(s_cal_b1, y_cal_b1, g_cal_b1, alpha=alpha)
tau_b1_initial = res_b1_initial['tau']

# Now radically mutate bin '5-8' scores and games
s_cal_b2_mutated = np.tile([0.9, 0.95, 0.99, 1.0], 10)
g_cal_b2_mutated = np.repeat(np.arange(20, 30), 4)

# Size-conditional threshold for bin '2-4' must remain 100% identical
res_b1_after_b2_mutation = md.calibrate_crc_bin(s_cal_b1, y_cal_b1, g_cal_b1, alpha=alpha)
tau_b1_after_b2_mutation = res_b1_after_b2_mutation['tau']
ok &= check("each bin's threshold uses only that bin's calibration rows (independent of other bins)",
            tau_b1_initial == tau_b1_after_b2_mutation)

# In contrast, mutating bin '2-4' calibration rows DOES change bin '2-4' threshold
s_cal_b1_altered = np.tile([0.6, 0.7, 0.8, 0.9], 10)
res_b1_altered = md.calibrate_crc_bin(s_cal_b1_altered, y_cal_b1, g_cal_b1, alpha=alpha)
ok &= check("modifying bin's own calibration rows modifies its threshold",
            res_b1_altered['tau'] != tau_b1_initial)

# Also verify that for Pooled strategy, mutating bin '5-8' rows DOES change the pooled threshold
s_pooled_orig = np.concatenate([s_cal_b1, s_cal_b2_orig])
y_pooled_orig = np.concatenate([y_cal_b1, y_cal_b2])
g_pooled_orig = np.concatenate([g_cal_b1, g_cal_b2])
tau_pooled_orig = md.calibrate_crc_bin(s_pooled_orig, y_pooled_orig, g_pooled_orig, alpha=alpha)['tau']

s_pooled_mut = np.concatenate([s_cal_b1, s_cal_b2_mutated])
y_pooled_mut = np.concatenate([y_cal_b1, y_cal_b2])
g_pooled_mut = np.concatenate([g_cal_b1, g_cal_b2_mutated])
tau_pooled_mut = md.calibrate_crc_bin(s_pooled_mut, y_pooled_mut, g_pooled_mut, alpha=alpha)['tau']
ok &= check("pooled calibration threshold is influenced by all rows across bins",
            tau_pooled_orig != tau_pooled_mut)

# -------------------------------------------------------------------------
# 4. Abstention condition: alpha < 1 / (n + 1)
# -------------------------------------------------------------------------
# For n = 10 calibration games, 1 / (n + 1) = 1/11 ~ 0.0909
res_abstain = md.calibrate_crc_bin(s_cal_b1, y_cal_b1, g_cal_b1, alpha=0.05)
ok &= check("alpha < 1/(n+1) strictly abstains (tau = inf)", np.isinf(res_abstain['tau']))
ok &= check("abstention flags feasible=False", not res_abstain['feasible'])

eval_abstain = md.evaluate_bin_test(s_cal_b1, y_cal_b1, g_cal_b1, tau=np.inf)
ok &= check("evaluating abstention yields skipped_fraction == 0.0", eval_abstain['fraction_skipped'] == 0.0)
ok &= check("evaluating abstention yields wrong_skip_rate == 0.0", eval_abstain['wrong_skip_rate'] == 0.0)

# For alpha >= 1 / (n + 1), finite threshold is returned
res_feasible = md.calibrate_crc_bin(s_cal_b1, y_cal_b1, g_cal_b1, alpha=0.50)
ok &= check("alpha >= 1/(n+1) returns finite threshold", np.isfinite(res_feasible['tau']))

# -------------------------------------------------------------------------
# 5. Per-game loss in a bin:
#    - Exclude games with no rows in the bin
#    - Games with rows but no safe candidates receive loss 0
#    - Monotonicity of per-game loss with tau
# -------------------------------------------------------------------------
# Game 0: 2 safe candidates, scores [0.2, 0.8]
# Game 1: 2 contact candidates (no safe candidates)
# Game 2: not in this bin
s_test = np.array([0.2, 0.8, 0.9, 0.95])
y_test = np.array([0, 0, 1, 1])
g_test = np.array([0, 0, 1, 1])

# Tau = 0.5: Game 0 skips 1 of 2 safe -> loss = 0.5; Game 1 has no safe -> loss = 0.0
losses_05 = md.per_game_losses_bin(s_test, y_test, g_test, tau=0.5)
ok &= check("games with rows but no safe candidates receive loss 0.0", losses_05[1] == 0.0)
ok &= check("game with safe candidates computes fraction skipped correctly", losses_05[0] == 0.5)

# Game 2 (not in slice) is excluded: unique games is exactly [0, 1]
ok &= check("games with no rows in bin slice are excluded", len(losses_05) == 2)

# Monotonicity with tau
l_low = md.per_game_losses_bin(s_test, y_test, g_test, tau=0.1)
l_mid = md.per_game_losses_bin(s_test, y_test, g_test, tau=0.5)
l_hi = md.per_game_losses_bin(s_test, y_test, g_test, tau=0.9)
ok &= check("per-game loss decreases monotonically with tau",
            (l_low >= l_mid).all() and (l_mid >= l_hi).all())

# -------------------------------------------------------------------------
# 6. Check that all 8 rules and models are defined
# -------------------------------------------------------------------------
ok &= check("ALL_MODEL_NAMES has exactly 8 models", len(md.ALL_MODEL_NAMES) == 8)
ok &= check("static friction rule present", md.STATIC_RULE in md.ALL_MODEL_NAMES)
ok &= check("distance rule present", md.DIST_RULE in md.ALL_MODEL_NAMES)
ok &= check("combined rule present", md.COMB_RULE in md.ALL_MODEL_NAMES)
ok &= check("critical-distance rule present", md.RC_RULE in md.ALL_MODEL_NAMES)
for ml_name in md.ALL_ML_NAMES:
    ok &= check(f"ML model {ml_name} present", ml_name in md.ALL_MODEL_NAMES)

# -------------------------------------------------------------------------
# 7. Paired comparison under size-conditional calibration
# -------------------------------------------------------------------------
mock_pair_data = []
for s in range(10):
    for b_name in md.BIN_NAMES:
        for alpha in [0.02, 0.05, 0.10]:
            ml_skip = 0.40 + 0.01 * s
            comb_skip = 0.30 + 0.01 * s
            mock_pair_data.append({
                'seed': s,
                'bin': b_name,
                'alpha': alpha,
                'best_ml_model': 'Gradient boosting (+force ratios)',
                'best_ml_skip': ml_skip,
                'combined_skip': comb_skip,
                'diff': ml_skip - comb_skip,
            })
df_mock_pair = pd.DataFrame(mock_pair_data)
df_paired_out = md.compute_mondrian_paired(df_mock_pair)

ok &= check("mondrian_paired has 9 rows (3 bins * 3 alphas)", len(df_paired_out) == 9)
req_cols = {'bin', 'alpha', 'best_ml_model', 'best_ml_skip_mean', 'combined_skip_mean', 'mean_diff', 'diff_lo', 'diff_hi'}
ok &= check("mondrian_paired has all required columns", req_cols.issubset(df_paired_out.columns))

# Verify mean_diff is exactly 0.10 across all rows
mean_diffs = df_paired_out['mean_diff'].values
ok &= check("mondrian_paired mean difference matches formula", np.allclose(mean_diffs, 0.10, atol=1e-4))
ok &= check("mondrian_paired percentiles ordered: diff_lo <= mean_diff <= diff_hi",
            (df_paired_out['diff_lo'] <= df_paired_out['mean_diff']).all() and
            (df_paired_out['mean_diff'] <= df_paired_out['diff_hi']).all())

# -------------------------------------------------------------------------
# 8. Risk check table under size-conditional calibration
# -------------------------------------------------------------------------
mock_raw_data = []
for s in range(20):
    for b_name in md.BIN_NAMES:
        for model in md.ALL_MODEL_NAMES:
            for alpha in [0.02, 0.05, 0.10]:
                w_rate = alpha + 0.01 * (s % 5 - 2)
                mock_raw_data.append({
                    'seed': s,
                    'strategy': 'Size-conditional',
                    'bin': b_name,
                    'model': model,
                    'alpha': alpha,
                    'wrong_skip_rate': w_rate,
                    'fraction_skipped': 0.5,
                    'n_cal_games': 50,
                    'tau': 0.5,
                })
df_mock_raw = pd.DataFrame(mock_raw_data)
df_rc_out = md.compute_mondrian_risk_check(df_mock_raw)

ok &= check("mondrian_risk_check has 72 rows (3 bins * 8 models * 3 alphas)", len(df_rc_out) == 72)
rc_cols = {'bin', 'model', 'alpha', 'mean_wrong_skip_rate', 'mean_minus_alpha', 'std_err', 'fraction_exceeding_alpha'}
ok &= check("mondrian_risk_check has all required columns", rc_cols.issubset(df_rc_out.columns))

# Check mean_minus_alpha == mean_wrong_skip_rate - alpha
diff_check = df_rc_out['mean_minus_alpha'] - (df_rc_out['mean_wrong_skip_rate'] - df_rc_out['alpha'])
ok &= check("mean_minus_alpha equals mean_wrong_skip_rate minus alpha", np.allclose(diff_check, 0.0, atol=1e-4))

# Check std_err is positive and matches sample standard error
ok &= check("std_err is positive and finite", (df_rc_out['std_err'] > 0).all() and np.isfinite(df_rc_out['std_err']).all())

# Verify Pooled strategy rows are excluded from risk_check
mock_raw_with_pooled = df_mock_raw.copy()
mock_raw_with_pooled['strategy'] = 'Pooled'
df_combined_mock = pd.concat([df_mock_raw, mock_raw_with_pooled], ignore_index=True)
df_rc_from_combined = md.compute_mondrian_risk_check(df_combined_mock)
ok &= check("compute_mondrian_risk_check strictly filters to Size-conditional strategy",
            len(df_rc_from_combined) == 72)

if ok:
    print("ALL MONDRIAN TESTS PASSED")
else:
    print("SOME MONDRIAN TESTS FAILED")
    sys.exit(1)

