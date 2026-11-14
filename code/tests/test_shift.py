"""Size-shift and ablation unit tests. Run:  python tests/test_shift.py   (all lines must print PASS)"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from kluster import shift as sh
from kluster.features import GEOM, PHYS


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


ok = True

# 1. Game splits: 50 / 15 / 20 / 15 and strictly disjoint
fake_df = pd.DataFrame({'game_id': np.repeat(np.arange(300), 10)})
tr_g, va_g, cal_g, te_g = sh.split_games_shift(fake_df, seed=7)
ok &= check("split sizes 150 / 45 / 60 / 45 for 300 games",
            (len(tr_g), len(va_g), len(cal_g), len(te_g)) == (150, 45, 60, 45))

all_games = set(tr_g) | set(va_g) | set(cal_g) | set(te_g)
ok &= check("all 300 games accounted for", len(all_games) == 300)

# Disjointness checks: No game appears in two splits
ok &= check("train and val games strictly disjoint", len(set(tr_g) & set(va_g)) == 0)
ok &= check("train and cal games strictly disjoint", len(set(tr_g) & set(cal_g)) == 0)
ok &= check("train and test games strictly disjoint", len(set(tr_g) & set(te_g)) == 0)
ok &= check("val and cal games strictly disjoint", len(set(va_g) & set(cal_g)) == 0)
ok &= check("val and test games strictly disjoint", len(set(va_g) & set(te_g)) == 0)
ok &= check("cal and test games strictly disjoint", len(set(cal_g) & set(te_g)) == 0)

# 2. Check no game appears in two splits across 20 repeated seeds
no_overlap_20 = True
for s in range(20):
    t_tr, t_va, t_cal, t_te = sh.split_games_shift(fake_df, seed=s)
    sets = [set(t_tr), set(t_va), set(t_cal), set(t_te)]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if len(sets[i] & sets[j]) > 0:
                no_overlap_20 = False
                break
ok &= check("no game appears in two splits across 20 distinct seeds", no_overlap_20)

# 3. Size partitioning logic (N <= 8 vs N >= 9)
mock_n = pd.Series([2, 5, 8, 9, 12, 14])
mask_le8 = mock_n <= 8
mask_ge9 = mock_n >= 9
ok &= check("N<=8 and N>=9 are mutually disjoint", not (mask_le8 & mask_ge9).any())
ok &= check("N<=8 and N>=9 partition all sizes 2..14", (mask_le8 | mask_ge9).all())

# 4. Feature ablation column sets
cols_full = list(PHYS)
cols_no_dopp = [c for c in cols_full if c != 'd_opp']
force_cols = ['f_ratio_max', 'f_ratio_new', 'n_over_fs']
cols_no_dopp_no_force = [c for c in cols_full if c != 'd_opp' and c not in force_cols]

ok &= check("full model has 18 features (PHYS)", len(cols_full) == 18)
ok &= check("ablation 1 removes d_opp (17 features)", len(cols_no_dopp) == 17 and 'd_opp' not in cols_no_dopp)
ok &= check("ablation 2 removes d_opp and force ratios (14 features)",
            len(cols_no_dopp_no_force) == 14 and 'd_opp' not in cols_no_dopp_no_force and not any(f in cols_no_dopp_no_force for f in force_cols))
ok &= check("ablation 2 equals GEOM without d_opp",
            set(cols_no_dopp_no_force) == set([c for c in GEOM if c != 'd_opp']))

# 5. Baseline threshold tuning
mock_X_tr = pd.DataFrame({
    'd_opp': np.array([0.02, 0.03, 0.04, 0.06, 0.08, 0.10]),
    'n_over_fs': np.array([1, 1, 1, 0, 0, 0]),
})
mock_y_tr = np.array([1, 1, 1, 0, 0, 0])
mock_tr_idx = np.array([True] * 6)
thr_dist, thr_comb = sh.tune_baseline_thresholds(mock_X_tr, mock_y_tr, mock_tr_idx)
ok &= check("tuned distance threshold in reasonable range [15 mm, 120 mm]", 0.015 <= thr_dist <= 0.12)
ok &= check("tuned combined threshold in reasonable range [15 mm, 120 mm]", 0.015 <= thr_comb <= 0.12)

# 6. CRC screening calibration and shifted evaluation
cal_scores = {'RuleA': np.array([0.1, 0.2, 0.3, 0.8, 0.9])}
cal_y = np.array([0, 0, 0, 1, 1])
cal_g = np.array([1, 1, 2, 2, 2])

te_scores = {'RuleA': np.array([0.15, 0.25, 0.85, 0.95])}
te_y = np.array([0, 0, 1, 1])
te_g = np.array([10, 10, 11, 11])

df_shift_crc = sh.evaluate_shift_crc(cal_scores, cal_y, cal_g, te_scores, te_y, te_g, alphas=[0.10, 0.50])
ok &= check("CRC returns evaluated alphas", set(df_shift_crc['alpha']) == {0.10, 0.50})
ok &= check("CRC skipped fraction in [0, 1]", ((df_shift_crc['fraction_skipped'] >= 0.0) & (df_shift_crc['fraction_skipped'] <= 1.0)).all())
ok &= check("CRC wrong skip rate in [0, 1]", ((df_shift_crc['wrong_skip_rate'] >= 0.0) & (df_shift_crc['wrong_skip_rate'] <= 1.0)).all())

# 7. Grouped bootstrap test metrics
mock_scores_te = {'ModelX': np.array([0.1, 0.2, 0.8, 0.9])}
mock_preds_te = {'ModelX': np.array([0, 0, 1, 1])}
mock_y_te = np.array([0, 0, 1, 1])
mock_g_te = np.array([1, 1, 2, 2])

df_boot_metrics = sh.compute_test_metrics_with_bootstrap(mock_scores_te, mock_preds_te, mock_y_te, mock_g_te, n_boot=50)
ok &= check("bootstrap metrics contains required columns",
            {'AUC', 'AUC_lo', 'AUC_hi', 'accuracy', 'acc_lo', 'acc_hi',
             'missed_contacts', 'missed_lo', 'missed_hi', 'false_alarms', 'false_alarm_lo', 'false_alarm_hi'}.issubset(df_boot_metrics.columns))
ok &= check("bootstrap CIs are correctly ordered",
            (df_boot_metrics['AUC_lo'] <= df_boot_metrics['AUC_hi']).all() and
            (df_boot_metrics['acc_lo'] <= df_boot_metrics['acc_hi']).all())

print("ALL SHIFT TESTS PASSED" if ok else "SOME SHIFT TESTS FAILED")
sys.exit(0 if ok else 1)

