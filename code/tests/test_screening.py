"""Screening unit tests. Run:  python tests/test_screening.py   (all lines must print PASS)"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from kluster import screening as sc

def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)

ok = True

# 1. Game splits: 50 / 15 / 20 / 15 and strictly disjoint
fake_df = pd.DataFrame({'game_id': np.repeat(np.arange(300), 10)})
tr_g, va_g, cal_g, te_g = sc.split_games_screening(fake_df, seed=7)
ok &= check("split sizes 150 / 45 / 60 / 45",
            (len(tr_g), len(va_g), len(cal_g), len(te_g)) == (150, 45, 60, 45))
all_games = set(tr_g) | set(va_g) | set(cal_g) | set(te_g)
ok &= check("all 300 games accounted for", len(all_games) == 300)
ok &= check("calibration never touches test games", len(set(cal_g) & set(te_g)) == 0)
ok &= check("calibration never touches train games", len(set(cal_g) & set(tr_g)) == 0)
ok &= check("calibration never touches validation games", len(set(cal_g) & set(va_g)) == 0)
ok &= check("test never touches train games", len(set(te_g) & set(tr_g)) == 0)

# 2. Monotonicity of per-game loss L_g(tau)
scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6])
y = np.array([0, 0, 0, 0, 1, 0, 0, 1])  # 6 safe (y=0), 2 contact (y=1)
g_ids = np.array([0, 0, 0, 0, 0, 1, 1, 1])

l_low = sc.per_game_losses(scores, y, g_ids, tau=0.2)
l_mid = sc.per_game_losses(scores, y, g_ids, tau=0.5)
l_high = sc.per_game_losses(scores, y, g_ids, tau=0.8)
l_inf = sc.per_game_losses(scores, y, g_ids, tau=np.inf)

ok &= check("L_g(tau) decreases monotonically with tau",
            (l_low >= l_mid).all() and (l_mid >= l_high).all() and (l_high >= l_inf).all())
ok &= check("L_g(inf) == 0 for all games", (l_inf == 0.0).all())
ok &= check("L_g(-inf) == 1 for games with safe candidates",
            (sc.per_game_losses(scores, y, g_ids, tau=-np.inf) == 1.0).all())

# 3. Conformal risk control calibration
# Under n=60 games, 1/(n+1) = 1/61 ~ 0.01639.
# For alpha = 0.01 < 1/61, CRC strictly cannot guarantee risk <= 0.01 without abstaining.
res_001 = sc.calibrate_crc(scores, y, g_ids, alpha=0.01)
ok &= check("alpha < 1/(n+1) correctly abstains (tau = inf)", np.isinf(res_001['tau']))

# For large enough alpha, CRC returns a finite tau satisfying the risk bound
res_050 = sc.calibrate_crc(scores, y, g_ids, alpha=0.50)
ok &= check("alpha=0.50 returns finite tau", np.isfinite(res_050['tau']))
ok &= check("CRC bound <= alpha holds for calibrated tau", res_050['crc_bound'] <= 0.50 + 1e-9)

# 4. First-safe calls logic
mock_df = pd.DataFrame({
    'game_id': [0, 0, 0, 0],
    'step': [0, 0, 0, 0],
    'label': ['contact', 'settled', 'contact', 'settled']
})
mock_scores = {
    'model_perfect': np.array([0.9, 0.1, 0.8, 0.2]),   # safe items have smallest scores -> first safe is index 0 -> 1 call
    'model_worst': np.array([0.1, 0.9, 0.2, 0.8]),     # contact items first -> first safe is at index 2 -> 3 calls
}
df_first = sc.evaluate_first_safe_calls(mock_df, mock_scores, n_boot=50)
res_dict = dict(zip(df_first['method'], df_first['mean_calls']))
# K = 4, M = 2 -> E[random] = (4+1)/(2+1) = 5/3 ~ 1.6667
ok &= check("expected random calls = (K+1)/(M+1)", abs(res_dict['random'] - (5.0/3.0)) < 1e-6)
ok &= check("perfect model finds safe candidate in 1 call", abs(res_dict['model_perfect'] - 1.0) < 1e-6)
ok &= check("worst model takes 3 calls", abs(res_dict['model_worst'] - 3.0) < 1e-6)

# 5. Repeated splits disjointness test
fake_df_large = pd.DataFrame({'game_id': np.repeat(np.arange(100), 10)})
repeated_splits_disjoint = True
for s in range(200):
    tr, va, cal, te = sc.split_games_screening(fake_df_large, seed=s)
    if len(set(cal) & set(te)) > 0:
        repeated_splits_disjoint = False
        break
ok &= check("repeated splits never share a game between calibration and test", repeated_splits_disjoint)

# 6. Risk check computation
df_mock_recs = pd.DataFrame({
    'model': ['RuleA'] * 10,
    'alpha': [0.05] * 10,
    'wrong_skip_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
})
df_rc = sc.compute_risk_check(df_mock_recs)
mean_w = df_rc.loc[0, 'mean_wrong_skip_rate']
frac_exc = df_rc.loc[0, 'fraction_exceeding_alpha']
p025 = df_rc.loc[0, 'wrong_skip_lo']
p975 = df_rc.loc[0, 'wrong_skip_hi']
ok &= check("risk check mean wrong-skip rate correct", abs(mean_w - 0.055) < 1e-6)
ok &= check("risk check fraction exceeding alpha correct", abs(frac_exc - 0.5) < 1e-6)
ok &= check("risk check percentiles ordered lo <= mean <= hi", p025 <= mean_w <= p975)

# 7. Paired net time saved calculation
df_mock_pair = pd.DataFrame({
    'alpha': [0.05, 0.05],
    'best_ml_model': ['BestML', 'BestML'],
    'best_ml_skip': [0.50, 0.60],
    'combined_skip': [0.30, 0.40],
})
df_nt_paired = sc.compute_net_time_paired(df_mock_pair, sim_ms=10.0, ml_eval_ms=1.0, comb_eval_ms=0.5)
# best_net = [0.5*10-1 = 4.0, 0.6*10-1 = 5.0]
# comb_net = [0.3*10-0.5 = 2.5, 0.4*10-0.5 = 3.5]
# diffs_ms = [4.0-2.5 = 1.5, 5.0-3.5 = 1.5] -> mean = 1.5
ok &= check("net time paired difference mean matches formula", abs(df_nt_paired.loc[0, 'mean_diff_ms'] - 1.5) < 1e-4)
ok &= check("net time paired percent matches (diff/sim)*100", abs(df_nt_paired.loc[0, 'mean_diff_pct'] - 15.0) < 1e-4)

# 8. Fast tuning hyperparameters
ok &= check("FAST_HGB_PARAMS has lr=0.05, depth=8, l2=1.0", sc.FAST_HGB_PARAMS == dict(lr=0.05, depth=8, l2=1.0))
ok &= check("FAST_LR_PARAMS has C=10.0", sc.FAST_LR_PARAMS == dict(C=10.0))

# 9. Timing measurement has 5 repeats and computes median
mock_row = {
    'n_total': 2, 'rw_mm': 100.0,
    'x0': 0.0, 'y0': 0.0, 's0': 1,
    'x1': 50.0, 'y1': 0.0, 's1': 1,
    'game_id': 0, 'step': 0, 'label': 'settled'
}
mock_timing_df = pd.DataFrame([mock_row] * 5)
from sklearn.dummy import DummyClassifier
dummy_m = DummyClassifier(strategy="constant", constant=0).fit(np.zeros((5, 1)), np.zeros(5))
df_net_mock, df_time_mock = sc.measure_screening_times(
    mock_timing_df, dummy_m, ['n_total'], 'Dummy', [0.05],
    pd.DataFrame([{'model': 'Dummy', 'alpha': 0.05, 'fraction_skipped_mean': 0.5}]),
    n_sim=5, n_eval=5, n_repeats=5, return_timing=True
)
ok &= check("screening timing returns 5 repeats and median",
            'repeat_1_ms' in df_time_mock.columns and 'repeat_5_ms' in df_time_mock.columns and 'median_ms' in df_time_mock.columns)
sim_median = df_time_mock.loc[df_time_mock['method'] == 'Simulator', 'median_ms'].values[0]
sim_r = [df_time_mock.loc[df_time_mock['method'] == 'Simulator', f'repeat_{i}_ms'].values[0] for i in range(1, 6)]
ok &= check("simulator median equals median of 5 repeats", abs(sim_median - np.median(sim_r)) < 1e-6)

print("ALL SCREENING TESTS PASSED" if ok else "SOME SCREENING TESTS FAILED")
sys.exit(0 if ok else 1)
