"""Unit tests for parameter-randomised dataset option.
Run:  python tests/test_params.py   (all lines must print PASS)
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from kluster import simulator as sm
from kluster.generate import play_game, generate
from kluster.features import featurize, GEOM, PHYS, PHYS_PARAMS


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


def two(r):
    return np.array([[-r/2, 0.0], [r/2, 0.0]]), np.array([1.0, -1.0])


ok = True

# 1. Option OFF reproducibility: game 0 with seed 2026 gives rows identical to data/kluster_dataset.csv
df_orig = pd.read_csv(ROOT / "data" / "kluster_dataset.csv")
g0_orig = df_orig[df_orig["game_id"] == 0].reset_index(drop=True)
g0_gen = pd.DataFrame(play_game(0, 2026, random_params=False))

ok &= check("game 0 option OFF row count matches", len(g0_gen) == len(g0_orig))
ok &= check("game 0 option OFF columns match", list(g0_gen.columns) == list(g0_orig.columns))
ok &= check("game 0 option OFF labels match exactly", (g0_gen["label"].values == g0_orig["label"].values).all())
ok &= check("game 0 option OFF static_flags match exactly", (g0_gen["static_flag"].values == g0_orig["static_flag"].values).all())
try:
    pd.testing.assert_frame_equal(g0_gen, g0_orig, check_exact=False, atol=1e-5)
    match_frame = True
except AssertionError:
    match_frame = False
ok &= check("game 0 option OFF values match within numerical tolerance (atol=1e-5)", match_frame)

# 2. Option ON: parameters drawn within specified distributions and stored as columns
rows_rand = play_game(0, 2026, random_params=True)
df_rand = pd.DataFrame(rows_rand)
cols_present = all(c in df_rand.columns for c in ['mu_s', 'mu_k', 'k_scale'])
ok &= check("mu_s, mu_k, k_scale columns present when random_params is ON", cols_present)

mu_s_val = float(df_rand['mu_s'].iloc[0])
mu_k_val = float(df_rand['mu_k'].iloc[0])
ks_val = float(df_rand['k_scale'].iloc[0])
ok &= check("mu_s in [0.15, 0.60]", 0.15 <= mu_s_val <= 0.60)
ok &= check("mu_k in [0.5, 0.8]*mu_s", 0.50 * mu_s_val - 1e-9 <= mu_k_val <= 0.80 * mu_s_val + 1e-9)
ok &= check("k_scale in [0.6, 1.6]", 0.60 <= ks_val <= 1.60)

# 3. Option ON: every settled base board is at rest under that game's own parameters
base_boards_at_rest = True
for gid in [0, 1, 2]:
    g_rows = play_game(gid, 2026, random_params=True)
    g_df = pd.DataFrame(g_rows)
    for step, grp in g_df.groupby('step'):
        if step == 0:
            continue
        r0 = grp.iloc[0]
        n = int(r0['n_total']) - 1
        Pb = np.array([[r0[f'x{i}'], r0[f'y{i}']] for i in range(n)]) / 1000.0
        Sb = np.array([r0[f's{i}'] for i in range(n)])
        # Check static baseline under that game's own parameters
        if n >= 2 and sm.static_baseline(Pb, Sb, r0['rw_mm'] / 1000.0, mu_s=r0['mu_s'], k_scale=r0['k_scale']):
            base_boards_at_rest = False
ok &= check("every settled base board is at rest under that game's own parameters", base_boards_at_rest)

# 4. Two-magnet critical distances from simulator match rc_mm within 1 mm for at least three parameter sets
param_sets = [
    (0.20, 0.14, 0.8),
    (0.35, 0.22, 1.2),
    (0.50, 0.35, 1.5),
    (0.55, 0.38, 0.7),
]
crit_match_all = True
for mu_s, mu_k, ks in param_sets:
    rc_m = sm.critical_distance(mu_s=mu_s, k_scale=ks)
    rc_mm = rc_m * 1000.0
    # Antiparallel pair at rc_mm - 1 mm must contact
    p_below, s_below = two((rc_mm - 1.0) / 1000.0)
    l_below = sm.simulate(p_below, s_below, 0.2, mu_s=mu_s, mu_k=mu_k, k_scale=ks)[0]
    # Antiparallel pair at rc_mm + 1 mm must settle
    p_above, s_above = two((rc_mm + 1.0) / 1000.0)
    l_above = sm.simulate(p_above, s_above, 0.2, mu_s=mu_s, mu_k=mu_k, k_scale=ks)[0]
    # Binary search to verify critical distance threshold
    lo, hi = 0.010, 0.150
    for _ in range(16):
        mid = (lo + hi) / 2.0
        pmid, smid = two(mid)
        lab = sm.simulate(pmid, smid, 0.2, mu_s=mu_s, mu_k=mu_k, k_scale=ks)[0]
        if lab == 'contact':
            lo = mid
        else:
            hi = mid
    r_sim_mm = (lo + hi) / 2.0 * 1000.0
    if l_below != 'contact' or l_above != 'settled' or abs(r_sim_mm - rc_mm) >= 1.0:
        crit_match_all = False

ok &= check("two-magnet critical distances from simulator match rc_mm within 1 mm for at least 3 param sets", crit_match_all)

# 5. Feature lists and featurize behavior
ok &= check("GEOM list has 15 features", len(GEOM) == 15)
ok &= check("PHYS list has 18 features", len(PHYS) == 18)
ok &= check("PHYS_PARAMS list has 20 features", len(PHYS_PARAMS) == 20)
ok &= check("PHYS_PARAMS equals PHYS + ['rc_mm', 'd_opp_over_rc']", PHYS_PARAMS == PHYS + ['rc_mm', 'd_opp_over_rc'])

# 6. Backward compatibility: absent parameter columns use defaults giving identical features
sample_old = df_orig.head(10).copy()
X_old = featurize(sample_old)
ok &= check("featurize on dataset without param cols outputs PHYS_PARAMS columns", list(X_old.columns) == list(PHYS_PARAMS))
rc_default = sm.critical_distance() * 1000.0
ok &= check("absent param columns yield default rc_mm", np.allclose(X_old['rc_mm'].values, rc_default, atol=1e-4))

# 7. Row-specific parameters in featurize and static_baseline
sample_new = df_rand.head(10).copy()
X_new = featurize(sample_new)
row0 = sample_new.iloc[0]
expected_rc0 = (3.0 * sm.K * row0['k_scale'] / (row0['mu_s'] * sm.MASS * sm.G))**0.25 * 1000.0
ok &= check("featurize uses row's own mu_s and k_scale for rc_mm", abs(X_new['rc_mm'].iloc[0] - expected_rc0) < 1e-4)
expected_ratio0 = (X_new['d_opp'].iloc[0] * 1000.0) / expected_rc0
ok &= check("d_opp_over_rc computed correctly", abs(X_new['d_opp_over_rc'].iloc[0] - expected_ratio0) < 1e-4)

# Static baseline with row object
ok &= check("static_baseline accepts row object directly", isinstance(sm.static_baseline(row0), bool))

print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
sys.exit(0 if ok else 1)

