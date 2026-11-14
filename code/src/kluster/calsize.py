"""Calibration-set size scaling analysis for Mondrian Conformal Risk Control (CRC).

Investigates how the finite-sample risk guarantee and empirical skip rate depend on
calibration-set size (n_cal in {20, 40, 80, 160, 320}) on data/kluster_dataset_v2.csv.

Main properties:
  - Game-level splits: fixed test set (225 games per repeat), models trained once
    on training split, and n_cal calibration games drawn at random from the remaining pool.
  - Size-conditional Mondrian CRC screening across board-size bins:
      '2-4' (2 <= n_total <= 4), '5-8' (5 <= n_total <= 8), '9-14' (9 <= n_total <= 14).
  - Models: Combined rule, Parameter-aware critical-distance rule,
    Gradient boosting (+force ratios), Logistic (+parameters).
  - Risk targets: alpha in {0.05, 0.10}.
  - Theoretical finite-sample slack: 1 / (n_bin + 1).
  - Strict abstention (tau = inf, skip_rate = 0) when n_bin + 1 < 1 / alpha.
"""
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from .features import PHYS, PHYS_PARAMS
from .screening import (
    COMB_RULE,
    FAST_HGB_PARAMS,
    FAST_LR_PARAMS,
)
from .mondrian import (
    BINS,
    BIN_NAMES,
    calibrate_crc_bin,
    evaluate_bin_test,
)
from .paramshift import (
    RC_PARAM_RULE,
    per_game_losses_group,
)

CAL_SIZES = [20, 40, 80, 160, 320]
DEFAULT_ALPHAS = [0.05, 0.10]

METHOD_COMBINED = COMB_RULE                               # 'Combined rule'
METHOD_RC_PARAM = RC_PARAM_RULE                           # 'Parameter-aware critical-distance rule'
METHOD_HGB_PHYS = 'Gradient boosting (+force ratios)'
METHOD_LR_PARAMS = 'Logistic (+parameters)'

METHODS = [
    METHOD_COMBINED,
    METHOD_RC_PARAM,
    METHOD_HGB_PHYS,
    METHOD_LR_PARAMS,
]


def theoretical_slack(n_games):
    """Compute exact finite-sample theoretical slack 1 / (n + 1)."""
    n = float(n_games)
    return 1.0 / (n + 1.0) if n >= 0 else 1.0


def split_games_calsize(df_or_games, seed=7, n_test=225, train_fraction=0.50):
    """Split games into training, pool (for calibration draws), and fixed test sets.
    
    Calibration games will be drawn at random from pool.
    Train, pool, and test sets are strictly disjoint.
    """
    if isinstance(df_or_games, pd.DataFrame):
        g = np.sort(df_or_games['game_id'].unique().copy())
    else:
        g = np.sort(np.unique(df_or_games).copy())
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    n = len(g)

    if n_test is not None and n_test < n:
        n_te = n_test
    else:
        n_te = max(1, int(round(0.15 * n)))

    n_tr = int(round(train_fraction * n))
    if n_tr + n_te > n:
        n_tr = max(1, int(round(0.50 * (n - n_te))))

    te_g = g[:n_te]
    tr_g = g[n_te:n_te + n_tr]
    pool_g = g[n_te + n_tr:]
    return tr_g, pool_g, te_g


def draw_calibration_games(pool_games, n_cal, seed=None, rng=None):
    """Draw n_cal calibration games at random without replacement from the pool."""
    if rng is None:
        rng = np.random.default_rng(seed)
    pool_arr = np.asarray(pool_games)
    if n_cal > len(pool_arr):
        raise ValueError(f"Requested n_cal={n_cal} games, but pool only has {len(pool_arr)} games.")
    return rng.choice(pool_arr, size=n_cal, replace=False)


def fit_calsize_models(X, y, in_tr, fast=True):
    """Fit ML models once per repeat on the training split."""
    models = {}
    # 1. Gradient boosting (+force ratios) on PHYS
    m_hgb = HistGradientBoostingClassifier(
        random_state=0,
        learning_rate=FAST_HGB_PARAMS['lr'],
        max_depth=FAST_HGB_PARAMS['depth'],
        l2_regularization=FAST_HGB_PARAMS['l2'],
        max_iter=200,
    ).fit(X.loc[in_tr, PHYS], y[in_tr])
    models[METHOD_HGB_PHYS] = m_hgb

    # 2. Logistic (+parameters) on PHYS_PARAMS
    m_lr = make_pipeline(StandardScaler(), LogisticRegression(C=FAST_LR_PARAMS['C'], max_iter=3000))
    m_lr.fit(X.loc[in_tr, PHYS_PARAMS], y[in_tr])
    models[METHOD_LR_PARAMS] = m_lr
    return models


def compute_scores_calsize(df_sub, X_sub, fitted_models=None):
    """Compute risk scores for all baselines and fitted ML models.
    
    Higher score indicates higher risk of contact.
    Candidate is skipped if score >= tau.
    """
    scores = {}
    # Non-ML rules
    scores[METHOD_COMBINED] = np.where(X_sub['n_over_fs'].values > 0, -X_sub['d_opp'].values, -1.0)
    scores[METHOD_RC_PARAM] = -X_sub['d_opp_over_rc'].values

    # ML models
    if fitted_models is not None:
        if METHOD_HGB_PHYS in fitted_models:
            scores[METHOD_HGB_PHYS] = fitted_models[METHOD_HGB_PHYS].predict_proba(X_sub[PHYS])[:, 1]
        if METHOD_LR_PARAMS in fitted_models:
            scores[METHOD_LR_PARAMS] = fitted_models[METHOD_LR_PARAMS].predict_proba(X_sub[PHYS_PARAMS])[:, 1]
    return scores


def run_single_calsize_repeat(
    seed, df, X, y,
    cal_sizes=CAL_SIZES,
    alphas=DEFAULT_ALPHAS,
    n_test_games=225,
    train_fraction=0.50,
    fast=True,
):
    """Execute one repeat across all calibration sizes, models, and alphas."""
    # 1. Disjoint split: test (fixed), train (fit once), and remaining pool
    tr_g, pool_g, te_g = split_games_calsize(df, seed=seed, n_test=n_test_games, train_fraction=train_fraction)

    ok = (df.label != 'unresolved').values
    in_tr = df.game_id.isin(tr_g).values & ok
    in_te = df.game_id.isin(te_g).values & ok
    in_pool = df.game_id.isin(pool_g).values & ok

    # 2. Train each ML model once on the training split
    models = fit_calsize_models(X, y, in_tr, fast=fast)

    # 3. Compute risk scores once on fixed test set and remaining pool
    df_te = df[in_te].reset_index(drop=True)
    X_te = X.loc[in_te].reset_index(drop=True)
    y_te = y[in_te]
    g_te = df_te['game_id'].values
    n_te = df_te['n_total'].values
    scores_te = compute_scores_calsize(df_te, X_te, models)

    df_pool = df[in_pool].reset_index(drop=True)
    X_pool = X.loc[in_pool].reset_index(drop=True)
    y_pool = y[in_pool]
    g_pool = df_pool['game_id'].values
    n_pool = df_pool['n_total'].values
    scores_pool = compute_scores_calsize(df_pool, X_pool, models)

    # Pre-shuffle pool games for nested calibration sets
    rng = np.random.default_rng(seed + 100000)
    pool_shuffled = pool_g.copy()
    rng.shuffle(pool_shuffled)

    records = []

    for n_cal in cal_sizes:
        if n_cal > len(pool_shuffled):
            continue
        cal_g = pool_shuffled[:n_cal]
        in_cal_sub = np.isin(g_pool, cal_g)

        y_cal = y_pool[in_cal_sub]
        g_cal = g_pool[in_cal_sub]
        n_cal_arr = n_pool[in_cal_sub]

        # Calibration bin information
        cal_bin_info = {}
        for b_name, (lo, hi) in BINS.items():
            m_c_b = (n_cal_arr >= lo) & (n_cal_arr <= hi)
            n_bin_cal = len(np.unique(g_cal[m_c_b]))
            cal_bin_info[b_name] = (m_c_b, n_bin_cal)

        # Calibrate size-conditional Mondrian thresholds
        taus = {m: {a: {} for a in alphas} for m in METHODS}
        for b_name, (lo, hi) in BINS.items():
            m_c_b, n_bin_cal = cal_bin_info[b_name]
            y_c_b = y_cal[m_c_b]
            g_c_b = g_cal[m_c_b]

            for m in METHODS:
                s_c_b = scores_pool[m][in_cal_sub][m_c_b]
                for a in alphas:
                    c_res = calibrate_crc_bin(s_c_b, y_c_b, g_c_b, a)
                    taus[m][a][b_name] = c_res['tau']

        # Evaluate on the test set
        for m in METHODS:
            s_te = scores_te[m]
            for a in alphas:
                # 1. Full test skip decisions for pooled game-level evaluation
                is_skipped_all = np.zeros(len(y_te), dtype=bool)
                for b_name, (lo, hi) in BINS.items():
                    m_te_b = (n_te >= lo) & (n_te <= hi)
                    tau_b = taus[m][a][b_name]
                    if not np.isinf(tau_b):
                        is_skipped_all[m_te_b] = (s_te[m_te_b] >= tau_b)

                losses_pooled = per_game_losses_group(is_skipped_all, y_te, g_te)
                wrong_skip_pooled = float(np.mean(losses_pooled)) if len(losses_pooled) > 0 else 0.0

                # 2. Within-bin evaluations
                for b_name, (lo, hi) in BINS.items():
                    m_te_b = (n_te >= lo) & (n_te <= hi)
                    tau_b = taus[m][a][b_name]
                    eval_b = evaluate_bin_test(s_te[m_te_b], y_te[m_te_b], g_te[m_te_b], tau_b)

                    skip_rate_bin = eval_b['fraction_skipped']
                    wrong_skip_bin = eval_b['wrong_skip_rate']
                    n_bin_cal = cal_bin_info[b_name][1]
                    slack = theoretical_slack(n_bin_cal)

                    records.append({
                        'seed': seed,
                        'n_cal': n_cal,
                        'method': m,
                        'alpha': a,
                        'bin': b_name,
                        'wrong_skip_bin': wrong_skip_bin,
                        'wrong_skip_pooled': wrong_skip_pooled,
                        'skip_rate': skip_rate_bin,
                        'n_bin_cal': n_bin_cal,
                        'slack_theory': slack,
                        'exceed': float(wrong_skip_bin > a),
                    })

    return records


def aggregate_calsize_records(records):
    """Aggregate per-split evaluation records into results/calsize.csv summary table."""
    df_raw = pd.DataFrame(records)
    cols = [
        'n_cal', 'method', 'alpha', 'bin', 'wrong_skip_bin', 'wrong_skip_pooled',
        'skip_rate', 'std_err', 'n_bin_cal_mean', 'slack_theory', 'frac_exceed', 'n_repeats'
    ]
    if len(df_raw) == 0:
        return pd.DataFrame(columns=cols)

    group_cols = ['n_cal', 'method', 'alpha', 'bin']
    grouped = df_raw.groupby(group_cols, sort=False)

    summary_rows = []
    for (n_cal, method, alpha, b_name), grp in grouped:
        n_reps = len(grp)
        w_bin = grp['wrong_skip_bin'].values
        w_pooled = grp['wrong_skip_pooled'].values
        s_rate = grp['skip_rate'].values
        n_bin_cal = grp['n_bin_cal'].values
        slack = grp['slack_theory'].values
        exc = grp['exceed'].values

        mean_w_bin = round(float(np.mean(w_bin)), 4)
        mean_w_pooled = round(float(np.mean(w_pooled)), 4)
        mean_s_rate = round(float(np.mean(s_rate)), 4)
        std_err = round(float(np.std(w_bin, ddof=1) / np.sqrt(n_reps)), 4) if n_reps > 1 else 0.0
        n_bin_cal_mean = round(float(np.mean(n_bin_cal)), 1)
        slack_theory = round(float(np.mean(slack)), 4)
        frac_exceed = round(float(np.mean(exc)), 4)

        summary_rows.append({
            'n_cal': int(n_cal),
            'method': method,
            'alpha': float(alpha),
            'bin': b_name,
            'wrong_skip_bin': mean_w_bin,
            'wrong_skip_pooled': mean_w_pooled,
            'skip_rate': mean_s_rate,
            'std_err': std_err,
            'n_bin_cal_mean': n_bin_cal_mean,
            'slack_theory': slack_theory,
            'frac_exceed': frac_exceed,
            'n_repeats': n_reps,
        })

    df_summary = pd.DataFrame(summary_rows)
    method_order = {m: i for i, m in enumerate(METHODS)}
    bin_order = {b: i for i, b in enumerate(BIN_NAMES)}
    df_summary['m_idx'] = df_summary['method'].map(method_order)
    df_summary['b_idx'] = df_summary['bin'].map(bin_order)
    df_summary = df_summary.sort_values(by=['n_cal', 'm_idx', 'alpha', 'b_idx']).drop(columns=['m_idx', 'b_idx']).reset_index(drop=True)
    return df_summary[cols]


def run_repeated_calsize(
    df, X, y,
    cal_sizes=CAL_SIZES,
    alphas=DEFAULT_ALPHAS,
    n_repeats=50,
    n_test_games=225,
    train_fraction=0.50,
    n_jobs=-1,
    fast=True,
    return_raw=False,
):
    """Run repeated calibration-set size experiments across multiple CPU cores."""
    if n_jobs == 1:
        results = [
            run_single_calsize_repeat(
                seed, df, X, y,
                cal_sizes=cal_sizes,
                alphas=alphas,
                n_test_games=n_test_games,
                train_fraction=train_fraction,
                fast=fast,
            )
            for seed in range(n_repeats)
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_calsize_repeat)(
                seed, df, X, y,
                cal_sizes=cal_sizes,
                alphas=alphas,
                n_test_games=n_test_games,
                train_fraction=train_fraction,
                fast=fast,
            )
            for seed in range(n_repeats)
        )

    all_records = [rec for rep in results for rec in rep]
    df_summary = aggregate_calsize_records(all_records)
    if return_raw:
        return df_summary, pd.DataFrame(all_records)
    return df_summary

