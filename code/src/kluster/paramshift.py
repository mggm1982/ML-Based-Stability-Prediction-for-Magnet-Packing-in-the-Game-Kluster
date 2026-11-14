"""Parameter shift experiment and Mondrian Conformal Risk Control screening.

Experiment A (fixed to random):
  - Train and calibrate on old data (default physics parameters) with old feature lists (GEOM, PHYS).
  - Test on v2 games (random mu_s, mu_k, k_scale).
  - Evaluate on three q groups (q = rc_mm / default_rc):
      q < 0.8, 0.8 <= q <= 1.25, q > 1.25.
  - Expect conformal guarantees to fail in the outer groups due to distribution shift.

Experiment B (random trained):
  - Train and calibrate on v2 only using PHYS_PARAMS features plus existing ones.
  - Game-level splits, 20 repeats.
  - Evaluate overall and by the three q groups.

Experiment C (parameter-aware critical-distance rule):
  - Rule using each row's own critical distance rc_mm (normalized distance d_opp / rc).
  - Evaluated inside Experiment B's splits.
"""
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from .features import GEOM, PHYS, PHYS_PARAMS
from . import simulator as sm
from .screening import (
    STATIC_RULE,
    DIST_RULE,
    COMB_RULE,
    FAST_HGB_PARAMS,
    FAST_LR_PARAMS,
)
from .mondrian import (
    BINS,
    BIN_NAMES,
    RC_RULE,
    split_games_mondrian,
    calibrate_crc_bin,
    per_game_losses_bin,
)

RC_PARAM_RULE = 'Parameter-aware critical-distance rule'

# Models and display names
NAMES_OLD = {
    'LR_geom': 'Logistic (geometry)',
    'LR_phys': 'Logistic (+force ratios)',
    'HGB_geom': 'Gradient boosting (geometry)',
    'HGB_phys': 'Gradient boosting (+force ratios)',
}

NAMES_PARAM = {
    'LR_geom': 'Logistic (geometry)',
    'LR_phys': 'Logistic (+force ratios)',
    'LR_params': 'Logistic (+parameters)',
    'HGB_geom': 'Gradient boosting (geometry)',
    'HGB_phys': 'Gradient boosting (+force ratios)',
    'HGB_params': 'Gradient boosting (+parameters)',
}

ALL_RULES_BASE = [STATIC_RULE, DIST_RULE, COMB_RULE, RC_RULE]

DEFAULT_ALPHAS = [0.02, 0.05, 0.10]
DEFAULT_RC_MM = sm.critical_distance() * 1000.0  # ~43.033 mm

GROUP_LOW = 'q < 0.8'
GROUP_MID = '0.8 <= q <= 1.25'
GROUP_HIGH = 'q > 1.25'
GROUP_OVERALL = 'overall'

Q_GROUPS = [GROUP_LOW, GROUP_MID, GROUP_HIGH]
ALL_EVAL_GROUPS = [GROUP_OVERALL, GROUP_LOW, GROUP_MID, GROUP_HIGH]


def compute_q_ratio(rc_mm_or_df):
    """Compute ratio q = rc_mm / default_rc."""
    if isinstance(rc_mm_or_df, (pd.DataFrame, pd.Series)):
        if 'rc_mm' in rc_mm_or_df:
            vals = rc_mm_or_df['rc_mm'].values
        else:
            vals = rc_mm_or_df.values
    else:
        vals = np.asarray(rc_mm_or_df)
    return vals / DEFAULT_RC_MM


def get_q_group_name(q):
    """Return group label ('q < 0.8', '0.8 <= q <= 1.25', 'q > 1.25') for ratio q."""
    if q < 0.8:
        return GROUP_LOW
    elif q <= 1.25:
        return GROUP_MID
    else:
        return GROUP_HIGH


def get_q_groups_mask(q_arr):
    """Return dictionary of boolean masks partitioning q_arr into the three q groups."""
    q = np.asarray(q_arr)
    return {
        GROUP_LOW: q < 0.8,
        GROUP_MID: (q >= 0.8) & (q <= 1.25),
        GROUP_HIGH: q > 1.25,
    }


def tune_models_paramshift(X, y, tr_idx, va_idx, fast=True, include_params=False):
    """Fit candidate models on train, select best hyperparameters on validation.
    
    When include_params=True, adds LR_params and HGB_params on PHYS_PARAMS.
    """
    fitted = {}
    hgb_params = [FAST_HGB_PARAMS] if fast else [
        dict(lr=lr, depth=depth, l2=l2)
        for lr, depth, l2 in itertools.product([0.05, 0.1], [4, 8], [0.0, 1.0])
    ]
    lr_params = [FAST_LR_PARAMS] if fast else [dict(C=c) for c in [0.01, 0.1, 1.0, 10.0]]

    # Gradient boosting
    hgb_configs = [
        ('HGB_geom', GEOM),
        ('HGB_phys', PHYS),
    ]
    if include_params:
        hgb_configs.append(('HGB_params', PHYS_PARAMS))

    for key, cols in hgb_configs:
        best = None
        for p in hgb_params:
            m = HistGradientBoostingClassifier(
                random_state=0, learning_rate=p['lr'], max_depth=p['depth'],
                l2_regularization=p['l2'], max_iter=200
            ).fit(X.loc[tr_idx, cols], y[tr_idx])
            a = roc_auc_score(y[va_idx], m.predict_proba(X.loc[va_idx, cols])[:, 1])
            if best is None or a > best[0]:
                best = (a, m, cols, p)
        fitted[key] = best

    # Logistic regression
    lr_configs = [
        ('LR_geom', GEOM),
        ('LR_phys', PHYS),
    ]
    if include_params:
        lr_configs.append(('LR_params', PHYS_PARAMS))

    for key, cols in lr_configs:
        best = None
        for p in lr_params:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=p['C'], max_iter=3000))
            m.fit(X.loc[tr_idx, cols], y[tr_idx])
            a = roc_auc_score(y[va_idx], m.predict_proba(X.loc[va_idx, cols])[:, 1])
            if best is None or a > best[0]:
                best = (a, m, cols, p)
        fitted[key] = best

    return fitted


def compute_scores_paramshift(df_sub, X_sub, fitted_models=None, include_param_rule=False):
    """Compute risk scores for all rules and fitted ML models.
    
    Higher score indicates higher risk of contact.
    """
    scores = {}
    # Non-ML baselines
    scores[STATIC_RULE] = X_sub['f_ratio_max'].values
    scores[DIST_RULE] = -X_sub['d_opp'].values
    scores[COMB_RULE] = np.where(X_sub['n_over_fs'].values > 0, -X_sub['d_opp'].values, -1.0)
    scores[RC_RULE] = -X_sub['d_opp'].values

    if include_param_rule:
        scores[RC_PARAM_RULE] = -X_sub['d_opp_over_rc'].values

    # ML models
    if fitted_models is not None:
        for key, (_, m, cols, _) in fitted_models.items():
            name = NAMES_PARAM.get(key, key)
            scores[name] = m.predict_proba(X_sub[cols])[:, 1]

    return scores


def per_game_losses_group(is_skipped, y, game_ids):
    """Compute per-game loss L_g for an arbitrary skip mask on a test subset.
    
    L_g is the fraction of safe candidates (y == 0) in game g that are skipped.
    If game g has no safe candidates in the subset, loss is 0.0.
    """
    if len(game_ids) == 0:
        return np.zeros(0, dtype=float)

    unique_games = np.unique(game_ids)
    losses = np.zeros(len(unique_games), dtype=float)

    is_safe = (y == 0)
    wrong_skip = is_safe & is_skipped

    for idx, g in enumerate(unique_games):
        g_mask = (game_ids == g)
        n_safe = np.count_nonzero(g_mask & is_safe)
        if n_safe == 0:
            losses[idx] = 0.0
        else:
            losses[idx] = np.count_nonzero(g_mask & wrong_skip) / n_safe

    return losses


def evaluate_paramshift_group(is_skipped, y, game_ids):
    """Evaluate fraction skipped and mean wrong-skip rate on a group slice."""
    if len(game_ids) == 0:
        return {
            'fraction_skipped': 0.0,
            'wrong_skip_rate': 0.0,
            'n_games': 0,
            'n_candidates': 0,
        }

    s_frac = float(np.mean(is_skipped))
    losses = per_game_losses_group(is_skipped, y, game_ids)
    w_rate = float(np.mean(losses)) if len(losses) > 0 else 0.0
    n_games = len(np.unique(game_ids))

    return {
        'fraction_skipped': s_frac,
        'wrong_skip_rate': w_rate,
        'n_games': n_games,
        'n_candidates': len(game_ids),
    }


def run_single_paramshift_A(seed, df_old, X_old, y_old, df_v2, X_v2, y_v2, alphas=DEFAULT_ALPHAS, fast=True):
    """Run one split for Experiment A (fixed to random).
    
    Train and calibrate on old data (default parameters).
    Test on v2 games, grouped by ratio q = rc_mm / default_rc.
    """
    tr_g, va_g, cal_g, _ = split_games_mondrian(df_old, seed=seed)
    ok_old = (df_old.label != 'unresolved').values
    in_tr = df_old.game_id.isin(tr_g).values & ok_old
    in_va = df_old.game_id.isin(va_g).values & ok_old
    in_cal = df_old.game_id.isin(cal_g).values & ok_old

    # Fit ML models on old train/val data with old feature lists (GEOM, PHYS)
    fitted = tune_models_paramshift(X_old, y_old, in_tr, in_va, fast=fast, include_params=False)

    # Compute scores on old calibration data
    scores_cal = compute_scores_paramshift(df_old[in_cal], X_old.loc[in_cal], fitted, include_param_rule=False)

    y_cal = y_old[in_cal]
    g_cal = df_old.game_id.values[in_cal]
    n_cal = df_old.n_total.values[in_cal]

    # Calibrate size-conditional Mondrian thresholds for each board-size bin
    # taus[model][bin][alpha] = tau
    model_names = list(scores_cal.keys())
    taus = {m: {b: {} for b in BIN_NAMES} for m in model_names}

    for b_name, (lo, hi) in BINS.items():
        m_c = (n_cal >= lo) & (n_cal <= hi)
        y_c_b = y_cal[m_c]
        g_c_b = g_cal[m_c]

        for m in model_names:
            s_c_b = scores_cal[m][m_c]
            for alpha in alphas:
                c_res = calibrate_crc_bin(s_c_b, y_c_b, g_c_b, alpha)
                taus[m][b_name][alpha] = c_res['tau']

    # Test on v2 data (all games)
    ok_v2 = (df_v2.label != 'unresolved').values
    df_te = df_v2[ok_v2].reset_index(drop=True)
    X_te = X_v2.loc[ok_v2].reset_index(drop=True)
    y_te = y_v2[ok_v2]
    g_te = df_te.game_id.values
    n_te = df_te.n_total.values

    # Compute risk scores on v2 test set
    scores_te = compute_scores_paramshift(df_te, X_te, fitted, include_param_rule=False)

    # Compute q = rc_mm / default_rc for test rows
    q_te = X_te['rc_mm'].values / DEFAULT_RC_MM
    q_masks = get_q_groups_mask(q_te)

    records = []
    for m in model_names:
        s_t = scores_te[m]
        for alpha in alphas:
            # Construct candidate skip decisions via size-conditional Mondrian thresholds
            is_skipped = np.zeros(len(df_te), dtype=bool)
            for b_name, (lo, hi) in BINS.items():
                m_t = (n_te >= lo) & (n_te <= hi)
                tau_val = taus[m][b_name][alpha]
                if not np.isinf(tau_val):
                    is_skipped[m_t] = (s_t[m_t] >= tau_val)

            # Evaluate per q group
            for grp_name in Q_GROUPS:
                mask_grp = q_masks[grp_name]
                eval_res = evaluate_paramshift_group(is_skipped[mask_grp], y_te[mask_grp], g_te[mask_grp])

                records.append({
                    'seed': seed,
                    'group': grp_name,
                    'model': m,
                    'alpha': alpha,
                    'wrong_skip_rate': eval_res['wrong_skip_rate'],
                    'skip_rate': eval_res['fraction_skipped'],
                    'n_games': eval_res['n_games'],
                })

    return records


def run_single_paramshift_B_C(seed, df_v2, X_v2, y_v2, alphas=DEFAULT_ALPHAS, fast=True):
    """Run one split for Experiment B (random trained) and Experiment C (parameter-aware critical-distance rule).
    
    Train and calibrate on v2 only, using PHYS_PARAMS features plus existing ones.
    Game-level splits: 50 / 15 / 20 / 15.
    Evaluate overall and by the three q groups.
    """
    tr_g, va_g, cal_g, te_g = split_games_mondrian(df_v2, seed=seed)
    ok = (df_v2.label != 'unresolved').values
    in_tr = df_v2.game_id.isin(tr_g).values & ok
    in_va = df_v2.game_id.isin(va_g).values & ok
    in_cal = df_v2.game_id.isin(cal_g).values & ok
    in_te = df_v2.game_id.isin(te_g).values & ok

    # Fit ML models on v2 train/val with PHYS_PARAMS
    fitted = tune_models_paramshift(X_v2, y_v2, in_tr, in_va, fast=fast, include_params=True)

    # Compute scores for calibration and test (including RC_PARAM_RULE for Exp C)
    scores_cal = compute_scores_paramshift(df_v2[in_cal], X_v2.loc[in_cal], fitted, include_param_rule=True)
    scores_te = compute_scores_paramshift(df_v2[in_te], X_v2.loc[in_te], fitted, include_param_rule=True)

    y_cal = y_v2[in_cal]
    g_cal = df_v2.game_id.values[in_cal]
    n_cal = df_v2.n_total.values[in_cal]

    y_te = y_v2[in_te]
    g_te = df_v2.game_id.values[in_te]
    n_te = df_v2.n_total.values[in_te]
    q_te = X_v2.loc[in_te, 'rc_mm'].values / DEFAULT_RC_MM

    # Precompute size-conditional Mondrian thresholds
    model_names = list(scores_cal.keys())
    taus = {m: {b: {} for b in BIN_NAMES} for m in model_names}

    for b_name, (lo, hi) in BINS.items():
        m_c = (n_cal >= lo) & (n_cal <= hi)
        y_c_b = y_cal[m_c]
        g_c_b = g_cal[m_c]

        for m in model_names:
            s_c_b = scores_cal[m][m_c]
            for alpha in alphas:
                c_res = calibrate_crc_bin(s_c_b, y_c_b, g_c_b, alpha)
                taus[m][b_name][alpha] = c_res['tau']

    # Precompute group masks for test set
    q_masks = get_q_groups_mask(q_te)
    eval_group_masks = {
        GROUP_OVERALL: np.ones(len(y_te), dtype=bool),
        GROUP_LOW: q_masks[GROUP_LOW],
        GROUP_MID: q_masks[GROUP_MID],
        GROUP_HIGH: q_masks[GROUP_HIGH],
    }

    records_B = []
    records_C = []

    for m in model_names:
        s_t = scores_te[m]
        target_records = records_C if m == RC_PARAM_RULE else records_B

        for alpha in alphas:
            is_skipped = np.zeros(len(y_te), dtype=bool)
            for b_name, (lo, hi) in BINS.items():
                m_t = (n_te >= lo) & (n_te <= hi)
                tau_val = taus[m][b_name][alpha]
                if not np.isinf(tau_val):
                    is_skipped[m_t] = (s_t[m_t] >= tau_val)

            for grp_name in ALL_EVAL_GROUPS:
                mask_grp = eval_group_masks[grp_name]
                eval_res = evaluate_paramshift_group(is_skipped[mask_grp], y_te[mask_grp], g_te[mask_grp])

                target_records.append({
                    'seed': seed,
                    'group': grp_name,
                    'model': m,
                    'alpha': alpha,
                    'wrong_skip_rate': eval_res['wrong_skip_rate'],
                    'skip_rate': eval_res['fraction_skipped'],
                    'n_games': eval_res['n_games'],
                })

    return records_B, records_C


def aggregate_experiment_results(raw_records):
    """Aggregate per-split evaluation records across repeats into summary table.
    
    Columns: group, model, alpha, mean_wrong_skip_rate, std_err, mean_skip_rate, n_games, n_repeats.
    """
    df_raw = pd.DataFrame(raw_records)
    summary_rows = []

    # Preserve order of groups
    known_groups = [GROUP_OVERALL, GROUP_LOW, GROUP_MID, GROUP_HIGH]
    unique_groups = [g for g in known_groups if g in df_raw['group'].values]
    extra_groups = [g for g in df_raw['group'].unique() if g not in known_groups]
    ordered_groups = unique_groups + extra_groups

    for grp_name in ordered_groups:
        df_grp = df_raw[df_raw['group'] == grp_name]
        for model in df_grp['model'].unique():
            df_m = df_grp[df_grp['model'] == model]
            for alpha in sorted(df_m['alpha'].unique()):
                sub = df_m[df_m['alpha'] == alpha]
                if len(sub) == 0:
                    continue
                wrongs = sub['wrong_skip_rate'].values
                skips = sub['skip_rate'].values
                n_games_arr = sub['n_games'].values
                n_rep = len(sub)

                mean_w = float(np.mean(wrongs))
                se = float(np.std(wrongs, ddof=1) / np.sqrt(n_rep)) if n_rep > 1 else 0.0
                mean_s = float(np.mean(skips))
                n_g_mean = int(round(float(np.mean(n_games_arr))))

                summary_rows.append({
                    'group': grp_name,
                    'model': model,
                    'alpha': alpha,
                    'mean_wrong_skip_rate': round(mean_w, 4),
                    'std_err': round(se, 4),
                    'mean_skip_rate': round(mean_s, 4),
                    'n_games': n_g_mean,
                    'n_repeats': n_rep,
                })

    df_summary = pd.DataFrame(summary_rows)
    # Provide 'bin' alias attribute for compatibility
    df_summary.attrs['group_column'] = 'group'
    return df_summary


def run_repeated_paramshift(df_old, X_old, y_old, df_v2, X_v2, y_v2, alphas=DEFAULT_ALPHAS,
                            n_repeats=20, n_jobs=-1, fast=True):
    """Run repeated evaluation of Experiments A, B, and C over n_repeats splits."""
    # 1. Experiment A
    res_A = Parallel(n_jobs=n_jobs)(
        delayed(run_single_paramshift_A)(s, df_old, X_old, y_old, df_v2, X_v2, y_v2, alphas=alphas, fast=fast)
        for s in range(n_repeats)
    )
    all_A = [r for sub in res_A for r in sub]
    df_A = aggregate_experiment_results(all_A)

    # 2. Experiments B and C
    res_BC = Parallel(n_jobs=n_jobs)(
        delayed(run_single_paramshift_B_C)(s, df_v2, X_v2, y_v2, alphas=alphas, fast=fast)
        for s in range(n_repeats)
    )
    all_B = [r for rec_b, _ in res_BC for r in rec_b]
    all_C = [r for _, rec_c in res_BC for r in rec_c]

    df_B = aggregate_experiment_results(all_B)
    df_C = aggregate_experiment_results(all_C)

    return df_A, df_B, df_C, pd.DataFrame(all_A), pd.DataFrame(all_B), pd.DataFrame(all_C)

