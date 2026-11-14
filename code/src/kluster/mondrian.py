"""Mondrian (size-conditional) Conformal Risk Control screening.

Compares two calibration strategies with Conformal Risk Control (Angelopoulos et al. 2022):
  (A) Pooled: one threshold from all calibration rows; evaluate on test rows separately in each bin.
  (B) Size-conditional: a separate threshold per bin, using only calibration rows in that bin; evaluate on test rows in the same bin.

Board-size bins defined by n_total:
  - '2-4': 2 <= n_total <= 4
  - '5-8': 5 <= n_total <= 8
  - '9-14': 9 <= n_total <= 14
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .features import GEOM, PHYS
from .screening import (
    NAMES,
    STATIC_RULE,
    DIST_RULE,
    COMB_RULE,
    FAST_HGB_PARAMS,
    FAST_LR_PARAMS,
    tune_models,
)

RC_RULE = 'Critical-distance rule'

ALL_RULES = [STATIC_RULE, DIST_RULE, COMB_RULE, RC_RULE]
ALL_ML_NAMES = [NAMES['LR_geom'], NAMES['LR_phys'], NAMES['HGB_geom'], NAMES['HGB_phys']]
ALL_MODEL_NAMES = ALL_RULES + ALL_ML_NAMES

BINS = {
    '2-4': (2, 4),
    '5-8': (5, 8),
    '9-14': (9, 14),
}
BIN_NAMES = list(BINS.keys())
DEFAULT_ALPHAS = [0.02, 0.05, 0.10]


def get_bin_name(n_total):
    """Return bin label ('2-4', '5-8', '9-14') for given board size n_total."""
    n = int(n_total)
    for b_name, (lo, hi) in BINS.items():
        if lo <= n <= hi:
            return b_name
    raise ValueError(f"n_total {n} not in any valid bin {list(BINS.values())}")


def get_bin_mask(df_or_n, bin_name):
    """Return boolean mask for rows belonging strictly to bin_name."""
    lo, hi = BINS[bin_name]
    if isinstance(df_or_n, pd.DataFrame):
        n_vals = df_or_n['n_total'].values
    elif isinstance(df_or_n, pd.Series):
        n_vals = df_or_n.values
    else:
        n_vals = np.asarray(df_or_n)
    return (n_vals >= lo) & (n_vals <= hi)


def split_games_mondrian(df, seed=7, fractions=(0.50, 0.15, 0.20, 0.15)):
    """Split games into train / validation / calibration / test (50 / 15 / 20 / 15).
    
    Calibration and test games are strictly disjoint and never used for fitting.
    """
    g = np.sort(df.game_id.unique().copy())
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    n = len(g)
    f_tr, f_va, f_cal, f_te = fractions
    a = int(round(f_tr * n))
    b = int(round((f_tr + f_va) * n))
    c = int(round((f_tr + f_va + f_cal) * n))
    tr_g, va_g, cal_g, te_g = g[:a], g[a:b], g[b:c], g[c:]
    return tr_g, va_g, cal_g, te_g


def compute_scores_mondrian(df_sub, X_sub, fitted_models=None):
    """Compute risk scores for all baselines and fitted ML models.
    
    Higher score indicates higher risk of contact.
    Candidate is skipped if score >= tau.
    """
    scores = {}
    # Non-ML baselines
    scores[STATIC_RULE] = X_sub['f_ratio_max'].values
    scores[DIST_RULE] = -X_sub['d_opp'].values
    scores[COMB_RULE] = np.where(X_sub['n_over_fs'].values > 0, -X_sub['d_opp'].values, -1.0)
    scores[RC_RULE] = -X_sub['d_opp'].values

    # ML models
    if fitted_models is not None:
        for key, (_, m, cols, _) in fitted_models.items():
            scores[NAMES[key]] = m.predict_proba(X_sub[cols])[:, 1]
    return scores


def per_game_losses_bin(scores, y, game_ids, tau):
    """Compute per-game loss L_g(tau) on a dataset slice.
    
    Per-game loss in a bin is the fraction of that game's safe candidates (y == 0) in the bin that are skipped.
    Exclude games with no rows in the bin (automatically satisfied when passing the bin's rows).
    Games with rows in the bin but no safe candidates receive loss 0.0.
    """
    if len(game_ids) == 0:
        return np.zeros(0, dtype=float)

    unique_games = np.unique(game_ids)
    losses = np.zeros(len(unique_games), dtype=float)

    if np.isinf(tau):
        # Abstention: no candidate skipped -> loss is 0.0 for all games
        return losses

    is_safe = (y == 0)
    is_skipped = (scores >= tau)
    wrong_skip = is_safe & is_skipped

    for idx, g in enumerate(unique_games):
        g_mask = (game_ids == g)
        n_safe = np.count_nonzero(g_mask & is_safe)
        if n_safe == 0:
            losses[idx] = 0.0
        else:
            losses[idx] = np.count_nonzero(g_mask & wrong_skip) / n_safe
    return losses


def calibrate_crc_bin(scores_cal, y_cal, game_ids_cal, alpha):
    """Calibrate screening threshold tau via Conformal Risk Control (Angelopoulos et al. 2022).
    
    Condition: (n / (n + 1)) * mean_g L_g(tau) + 1 / (n + 1) <= alpha,
    where n is the number of calibration games in this calibration slice.
    
    Abstains (returns tau = inf) when alpha < 1 / (n + 1).
    """
    unique_games = np.unique(game_ids_cal)
    n = len(unique_games)
    min_possible_bound = 1.0 / (n + 1) if n > 0 else 1.0

    if n == 0 or alpha < min_possible_bound:
        # Strict finite-sample guarantee cannot be satisfied with n calibration games.
        # Fallback to no-skip (tau = inf) guaranteeing risk = 0 <= alpha.
        return {
            'tau': np.inf,
            'mean_loss': 0.0,
            'crc_bound': min_possible_bound,
            'alpha': alpha,
            'feasible': False,
            'n_games': n,
        }

    # Safe candidate scores in calibration slice
    safe_scores = scores_cal[y_cal == 0]
    sorted_thresholds = np.sort(np.unique(safe_scores))

    # Pre-index sorted safe scores per game for speed
    game_safe_scores = {}
    for g in unique_games:
        mask = (game_ids_cal == g) & (y_cal == 0)
        game_safe_scores[g] = np.sort(scores_cal[mask])

    def eval_crc(tau):
        losses = []
        for g in unique_games:
            s_arr = game_safe_scores[g]
            if len(s_arr) == 0:
                losses.append(0.0)
            else:
                num_ge = len(s_arr) - np.searchsorted(s_arr, tau, side='left')
                losses.append(num_ge / len(s_arr))
        mean_l = float(np.mean(losses))
        crc = (n / (n + 1)) * mean_l + (1.0 / (n + 1))
        return mean_l, crc

    best_tau = np.inf
    best_mean_l = 0.0
    best_crc = min_possible_bound

    # Binary search for the smallest threshold satisfying CRC bound
    low = 0
    high = len(sorted_thresholds) - 1
    found_idx = -1

    while low <= high:
        mid = (low + high) // 2
        tau_cand = sorted_thresholds[mid]
        ml, crc = eval_crc(tau_cand)
        if crc <= alpha:
            found_idx = mid
            best_tau = float(tau_cand)
            best_mean_l = ml
            best_crc = crc
            high = mid - 1  # try to find smaller tau
        else:
            low = mid + 1

    if found_idx == -1:
        # tau > max(safe_scores) guarantees mean_loss = 0 and crc = 1/(n+1) <= alpha
        best_tau = float(sorted_thresholds[-1] + 1e-6) if len(sorted_thresholds) > 0 else np.inf
        best_mean_l, best_crc = eval_crc(best_tau)

    return {
        'tau': best_tau,
        'mean_loss': best_mean_l,
        'crc_bound': best_crc,
        'alpha': alpha,
        'feasible': True,
        'n_games': n,
    }


def evaluate_bin_test(scores_te_bin, y_te_bin, game_ids_te_bin, tau):
    """Evaluate screening on test rows of a specific bin.
    
    Returns:
      fraction_skipped: fraction of test candidates in this bin skipped.
      wrong_skip_rate: mean per-game loss over test games that have rows in this bin.
    """
    if len(scores_te_bin) == 0:
        return {
            'fraction_skipped': 0.0,
            'wrong_skip_rate': 0.0,
            'n_test_candidates': 0,
            'n_test_games': 0,
        }

    if np.isinf(tau):
        return {
            'fraction_skipped': 0.0,
            'wrong_skip_rate': 0.0,
            'n_test_candidates': len(scores_te_bin),
            'n_test_games': len(np.unique(game_ids_te_bin)),
        }

    frac_skipped = float(np.mean(scores_te_bin >= tau))
    losses = per_game_losses_bin(scores_te_bin, y_te_bin, game_ids_te_bin, tau)
    mean_wrong_skip = float(np.mean(losses)) if len(losses) > 0 else 0.0

    return {
        'fraction_skipped': frac_skipped,
        'wrong_skip_rate': mean_wrong_skip,
        'n_test_candidates': len(scores_te_bin),
        'n_test_games': len(np.unique(game_ids_te_bin)),
    }


def run_single_mondrian_repeat(seed, df, X, y, alphas=DEFAULT_ALPHAS, fast=True):
    """Run one split, fit models on ALL train rows across all board sizes, and evaluate both calibration strategies."""
    tr_g, va_g, cal_g, te_g = split_games_mondrian(df, seed=seed)
    ok = (df.label != 'unresolved').values
    in_tr = df.game_id.isin(tr_g).values & ok
    in_va = df.game_id.isin(va_g).values & ok
    in_cal = df.game_id.isin(cal_g).values & ok
    in_te = df.game_id.isin(te_g).values & ok

    # Train and tune models on ALL train rows across all board sizes
    fitted = tune_models(X, y, in_tr, in_va, fast=fast)
    best_ml_key = max(fitted, key=lambda k: fitted[k][0])
    best_ml_name = NAMES[best_ml_key]

    # Compute scores for all rules and ML models on calibration and test sets
    scores_cal = compute_scores_mondrian(df[in_cal], X.loc[in_cal], fitted)
    scores_te = compute_scores_mondrian(df[in_te], X.loc[in_te], fitted)

    y_cal = y[in_cal]
    g_cal = df.game_id.values[in_cal]
    n_cal = df.n_total.values[in_cal]

    y_te = y[in_te]
    g_te = df.game_id.values[in_te]
    n_te = df.n_total.values[in_te]

    records = []

    # Pre-compute bin masks
    cal_bin_masks = {}
    te_bin_masks = {}
    n_cal_games_by_bin = {}

    for b_name, (lo, hi) in BINS.items():
        m_c = (n_cal >= lo) & (n_cal <= hi)
        m_t = (n_te >= lo) & (n_te <= hi)
        cal_bin_masks[b_name] = m_c
        te_bin_masks[b_name] = m_t
        n_cal_games_by_bin[b_name] = len(np.unique(g_cal[m_c]))

    # --- Strategy (A) Pooled: one threshold from ALL calibration rows, evaluated per bin ---
    for model_name in ALL_MODEL_NAMES:
        s_c = scores_cal[model_name]
        s_t = scores_te[model_name]

        for alpha in alphas:
            c_res = calibrate_crc_bin(s_c, y_cal, g_cal, alpha)
            tau_pooled = c_res['tau']

            for b_name in BIN_NAMES:
                m_t = te_bin_masks[b_name]
                eval_res = evaluate_bin_test(s_t[m_t], y_te[m_t], g_te[m_t], tau_pooled)

                records.append({
                    'seed': seed,
                    'strategy': 'Pooled',
                    'bin': b_name,
                    'model': model_name,
                    'alpha': alpha,
                    'n_cal_games': n_cal_games_by_bin[b_name],
                    'tau': tau_pooled,
                    'fraction_skipped': eval_res['fraction_skipped'],
                    'wrong_skip_rate': eval_res['wrong_skip_rate'],
                    'best_ml_model': best_ml_name,
                })

    # --- Strategy (B) Size-conditional: separate threshold per bin from calibration rows in that bin ---
    paired_records = []
    for b_name in BIN_NAMES:
        m_c = cal_bin_masks[b_name]
        m_t = te_bin_masks[b_name]

        y_c_b = y_cal[m_c]
        g_c_b = g_cal[m_c]
        y_t_b = y_te[m_t]
        g_t_b = g_te[m_t]
        n_cal_b = n_cal_games_by_bin[b_name]

        for alpha in alphas:
            best_ml_skip = None
            comb_skip = None

            for model_name in ALL_MODEL_NAMES:
                s_c_b = scores_cal[model_name][m_c]
                s_t_b = scores_te[model_name][m_t]

                c_res = calibrate_crc_bin(s_c_b, y_c_b, g_c_b, alpha)
                tau_b = c_res['tau']
                eval_res = evaluate_bin_test(s_t_b, y_t_b, g_t_b, tau_b)

                records.append({
                    'seed': seed,
                    'strategy': 'Size-conditional',
                    'bin': b_name,
                    'model': model_name,
                    'alpha': alpha,
                    'n_cal_games': n_cal_b,
                    'tau': tau_b,
                    'fraction_skipped': eval_res['fraction_skipped'],
                    'wrong_skip_rate': eval_res['wrong_skip_rate'],
                    'best_ml_model': best_ml_name,
                })

                if model_name == best_ml_name:
                    best_ml_skip = eval_res['fraction_skipped']
                if model_name == COMB_RULE:
                    comb_skip = eval_res['fraction_skipped']

            if best_ml_skip is not None and comb_skip is not None:
                paired_records.append({
                    'seed': seed,
                    'bin': b_name,
                    'alpha': alpha,
                    'best_ml_model': best_ml_name,
                    'best_ml_skip': best_ml_skip,
                    'combined_skip': comb_skip,
                    'diff': best_ml_skip - comb_skip,
                })

    return records, paired_records


def compute_mondrian_paired(df_pair_or_raw):
    """Compute paired comparison of skipped fraction between best ML model and combined rule under size-conditional calibration.
    
    Returns DataFrame with:
      bin, alpha, best_ml_model, best_ml_skip_mean, combined_skip_mean, mean_diff, diff_lo, diff_hi
    """
    if isinstance(df_pair_or_raw, pd.DataFrame) and 'diff' in df_pair_or_raw.columns and 'best_ml_skip' in df_pair_or_raw.columns:
        df_pair = df_pair_or_raw
    else:
        # Reconstruct paired records from raw split records
        df_cond = df_pair_or_raw[df_pair_or_raw['strategy'] == 'Size-conditional']
        pair_rows = []
        for (s, b_name, alpha), grp in df_cond.groupby(['seed', 'bin', 'alpha'], sort=False):
            best_name = grp['best_ml_model'].iloc[0] if 'best_ml_model' in grp.columns else None
            if best_name is None:
                ml_rows = grp[grp['model'].isin(ALL_ML_NAMES)]
                best_name = ml_rows.sort_values('fraction_skipped', ascending=False)['model'].iloc[0]
            ml_row = grp[grp['model'] == best_name]
            comb_row = grp[grp['model'] == COMB_RULE]
            if len(ml_row) > 0 and len(comb_row) > 0:
                s_ml = ml_row['fraction_skipped'].iloc[0]
                s_comb = comb_row['fraction_skipped'].iloc[0]
                pair_rows.append({
                    'seed': s,
                    'bin': b_name,
                    'alpha': alpha,
                    'best_ml_model': best_name,
                    'best_ml_skip': s_ml,
                    'combined_skip': s_comb,
                    'diff': s_ml - s_comb,
                })
        df_pair = pd.DataFrame(pair_rows)

    rows = []
    for b_name in BIN_NAMES:
        for alpha in sorted(df_pair['alpha'].unique()):
            grp = df_pair[(df_pair['bin'] == b_name) & (df_pair['alpha'] == alpha)]
            if len(grp) == 0:
                continue
            diffs = grp['diff'].values
            best_name = grp['best_ml_model'].mode()[0]
            best_skips = grp['best_ml_skip'].values
            comb_skips = grp['combined_skip'].values

            rows.append({
                'bin': b_name,
                'alpha': alpha,
                'best_ml_model': best_name,
                'best_ml_skip_mean': float(np.mean(best_skips)),
                'combined_skip_mean': float(np.mean(comb_skips)),
                'mean_diff': float(np.mean(diffs)),
                'diff_lo': float(np.percentile(diffs, 2.5)),
                'diff_hi': float(np.percentile(diffs, 97.5)),
            })
    return pd.DataFrame(rows).round(4)


def compute_mondrian_risk_check(df_raw):
    """Compute wrong-skip risk check metrics under size-conditional calibration.
    
    Table of the mean wrong-skip rate under size-conditional calibration minus alpha,
    per bin, model and alpha, with the standard error over the repeats.
    """
    df_cond = df_raw[df_raw['strategy'] == 'Size-conditional']
    rows = []

    for b_name in BIN_NAMES:
        for model in ALL_MODEL_NAMES:
            for alpha in sorted(df_cond['alpha'].unique()):
                grp = df_cond[(df_cond['bin'] == b_name) & 
                              (df_cond['model'] == model) & 
                              (df_cond['alpha'] == alpha)]
                if len(grp) == 0:
                    continue
                wrongs = grp['wrong_skip_rate'].values
                mean_w = float(np.mean(wrongs))
                n_rep = len(wrongs)
                se = float(np.std(wrongs, ddof=1) / np.sqrt(n_rep)) if n_rep > 1 else 0.0
                diff_alpha = mean_w - alpha

                rows.append({
                    'bin': b_name,
                    'model': model,
                    'alpha': alpha,
                    'mean_wrong_skip_rate': mean_w,
                    'mean_minus_alpha': diff_alpha,
                    'risk_minus_alpha': diff_alpha,
                    'std_err': se,
                    'standard_error': se,
                    'wrong_skip_lo': float(np.percentile(wrongs, 2.5)),
                    'wrong_skip_hi': float(np.percentile(wrongs, 97.5)),
                    'fraction_exceeding_alpha': float(np.mean(wrongs > alpha)),
                    'n_repeats': n_rep,
                })
    return pd.DataFrame(rows).round(4)


def run_repeated_mondrian(df, X, y, alphas=DEFAULT_ALPHAS, n_repeats=20, n_jobs=-1, fast=True, return_all=False):
    """Repeat the Mondrian conformal screening experiment over n_repeats different game splits."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_mondrian_repeat)(s, df, X, y, alphas=alphas, fast=fast)
        for s in range(n_repeats)
    )

    all_records = []
    all_paired = []
    for res in results:
        if isinstance(res, tuple) and len(res) == 2:
            recs, paired = res
            all_records.extend(recs)
            all_paired.extend(paired)
        else:
            all_records.extend(res)

    df_all = pd.DataFrame(all_records)
    df_pair_raw = pd.DataFrame(all_paired) if len(all_paired) > 0 else df_all

    # Aggregate metrics across the n_repeats splits
    summary_rows = []
    for (strategy, b_name, model, alpha), grp in df_all.groupby(['strategy', 'bin', 'model', 'alpha'], sort=False):
        skips = grp['fraction_skipped'].values
        wrongs = grp['wrong_skip_rate'].values
        n_cals = grp['n_cal_games'].values
        taus = grp['tau'].values

        mean_tau = float(np.mean(taus)) if np.all(np.isfinite(taus)) else float(np.median(taus))

        summary_rows.append({
            'strategy': strategy,
            'bin': b_name,
            'model': model,
            'alpha': alpha,
            'n_cal_games': float(np.mean(n_cals)),
            'skipped_fraction_mean': float(np.mean(skips)),
            'skip_lo': float(np.percentile(skips, 2.5)),
            'skip_hi': float(np.percentile(skips, 97.5)),
            'wrong_skip_rate_mean': float(np.mean(wrongs)),
            'wrong_skip_lo': float(np.percentile(wrongs, 2.5)),
            'wrong_skip_hi': float(np.percentile(wrongs, 97.5)),
            'fraction_exceeding_alpha': float(np.mean(wrongs > alpha)),
            'tau_mean': mean_tau,
            # Dual column aliases for compatibility
            'fraction_skipped_mean': float(np.mean(skips)),
            'mean_wrong_skip_rate': float(np.mean(wrongs)),
        })

    df_summary = pd.DataFrame(summary_rows).round(4)
    df_paired = compute_mondrian_paired(df_pair_raw)
    df_risk_check = compute_mondrian_risk_check(df_all)

    if return_all:
        return df_summary, df_all, df_paired, df_risk_check
    return df_summary, df_all

