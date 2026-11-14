"""Phase 4: Guaranteed-skip screening via Conformal Risk Control (CRC).

Goal: skip simulations safely. Simulate a candidate only if score < tau.
Risk to control: the fraction of truly safe (non-contact) candidates wrongly skipped.
Per-game loss L_g(tau) = fraction of safe candidates in game g with score >= tau.

Conformal risk control (Angelopoulos et al. 2022, arXiv:2208.02814):
    tau_hat = inf { tau : (n / (n + 1)) * mean_g L_g(tau) + 1 / (n + 1) <= alpha }
Guarantee holds as an expectation over games exchangeable with calibration games.
"""
import itertools
import time
from contextlib import nullcontext
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from .features import featurize, arrangement, GEOM, PHYS
from . import simulator as sm

NAMES = {
    'LR_geom': 'Logistic (geometry)',
    'LR_phys': 'Logistic (+force ratios)',
    'HGB_geom': 'Gradient boosting (geometry)',
    'HGB_phys': 'Gradient boosting (+force ratios)',
}
STATIC_RULE = 'Static-friction rule'
DIST_RULE = 'Distance rule'
COMB_RULE = 'Combined rule'

# Exact fixed hyperparameters used in fast mode (fast=True):
# HistGradientBoostingClassifier: learning_rate=0.05, max_depth=8, l2_regularization=1.0, max_iter=200, random_state=0
# LogisticRegression: C=10.0, max_iter=3000 with StandardScaler
FAST_HGB_PARAMS = dict(lr=0.05, depth=8, l2=1.0)
FAST_LR_PARAMS = dict(C=10.0)


def split_games_screening(df, seed=7):
    """Split games into train / validation / calibration / test (50 / 15 / 20 / 15).
    
    Calibration and test games are strictly disjoint and never used for fitting.
    """
    g = np.sort(df.game_id.unique().copy())
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    n = len(g)
    a = int(0.50 * n)
    b = int(0.65 * n)
    c = int(0.85 * n)
    tr_g, va_g, cal_g, te_g = g[:a], g[a:b], g[b:c], g[c:]
    return tr_g, va_g, cal_g, te_g


def tune_models(X, y, tr_idx, va_idx, fast=False):
    """Fit candidate models on train, select best hyperparameters on validation.
    
    When fast=True, uses fixed single hyperparameters:
      HistGradientBoostingClassifier: lr=0.05, depth=8, l2=1.0, max_iter=200, random_state=0
      LogisticRegression: C=10.0, max_iter=3000 (with StandardScaler)
    When fast=False (via --full-tuning), runs the full grid search over learning_rate,
    depth, l2_regularization, and C.
    """
    fitted = {}
    hgb_params = [FAST_HGB_PARAMS] if fast else [
        dict(lr=lr, depth=depth, l2=l2)
        for lr, depth, l2 in itertools.product([0.05, 0.1], [4, 8], [0.0, 1.0])
    ]
    lr_params = [FAST_LR_PARAMS] if fast else [dict(C=c) for c in [0.01, 0.1, 1.0, 10.0]]

    # Gradient boosting
    for key, cols in [('HGB_geom', GEOM), ('HGB_phys', PHYS)]:
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
    for key, cols in [('LR_geom', GEOM), ('LR_phys', PHYS)]:
        best = None
        for p in lr_params:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=p['C'], max_iter=3000))
            m.fit(X.loc[tr_idx, cols], y[tr_idx])
            a = roc_auc_score(y[va_idx], m.predict_proba(X.loc[va_idx, cols])[:, 1])
            if best is None or a > best[0]:
                best = (a, m, cols, p)
        fitted[key] = best

    return fitted


def compute_scores(df_sub, X_sub, fitted_models=None):
    """Compute risk scores for all baselines and ML models on a dataset slice.
    
    Higher score indicates higher risk of contact.
    """
    scores = {}
    # Non-ML baselines
    scores[STATIC_RULE] = X_sub['f_ratio_max'].values
    scores[DIST_RULE] = -X_sub['d_opp'].values
    scores[COMB_RULE] = np.where(X_sub['n_over_fs'].values > 0, -X_sub['d_opp'].values, -1.0)

    # ML models
    if fitted_models is not None:
        for key, (_, m, cols, _) in fitted_models.items():
            scores[NAMES[key]] = m.predict_proba(X_sub[cols])[:, 1]
    return scores


def per_game_losses(scores, y, game_ids, tau):
    """Compute per-game loss L_g(tau).
    
    L_g(tau) = fraction of safe candidates (y == 0) in game g with score >= tau.
    If a game has no safe candidates, loss is 0.0.
    """
    unique_games = np.unique(game_ids)
    losses = np.zeros(len(unique_games), dtype=float)
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


def calibrate_crc(scores_cal, y_cal, game_ids_cal, alpha):
    """Calibrate screening threshold tau via Conformal Risk Control (Angelopoulos et al. 2022).
    
    Condition: (n / (n + 1)) * mean_g L_g(tau) + 1 / (n + 1) <= alpha.
    Selects smallest tau (infimum) satisfying the condition to maximize safe skips.
    If alpha < 1 / (n + 1), returns inf (abstains from skipping).
    """
    unique_games = np.unique(game_ids_cal)
    n = len(unique_games)
    min_possible_bound = 1.0 / (n + 1)

    if alpha < min_possible_bound:
        # Strict finite-sample guarantee cannot be satisfied with n calibration games.
        # Fallback to no-skip (tau = inf) to guarantee risk = 0 <= alpha.
        return {
            'tau': np.inf,
            'mean_loss': 0.0,
            'crc_bound': min_possible_bound,
            'alpha': alpha,
            'feasible': False,
        }

    # Evaluate at sorted unique scores of safe candidates in calibration set
    safe_scores = scores_cal[y_cal == 0]
    sorted_thresholds = np.sort(np.unique(safe_scores))

    # Binary search or scan for smallest threshold satisfying CRC bound
    # Pre-index safe items per game for speed
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
                # fraction with score >= tau
                num_ge = len(s_arr) - np.searchsorted(s_arr, tau, side='left')
                losses.append(num_ge / len(s_arr))
        mean_l = float(np.mean(losses))
        crc = (n / (n + 1)) * mean_l + (1.0 / (n + 1))
        return mean_l, crc

    best_tau = np.inf
    best_mean_l = 0.0
    best_crc = min_possible_bound

    # Scan sorted candidate thresholds
    # We want the SMALLEST tau such that crc <= alpha
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
            high = mid - 1  # look for a smaller tau that also satisfies the bound
        else:
            low = mid + 1

    if found_idx == -1:
        # If no single threshold from the array met the criterion,
        # tau > max(safe_scores) guarantees mean_loss = 0 and crc = 1/(n+1) <= alpha
        best_tau = float(sorted_thresholds[-1] + 1e-6) if len(sorted_thresholds) > 0 else np.inf
        best_mean_l, best_crc = eval_crc(best_tau)

    return {
        'tau': best_tau,
        'mean_loss': best_mean_l,
        'crc_bound': best_crc,
        'alpha': alpha,
        'feasible': True,
    }


def evaluate_screening_test(scores, y_test, game_ids_test, tau, n_boot=500, seed=3):
    """Evaluate screening on test games with grouped-bootstrap confidence intervals."""
    unique_games = np.unique(game_ids_test)

    if np.isinf(tau):
        # Abstention: no candidates skipped
        return {
            'fraction_skipped': 0.0,
            'skip_lo': 0.0,
            'skip_hi': 0.0,
            'wrong_skip_rate': 0.0,
            'wrong_skip_lo': 0.0,
            'wrong_skip_hi': 0.0,
        }

    # Point estimates
    skipped_mask = (scores >= tau)
    frac_skipped = float(np.mean(skipped_mask))

    losses = per_game_losses(scores, y_test, game_ids_test, tau)
    mean_wrong_skip = float(np.mean(losses))

    if n_boot == 0 or n_boot is None:
        return {
            'fraction_skipped': frac_skipped,
            'skip_lo': frac_skipped,
            'skip_hi': frac_skipped,
            'wrong_skip_rate': mean_wrong_skip,
            'wrong_skip_lo': mean_wrong_skip,
            'wrong_skip_hi': mean_wrong_skip,
        }

    # Grouped bootstrap over test games
    rng = np.random.default_rng(seed)
    boot_games = [rng.choice(unique_games, len(unique_games), replace=True) for _ in range(n_boot)]

    game_indices = {g: np.where(game_ids_test == g)[0] for g in unique_games}
    loss_by_game = {g: losses[idx] for idx, g in enumerate(unique_games)}

    boot_skips = []
    boot_wrong_skips = []

    for b in boot_games:
        b_losses = [loss_by_game[g] for g in b]
        boot_wrong_skips.append(float(np.mean(b_losses)))

        idx_concat = np.concatenate([game_indices[g] for g in b])
        boot_skips.append(float(np.mean(scores[idx_concat] >= tau)))

    return {
        'fraction_skipped': frac_skipped,
        'skip_lo': float(np.percentile(boot_skips, 2.5)),
        'skip_hi': float(np.percentile(boot_skips, 97.5)),
        'wrong_skip_rate': mean_wrong_skip,
        'wrong_skip_lo': float(np.percentile(boot_wrong_skips, 2.5)),
        'wrong_skip_hi': float(np.percentile(boot_wrong_skips, 97.5)),
    }


def evaluate_first_safe_calls(df_test, score_dict, n_boot=500, seed=3):
    """Compute simulator calls needed to find the first verified-safe placement in a state.
    
    Evaluated across test states (game_id, step) that contain at least one settled placement.
    Returns point estimates and grouped-bootstrap 95% CIs across test games.
    """
    df_test = df_test.reset_index(drop=True)
    states = []
    unique_games = np.sort(df_test.game_id.unique())

    for (gid, step), grp in df_test.groupby(['game_id', 'step']):
        is_safe = (grp['label'] == 'settled').values
        if not is_safe.any():
            continue
        K = len(grp)
        M = int(is_safe.sum())
        exp_rnd = (K + 1) / (M + 1)

        row_dict = {'game_id': gid, 'step': step, 'random': exp_rnd}
        for model_name, s_vals in score_dict.items():
            sub_scores = s_vals[grp.index.values]
            order = np.argsort(sub_scores)  # p ascending
            first_safe_idx = int(np.where(is_safe[order])[0][0]) + 1  # 1-based call count
            row_dict[model_name] = first_safe_idx
        states.append(row_dict)

    df_states = pd.DataFrame(states)
    model_keys = ['random'] + list(score_dict.keys())

    per_game_means = df_states.groupby('game_id')[model_keys].mean()
    overall_means = df_states[model_keys].mean()

    rng = np.random.default_rng(seed)
    boot_games = [rng.choice(unique_games, len(unique_games), replace=True) for _ in range(n_boot)]

    results = []
    for k in model_keys:
        b_means = [np.mean([per_game_means.loc[g, k] for g in b]) for b in boot_games]
        results.append({
            'method': k,
            'mean_calls': float(overall_means[k]),
            'calls_lo': float(np.percentile(b_means, 2.5)),
            'calls_hi': float(np.percentile(b_means, 97.5)),
        })

    return pd.DataFrame(results)


def run_single_screening_repeat(seed, df, X, y, alphas, fast=True):
    """Run one split, fit models, calibrate CRC, and evaluate test metrics."""
    tr_g, va_g, cal_g, te_g = split_games_screening(df, seed=seed)
    ok = (df.label != 'unresolved').values
    in_tr = df.game_id.isin(tr_g).values & ok
    in_va = df.game_id.isin(va_g).values & ok
    in_cal = df.game_id.isin(cal_g).values & ok
    in_te = df.game_id.isin(te_g).values & ok

    fitted = tune_models(X, y, in_tr, in_va, fast=fast)
    best_ml_key = max(fitted, key=lambda k: fitted[k][0])
    best_ml_name = NAMES[best_ml_key]

    scores_cal = compute_scores(df[in_cal], X.loc[in_cal], fitted)
    scores_te = compute_scores(df[in_te], X.loc[in_te], fitted)

    y_cal = y[in_cal]
    g_cal = df.game_id.values[in_cal]
    y_te = y[in_te]
    g_te = df.game_id.values[in_te]

    records = []
    paired_records = []

    for alpha in alphas:
        comb_skip = None
        best_ml_skip = None

        for rule_name, s_cal in scores_cal.items():
            s_te = scores_te[rule_name]
            c_res = calibrate_crc(s_cal, y_cal, g_cal, alpha)
            tau = c_res['tau']

            if np.isinf(tau):
                s_frac = 0.0
                w_rate = 0.0
            else:
                s_frac = float(np.mean(s_te >= tau))
                w_rate = float(np.mean(per_game_losses(s_te, y_te, g_te, tau)))

            records.append({
                'seed': seed,
                'model': rule_name,
                'alpha': alpha,
                'fraction_skipped': s_frac,
                'wrong_skip_rate': w_rate,
            })

            if rule_name == COMB_RULE:
                comb_skip = s_frac
            if rule_name == best_ml_name:
                best_ml_skip = s_frac

        if comb_skip is not None and best_ml_skip is not None:
            paired_records.append({
                'seed': seed,
                'alpha': alpha,
                'best_ml_model': best_ml_name,
                'best_ml_skip': best_ml_skip,
                'combined_skip': comb_skip,
                'diff': best_ml_skip - comb_skip,
            })

    return records, paired_records


def compute_risk_check(df_all_records):
    """Compute wrong-skip risk check metrics across repeated splits.
    
    Returns DataFrame with columns:
      model, alpha, mean_wrong_skip_rate, wrong_skip_lo, wrong_skip_hi, fraction_exceeding_alpha
    """
    risk_rows = []
    for (model, alpha), grp in df_all_records.groupby(['model', 'alpha']):
        wrongs = grp['wrong_skip_rate'].values
        risk_rows.append({
            'model': model,
            'alpha': alpha,
            'mean_wrong_skip_rate': float(np.mean(wrongs)),
            'wrong_skip_lo': float(np.percentile(wrongs, 2.5)),
            'wrong_skip_hi': float(np.percentile(wrongs, 97.5)),
            'fraction_exceeding_alpha': float(np.mean(wrongs > alpha)),
        })
    return pd.DataFrame(risk_rows).sort_values(['model', 'alpha']).reset_index(drop=True).round(4)


def compute_net_time_paired(df_pair, sim_ms, ml_eval_ms, comb_eval_ms, alphas=None):
    """Compute paired difference in net time saved between best ML model and combined rule across repeats.
    
    Confidence intervals are the 2.5th to 97.5th percentiles over repeats.
    """
    if alphas is None:
        alphas = np.sort(df_pair['alpha'].unique())

    rows = []
    for alpha in alphas:
        grp = df_pair[df_pair['alpha'] == alpha]
        if len(grp) == 0:
            continue
        best_name = grp['best_ml_model'].mode()[0]
        best_skips = grp['best_ml_skip'].values
        comb_skips = grp['combined_skip'].values

        # Net time saved per candidate in ms: s_frac * sim_ms - eval_ms
        best_net_ms = best_skips * sim_ms - ml_eval_ms
        comb_net_ms = comb_skips * sim_ms - comb_eval_ms
        diffs_ms = best_net_ms - comb_net_ms

        # Net time saved in percent of simulator time
        diffs_pct = (diffs_ms / sim_ms) * 100.0 if sim_ms > 0 else np.zeros_like(diffs_ms)

        rows.append({
            'alpha': alpha,
            'best_ml_model': best_name,
            'best_ml_net_time_ms_mean': float(np.mean(best_net_ms)),
            'combined_net_time_ms_mean': float(np.mean(comb_net_ms)),
            'mean_diff_ms': float(np.mean(diffs_ms)),
            'diff_lo_ms': float(np.percentile(diffs_ms, 2.5)),
            'diff_hi_ms': float(np.percentile(diffs_ms, 97.5)),
            'mean_diff_pct': float(np.mean(diffs_pct)),
            'diff_lo_pct': float(np.percentile(diffs_pct, 2.5)),
            'diff_hi_pct': float(np.percentile(diffs_pct, 97.5)),
            'mean_diff': float(np.mean(diffs_ms)),
            'diff_lo': float(np.percentile(diffs_ms, 2.5)),
            'diff_hi': float(np.percentile(diffs_ms, 97.5)),
        })
    return pd.DataFrame(rows).sort_values('alpha').reset_index(drop=True).round(4)


def run_repeated_screening(df, X, y, alphas, n_repeats=200, n_jobs=-1, fast=True, return_risk_check=False, return_raw=False):
    """Repeat the screening procedure n_repeats times with different split seeds.
    
    When fast=True (default), models use fixed hyperparameters (FAST_HGB_PARAMS, FAST_LR_PARAMS).
    Set fast=False for full grid tuning.
    """
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_screening_repeat)(s, df, X, y, alphas, fast=fast)
        for s in range(n_repeats)
    )
    all_records = []
    all_paired = []
    for recs, paired in results:
        all_records.extend(recs)
        all_paired.extend(paired)

    df_all = pd.DataFrame(all_records)
    df_pair = pd.DataFrame(all_paired)

    rep_rows = []
    for (model, alpha), grp in df_all.groupby(['model', 'alpha']):
        skips = grp['fraction_skipped'].values
        wrongs = grp['wrong_skip_rate'].values
        rep_rows.append({
            'model': model,
            'alpha': alpha,
            'fraction_skipped_mean': float(np.mean(skips)),
            'skip_lo': float(np.percentile(skips, 2.5)),
            'skip_hi': float(np.percentile(skips, 97.5)),
            'wrong_skip_rate_mean': float(np.mean(wrongs)),
            'wrong_skip_lo': float(np.percentile(wrongs, 2.5)),
            'wrong_skip_hi': float(np.percentile(wrongs, 97.5)),
            'fraction_exceeding_alpha': float(np.mean(wrongs > alpha)),
        })
    df_repeated = pd.DataFrame(rep_rows).sort_values(['model', 'alpha']).reset_index(drop=True).round(4)

    paired_rows = []
    for alpha, grp in df_pair.groupby('alpha'):
        diffs = grp['diff'].values
        best_name = grp['best_ml_model'].mode()[0]
        paired_rows.append({
            'alpha': alpha,
            'best_ml_model': best_name,
            'best_ml_skip_mean': float(np.mean(grp['best_ml_skip'].values)),
            'combined_skip_mean': float(np.mean(grp['combined_skip'].values)),
            'mean_diff': float(np.mean(diffs)),
            'diff_lo': float(np.percentile(diffs, 2.5)),
            'diff_hi': float(np.percentile(diffs, 97.5)),
        })
    df_paired = pd.DataFrame(paired_rows).sort_values('alpha').reset_index(drop=True).round(4)

    df_risk_check = compute_risk_check(df_all)

    # Attach raw and risk check data for convenience
    df_repeated.attrs['raw_records'] = df_all
    df_paired.attrs['raw_pairs'] = df_pair
    df_repeated.attrs['risk_check'] = df_risk_check

    if return_raw:
        return df_repeated, df_paired, df_risk_check, df_pair
    if return_risk_check:
        return df_repeated, df_paired, df_risk_check
    return df_repeated, df_paired


def measure_screening_times(df, best_model, best_cols, best_model_name, alphas, df_repeated,
                            n_sim=150, n_eval=500, seed=42, n_repeats=5, return_timing=True):
    """Measure evaluation time per candidate and compute net time saved vs simulating every candidate.
    
    Single-threaded execution with no parallel workers running, taking median over n_repeats (default 5).
    """
    try:
        import threadpoolctl
        has_threadpoolctl = True
    except ImportError:
        has_threadpoolctl = False

    context = threadpoolctl.threadpool_limits(limits=1) if has_threadpoolctl else nullcontext()

    with context:
        # 1. Simulator time per candidate across n_repeats
        sim_sample = df.sample(min(n_sim, len(df)), random_state=seed)
        sim_records = sim_sample.to_dict('records')
        sim_repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for r in sim_records:
                P, S = arrangement(r)
                sm.simulate(P, S, r['rw_mm'] / 1000.0)
            t_sim = (time.perf_counter() - t0) / len(sim_records)
            sim_repeats.append(t_sim * 1000.0)
        sim_ms = float(np.median(sim_repeats))

        # 2. Candidate evaluation time for rules and best ML model across n_repeats
        eval_sample = df.sample(min(n_eval, len(df)), random_state=seed)
        eval_records = eval_sample.to_dict('records')

        # Distance rule
        dist_repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for row in eval_records:
                P, S = arrangement(row)
                n = len(S); new = n - 1
                diff = P[:new] - P[new]
                dists = np.linalg.norm(diff, axis=1)
                opp = dists[S[:new] != S[new]]
                _ = -(opp.min() if len(opp) else 0.2)
            t_dist = (time.perf_counter() - t0) / len(eval_records)
            dist_repeats.append(t_dist * 1000.0)
        dist_ms = float(np.median(dist_repeats))

        # Static-friction rule
        static_repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for row in eval_records:
                P, S = arrangement(row)
                rw = row['rw_mm'] / 1000.0
                F = sm.effective_forces(P, S, rw)
                _ = F.max() / sm.FS
            t_static = (time.perf_counter() - t0) / len(eval_records)
            static_repeats.append(t_static * 1000.0)
        static_ms = float(np.median(static_repeats))

        # Combined rule
        comb_repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            for row in eval_records:
                P, S = arrangement(row)
                n = len(S); new = n - 1
                diff = P[:new] - P[new]
                dists = np.linalg.norm(diff, axis=1)
                opp = dists[S[:new] != S[new]]
                d_opp = opp.min() if len(opp) else 0.2
                rw = row['rw_mm'] / 1000.0
                F = sm.effective_forces(P, S, rw)
                _ = -d_opp if (F > sm.FS).any() else -1.0
            t_comb = (time.perf_counter() - t0) / len(eval_records)
            comb_repeats.append(t_comb * 1000.0)
        comb_ms = float(np.median(comb_repeats))

        # Best ML model: featurize + predict
        ml_repeats = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            Xs = featurize(eval_sample)
            best_model.predict_proba(Xs[best_cols])
            t_ml = (time.perf_counter() - t0) / len(eval_sample)
            ml_repeats.append(t_ml * 1000.0)
        ml_ms = float(np.median(ml_repeats))

    timing_rows = [
        {
            'method': 'Simulator',
            'model': 'Simulator',
            'median_ms': round(sim_ms, 5),
            'time_ms': round(sim_ms, 5),
            'repeat_1_ms': round(sim_repeats[0], 5),
            'repeat_2_ms': round(sim_repeats[1], 5),
            'repeat_3_ms': round(sim_repeats[2], 5),
            'repeat_4_ms': round(sim_repeats[3], 5),
            'repeat_5_ms': round(sim_repeats[4], 5),
        },
        {
            'method': STATIC_RULE,
            'model': STATIC_RULE,
            'median_ms': round(static_ms, 5),
            'time_ms': round(static_ms, 5),
            'repeat_1_ms': round(static_repeats[0], 5),
            'repeat_2_ms': round(static_repeats[1], 5),
            'repeat_3_ms': round(static_repeats[2], 5),
            'repeat_4_ms': round(static_repeats[3], 5),
            'repeat_5_ms': round(static_repeats[4], 5),
        },
        {
            'method': DIST_RULE,
            'model': DIST_RULE,
            'median_ms': round(dist_ms, 5),
            'time_ms': round(dist_ms, 5),
            'repeat_1_ms': round(dist_repeats[0], 5),
            'repeat_2_ms': round(dist_repeats[1], 5),
            'repeat_3_ms': round(dist_repeats[2], 5),
            'repeat_4_ms': round(dist_repeats[3], 5),
            'repeat_5_ms': round(dist_repeats[4], 5),
        },
        {
            'method': COMB_RULE,
            'model': COMB_RULE,
            'median_ms': round(comb_ms, 5),
            'time_ms': round(comb_ms, 5),
            'repeat_1_ms': round(comb_repeats[0], 5),
            'repeat_2_ms': round(comb_repeats[1], 5),
            'repeat_3_ms': round(comb_repeats[2], 5),
            'repeat_4_ms': round(comb_repeats[3], 5),
            'repeat_5_ms': round(comb_repeats[4], 5),
        },
        {
            'method': best_model_name,
            'model': best_model_name,
            'median_ms': round(ml_ms, 5),
            'time_ms': round(ml_ms, 5),
            'repeat_1_ms': round(ml_repeats[0], 5),
            'repeat_2_ms': round(ml_repeats[1], 5),
            'repeat_3_ms': round(ml_repeats[2], 5),
            'repeat_4_ms': round(ml_repeats[3], 5),
            'repeat_5_ms': round(ml_repeats[4], 5),
        },
    ]
    df_timing = pd.DataFrame(timing_rows)

    rule_times = {
        STATIC_RULE: static_ms,
        DIST_RULE: dist_ms,
        COMB_RULE: comb_ms,
        best_model_name: ml_ms,
    }

    net_rows = []
    for model_name, eval_ms in rule_times.items():
        sub_rep = df_repeated[df_repeated['model'] == model_name]
        for alpha in alphas:
            alpha_match = sub_rep[sub_rep['alpha'] == alpha]
            if len(alpha_match) > 0:
                s_frac = float(alpha_match['fraction_skipped_mean'].values[0])
            else:
                s_frac = 0.0

            net_saved_ms = s_frac * sim_ms - eval_ms
            net_saved_pct = (net_saved_ms / sim_ms) * 100.0 if sim_ms > 0 else 0.0
            screened_time_ms = eval_ms + (1.0 - s_frac) * sim_ms
            speedup = sim_ms / screened_time_ms if screened_time_ms > 0 else np.nan

            net_rows.append({
                'model': model_name,
                'alpha': alpha,
                'eval_time_ms': round(eval_ms, 4),
                'sim_time_ms': round(sim_ms, 4),
                'skipped_fraction': round(s_frac, 4),
                'net_time_saved_ms': round(net_saved_ms, 4),
                'net_time_saved_pct': round(net_saved_pct, 2),
                'speedup': round(speedup, 2),
            })

    df_net_time = pd.DataFrame(net_rows)
    df_net_time.attrs['timing'] = df_timing

    if return_timing:
        return df_net_time, df_timing
    return df_net_time
