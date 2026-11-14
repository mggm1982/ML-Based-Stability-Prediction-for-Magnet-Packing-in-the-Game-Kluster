"""Phase 5: Generalisation under size shift and feature ablations.

Size shift setup:
  - Split games into train / validation / calibration / test (50 / 15 / 20 / 15).
  - Train and tune on candidates with n_total <= 8 from train and validation games.
  - Calibrate screening threshold (CRC) on calibration games with n_total <= 8.
  - Test on held-out test games with n_total >= 9 (shifted), and n_total <= 8 (comparison).
  - Ablations: refit best model type without d_opp, and without d_opp & force-ratio features.
  - Repeat everything over 20 different game splits and report mean and 2.5th-97.5th percentiles.
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, confusion_matrix
from joblib import Parallel, delayed

from .features import GEOM, PHYS
from . import simulator as sm
from .screening import calibrate_crc, per_game_losses, FAST_HGB_PARAMS, FAST_LR_PARAMS

NAMES = {
    'LR_geom': 'Logistic (geometry)',
    'LR_phys': 'Logistic (+force ratios)',
    'HGB_geom': 'Gradient boosting (geometry)',
    'HGB_phys': 'Gradient boosting (+force ratios)',
}

STATIC_RULE = 'Static-friction rule'
DIST_RULE = 'Distance rule'
COMB_RULE = 'Combined rule'
RC_RULE = 'Critical-distance rule (r_c = 43 mm)'


def split_games_shift(df, seed=7, fractions=(0.50, 0.15, 0.20, 0.15)):
    """Split unique games into train, validation, calibration, and test games.
    
    All splits are strictly disjoint partitions of the games.
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


def tune_models_shift(X, y, tr_idx, va_idx, fast=True):
    """Fit candidate models on train (N <= 8), select best hyperparameters on validation (N <= 8)."""
    fitted = {}
    hgb_params = [FAST_HGB_PARAMS] if fast else [
        dict(lr=lr, depth=depth, l2=l2)
        for lr, depth, l2 in itertools.product([0.05, 0.1], [4, 8], [0.0, 1.0])
    ]
    lr_params = [FAST_LR_PARAMS] if fast else [dict(C=c) for c in [0.01, 0.1, 1.0, 10.0]]

    # Gradient boosting models
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

    # Logistic regression models
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


def tune_baseline_thresholds(X, y, tr_idx):
    """Tune distance rule and combined rule thresholds on training set (N <= 8)."""
    grid = np.linspace(0.015, 0.12, 211)
    d_tr = X.loc[tr_idx, 'd_opp'].values
    y_tr = y[tr_idx]

    # Distance rule threshold (maximize accuracy)
    accs_dist = [((d_tr < c) == (y_tr == 1)).mean() for c in grid]
    thr_dist = grid[int(np.argmax(accs_dist))]

    # Combined rule threshold (maximize accuracy given n_over_fs > 0)
    stat_tr = (X.loc[tr_idx, 'n_over_fs'].values > 0)
    accs_comb = [(((d_tr < c) & stat_tr) == (y_tr == 1)).mean() for c in grid]
    thr_comb = grid[int(np.argmax(accs_comb))]

    return thr_dist, thr_comb


def compute_scores_and_preds(df_sub, X_sub, fitted_models, thr_dist, thr_comb):
    """Compute risk scores and binary predictions for all baselines and fitted ML models."""
    rc = sm.critical_distance()
    scores = {}
    preds = {}

    # Static-friction rule
    scores[STATIC_RULE] = X_sub['f_ratio_max'].values
    preds[STATIC_RULE] = (X_sub['n_over_fs'].values > 0).astype(int)

    # Distance rule
    scores[DIST_RULE] = -X_sub['d_opp'].values
    preds[DIST_RULE] = (X_sub['d_opp'].values < thr_dist).astype(int)

    # Combined rule
    scores[COMB_RULE] = np.where(X_sub['n_over_fs'].values > 0, -X_sub['d_opp'].values, -1.0)
    preds[COMB_RULE] = ((X_sub['d_opp'].values < thr_comb) & (X_sub['n_over_fs'].values > 0)).astype(int)

    # Critical-distance rule (fixed 43 mm)
    scores[RC_RULE] = -X_sub['d_opp'].values
    preds[RC_RULE] = (X_sub['d_opp'].values < rc).astype(int)

    # ML models
    if fitted_models is not None:
        for key, (_, m, cols, _) in fitted_models.items():
            name = NAMES[key]
            prob = m.predict_proba(X_sub[cols])[:, 1]
            scores[name] = prob
            preds[name] = (prob >= 0.5).astype(int)

    return scores, preds


def compute_test_metrics_with_bootstrap(scores_dict, preds_dict, y_test, game_ids_test, n_boot=500, seed=3):
    """Compute AUC, accuracy, missed contacts, and false alarms with grouped-bootstrap CIs over test games."""
    unique_games = np.unique(game_ids_test)
    n_games = len(unique_games)

    game_indices = {g: np.where(game_ids_test == g)[0] for g in unique_games}
    rng = np.random.default_rng(seed)
    boot_games = [rng.choice(unique_games, n_games, replace=True) for _ in range(n_boot)]
    boot_indices = [np.concatenate([game_indices[g] for g in b]) for b in boot_games]

    rows = []
    for name in scores_dict:
        s = scores_dict[name]
        p = preds_dict[name]

        auc = float(roc_auc_score(y_test, s))
        acc = float((p == y_test).mean())
        tn, fp, fn, tp = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        missed = int(fn)
        fa = int(fp)

        b_aucs = []
        b_accs = []
        b_miss = []
        b_fa = []

        for idx in boot_indices:
            y_b = y_test[idx]
            s_b = s[idx]
            p_b = p[idx]
            if len(np.unique(y_b)) > 1:
                b_aucs.append(roc_auc_score(y_b, s_b))
            b_accs.append((p_b == y_b).mean())
            b_miss.append(int(((p_b == 0) & (y_b == 1)).sum()))
            b_fa.append(int(((p_b == 1) & (y_b == 0)).sum()))

        rows.append({
            'model': name,
            'AUC': round(auc, 4),
            'AUC_lo': round(float(np.percentile(b_aucs, 2.5)), 4) if b_aucs else np.nan,
            'AUC_hi': round(float(np.percentile(b_aucs, 97.5)), 4) if b_aucs else np.nan,
            'accuracy': round(acc, 4),
            'acc_lo': round(float(np.percentile(b_accs, 2.5)), 4),
            'acc_hi': round(float(np.percentile(b_accs, 97.5)), 4),
            'missed_contacts': missed,
            'missed_lo': round(float(np.percentile(b_miss, 2.5)), 1),
            'missed_hi': round(float(np.percentile(b_miss, 97.5)), 1),
            'false_alarms': fa,
            'false_alarm_lo': round(float(np.percentile(b_fa, 2.5)), 1),
            'false_alarm_hi': round(float(np.percentile(b_fa, 97.5)), 1),
        })
    return pd.DataFrame(rows)


def evaluate_shift_crc(scores_cal, y_cal, g_cal, scores_te, y_te, g_te, alphas=(0.02, 0.05, 0.10)):
    """Calibrate CRC screening threshold on calibration rows (N <= 8) and measure performance on test rows."""
    rows = []
    for model_name in scores_cal:
        s_c = scores_cal[model_name]
        s_t = scores_te[model_name]
        for alpha in alphas:
            c_res = calibrate_crc(s_c, y_cal, g_cal, alpha)
            tau = c_res['tau']
            if np.isinf(tau):
                s_frac = 0.0
                w_rate = 0.0
            else:
                s_frac = float(np.mean(s_t >= tau))
                losses = per_game_losses(s_t, y_te, g_te, tau)
                w_rate = float(np.mean(losses))

            rows.append({
                'model': model_name,
                'alpha': alpha,
                'tau': tau,
                'fraction_skipped': s_frac,
                'wrong_skip_rate': w_rate,
                'guarantee_survived': (w_rate <= alpha),
            })
    return pd.DataFrame(rows)


def run_ablation_shift(X, y, tr_idx, te_idx, best_fitted_model, best_cols):
    """Refit best model type without d_opp, and without d_opp & force-ratio features."""
    force_cols = ['f_ratio_max', 'f_ratio_new', 'n_over_fs']
    cols_full = list(best_cols)
    cols_no_dopp = [c for c in cols_full if c != 'd_opp']
    cols_no_dopp_no_force = [c for c in cols_full if c != 'd_opp' and c not in force_cols]

    y_tr = y[tr_idx]
    y_te = y[te_idx]

    # 1. Full model
    m_full = clone(best_fitted_model).fit(X.loc[tr_idx, cols_full], y_tr)
    auc_full = float(roc_auc_score(y_te, m_full.predict_proba(X.loc[te_idx, cols_full])[:, 1]))

    # 2. Without d_opp
    m_no_dopp = clone(best_fitted_model).fit(X.loc[tr_idx, cols_no_dopp], y_tr)
    auc_no_dopp = float(roc_auc_score(y_te, m_no_dopp.predict_proba(X.loc[te_idx, cols_no_dopp])[:, 1]))

    # 3. Without d_opp and force-ratio features
    m_no_force = clone(best_fitted_model).fit(X.loc[tr_idx, cols_no_dopp_no_force], y_tr)
    auc_no_force = float(roc_auc_score(y_te, m_no_force.predict_proba(X.loc[te_idx, cols_no_dopp_no_force])[:, 1]))

    return [
        {
            'ablation': 'Full model',
            'features': f"{len(cols_full)} features",
            'AUC': auc_full,
            'model_obj': m_full,
            'cols': cols_full,
        },
        {
            'ablation': 'Without d_opp',
            'features': f"{len(cols_no_dopp)} features",
            'AUC': auc_no_dopp,
            'model_obj': m_no_dopp,
            'cols': cols_no_dopp,
        },
        {
            'ablation': 'Without d_opp and force ratios',
            'features': f"{len(cols_no_dopp_no_force)} features",
            'AUC': auc_no_force,
            'model_obj': m_no_force,
            'cols': cols_no_dopp_no_force,
        },
    ]


def run_single_shift_repeat(seed, df, X, y, alphas=(0.02, 0.05, 0.10), fast=True):
    """Execute one full shift experiment for a given game split seed."""
    tr_g, va_g, cal_g, te_g = split_games_shift(df, seed=seed)

    ok = (df.label != 'unresolved').values
    y_arr = y if isinstance(y, np.ndarray) else (df.label == 'contact').astype(int).values
    n_total = df.n_total.values
    game_ids = df.game_id.values

    # Masks for splits and size slices
    tr_mask = df.game_id.isin(tr_g).values & ok & (n_total <= 8)
    va_mask = df.game_id.isin(va_g).values & ok & (n_total <= 8)
    cal_mask = df.game_id.isin(cal_g).values & ok & (n_total <= 8)
    te_ge9_mask = df.game_id.isin(te_g).values & ok & (n_total >= 9)
    te_le8_mask = df.game_id.isin(te_g).values & ok & (n_total <= 8)

    # 1. Fit & tune models on train/val N <= 8
    fitted = tune_models_shift(X, y_arr, tr_mask, va_mask, fast=fast)
    best_ml_key = max(fitted, key=lambda k: fitted[k][0])
    best_ml_model = fitted[best_ml_key][1]
    best_ml_cols = fitted[best_ml_key][2]

    # 2. Tune baseline thresholds on train N <= 8
    thr_dist, thr_comb = tune_baseline_thresholds(X, y_arr, tr_mask)

    # 3. Compute scores & predictions for cal, test N >= 9, test N <= 8
    scores_cal, _ = compute_scores_and_preds(df[cal_mask], X.loc[cal_mask], fitted, thr_dist, thr_comb)
    scores_te_ge9, preds_te_ge9 = compute_scores_and_preds(df[te_ge9_mask], X.loc[te_ge9_mask], fitted, thr_dist, thr_comb)
    scores_te_le8, preds_te_le8 = compute_scores_and_preds(df[te_le8_mask], X.loc[te_le8_mask], fitted, thr_dist, thr_comb)

    y_cal, g_cal = y_arr[cal_mask], game_ids[cal_mask]
    y_ge9, g_ge9 = y_arr[te_ge9_mask], game_ids[te_ge9_mask]
    y_le8, g_le8 = y_arr[te_le8_mask], game_ids[te_le8_mask]

    # Metrics on test N >= 9 (shifted)
    metric_records = []
    for test_set_name, s_dict, p_dict, y_t in [('N>=9 (shifted)', scores_te_ge9, preds_te_ge9, y_ge9),
                                               ('N<=8 (comparison)', scores_te_le8, preds_te_le8, y_le8)]:
        for model_name in s_dict:
            s_ = s_dict[model_name]
            p_ = p_dict[model_name]
            auc = float(roc_auc_score(y_t, s_))
            acc = float((p_ == y_t).mean())
            tn, fp, fn, tp = confusion_matrix(y_t, p_, labels=[0, 1]).ravel()
            metric_records.append({
                'seed': seed,
                'test_set': test_set_name,
                'model': model_name,
                'AUC': auc,
                'accuracy': acc,
                'missed_contacts': int(fn),
                'false_alarms': int(fp),
            })

    # CRC screening evaluation on test N >= 9 (and N <= 8)
    crc_records = []
    for test_set_name, s_t_dict, y_t, g_t in [('N>=9 (shifted)', scores_te_ge9, y_ge9, g_ge9),
                                              ('N<=8 (comparison)', scores_te_le8, y_le8, g_le8)]:
        df_crc = evaluate_shift_crc(scores_cal, y_cal, g_cal, s_t_dict, y_t, g_t, alphas=alphas)
        for r in df_crc.to_dict('records'):
            r['seed'] = seed
            r['test_set'] = test_set_name
            crc_records.append(r)

    # Ablation evaluation on best model type
    ablation_records = []
    for test_set_name, te_m in [('N>=9 (shifted)', te_ge9_mask), ('N<=8 (comparison)', te_le8_mask)]:
        abl_list = run_ablation_shift(X, y_arr, tr_mask, te_m, best_ml_model, best_ml_cols)
        for r in abl_list:
            ablation_records.append({
                'seed': seed,
                'test_set': test_set_name,
                'ablation': r['ablation'],
                'features': r['features'],
                'AUC': r['AUC'],
            })

    return {
        'metrics': metric_records,
        'crc': crc_records,
        'ablation': ablation_records,
        'best_ml_model': NAMES[best_ml_key],
    }


def run_repeated_shift(df, X, y, alphas=(0.02, 0.05, 0.10), n_repeats=20, n_jobs=-1, fast=True):
    """Repeat the size shift experiment over n_repeats different game splits."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_shift_repeat)(s, df, X, y, alphas=alphas, fast=fast)
        for s in range(n_repeats)
    )

    all_metrics = []
    all_crc = []
    all_ablation = []

    for res in results:
        all_metrics.extend(res['metrics'])
        all_crc.extend(res['crc'])
        all_ablation.extend(res['ablation'])

    df_all_metrics = pd.DataFrame(all_metrics)
    df_all_crc = pd.DataFrame(all_crc)
    df_all_ablation = pd.DataFrame(all_ablation)

    # Aggregate test metrics across splits
    rep_metrics = []
    for (test_set, model), grp in df_all_metrics.groupby(['test_set', 'model']):
        aucs = grp['AUC'].values
        accs = grp['accuracy'].values
        miss = grp['missed_contacts'].values
        fa = grp['false_alarms'].values

        rep_metrics.append({
            'test_set': test_set,
            'model': model,
            'AUC_mean': round(float(np.mean(aucs)), 4),
            'AUC_lo': round(float(np.percentile(aucs, 2.5)), 4),
            'AUC_hi': round(float(np.percentile(aucs, 97.5)), 4),
            'accuracy_mean': round(float(np.mean(accs)), 4),
            'acc_lo': round(float(np.percentile(accs, 2.5)), 4),
            'acc_hi': round(float(np.percentile(accs, 97.5)), 4),
            'missed_mean': round(float(np.mean(miss)), 1),
            'missed_lo': round(float(np.percentile(miss, 2.5)), 1),
            'missed_hi': round(float(np.percentile(miss, 97.5)), 1),
            'false_alarms_mean': round(float(np.mean(fa)), 1),
            'false_alarm_lo': round(float(np.percentile(fa, 2.5)), 1),
            'false_alarm_hi': round(float(np.percentile(fa, 97.5)), 1),
        })
    df_metrics_summary = pd.DataFrame(rep_metrics).sort_values(['test_set', 'AUC_mean'], ascending=[False, False]).reset_index(drop=True)

    # Aggregate CRC screening metrics across splits
    rep_crc = []
    for (test_set, model, alpha), grp in df_all_crc.groupby(['test_set', 'model', 'alpha']):
        skips = grp['fraction_skipped'].values
        wrongs = grp['wrong_skip_rate'].values

        rep_crc.append({
            'test_set': test_set,
            'model': model,
            'alpha': alpha,
            'fraction_skipped_mean': round(float(np.mean(skips)), 4),
            'skip_lo': round(float(np.percentile(skips, 2.5)), 4),
            'skip_hi': round(float(np.percentile(skips, 97.5)), 4),
            'wrong_skip_rate_mean': round(float(np.mean(wrongs)), 4),
            'wrong_skip_lo': round(float(np.percentile(wrongs, 2.5)), 4),
            'wrong_skip_hi': round(float(np.percentile(wrongs, 97.5)), 4),
            'fraction_exceeding_alpha': round(float(np.mean(wrongs > alpha)), 4),
            'guarantee_survived': bool(np.mean(wrongs) <= alpha),
        })
    df_crc_summary = pd.DataFrame(rep_crc).sort_values(['test_set', 'model', 'alpha']).reset_index(drop=True)

    # Aggregate ablation metrics across splits
    rep_ablation = []
    for (test_set, ablation), grp in df_all_ablation.groupby(['test_set', 'ablation']):
        aucs = grp['AUC'].values
        feats = grp['features'].iloc[0]

        rep_ablation.append({
            'test_set': test_set,
            'ablation': ablation,
            'features': feats,
            'AUC_mean': round(float(np.mean(aucs)), 4),
            'AUC_lo': round(float(np.percentile(aucs, 2.5)), 4),
            'AUC_hi': round(float(np.percentile(aucs, 97.5)), 4),
        })
    df_ablation_summary = pd.DataFrame(rep_ablation).sort_values(['test_set', 'AUC_mean'], ascending=[False, False]).reset_index(drop=True)

    return {
        'metrics_repeated': df_metrics_summary,
        'crc_repeated': df_crc_summary,
        'ablation_repeated': df_ablation_summary,
        'raw_metrics': df_all_metrics,
        'raw_crc': df_all_crc,
        'raw_ablation': df_all_ablation,
    }

