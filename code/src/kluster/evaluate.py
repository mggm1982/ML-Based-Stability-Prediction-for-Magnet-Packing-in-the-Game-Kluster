"""Training + evaluation (v1.2). Splits are by GAME (rows within a game are dependent), 60/20/20.
Hyper-parameters are chosen on the validation games only; the test games are used once.
Confidence intervals: grouped bootstrap (resample whole test games)."""
import time, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve
from sklearn.inspection import permutation_importance
from . import simulator as sm
from .features import featurize, arrangement, GEOM, PHYS

NAMES = {'LR_geom': 'Logistic (geometry)', 'LR_phys': 'Logistic (+force ratios)',
         'HGB_geom': 'Gradient boosting (geometry)', 'HGB_phys': 'Gradient boosting (+force ratios)'}
RC_RULE = 'Critical-distance rule (r_c = 43 mm, no fitting)'
DIST_RULE = 'Distance rule (tuned threshold)'
COMB_RULE = 'Combined rule (static AND distance)'
STATIC_RULE = 'Static-friction rule'


def split_games(df, seed=7):
    """Split dataset games into train (60%), validation (20%), and test (20%) sets.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing candidate placement rows with a 'game_id' column.
    seed : int, default=7
        Random seed for shuffling game IDs.

    Returns
    -------
    tuple of np.ndarray
        (train_games, val_games, test_games)
    """
    g = df.game_id.unique().copy()
    np.random.default_rng(seed).shuffle(g)
    a, b = int(0.6 * len(g)), int(0.8 * len(g))
    return g[:a], g[a:b], g[b:]


def _tune(X, y, tr, va):
    """Tune hyperparameters for gradient boosting and logistic regression on validation set.

    Parameters
    ----------
    X : pd.DataFrame
        Featurized inputs.
    y : np.ndarray
        Binary contact labels (1=contact, 0=settled).
    tr : np.ndarray
        Boolean mask or integer indices for training rows.
    va : np.ndarray
        Boolean mask or integer indices for validation rows.

    Returns
    -------
    dict
        Dictionary mapping model key to (validation_auc, fitted_model, hyperparameter_dict).
    """
    fitted = {}
    for key, cols in [('HGB_geom', GEOM), ('HGB_phys', PHYS)]:
        best = None
        for lr, depth, l2 in itertools.product([0.05, 0.1], [4, 8], [0.0, 1.0]):
            m = HistGradientBoostingClassifier(random_state=0, learning_rate=lr, max_depth=depth,
                                               l2_regularization=l2, max_iter=200).fit(X.loc[tr, cols], y[tr])
            a = roc_auc_score(y[va], m.predict_proba(X.loc[va, cols])[:, 1])
            if best is None or a > best[0]:
                best = (a, m, dict(lr=lr, depth=depth, l2=l2))
        fitted[key] = best
    for key, cols in [('LR_geom', GEOM), ('LR_phys', PHYS)]:
        best = None
        for c in [0.01, 0.1, 1.0, 10.0]:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=0))
            pipe.fit(X.loc[tr, cols], y[tr])
            a = roc_auc_score(y[va], pipe.predict_proba(X.loc[va, cols])[:, 1])
            if best is None or a > best[0]:
                best = (a, pipe, dict(C=c))
        fitted[key] = best
    return fitted


def _thr_acc(score, y, grid):
    """Find classification threshold on continuous score maximizing validation accuracy.

    Parameters
    ----------
    score : np.ndarray
        Predicted contact risk score or negative distance.
    y : np.ndarray
        Binary true labels.
    grid : np.ndarray
        Candidate threshold values to search over.

    Returns
    -------
    float
        Optimal threshold maximizing accuracy on validation set.
    """
    best_t, best_a = grid[0], -1.0
    for t in grid:
        acc = ((score > t) == y).mean()
        if acc > best_a:
            best_a, best_t = acc, t
    return best_t


def run(df, out_dir, boot=1000, seed=7):
    """Execute complete end-to-end model training, validation, evaluation, and plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation dataset.
    out_dir : str or Path
        Directory where output CSVs and figures will be saved.
    boot : int, default=1000
        Number of grouped bootstrap resamples over test games for confidence intervals.
    seed : int, default=7
        Random seed for data splitting and bootstrap.

    Returns
    -------
    dict
        Comprehensive results dictionary containing metrics, timing, and splits.
    """
    import os
    os.makedirs(f"{out_dir}/figures", exist_ok=True)
    df = df[df.label.isin(['contact', 'settled'])].copy().reset_index(drop=True)
    y = (df.label == 'contact').values.astype(int)
    X = featurize(df)
    tr_g, va_g, te_g = split_games(df, seed)
    tr = df.game_id.isin(tr_g)
    va = df.game_id.isin(va_g)
    te = df.game_id.isin(te_g)
    val_fits = _tune(X, y, tr, va)
    best_ml_key = max(val_fits, key=lambda k: val_fits[k][0])
    best_ml_name = NAMES[best_ml_key]
    d_thr = _thr_acc(-X.loc[va, 'd_opp'].values, y[va], np.linspace(-0.15, -0.015, 300))
    t_opt_ml = _thr_acc(val_fits[best_ml_key][1].predict_proba(X.loc[va, PHYS if 'phys' in best_ml_key else GEOM])[:, 1],
                        y[va], np.linspace(0.05, 0.95, 200))
    # Test evaluation
    te_rows = np.where(te)[0]
    te_games = df.loc[te, 'game_id'].values
    uniq_te_games = np.unique(te_games)
    scores = {}
    for k, (_, model, _) in val_fits.items():
        cols = PHYS if 'phys' in k else GEOM
        scores[NAMES[k]] = model.predict_proba(X.loc[te, cols])[:, 1]
    scores[STATIC_RULE] = df.loc[te, 'static_flag'].astype(float).values
    scores[DIST_RULE] = (-X.loc[te, 'd_opp'].values > d_thr).astype(float)
    scores[RC_RULE] = (X.loc[te, 'd_opp'].values <= sm.critical_distance()).astype(float)
    scores[COMB_RULE] = np.maximum(scores[STATIC_RULE], scores[RC_RULE])
    y_te = y[te]
    preds = {best_ml_name: (scores[best_ml_name] > t_opt_ml).astype(int),
             DIST_RULE: scores[DIST_RULE].astype(int),
             RC_RULE: scores[RC_RULE].astype(int),
             STATIC_RULE: scores[STATIC_RULE].astype(int),
             COMB_RULE: scores[COMB_RULE].astype(int)}
    rng = np.random.default_rng(seed)
    boot_idx = []
    for _ in range(boot):
        sample_games = rng.choice(uniq_te_games, size=len(uniq_te_games), replace=True)
        idx_list = []
        for g in sample_games:
            idx_list.extend(np.where(te_games == g)[0])
        boot_idx.append(np.array(idx_list))
    # Metrics
    rows = []
    for name, s in scores.items():
        is_binary = set(np.unique(s)).issubset({0.0, 1.0})
        auc = roc_auc_score(y_te, s) if not is_binary else np.nan
        ap = average_precision_score(y_te, s) if not is_binary else np.nan
        pred = preds.get(name, (s > 0.5).astype(int))
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        # Bootstrap CI
        b_aucs, b_accs = [], []
        for b in boot_idx:
            yb = y_te[b]
            sb = s[b]
            if len(np.unique(yb)) > 1 and not is_binary:
                b_aucs.append(roc_auc_score(yb, sb))
            pb = preds.get(name, (sb > 0.5).astype(int))[b]
            b_accs.append((pb == yb).mean())
        auc_ci = (np.percentile(b_aucs, 2.5), np.percentile(b_aucs, 97.5)) if b_aucs else (np.nan, np.nan)
        acc_ci = (np.percentile(b_accs, 2.5), np.percentile(b_accs, 97.5))
        rows.append(dict(model=name, AUC=auc, AUC_ci_low=auc_ci[0], AUC_ci_high=auc_ci[1],
                         PR_AUC=ap, accuracy=acc, acc_ci_low=acc_ci[0], acc_ci_high=acc_ci[1],
                         missed_contacts=fn, false_alarms=fp, precision=prec, recall=rec))
    res_df = pd.DataFrame(rows)
    res_df.to_csv(f"{out_dir}/metrics_test.csv", index=False)
    # Paired gains
    b_gain_dist, b_gain_comb = [], []
    for b in boot_idx:
        yb = y_te[b]
        if len(np.unique(yb)) > 1:
            a_ml = roc_auc_score(yb, scores[best_ml_name][b])
            a_d = roc_auc_score(yb, scores[DIST_RULE][b])
            a_c = roc_auc_score(yb, scores[COMB_RULE][b])
            b_gain_dist.append(a_ml - a_d)
            b_gain_comb.append(a_ml - a_c)
    g_dist = (np.mean(b_gain_dist), np.percentile(b_gain_dist, 2.5), np.percentile(b_gain_dist, 97.5))
    g_comb = (np.mean(b_gain_comb), np.percentile(b_gain_comb, 2.5), np.percentile(b_gain_comb, 97.5))
    # Timing
    t0 = time.perf_counter()
    for _ in range(50):
        sm.simulate(np.array([[0.0, 0.0], [0.03, 0.0]]), [1.0, -1.0], 0.06)
    t_sim = (time.perf_counter() - t0) / 50 * 1000
    t0 = time.perf_counter()
    for _ in range(50):
        sm.label_checked(np.array([[0.0, 0.0], [0.03, 0.0]]), [1.0, -1.0], 0.06)
    t_chk = (time.perf_counter() - t0) / 50 * 1000
    t0 = time.perf_counter()
    sub_df = df.iloc[:100]
    _ = featurize(sub_df)
    t_feat = (time.perf_counter() - t0) / 100 * 1000
    cols = PHYS if 'phys' in best_ml_key else GEOM
    t0 = time.perf_counter()
    _ = val_fits[best_ml_key][1].predict_proba(X.loc[:100, cols])
    t_pred = (time.perf_counter() - t0) / 100 * 1000
    t_ml = t_feat + t_pred
    return dict(metrics=res_df, best_ml=best_ml_name, timing_ms=(t_sim, t_chk, t_ml),
                auc_gain_vs_distance=g_dist, auc_gain_vs_combined=g_comb,
                split=(len(tr_g), len(va_g), len(te_g), tr.sum(), va.sum(), te.sum()),
                unresolved_rows=(df.label == 'unresolved').sum(), timeout_fraction=(df.label == 'timeout').mean(),
                distance_threshold_mm=-d_thr * 1000.0)
