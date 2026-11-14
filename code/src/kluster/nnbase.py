"""Neural network baseline for Mondrian Conformal Risk Control screening.

Compares a small neural network against Gradient boosting (+parameters)
and Logistic (+parameters) under size-conditional Mondrian CRC screening on
data/kluster_dataset_v2.csv.

Model specification:
  - Neural network: scikit-learn MLPClassifier(hidden_layer_sizes=(32, 16),
    early_stopping=True, max_iter=60, random_state=seed) preceded by
    StandardScaler() in a pipeline.
  - Gradient boosting (+parameters): HistGradientBoostingClassifier on PHYS_PARAMS.
  - Logistic (+parameters): LogisticRegression with StandardScaler() on PHYS_PARAMS.

Evaluation:
  - Game-level splits (50 / 15 / 20 / 15: train / val / cal / test).
  - Mondrian size bins: '2-4', '5-8', '9-14'.
  - Alphas: 0.02, 0.05, 0.10.
  - Reports within-bin and pooled wrong-skip rate, mean skip rate, std_err,
    and training time per repeat.
  - Reports paired difference in skip rate between Neural network and
    Gradient boosting (+parameters) with a 95% interval per size bin and alpha.
"""
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .features import PHYS_PARAMS
from .mondrian import (
    BINS,
    BIN_NAMES,
    split_games_mondrian,
    calibrate_crc_bin,
    evaluate_bin_test,
)
from .paramshift import per_game_losses_group
from .screening import FAST_HGB_PARAMS, FAST_LR_PARAMS

NN_MODEL = 'Neural network'
HGB_MODEL = 'Gradient boosting (+parameters)'
LR_MODEL = 'Logistic (+parameters)'

MODELS = [NN_MODEL, HGB_MODEL, LR_MODEL]
DEFAULT_ALPHAS = [0.02, 0.05, 0.10]
FEATURE_COLS = PHYS_PARAMS


def build_models(seed):
    """Instantiate candidate ML models for repeat with given seed."""
    models = {
        NN_MODEL: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                early_stopping=True,
                max_iter=60,
                random_state=seed,
            ),
        ),
        HGB_MODEL: HistGradientBoostingClassifier(
            random_state=0,
            learning_rate=FAST_HGB_PARAMS['lr'],
            max_depth=FAST_HGB_PARAMS['depth'],
            l2_regularization=FAST_HGB_PARAMS['l2'],
            max_iter=200,
        ),
        LR_MODEL: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=FAST_LR_PARAMS['C'],
                max_iter=3000,
                random_state=0,
            ),
        ),
    }
    return models


def run_single_nn_repeat(seed, df, X, y, alphas=DEFAULT_ALPHAS):
    """Execute one repeated split for neural network baseline evaluation.
    
    Reuses game-level splits and label handling (unresolved dropped,
    timeout treated as safe y=0) from Experiment B.
    """
    tr_g, va_g, cal_g, te_g = split_games_mondrian(df, seed=seed)
    ok = (df['label'] != 'unresolved').values

    in_tr = df['game_id'].isin(tr_g).values & ok
    in_cal = df['game_id'].isin(cal_g).values & ok
    in_te = df['game_id'].isin(te_g).values & ok

    X_tr = X.loc[in_tr, FEATURE_COLS]
    y_tr = y[in_tr]

    X_cal = X.loc[in_cal, FEATURE_COLS]
    y_cal = y[in_cal]
    g_cal = df['game_id'].values[in_cal]
    n_cal = df['n_total'].values[in_cal]

    X_te = X.loc[in_te, FEATURE_COLS]
    y_te = y[in_te]
    g_te = df['game_id'].values[in_te]
    n_te = df['n_total'].values[in_te]

    # Instantiate and fit models, measuring training time per model
    models = build_models(seed)
    train_times = {}
    scores_cal = {}
    scores_te = {}

    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0
        train_times[name] = t_fit

        # Predict risk scores (probability of contact, score >= tau implies skip)
        scores_cal[name] = m.predict_proba(X_cal)[:, 1]
        scores_te[name] = m.predict_proba(X_te)[:, 1]

    # Calibrate size-conditional Mondrian thresholds
    # taus[model][alpha][bin] = tau
    taus = {m: {a: {} for a in alphas} for m in MODELS}
    for b_name, (lo, hi) in BINS.items():
        m_c_b = (n_cal >= lo) & (n_cal <= hi)
        y_c_b = y_cal[m_c_b]
        g_c_b = g_cal[m_c_b]

        for m in MODELS:
            s_c_b = scores_cal[m][m_c_b]
            for a in alphas:
                c_res = calibrate_crc_bin(s_c_b, y_c_b, g_c_b, a)
                taus[m][a][b_name] = c_res['tau']

    records = []
    for m in MODELS:
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

                records.append({
                    'seed': seed,
                    'model': m,
                    'alpha': a,
                    'bin': b_name,
                    'wrong_skip_bin': eval_b['wrong_skip_rate'],
                    'wrong_skip_pooled': wrong_skip_pooled,
                    'skip_rate': eval_b['fraction_skipped'],
                    'train_time': train_times[m],
                })

    return records


def aggregate_nn_baseline(raw_records, alphas=DEFAULT_ALPHAS):
    """Aggregate per-split records into results/nn_baseline.csv summary table."""
    df_raw = pd.DataFrame(raw_records)
    summary_rows = []

    for model in MODELS:
        for a in sorted(alphas):
            for b_name in BIN_NAMES:
                sub = df_raw[
                    (df_raw['model'] == model) &
                    (df_raw['alpha'] == a) &
                    (df_raw['bin'] == b_name)
                ]
                if len(sub) == 0:
                    continue

                w_bin = sub['wrong_skip_bin'].values
                w_pooled = sub['wrong_skip_pooled'].values
                s_rate = sub['skip_rate'].values
                t_time = sub['train_time'].values
                n_rep = len(sub)

                mean_w_bin = round(float(np.mean(w_bin)), 4)
                mean_w_pooled = round(float(np.mean(w_pooled)), 4)
                mean_s_rate = round(float(np.mean(s_rate)), 4)
                std_err = round(float(np.std(w_bin, ddof=1) / np.sqrt(n_rep)), 4) if n_rep > 1 else 0.0
                mean_t_time = round(float(np.mean(t_time)), 4)

                summary_rows.append({
                    'model': model,
                    'alpha': a,
                    'bin': b_name,
                    'wrong_skip_bin': mean_w_bin,
                    'wrong_skip_pooled': mean_w_pooled,
                    'mean_skip_rate': mean_s_rate,
                    'skip_rate': mean_s_rate,
                    'std_err': std_err,
                    'train_time_s': mean_t_time,
                    'n_repeats': n_rep,
                })

    return pd.DataFrame(summary_rows)


def compute_paired_diff(raw_records, model_a=NN_MODEL, model_b=HGB_MODEL, alphas=DEFAULT_ALPHAS):
    """Compute paired differences in skip rate between model_a and model_b over identical splits.
    
    Returns summary DataFrame with 95% percentile interval per size bin and alpha.
    """
    df_raw = pd.DataFrame(raw_records)
    rows = []

    for b_name in BIN_NAMES:
        for a in sorted(alphas):
            sub_a = df_raw[(df_raw['model'] == model_a) & (df_raw['bin'] == b_name) & (df_raw['alpha'] == a)]
            sub_b = df_raw[(df_raw['model'] == model_b) & (df_raw['bin'] == b_name) & (df_raw['alpha'] == a)]

            if len(sub_a) == 0 or len(sub_b) == 0:
                continue

            merged = pd.merge(
                sub_a[['seed', 'skip_rate']],
                sub_b[['seed', 'skip_rate']],
                on='seed',
                suffixes=('_a', '_b'),
            )

            if len(merged) == 0:
                continue

            skips_a = merged['skip_rate_a'].values
            skips_b = merged['skip_rate_b'].values
            diffs = skips_a - skips_b
            n_rep = len(diffs)

            mean_a = round(float(np.mean(skips_a)), 4)
            mean_b = round(float(np.mean(skips_b)), 4)
            mean_diff = round(float(np.mean(diffs)), 4)

            # If difference is identical (e.g. self-paired), exact zero intervals
            if np.allclose(diffs, 0.0, atol=1e-12):
                diff_lo = 0.0
                diff_hi = 0.0
                mean_diff = 0.0
            else:
                diff_lo = round(float(np.percentile(diffs, 2.5)), 4)
                diff_hi = round(float(np.percentile(diffs, 97.5)), 4)

            rows.append({
                'bin': b_name,
                'alpha': a,
                'model_a': model_a,
                'model_b': model_b,
                'nn_skip_mean': mean_a if model_a == NN_MODEL else mean_b,
                'hgb_skip_mean': mean_b if model_b == HGB_MODEL else mean_a,
                'mean_skip_a': mean_a,
                'mean_skip_b': mean_b,
                'mean_diff': mean_diff,
                'diff_lo': diff_lo,
                'diff_hi': diff_hi,
                'n_repeats': n_rep,
            })

    return pd.DataFrame(rows)


def run_repeated_nn(
    df, X, y,
    alphas=DEFAULT_ALPHAS,
    n_repeats=20,
    n_jobs=-1,
    return_raw=False,
):
    """Run repeated neural network baseline experiments over n_repeats splits."""
    if n_jobs == 1:
        results = [
            run_single_nn_repeat(seed, df, X, y, alphas=alphas)
            for seed in range(n_repeats)
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_nn_repeat)(seed, df, X, y, alphas=alphas)
            for seed in range(n_repeats)
        )

    all_records = [rec for rep in results for rec in rep]
    df_baseline = aggregate_nn_baseline(all_records, alphas=alphas)
    df_paired = compute_paired_diff(all_records, model_a=NN_MODEL, model_b=HGB_MODEL, alphas=alphas)

    if return_raw:
        return df_baseline, df_paired, pd.DataFrame(all_records)
    return df_baseline, df_paired

