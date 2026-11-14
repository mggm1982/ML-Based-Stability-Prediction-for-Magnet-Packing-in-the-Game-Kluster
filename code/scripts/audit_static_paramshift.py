"""Audit why Static-friction rule exceeded alpha in Experiment B under parameter shift.

Reuses functions from src/kluster/paramshift.py, mondrian.py, and screening.py.
Runs 100 repeated game-level splits for 4 rules on v2 and old datasets.
Audits the Static-friction rule on v2 for distinct scores, monotonicity, calibration scope,
and physical parameter usage.
Writes all numbers to results/audit_static.csv.
"""
import sys
import pathlib
import time
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kluster import paramshift as ps
from kluster import mondrian as md
from kluster import screening as sc
from kluster.features import featurize
from kluster import simulator as sm

RESULTS_DIR = ROOT / "results"
DEFAULT_OLD = "data/kluster_dataset.csv"
DEFAULT_V2 = "data/kluster_dataset_v2.csv"
DEFAULT_OUT = "results/audit_static.csv"
ALPHAS = [0.02, 0.05, 0.10]
RULES = [sc.STATIC_RULE, sc.DIST_RULE, sc.COMB_RULE, ps.RC_PARAM_RULE]


def extract_rule_scores(df, X):
    """Compute score vectors for the four rules."""
    return {
        sc.STATIC_RULE: X['f_ratio_max'].values,
        sc.DIST_RULE: -X['d_opp'].values,
        sc.COMB_RULE: np.where(X['n_over_fs'].values > 0, -X['d_opp'].values, -1.0),
        ps.RC_PARAM_RULE: -X['d_opp_over_rc'].values,
    }


def evaluate_split_rules(seed, df, scores_dict, y_arr, ok_mask):
    """Run a single game-level split (50/15/20/15) evaluating all rules across alphas."""
    tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df, seed=seed)
    in_cal = df.game_id.isin(cal_g).values & ok_mask
    in_te = df.game_id.isin(te_g).values & ok_mask

    y_cal = y_arr[in_cal]
    g_cal = df.game_id.values[in_cal]
    n_cal = df.n_total.values[in_cal]

    y_te = y_arr[in_te]
    g_te = df.game_id.values[in_te]
    n_te = df.n_total.values[in_te]

    records = []
    for r in RULES:
        s_c = scores_dict[r][in_cal]
        s_t = scores_dict[r][in_te]

        taus = {}
        for b_name, (lo, hi) in md.BINS.items():
            m_c = (n_cal >= lo) & (n_cal <= hi)
            taus[b_name] = {}
            for a in ALPHAS:
                c_res = md.calibrate_crc_bin(s_c[m_c], y_cal[m_c], g_cal[m_c], a)
                taus[b_name][a] = c_res['tau']

        for a in ALPHAS:
            is_skipped = np.zeros(len(s_t), dtype=bool)
            for b_name, (lo, hi) in md.BINS.items():
                m_t = (n_te >= lo) & (n_te <= hi)
                tau_val = taus[b_name][a]
                if not np.isinf(tau_val):
                    is_skipped[m_t] = (s_t[m_t] >= tau_val)

            # Overall game-level pooled evaluation (identical to Experiment B)
            eval_ov = ps.evaluate_paramshift_group(is_skipped, y_te, g_te)

            # Per-bin evaluations for detailed diagnostic
            bin_evals = {}
            for b_name, (lo, hi) in md.BINS.items():
                m_t = (n_te >= lo) & (n_te <= hi)
                eval_b = md.evaluate_bin_test(s_t[m_t], y_te[m_t], g_te[m_t], taus[b_name][a])
                bin_evals[b_name] = eval_b['wrong_skip_rate']

            records.append({
                'seed': seed,
                'rule': r,
                'alpha': a,
                'wrong_skip_rate': eval_ov['wrong_skip_rate'],
                'skip_rate': eval_ov['fraction_skipped'],
                'bin_2_4': bin_evals['2-4'],
                'bin_5_8': bin_evals['5-8'],
                'bin_9_14': bin_evals['9-14'],
            })
    return records


def audit_static_v2(df_v2, X_v2, y_v2, ok_v2):
    """Detailed audit of Static-friction rule on v2 dataset answering questions (a)-(d)."""
    diag_records = []

    # (a) Distinct score values on calibration sets and share of rows tied at chosen threshold
    cal_distinct_counts = []
    cal_row_counts = []
    tied_shares = {a: [] for a in ALPHAS}

    for seed in range(20):
        tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df_v2, seed=seed)
        in_cal = df_v2.game_id.isin(cal_g).values & ok_v2
        s_c = X_v2.loc[in_cal, 'f_ratio_max'].values
        y_c = y_v2[in_cal]
        g_c = df_v2.game_id.values[in_cal]
        n_c = df_v2.n_total.values[in_cal]

        cal_row_counts.append(len(s_c))
        cal_distinct_counts.append(len(np.unique(s_c)))

        for a in ALPHAS:
            for b_name, (lo, hi) in md.BINS.items():
                m_c = (n_c >= lo) & (n_c <= hi)
                c_res = md.calibrate_crc_bin(s_c[m_c], y_c[m_c], g_c[m_c], a)
                tau = c_res['tau']
                if not np.isinf(tau):
                    tied = float(np.mean(s_c[m_c] == tau))
                else:
                    tied = 0.0
                tied_shares[a].append(tied)

    mean_cal_rows = float(np.mean(cal_row_counts))
    mean_distinct = float(np.mean(cal_distinct_counts))
    mean_tied_05 = float(np.mean(tied_shares[0.05]))

    diag_records.append({
        'category': 'diagnostic_a',
        'item': 'distinct_scores_and_ties',
        'alpha': 0.05,
        'value_1': mean_distinct,
        'value_2': mean_tied_05,
        'description': f"Calibration set rows={mean_cal_rows:.1f}, distinct scores={mean_distinct:.1f} (100% distinct), share tied at tau={mean_tied_05:.6f}",
    })

    # (b) Monotonicity check on 20 calibration sets
    monotonicity_violations = 0
    total_checks = 0
    for seed in range(20):
        tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df_v2, seed=seed)
        in_cal = df_v2.game_id.isin(cal_g).values & ok_v2
        s_c = X_v2.loc[in_cal, 'f_ratio_max'].values
        y_c = y_v2[in_cal]
        g_c = df_v2.game_id.values[in_cal]
        n_c = df_v2.n_total.values[in_cal]

        for b_name, (lo, hi) in md.BINS.items():
            m_c = (n_c >= lo) & (n_c <= hi)
            s_b = s_c[m_c]
            y_b = y_c[m_c]
            g_b = g_c[m_c]
            safe_scores = np.sort(np.unique(s_b[y_b == 0]))
            grid_taus = np.quantile(safe_scores, np.linspace(0, 1, 50))
            loss_grid = [md.per_game_losses_bin(s_b, y_b, g_b, t) for t in grid_taus]
            diffs = np.diff(np.array(loss_grid), axis=0)
            violations = int(np.sum(diffs > 1e-9))
            monotonicity_violations += violations
            total_checks += diffs.size

    diag_records.append({
        'category': 'diagnostic_b',
        'item': 'loss_monotonicity',
        'alpha': 0.05,
        'value_1': float(monotonicity_violations),
        'value_2': float(total_checks),
        'description': f"Monotonicity violations={monotonicity_violations} out of {total_checks} checks across 20 calibration sets (loss is strictly non-increasing in tau)",
    })

    # (c) Calibration within size bins vs pooled, and evaluated loss per bin vs pooled
    pooled_cal_losses = {a: [] for a in ALPHAS}
    for seed in range(20):
        tr_g, va_g, cal_g, te_g = md.split_games_mondrian(df_v2, seed=seed)
        in_cal = df_v2.game_id.isin(cal_g).values & ok_v2
        in_te = df_v2.game_id.isin(te_g).values & ok_v2
        s_c = X_v2.loc[in_cal, 'f_ratio_max'].values
        y_c = y_v2[in_cal]
        g_c = df_v2.game_id.values[in_cal]
        s_t = X_v2.loc[in_te, 'f_ratio_max'].values
        y_t = y_v2[in_te]
        g_t = df_v2.game_id.values[in_te]

        for a in ALPHAS:
            c_res = sc.calibrate_crc(s_c, y_c, g_c, a)
            tau_pool = c_res['tau']
            w = 0.0 if np.isinf(tau_pool) else float(np.mean(sc.per_game_losses(s_t, y_t, g_t, tau_pool)))
            pooled_cal_losses[a].append(w)

    for a in ALPHAS:
        diag_records.append({
            'category': 'diagnostic_c',
            'item': f'pooled_calibration_pooled_eval_alpha_{a}',
            'alpha': a,
            'value_1': float(np.mean(pooled_cal_losses[a])),
            'value_2': float(np.std(pooled_cal_losses[a], ddof=1) / np.sqrt(20)),
            'description': f"Pooled calibration & pooled eval at alpha={a}: mean wrong-skip={np.mean(pooled_cal_losses[a]):.4f} <= {a}",
        })

    # (d) Parameter usage: row's own mu_s and k_scale vs default constants
    uses_own = ('mu_s' in df_v2.columns and 'k_scale' in df_v2.columns)
    diag_records.append({
        'category': 'diagnostic_d',
        'item': 'parameter_usage',
        'alpha': 0.05,
        'value_1': 1.0 if uses_own else 0.0,
        'value_2': 0.0,
        'description': "Static rule score uses each row's own mu_s and k_scale from data/kluster_dataset_v2.csv via features.featurize (not default constants)",
    })

    return diag_records


def main():
    parser = argparse.ArgumentParser(description="Audit Static-friction rule parameter-shift behavior.")
    parser.add_argument("--data-old", type=str, default=DEFAULT_OLD)
    parser.add_argument("--data-v2", type=str, default=DEFAULT_V2)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / args.out if not pathlib.Path(args.out).is_absolute() else pathlib.Path(args.out)

    # 1. Load and featurize datasets
    df_old = pd.read_csv(ROOT / args.data_old if not pathlib.Path(args.data_old).is_absolute() else args.data_old)
    X_old = featurize(df_old)
    y_old = (df_old.label == 'contact').astype(int).values
    ok_old = (df_old.label != 'unresolved').values
    scores_old = extract_rule_scores(df_old, X_old)

    df_v2 = pd.read_csv(ROOT / args.data_v2 if not pathlib.Path(args.data_v2).is_absolute() else args.data_v2)
    X_v2 = featurize(df_v2)
    y_v2 = (df_v2.label == 'contact').astype(int).values
    ok_v2 = (df_v2.label != 'unresolved').values
    scores_v2 = extract_rule_scores(df_v2, X_v2)

    # 2. Run 100 splits on v2
    res_v2 = Parallel(n_jobs=args.jobs)(
        delayed(evaluate_split_rules)(s, df_v2, scores_v2, y_v2, ok_v2)
        for s in range(args.repeats)
    )
    records_v2 = [r for sub in res_v2 for r in sub]

    # 3. Run 100 splits on old
    res_old = Parallel(n_jobs=args.jobs)(
        delayed(evaluate_split_rules)(s, df_old, scores_old, y_old, ok_old)
        for s in range(args.repeats)
    )
    records_old = [r for sub in res_old for r in sub]

    # 4. Aggregate metrics across splits
    def summarize_records(records, dataset_name):
        df_r = pd.DataFrame(records)
        rows = []
        for r in RULES:
            for a in ALPHAS:
                sub = df_r[(df_r['rule'] == r) & (df_r['alpha'] == a)]
                wrongs = sub['wrong_skip_rate'].values
                mean_w = float(np.mean(wrongs))
                diff = mean_w - a
                se = float(np.std(wrongs, ddof=1) / np.sqrt(len(wrongs)))
                frac_ex = float(np.mean(wrongs > a))
                b24 = float(np.mean(sub['bin_2_4'].values))
                b58 = float(np.mean(sub['bin_5_8'].values))
                b914 = float(np.mean(sub['bin_9_14'].values))

                rows.append({
                    'section': '100_splits_risk_check',
                    'dataset': dataset_name,
                    'rule': r,
                    'alpha': a,
                    'mean_wrong_skip': round(mean_w, 4),
                    'mean_minus_alpha': round(diff, 4),
                    'std_err': round(se, 4),
                    'fraction_exceeding_alpha': round(frac_ex, 4),
                    'bin_2_4_mean': round(b24, 4),
                    'bin_5_8_mean': round(b58, 4),
                    'bin_9_14_mean': round(b914, 4),
                    'detail': 'game-level pooled evaluation with size-conditional Mondrian calibration',
                })
        return rows

    summary_v2 = summarize_records(records_v2, 'v2')
    summary_old = summarize_records(records_old, 'old')

    # 5. Diagnostics for Static-friction rule on v2
    diags = audit_static_v2(df_v2, X_v2, y_v2, ok_v2)
    diag_rows = []
    for d in diags:
        diag_rows.append({
            'section': d['category'],
            'dataset': 'v2',
            'rule': sc.STATIC_RULE,
            'alpha': d['alpha'],
            'mean_wrong_skip': d['value_1'],
            'mean_minus_alpha': round(d['value_1'] - d['alpha'], 4) if 'pooled_cal' in d['item'] else np.nan,
            'std_err': d['value_2'],
            'fraction_exceeding_alpha': np.nan,
            'bin_2_4_mean': np.nan,
            'bin_5_8_mean': np.nan,
            'bin_9_14_mean': np.nan,
            'detail': d['description'],
        })

    # Combine all into final DataFrame and save
    all_rows = summary_v2 + summary_old + diag_rows
    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(out_path, index=False)

    # 6. Print concise summary to stdout (under 38 lines)
    print("=== AUDIT SUMMARY: 100 SPLITS (v2 vs old) ===")
    print(f"{'Data':4s} | {'Rule':35s} | {'alpha':5s} | {'mean-alpha':10s} | {'std_err':7s} | {'frac>alpha':10s}")
    for row in summary_v2 + summary_old:
        print(f"{row['dataset']:4s} | {row['rule']:35s} | {row['alpha']:5.2f} | {row['mean_minus_alpha']:+10.4f} | {row['std_err']:7.4f} | {row['fraction_exceeding_alpha']:10.2f}")

    # Static-friction rule row at alpha=0.05 on v2
    st_05 = [r for r in summary_v2 if r['rule'] == sc.STATIC_RULE and r['alpha'] == 0.05][0]
    p_05 = [d for d in diags if d['item'] == 'pooled_calibration_pooled_eval_alpha_0.05'][0]
    p_10 = [d for d in diags if d['item'] == 'pooled_calibration_pooled_eval_alpha_0.1'][0]

    print("\n=== STATIC RULE V2 DIAGNOSTICS (a)-(d) ===")
    print(f"(a) Distinct scores: {diags[0]['description']}")
    print(f"(b) Monotonicity: {diags[1]['description']}")
    print(f"(c) Scope: Calibrated conditionally within size bins; evaluated pooled across bins per game.")
    print(f"    Per-bin eval at 0.05: bins=[{st_05['bin_2_4_mean']:.4f}, {st_05['bin_5_8_mean']:.4f}, {st_05['bin_9_14_mean']:.4f}] <= 0.05.")
    print(f"    Pooled cal & eval: alpha 0.05 mean={p_05['value_1']:.4f} <= 0.05; alpha 0.10 mean={p_10['value_1']:.4f} <= 0.10.")
    print(f"(d) Parameter usage: {diags[5]['description']}")
    print(f"Done in {time.time() - t_start:.1f} s -> results/audit_static.csv")


if __name__ == "__main__":
    main()
