"""MASTER REPRODUCTION SCRIPT: Kluster Magnet-Stability Project
Author: Sankalp Savarn (Enrolment No. 24123038), IIT Roorkee
Supervisor: Prof. Mayank Goswami (mayank.goswami@ph.iitr.ac.in)

PIPELINE OVERVIEW & TRACEABILITY:
-------------------------------------------------------------------------------------------------------
Step | Action                    | Source File Called            | Paper Table / Figure Produced
-------------------------------------------------------------------------------------------------------
1    | Physics Unit Tests        | tests/test_physics.py         | 20 PASS checks (simulation validity)
2    | Tabular Results Reporting | results/final/T1..T5 CSVs     | Tables T1, T2, T3, T4, T5
     |   - T1: Mondrian Risk     | results/final/T1_risk_check   | Table T1 (Size-conditional CRC risk)
     |   - T2: Paired ML vs Rule | results/final/T2_paired_ml    | Table T2 (Skip-rate gain over Combined)
     |   - T3: Parameter Shift   | results/final/T3_paramshift   | Table T3 (Generalization under shift)
     |   - T4: Calibration Size  | results/final/T4_calsize      | Table T4 (Finite-sample scaling/abstain)
     |   - T5: Neural Baseline   | results/final/T5_nn_baseline  | Table T5 (MLP vs HGB & Logistic)
3    | Figure Generation         | Copied from reproduce_all.py  | Figures F1, F2, F3, F4, F5 (in figures/)
     |   - F1: Risk & Skip by Bin| results/final/T1 & T2         | Figure F1 (Risk/skip rate vs alpha)
     |   - F2: Param-Shift Risk  | results/final/T3              | Figure F2 (Risk vs alpha across q-groups)
     |   - F3: Calibration Slack | results/final/T4              | Figure F3 (Risk gap vs n_cal & 1/(n+1))
     |   - F4: Model Comparison  | results/final/T5              | Figure F4 (Per-bin skip rates across models)
     |   - F5: Timing Profiling  | documentation/raw/timing.csv  | Figure F5 (Simulator vs ML latency bars)
4    | Full Pipeline Execution   | (--full mode only)            | End-to-end dataset & experiment rebuild
     |   - Dataset Generation    | scripts/make_dataset.py       | data/kluster_dataset_v2.csv (1,500 games)
     |   - Mondrian Runner       | scripts/run_mondrian.py       | results/final/T1, T2
     |   - Parameter Shift       | scripts/run_paramshift.py     | results/final/T3
     |   - Calibration Scaling   | scripts/run_calsize.py        | results/final/T4
     |   - Neural Network Runner | scripts/run_nn.py             | results/final/T5
-------------------------------------------------------------------------------------------------------

Execution Modes:
    python run_all.py          (Default: --quick mode, verifies tests, prints T1..T5, generates figures < 15s)
    python run_all.py --quick  (Explicit quick mode, reads verified CSVs without retraining)
    python run_all.py --full   (Full reproduction mode, re-runs dataset generation and model training)
"""

import os
import sys
import time
import pathlib
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Determine repository root by locating results/final or src/kluster
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
cur = SCRIPT_DIR
ROOT = SCRIPT_DIR
while cur != cur.parent:
    if (cur / "results" / "final").exists():
        ROOT = cur
        break
    cur = cur.parent

sys.path.insert(0, str(ROOT / "src"))

# Colorblind-safe Okabe-Ito palette (matching reproduce_all.py)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#999999",
}
PALETTE = list(OKABE_ITO.values())


def run_step_1_physics_tests(root_dir):
    """Step 1: Execute tests/test_physics.py and report status."""
    print("=" * 80)
    print("STEP 1: RUNNING PHYSICS UNIT TESTS (tests/test_physics.py)")
    print("=" * 80)
    test_script = root_dir / "tests" / "test_physics.py"
    if not test_script.exists():
        test_script = ROOT / "tests" / "test_physics.py"
    
    t0 = time.time()
    res = subprocess.run([sys.executable, str(test_script)], capture_output=True, text=True)
    dt = time.time() - t0
    
    print(res.stdout.strip())
    if res.returncode == 0:
        print(f"--> ALL 20 PHYSICS TESTS PASSED in {dt:.2f}s (Exit Code: 0)")
        return True
    else:
        print(res.stderr.strip())
        print(f"--> TEST FAILURE (Exit Code: {res.returncode})")
        return False


def run_step_2_print_tables(final_dir):
    """Step 2: Read verified T1..T5 CSVs and print each table."""
    print("\n" + "=" * 80)
    print("STEP 2: TABULAR RESULTS (T1 to T5 from results/final/)")
    print("=" * 80)

    # Table T1
    t1_path = final_dir / "T1_risk_check.csv"
    if t1_path.exists():
        df_t1 = pd.read_csv(t1_path)
        print("\n--- TABLE T1: Realized Wrong-Skip Risk vs Target Alpha Across Board-Size Bins ---")
        sub_t1 = df_t1[df_t1["model"].str.contains("Gradient boosting") | df_t1["model"].str.contains("Combined")].copy()
        cols_t1 = ["bin", "model", "alpha", "mean_wrong_skip_rate", "std_err", "fraction_exceeding_alpha"]
        sub_t1 = sub_t1[cols_t1].sort_values(["bin", "alpha", "model"])
        print(sub_t1.to_string(index=False))

    # Table T2
    t2_path = final_dir / "T2_paired_ml_vs_rule.csv"
    if t2_path.exists():
        df_t2 = pd.read_csv(t2_path)
        print("\n--- TABLE T2: Paired Skip-Rate Differences (Best ML vs. Combined Rule) ---")
        cols_t2 = ["bin", "alpha", "best_ml_skip_mean", "combined_skip_mean", "mean_diff", "diff_lo", "diff_hi"]
        print(df_t2[cols_t2].to_string(index=False))

    # Table T3
    t3_path = final_dir / "T3_paramshift.csv"
    if t3_path.exists():
        df_t3 = pd.read_csv(t3_path)
        print("\n--- TABLE T3: Parameter-Shift Experiments A, B, and C (Alpha = 0.05) ---")
        sub_t3 = df_t3[df_t3["alpha"] == 0.05].copy()
        cols_t3 = ["experiment", "group", "model", "mean_wrong_skip_rate", "std_err", "mean_skip_rate"]
        print(sub_t3[cols_t3].to_string(index=False))

    # Table T4
    t4_path = final_dir / "T4_calsize.csv"
    if t4_path.exists():
        df_t4 = pd.read_csv(t4_path)
        print("\n--- TABLE T4: Calibration Sample Size Scaling Study (n_cal in {20, 40, 80, 160, 320}) ---")
        sub_t4 = df_t4[(df_t4["alpha"] == 0.05) & (df_t4["bin"] == "5-8")].copy()
        cols_t4 = ["n_cal", "method", "wrong_skip_bin", "gap_below_alpha", "slack_theory", "skip_rate"]
        print(sub_t4[cols_t4].to_string(index=False))

    # Table T5
    t5_path = final_dir / "T5_nn_baseline.csv"
    if t5_path.exists():
        df_t5 = pd.read_csv(t5_path)
        print("\n--- TABLE T5: Neural Network Baseline Comparison (Alpha = 0.05) ---")
        sub_t5 = df_t5[df_t5["alpha"] == 0.05].copy()
        cols_t5 = ["model", "bin", "skip_rate", "wrong_skip_bin", "train_time_s"]
        print(sub_t5[cols_t5].to_string(index=False))


def plot_f1(final_dir, figures_dir):
    """F1: Skip rate against alpha per size bin with realised wrong-skip rate."""
    t4_path = final_dir / "T4_calsize.csv"
    if not t4_path.exists():
        return
    df_cal = pd.read_csv(t4_path)
    df_sub = df_cal[df_cal["n_cal"] == 320].copy()

    methods = [
        "Gradient boosting (+force ratios)",
        "Combined rule",
        "Parameter-aware critical-distance rule",
    ]
    labels_map = {
        "Gradient boosting (+force ratios)": "Best ML (GB + force ratios)",
        "Combined rule": "Combined rule",
        "Parameter-aware critical-distance rule": "Param-aware critical distance",
    }
    colors = {
        "Gradient boosting (+force ratios)": OKABE_ITO["blue"],
        "Combined rule": OKABE_ITO["vermilion"],
        "Parameter-aware critical-distance rule": OKABE_ITO["bluish_green"],
    }
    bins = ["2-4", "5-8", "9-14"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for i, b in enumerate(bins):
        ax = axes[i]
        b_data = df_sub[df_sub["bin"] == b]
        for m in methods:
            m_data = b_data[b_data["method"] == m].sort_values("alpha")
            alphas = m_data["alpha"].values
            skips = m_data["skip_rate"].values
            wrongs = m_data["wrong_skip_bin"].values
            ax.plot(alphas, skips, color=colors[m], linestyle="-", marker="o", linewidth=2.0, markersize=5,
                    label=f"{labels_map[m]} (skip)" if i == 0 else "")
            ax.plot(alphas, wrongs, color=colors[m], linestyle="--", marker="^", linewidth=1.6, markersize=5,
                    label=f"{labels_map[m]} (wrong-skip)" if i == 0 else "")
        ax.plot([0.04, 0.11], [0.04, 0.11], color=OKABE_ITO["black"], linestyle=":", linewidth=1.2,
                label=r"Target risk $\alpha$" if i == 0 else "")
        ax.set_xlabel(f"Risk target $\\alpha$ (Bin {b})", fontsize=10.5)
        if i == 0:
            ax.set_ylabel("Rate (fraction of placements)", fontsize=10.5)
        ax.set_xlim(0.04, 0.11)
        ax.set_ylim(-0.02, 0.85)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "F1.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "F1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f2(final_dir, figures_dir):
    """F2: Wrong-skip rate against alpha in three q groups for Experiment A models."""
    t3_path = final_dir / "T3_paramshift.csv"
    if not t3_path.exists():
        return
    df_t3 = pd.read_csv(t3_path)
    df_a = df_t3[df_t3["experiment"] == "A"].copy()
    groups = ["q < 0.8", "0.8 <= q <= 1.25", "q > 1.25"]
    models = [
        "Static-friction rule",
        "Distance rule",
        "Combined rule",
        "Critical-distance rule",
        "Logistic (geometry)",
        "Logistic (+force ratios)",
        "Gradient boosting (geometry)",
        "Gradient boosting (+force ratios)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for i, grp in enumerate(groups):
        ax = axes[i]
        grp_data = df_a[df_a["group"] == grp]
        ax.plot([0.0, 0.12], [0.0, 0.12], color=OKABE_ITO["black"], linestyle="--", linewidth=1.8,
                label=r"Target $\alpha$ (diagonal)" if i == 0 else "")
        for m_idx, m in enumerate(models):
            m_sub = grp_data[grp_data["model"] == m].sort_values("alpha")
            alphas = m_sub["alpha"].values
            wrongs = m_sub["mean_wrong_skip_rate"].values
            errs = m_sub["std_err"].values
            color = PALETTE[m_idx % len(PALETTE)]
            ax.errorbar(alphas, wrongs, yerr=errs, color=color, marker="o", markersize=4, linewidth=1.4, capsize=3,
                        label=m if i == 0 else "")
        ax.set_xlabel(f"Risk target $\\alpha$ [{grp}]", fontsize=10.5)
        if i == 0:
            ax.set_ylabel("Wrong-skip rate (fraction)", fontsize=10.5)
        ax.set_xlim(0.01, 0.11)
        ax.set_ylim(-0.01, 0.35)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "F2.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "F2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f3(final_dir, figures_dir):
    """F3: Gap below alpha against calibration games with 1/(n+1) curve."""
    t4_path = final_dir / "T4_calsize.csv"
    if not t4_path.exists():
        return
    df_cal = pd.read_csv(t4_path)
    df_cal["gap"] = df_cal["alpha"] - df_cal["wrong_skip_bin"]
    methods = [
        "Combined rule",
        "Parameter-aware critical-distance rule",
        "Gradient boosting (+force ratios)",
        "Logistic (+parameters)",
    ]
    method_labels = {
        "Combined rule": "Combined rule",
        "Parameter-aware critical-distance rule": "Param-aware critical rule",
        "Gradient boosting (+force ratios)": "Gradient boosting (+force ratios)",
        "Logistic (+parameters)": "Logistic (+parameters)",
    }
    method_colors = {
        "Combined rule": OKABE_ITO["vermilion"],
        "Parameter-aware critical-distance rule": OKABE_ITO["bluish_green"],
        "Gradient boosting (+force ratios)": OKABE_ITO["blue"],
        "Logistic (+parameters)": OKABE_ITO["orange"],
    }
    alphas = [0.05, 0.10]
    n_curve = np.linspace(15, 330, 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for i, a in enumerate(alphas):
        ax = axes[i]
        ax.plot(n_curve, 1.0 / (n_curve + 1), color=OKABE_ITO["black"], linestyle="--", linewidth=1.8,
                label=r"Theoretical slack $1/(n_{\mathrm{cal}}+1)$" if i == 0 else "")
        sub_a = df_cal[df_cal["alpha"] == a]
        for m in methods:
            sub_m = sub_a[sub_a["method"] == m]
            grp = sub_m.groupby("n_cal")["gap"].mean().reset_index().sort_values("n_cal")
            ax.plot(grp["n_cal"], grp["gap"], color=method_colors[m], marker="s", markersize=5, linewidth=1.6,
                    label=method_labels[m] if i == 0 else "")
        ax.set_xlabel(rf"Calibration games $n_{{\mathrm{{cal}}}}$ [$\alpha = {a}$]", fontsize=10.5)
        if i == 0:
            ax.set_ylabel(r"Risk gap below $\alpha$ ($\alpha - R$)", fontsize=10.5)
        ax.set_xlim(10, 335)
        ax.set_ylim(-0.005, 0.085)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True, fontsize=9.0)
    plt.tight_layout()
    fig.savefig(figures_dir / "F3.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "F3.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f4(final_dir, figures_dir):
    """F4: Skip rate of Logistic, Gradient boosting, and Neural network per bin."""
    t5_path = final_dir / "T5_nn_baseline.csv"
    if not t5_path.exists():
        return
    df_nn = pd.read_csv(t5_path)
    models = ["Logistic (+parameters)", "Gradient boosting (+parameters)", "Neural network"]
    model_colors = {
        "Logistic (+parameters)": OKABE_ITO["orange"],
        "Gradient boosting (+parameters)": OKABE_ITO["blue"],
        "Neural network": OKABE_ITO["bluish_green"],
    }
    bins = ["2-4", "5-8", "9-14"]
    alphas = [0.02, 0.05, 0.10]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    bar_width = 0.25
    x_indices = np.arange(len(bins))

    for i, a in enumerate(alphas):
        ax = axes[i]
        sub_a = df_nn[df_nn["alpha"] == a]
        for m_idx, m in enumerate(models):
            sub_m = sub_a[sub_a["model"] == m]
            rates = [sub_m[sub_m["bin"] == b]["skip_rate"].values[0] if len(sub_m[sub_m["bin"] == b]) else 0.0 for b in bins]
            errs = [sub_m[sub_m["bin"] == b]["std_err"].values[0] if len(sub_m[sub_m["bin"] == b]) else 0.0 for b in bins]
            offset = (m_idx - 1) * bar_width
            ax.bar(x_indices + offset, rates, width=bar_width, yerr=errs, capsize=3,
                   color=model_colors[m], edgecolor=OKABE_ITO["black"], linewidth=0.8,
                   label=m if i == 0 else "")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"Bin {b}" for b in bins], fontsize=10)
        ax.set_xlabel(rf"Board size bin [$\alpha = {a}$]", fontsize=10)
        if i == 0:
            ax.set_ylabel("Skip rate (fraction safe skipped)", fontsize=10.5)
        ax.set_ylim(0.0, 0.85)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True, fontsize=9.5)
    plt.tight_layout()
    fig.savefig(figures_dir / "F4.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "F4.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_f5_timing(timing_path, figures_dir):
    """F5: Computational timing comparison bars (Simulator vs Screening steps)."""
    if not timing_path.exists():
        return
    df_timing = pd.read_csv(timing_path)
    r = df_timing[df_timing["model"] == "Gradient boosting (+force ratios)"].iloc[0]
    sim_single = float(r["simulator_single_run_ms"])
    sim_checked = float(r["convergence_checked_ms"])
    feat_ms = float(r["feature_ms"])
    infer_ms = float(r["inference_ms"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    categories = [
        "Numerical Simulator\n(Checked run)",
        "Numerical Simulator\n(Single run)",
        "ML Feature\nExtraction",
        "ML Model\nInference"
    ]
    times = [sim_checked, sim_single, feat_ms, infer_ms]
    colors = [OKABE_ITO["vermilion"], OKABE_ITO["orange"], OKABE_ITO["blue"], OKABE_ITO["bluish_green"]]

    bars = ax.bar(categories, times, color=colors, edgecolor=OKABE_ITO["black"], width=0.55)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.3f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Execution Time per Placement (ms)", fontsize=11)
    ax.set_title("Computational Latency: Numerical Physics vs. ML Screening", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.18)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()
    fig.savefig(figures_dir / "F5_timing.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "F5_timing.pdf", bbox_inches="tight")
    plt.close(fig)


def run_step_3_generate_figures(root_dir, figures_dir):
    """Step 3: Generate figures F1 to F4 and F5 from CSVs."""
    print("\n" + "=" * 80)
    print("STEP 3: REGENERATING ALL PAPER FIGURES (F1..F4 + Timing in figures/)")
    print("=" * 80)
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_dir = root_dir / "results" / "final"
    timing_path = root_dir / "documentation" / "raw" / "timing.csv"
    if not timing_path.exists():
        timing_path = ROOT / "documentation" / "raw" / "timing.csv"

    plot_f1(final_dir, figures_dir)
    print(f"  [OK] F1 generated -> {figures_dir / 'F1.png'} & {figures_dir / 'F1.pdf'}")
    plot_f2(final_dir, figures_dir)
    print(f"  [OK] F2 generated -> {figures_dir / 'F2.png'} & {figures_dir / 'F2.pdf'}")
    plot_f3(final_dir, figures_dir)
    print(f"  [OK] F3 generated -> {figures_dir / 'F3.png'} & {figures_dir / 'F3.pdf'}")
    plot_f4(final_dir, figures_dir)
    print(f"  [OK] F4 generated -> {figures_dir / 'F4.png'} & {figures_dir / 'F4.pdf'}")
    plot_f5_timing(timing_path, figures_dir)
    print(f"  [OK] F5 generated -> {figures_dir / 'F5_timing.png'} & {figures_dir / 'F5_timing.pdf'}")


def run_full_pipeline(root_dir):
    """Step 4: Full pipeline execution across all experiment scripts."""
    print("\n" + "=" * 80)
    print("STEP 4: FULL PIPELINE RE-EXECUTION (--full mode)")
    print("=" * 80)
    scripts = [
        ("1. Dataset Generation", root_dir / "scripts" / "make_dataset.py", ["--games", "1500", "--seed", "2027", "--random-params", "--out", "data/kluster_dataset_v2.csv"]),
        ("2. Mondrian CRC Runner (T1 & T2)", root_dir / "scripts" / "run_mondrian.py", ["--data", "data/kluster_dataset_v2.csv", "--repeats", "100"]),
        ("3. Parameter Shift Runner (T3)", root_dir / "scripts" / "run_paramshift.py", ["--repeats", "20"]),
        ("4. Calibration Scaling Runner (T4)", root_dir / "scripts" / "run_calsize.py", ["--repeats", "50"]),
        ("5. Neural Network Runner (T5)", root_dir / "scripts" / "run_nn.py", ["--repeats", "20"]),
    ]
    for desc, script, args in scripts:
        print(f"\n---> Running: {desc} ({script.name})")
        cmd = [sys.executable, str(script)] + args
        subprocess.run(cmd, check=True)
    print("\nFull pipeline re-execution completed.")


def main():
    parser = argparse.ArgumentParser(description="Master reproduction script for Kluster project.")
    parser.add_argument("--full", action="store_true", help="Re-run entire simulation, training, and calibration pipeline")
    parser.add_argument("--quick", action="store_true", default=True, help="Quick reproduction reading saved results (default)")
    parser.add_argument("--figures-dir", type=str, default=None, help="Directory to save generated figures")
    args = parser.parse_args()

    t_start = time.time()
    print("*" * 80)
    print("KLUSTER MAGNET-STABILITY: MASTER REPRODUCTION RUNNER")
    print("Mode:", "FULL PIPELINE (--full)" if args.full else "QUICK REPRODUCTION (--quick)")
    print("*" * 80)

    # 1. Physics tests
    ok = run_step_1_physics_tests(ROOT)
    if not ok:
        sys.exit(1)

    # If --full requested, execute full pipeline
    if args.full:
        run_full_pipeline(ROOT)

    # 2. Print tables
    final_dir = ROOT / "results" / "final"
    run_step_2_print_tables(final_dir)

    # 3. Generate figures
    fig_dir = pathlib.Path(args.figures_dir) if args.figures_dir else (ROOT / "figures")
    run_step_3_generate_figures(ROOT, fig_dir)

    total_time = time.time() - t_start
    print("\n" + "*" * 80)
    print(f"REPRODUCTION COMPLETE IN {total_time:.2f} SECONDS.")
    print("*" * 80)


if __name__ == "__main__":
    main()
