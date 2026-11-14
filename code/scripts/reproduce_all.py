"""Master reproduction script for Kluster magnet-stability project.

By default, does NOT rerun experiments; reads existing result files and writes:
  - Tidy tables in results/final/:
      T1_risk_check.csv (within-bin, per alpha and model)
      T2_paired_ml_vs_rule.csv
      T3_paramshift.csv (A, B, C with group)
      T4_calsize.csv (gap below alpha next to 1/(n+1))
      T5_nn_baseline.csv
  - Figures in figures/ as both PNG (300 dpi) and PDF:
      F1: Skip rate vs alpha per size bin with realised within-bin wrong-skip rate
      F2: Wrong-skip rate vs alpha in three q groups with diagonal wrong-skip = alpha
      F3: Gap below alpha vs calibration games with 1/(n+1) theoretical curve
      F4: Skip rate of Logistic, Gradient boosting, and Neural network per bin
  - Manifest in results/final/MANIFEST.md

Usage:
    python scripts/reproduce_all.py          # default mode: build tables and figures from existing results
    python scripts/reproduce_all.py --rerun  # rerun all experiment scripts in order, then build tables/figures
"""
import sys
import os
import pathlib
import hashlib
import argparse
import subprocess
import platform

# Ensure matplotlib runs headlessly before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FINAL_DIR = RESULTS_DIR / "final"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

# Standard Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "bluish_green": "#009E73",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "reddish_purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}
PALETTE = list(OKABE_ITO.values())

EXPERIMENT_SCRIPTS = [
    "run_mondrian.py",
    "run_paramshift.py",
    "audit_static_paramshift.py",
    "run_calsize.py",
    "run_nn.py",
]


def run_experiment_scripts():
    """Run all existing experiment scripts in order."""
    print("=" * 70)
    print("RERUN MODE: Running experiment scripts in order...")
    print("=" * 70)
    for idx, script_name in enumerate(EXPERIMENT_SCRIPTS, 1):
        script_path = ROOT / "scripts" / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Experiment script not found: {script_path}")
        print(f"\n[{idx}/{len(EXPERIMENT_SCRIPTS)}] Executing {script_name}...")
        cmd = [sys.executable, str(script_path)]
        res = subprocess.run(cmd, cwd=str(ROOT), check=True)
        if res.returncode != 0:
            raise RuntimeError(f"Script {script_name} failed with exit code {res.returncode}")
    print("\nAll experiment scripts completed successfully.\n")


def compute_sha256(filepath):
    """Compute SHA256 hex digest of a file."""
    p = pathlib.Path(filepath)
    if not p.exists():
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_source_csvs():
    """Load all required source CSV files from results/."""
    sources = {}
    csv_names = [
        "mondrian_risk_check.csv",
        "mondrian_paired.csv",
        "paramshift_A.csv",
        "paramshift_B.csv",
        "paramshift_C.csv",
        "audit_static.csv",
        "calsize.csv",
        "nn_baseline.csv",
        "nn_paired.csv",
    ]
    for name in csv_names:
        p = RESULTS_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"Required result file not found: {p}")
        sources[name] = pd.read_csv(p)
    return sources


def build_tables(sources):
    """Build tidy tables T1 through T5 and write to results/final/."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}

    # -------------------------------------------------------------------------
    # T1_risk_check: within-bin risk check per alpha and model
    # -------------------------------------------------------------------------
    df_t1 = sources["mondrian_risk_check.csv"].copy()
    p_t1 = FINAL_DIR / "T1_risk_check.csv"
    df_t1.to_csv(p_t1, index=False)
    tables["T1_risk_check.csv"] = df_t1

    # -------------------------------------------------------------------------
    # T2_paired_ml_vs_rule: paired comparison between best ML and Combined rule
    # -------------------------------------------------------------------------
    df_t2 = sources["mondrian_paired.csv"].copy()
    p_t2 = FINAL_DIR / "T2_paired_ml_vs_rule.csv"
    df_t2.to_csv(p_t2, index=False)
    tables["T2_paired_ml_vs_rule.csv"] = df_t2

    # -------------------------------------------------------------------------
    # T3_paramshift: Experiments A, B, C with group
    # -------------------------------------------------------------------------
    df_a = sources["paramshift_A.csv"].copy()
    df_a["experiment"] = "A"
    df_b = sources["paramshift_B.csv"].copy()
    df_b["experiment"] = "B"
    df_c = sources["paramshift_C.csv"].copy()
    df_c["experiment"] = "C"

    cols_order = [
        "experiment", "group", "model", "alpha",
        "mean_wrong_skip_rate", "std_err", "mean_skip_rate",
        "n_games", "n_repeats"
    ]
    df_t3 = pd.concat([df_a, df_b, df_c], ignore_index=True)[cols_order]
    p_t3 = FINAL_DIR / "T3_paramshift.csv"
    df_t3.to_csv(p_t3, index=False)
    tables["T3_paramshift.csv"] = df_t3

    # -------------------------------------------------------------------------
    # T4_calsize: calibration set scaling with gap below alpha next to 1/(n+1)
    # -------------------------------------------------------------------------
    df_t4 = sources["calsize.csv"].copy()
    df_t4["gap_below_alpha"] = (df_t4["alpha"] - df_t4["wrong_skip_bin"]).round(4)
    # Position gap_below_alpha right next to slack_theory (1/(n+1))
    cols_t4 = [
        "n_cal", "method", "alpha", "bin",
        "wrong_skip_bin", "wrong_skip_pooled", "skip_rate", "std_err",
        "n_bin_cal_mean", "gap_below_alpha", "slack_theory",
        "frac_exceed", "n_repeats"
    ]
    df_t4 = df_t4[cols_t4]
    p_t4 = FINAL_DIR / "T4_calsize.csv"
    df_t4.to_csv(p_t4, index=False)
    tables["T4_calsize.csv"] = df_t4

    # -------------------------------------------------------------------------
    # T5_nn_baseline: neural network baseline comparison
    # -------------------------------------------------------------------------
    df_t5 = sources["nn_baseline.csv"].copy()
    p_t5 = FINAL_DIR / "T5_nn_baseline.csv"
    df_t5.to_csv(p_t5, index=False)
    tables["T5_nn_baseline.csv"] = df_t5

    return tables


def plot_figure_1(sources):
    """F1: Skip rate against alpha per size bin for best ML model, Combined rule,

    and Parameter-aware critical-distance rule, with a line showing the realised
    within-bin wrong-skip rate.
    """
    df_cal = sources["calsize.csv"].copy()
    # Use full calibration set size n_cal = 320 where all three methods are evaluated
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

            # Skip rate: solid line with circle
            ax.plot(
                alphas, skips,
                color=colors[m], linestyle="-", marker="o", linewidth=2.0, markersize=6,
                label=f"{labels_map[m]} (skip rate)" if i == 0 else ""
            )
            # Realised wrong-skip rate: dashed line with triangle
            ax.plot(
                alphas, wrongs,
                color=colors[m], linestyle="--", marker="^", linewidth=1.8, markersize=6,
                label=f"{labels_map[m]} (wrong-skip)" if i == 0 else ""
            )

        # Plot alpha reference line
        ax.plot([0.04, 0.11], [0.04, 0.11], color=OKABE_ITO["black"], linestyle=":", linewidth=1.2,
                label=r"Target risk $\alpha$" if i == 0 else "")

        ax.set_xlabel(f"Risk target $\\alpha$ (Bin {b})", fontsize=11)
        if i == 0:
            ax.set_ylabel("Rate (fraction of placements)", fontsize=11)
        ax.set_xlim(0.04, 0.11)
        ax.set_ylim(-0.02, 0.85)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=True, fontsize=9.5)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F1.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "F1.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_2(sources):
    """F2: Wrong-skip rate against alpha in three q groups for Experiment A models,

    with the diagonal wrong-skip = alpha drawn.
    """
    df_a = sources["paramshift_A.csv"].copy()
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

        # Draw diagonal wrong-skip = alpha
        ax.plot([0.0, 0.12], [0.0, 0.12], color=OKABE_ITO["black"], linestyle="--", linewidth=1.8,
                label=r"Target $\alpha$ (diagonal)" if i == 0 else "")

        for m_idx, m in enumerate(models):
            m_sub = grp_data[grp_data["model"] == m].sort_values("alpha")
            alphas = m_sub["alpha"].values
            wrongs = m_sub["mean_wrong_skip_rate"].values
            errs = m_sub["std_err"].values

            color = PALETTE[m_idx % len(PALETTE)]
            ax.errorbar(
                alphas, wrongs, yerr=errs,
                color=color, marker="o", markersize=5, linewidth=1.5, capsize=3,
                label=m if i == 0 else ""
            )

        ax.set_xlabel(f"Risk target $\\alpha$ [{grp}]", fontsize=11)
        if i == 0:
            ax.set_ylabel("Wrong-skip rate (fraction)", fontsize=11)
        ax.set_xlim(0.01, 0.11)
        ax.set_ylim(-0.01, 0.35)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True, fontsize=9.5)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F2.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "F2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_3(sources):
    """F3: Gap below alpha against calibration games with the 1/(n+1) curve."""
    df_cal = sources["calsize.csv"].copy()
    # Compute gap below alpha for each configuration
    df_cal["gap"] = df_cal["alpha"] - df_cal["wrong_skip_bin"]

    methods = [
        "Combined rule",
        "Parameter-aware critical-distance rule",
        "Gradient boosting (+force ratios)",
        "Logistic (+parameters)",
    ]
    method_labels = {
        "Combined rule": "Combined rule",
        "Parameter-aware critical-distance rule": "Param-aware critical distance",
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    n_smooth = np.linspace(18, 330, 200)
    y_smooth = 1.0 / (n_smooth + 1.0)

    for i, a in enumerate(alphas):
        ax = axes[i]
        # Smooth theoretical 1/(n+1) curve
        ax.plot(
            n_smooth, y_smooth,
            color=OKABE_ITO["black"], linestyle="--", linewidth=2.0,
            label=r"Theoretical slack $1/(n_{\mathrm{cal}}+1)$" if i == 0 else ""
        )

        sub_a = df_cal[df_cal["alpha"] == a]
        for m in methods:
            sub_m = sub_a[sub_a["method"] == m]
            # Mean gap across board-size bins for each n_cal
            grp = sub_m.groupby("n_cal")["gap"].mean().reset_index().sort_values("n_cal")
            ax.plot(
                grp["n_cal"], grp["gap"],
                color=method_colors[m], marker="s", markersize=6, linewidth=1.8,
                label=method_labels[m] if i == 0 else ""
            )

        ax.set_xlabel(rf"Calibration games $n_{{\mathrm{{cal}}}}$ (games) [$\alpha = {a}$]", fontsize=11)
        if i == 0:
            ax.set_ylabel(r"Risk gap below $\alpha$ ($\alpha - R$)", fontsize=11)
        ax.set_xlim(10, 335)
        ax.set_ylim(-0.005, 0.085)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True, fontsize=9.5)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F3.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "F3.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure_4(sources):
    """F4: Skip rate of Logistic, Gradient boosting, and Neural network per bin."""
    df_nn = sources["nn_baseline.csv"].copy()
    models = [
        "Logistic (+parameters)",
        "Gradient boosting (+parameters)",
        "Neural network",
    ]
    model_labels = {
        "Logistic (+parameters)": "Logistic (+parameters)",
        "Gradient boosting (+parameters)": "Gradient boosting (+parameters)",
        "Neural network": "Neural network",
    }
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
            # Order by bins
            rates = []
            errs = []
            for b in bins:
                row = sub_m[sub_m["bin"] == b]
                if len(row) > 0:
                    rates.append(row["mean_skip_rate"].values[0])
                    errs.append(row["std_err"].values[0])
                else:
                    rates.append(0.0)
                    errs.append(0.0)

            offset = (m_idx - 1) * bar_width
            ax.bar(
                x_indices + offset, rates, width=bar_width, yerr=errs, capsize=4,
                color=model_colors[m], edgecolor=OKABE_ITO["black"], linewidth=0.8,
                label=model_labels[m] if i == 0 else ""
            )

        ax.set_xticks(x_indices)
        ax.set_xticklabels([f"Bin {b}" for b in bins], fontsize=10.5)
        ax.set_xlabel(rf"Board size bin $N$ (magnets on board) [$\alpha = {a}$]", fontsize=10.5)
        if i == 0:
            ax.set_ylabel("Skip rate (fraction safe skipped)", fontsize=11)
        ax.set_ylim(0.0, 0.85)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True, fontsize=10)
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "F4.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "F4.pdf", bbox_inches="tight")
    plt.close(fig)


def build_manifest(tables):
    """Write results/final/MANIFEST.md documenting all artifacts, commands, environment, seeds, and SHA256 checksums."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = FINAL_DIR / "MANIFEST.md"

    csv_old_path = DATA_DIR / "kluster_dataset.csv"
    csv_v2_path = DATA_DIR / "kluster_dataset_v2.csv"
    sha_old = compute_sha256(csv_old_path)
    sha_v2 = compute_sha256(csv_v2_path)

    import scipy, sklearn, numba, joblib

    packages = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "numba": numba.__version__,
        "joblib": joblib.__version__,
    }

    manifest_lines = [
        "# Deliverables Manifest: Kluster Magnet-Stability Reproduction Package",
        "",
        "This manifest catalogues all final tables, figures, regeneration commands, execution environments, random seeds, and cryptographic dataset checksums.",
        "",
        "## 1. Final Output Files",
        "",
        "### Tables (results/final/)",
        "- `T1_risk_check.csv`: Within-size-bin risk check across 8 models, 3 alphas (0.02, 0.05, 0.10), and 3 bins ('2-4', '5-8', '9-14') under Mondrian Conformal Risk Control.",
        "- `T2_paired_ml_vs_rule.csv`: Paired differences in skip rate between the best ML model and the Combined rule over identical game splits with 95% bootstrap intervals.",
        "- `T3_paramshift.csv`: Parameter-shift generalisation across Experiments A (fixed-to-random), B (random-trained with PHYS_PARAMS), and C (parameter-aware critical distance) evaluated across q groups ('q < 0.8', '0.8 <= q <= 1.25', 'q > 1.25') and overall.",
        "- `T4_calsize.csv`: Finite-sample calibration size scaling (n_cal in {20, 40, 80, 160, 320}) presenting the empirical gap below alpha next to the exact theoretical slack 1/(n+1).",
        "- `T5_nn_baseline.csv`: Small neural network baseline (MLPClassifier (32, 16)) compared against Gradient boosting (+parameters) and Logistic (+parameters) across size bins and alphas.",
        "",
        "### Figures (figures/ - available as 300 dpi PNG and vector PDF)",
        "- `F1.png` / `F1.pdf`: Skip rate against alpha per size bin for the best ML model, the Combined rule, and the Parameter-aware critical-distance rule, with dashed lines depicting realised within-bin wrong-skip rates.",
        "- `F2.png` / `F2.pdf`: Realised wrong-skip rate against alpha across the three q groups for Experiment A models, displaying the diagonal wrong-skip = alpha line.",
        "- `F3.png` / `F3.pdf`: Empirical risk gap below alpha against calibration games alongside the smooth theoretical 1/(n+1) curve.",
        "- `F4.png` / `F4.pdf`: Skip rates of Logistic (+parameters), Gradient boosting (+parameters), and Neural network per size bin across alphas.",
        "",
        "## 2. Regeneration Commands",
        "",
        "### Default Reproduction (Fast, reads existing results and regenerates final tables/figures)",
        "```bash",
        "python scripts/reproduce_all.py",
        "```",
        "",
        "### Full End-to-End Rerun (Executes all underlying experiment scripts in sequence, then rebuilds tables/figures)",
        "```bash",
        "python scripts/reproduce_all.py --rerun",
        "```",
        "",
        "### Underlying Individual Experiment Scripts",
        "- Mondrian CRC: `python scripts/run_mondrian.py`",
        "- Parameter Shift: `python scripts/run_paramshift.py`",
        "- Static-Friction Audit: `python scripts/audit_static_paramshift.py`",
        "- Calibration Size Scaling: `python scripts/run_calsize.py`",
        "- Neural Network Baseline: `python scripts/run_nn.py`",
        "",
        "## 3. Environment and Package Versions",
        "",
        f"- **Python**: {packages['python']} ({packages['platform']})",
        f"- **numpy**: {packages['numpy']}",
        f"- **scipy**: {packages['scipy']}",
        f"- **pandas**: {packages['pandas']}",
        f"- **scikit-learn**: {packages['scikit-learn']}",
        f"- **matplotlib**: {packages['matplotlib']}",
        f"- **numba**: {packages['numba']}",
        f"- **joblib**: {packages['joblib']}",
        "",
        "## 4. Random Seeds",
        "",
        "- **Dataset Generation**:",
        "  - `data/kluster_dataset.csv`: seed = 2026",
        "  - `data/kluster_dataset_v2.csv`: seed = 42",
        "- **Mondrian CRC Splits** (`run_mondrian.py`): seeds 0..19 (20 repeated splits)",
        "- **Parameter Shift Splits** (`run_paramshift.py`): seeds 0..19 (20 repeated splits)",
        "- **Static-Friction Audit** (`audit_static_paramshift.py`): seeds 0..99 (100 repeated splits)",
        "- **Calibration Scaling Splits** (`run_calsize.py`): seeds 0..49 (50 repeated splits)",
        "- **Neural Network Baseline** (`run_nn.py`): seeds 0..19 (20 repeated splits, models initialized with random_state=seed)",
        "",
        "## 5. Dataset Checksums (SHA256)",
        "",
        f"- `data/kluster_dataset.csv`: `{sha_old}`",
        f"- `data/kluster_dataset_v2.csv`: `{sha_v2}`",
        "",
    ]

    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Reproduce all final tables, figures, and manifest for Kluster project.")
    parser.add_argument("--rerun", action="store_true", help="Rerun all underlying experiment scripts in order before building deliverables")
    args = parser.parse_args()

    if args.rerun:
        run_experiment_scripts()

    print("[1/4] Reading existing result files...")
    sources = load_source_csvs()
    for name, df in sources.items():
        print(f"      Loaded results/{name}: {len(df)} rows, {len(df.columns)} columns.")

    print("\n[2/4] Generating tidy final tables in results/final/...")
    tables = build_tables(sources)
    for name, df in tables.items():
        print(f"      Saved results/final/{name}: {len(df)} rows, {list(df.columns)}")

    print("\n[3/4] Generating publication-quality figures in figures/...")
    plot_figure_1(sources)
    print("      Saved figures/F1.png and figures/F1.pdf")
    plot_figure_2(sources)
    print("      Saved figures/F2.png and figures/F2.pdf")
    plot_figure_3(sources)
    print("      Saved figures/F3.png and figures/F3.pdf")
    plot_figure_4(sources)
    print("      Saved figures/F4.png and figures/F4.pdf")

    print("\n[4/4] Writing results/final/MANIFEST.md...")
    manifest_path = build_manifest(tables)
    print(f"      Saved {manifest_path}")

    print("\n" + "=" * 70)
    print("REPRODUCTION SUMMARY")
    print("=" * 70)
    print(f"Tables generated: {len(tables)} files in {FINAL_DIR}")
    print(f"Figures generated: 8 files (4 PNG + 4 PDF) in {FIGURES_DIR}")
    print(f"Manifest created: {manifest_path}")
    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()
