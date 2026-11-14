"""Unit tests for reproduction package deliverables.

Verifies:
  1. Every table (T1..T5) in results/final/ has no NaN in numeric columns except where a bin abstains.
  2. Every figure file (F1..F4 in PNG and PDF) in figures/ exists and is non-empty.
  3. The numbers for 3 randomly chosen cells in each table match the source CSV exactly.
  4. MANIFEST.md exists and contains all required sections, regeneration commands, package versions, random seeds, and SHA256 checksums.

Run:
    python tests/test_reproduce.py
"""
import sys
import pathlib
import hashlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "final"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)


def test_tables_no_nan():
    ok = True
    table_files = [
        "T1_risk_check.csv",
        "T2_paired_ml_vs_rule.csv",
        "T3_paramshift.csv",
        "T4_calsize.csv",
        "T5_nn_baseline.csv",
    ]

    for fname in table_files:
        p = FINAL_DIR / fname
        if not p.exists():
            ok &= check(f"table {fname} exists", False)
            continue

        df = pd.read_csv(p)
        ok &= check(f"table {fname} has rows ({len(df)} rows)", len(df) > 0)

        # Identify numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        ok &= check(f"table {fname} has numeric columns", len(num_cols) > 0)

        # In case of abstention (tau = inf / feasible = False), certain cells could be NaN;
        # otherwise numeric columns must have zero NaNs.
        nan_counts = df[num_cols].isna().sum()
        total_nans = nan_counts.sum()
        ok &= check(f"table {fname} has no NaN in numeric columns (found {total_nans})", total_nans == 0)

    return ok


def test_figures_exist_and_nonempty():
    ok = True
    figures = [
        "F1.png", "F1.pdf",
        "F2.png", "F2.pdf",
        "F3.png", "F3.pdf",
        "F4.png", "F4.pdf",
    ]

    for fig in figures:
        p = FIGURES_DIR / fig
        exists = p.is_file()
        ok &= check(f"figure {fig} exists", exists)
        if exists:
            size = p.stat().st_size
            ok &= check(f"figure {fig} is non-empty ({size} bytes)", size > 0)

    return ok


def test_cell_exact_match():
    ok = True
    rng = np.random.default_rng(2026)

    # 1. T1_risk_check vs mondrian_risk_check.csv
    df_t1 = pd.read_csv(FINAL_DIR / "T1_risk_check.csv")
    src_t1 = pd.read_csv(RESULTS_DIR / "mondrian_risk_check.csv")
    common_num_t1 = [c for c in df_t1.select_dtypes(include=[np.number]).columns if c in src_t1.columns]

    for i in range(3):
        r = rng.integers(0, len(df_t1))
        c = rng.choice(common_num_t1)
        val_t1 = df_t1.loc[r, c]
        val_src = src_t1.loc[r, c]
        match = (val_t1 == val_src) or np.isclose(val_t1, val_src, atol=1e-12)
        ok &= check(f"T1 cell [row {r}, {c}] matches source ({val_t1} == {val_src})", match)

    # 2. T2_paired_ml_vs_rule vs mondrian_paired.csv
    df_t2 = pd.read_csv(FINAL_DIR / "T2_paired_ml_vs_rule.csv")
    src_t2 = pd.read_csv(RESULTS_DIR / "mondrian_paired.csv")
    common_num_t2 = [c for c in df_t2.select_dtypes(include=[np.number]).columns if c in src_t2.columns]

    for i in range(3):
        r = rng.integers(0, len(df_t2))
        c = rng.choice(common_num_t2)
        val_t2 = df_t2.loc[r, c]
        val_src = src_t2.loc[r, c]
        match = (val_t2 == val_src) or np.isclose(val_t2, val_src, atol=1e-12)
        ok &= check(f"T2 cell [row {r}, {c}] matches source ({val_t2} == {val_src})", match)

    # 3. T3_paramshift vs paramshift_A/B/C.csv
    df_t3 = pd.read_csv(FINAL_DIR / "T3_paramshift.csv")
    src_map = {
        "A": pd.read_csv(RESULTS_DIR / "paramshift_A.csv"),
        "B": pd.read_csv(RESULTS_DIR / "paramshift_B.csv"),
        "C": pd.read_csv(RESULTS_DIR / "paramshift_C.csv"),
    }
    num_t3 = df_t3.select_dtypes(include=[np.number]).columns

    for i in range(3):
        r = rng.integers(0, len(df_t3))
        exp = df_t3.loc[r, "experiment"]
        src_df = src_map[exp]
        c = rng.choice(num_t3)
        val_t3 = df_t3.loc[r, c]

        # Match row in source by (group, model, alpha)
        grp = df_t3.loc[r, "group"]
        mod = df_t3.loc[r, "model"]
        alp = df_t3.loc[r, "alpha"]
        src_row = src_df[(src_df["group"] == grp) & (src_df["model"] == mod) & (np.isclose(src_df["alpha"], alp))]
        val_src = src_row[c].values[0]

        match = (val_t3 == val_src) or np.isclose(val_t3, val_src, atol=1e-12)
        ok &= check(f"T3 cell [row {r} (exp {exp}), {c}] matches source ({val_t3} == {val_src})", match)

    # 4. T4_calsize vs calsize.csv
    df_t4 = pd.read_csv(FINAL_DIR / "T4_calsize.csv")
    src_t4 = pd.read_csv(RESULTS_DIR / "calsize.csv")
    common_num_t4 = [c for c in df_t4.select_dtypes(include=[np.number]).columns if c in src_t4.columns]

    for i in range(3):
        r = rng.integers(0, len(df_t4))
        c = rng.choice(common_num_t4)
        val_t4 = df_t4.loc[r, c]
        val_src = src_t4.loc[r, c]
        match = (val_t4 == val_src) or np.isclose(val_t4, val_src, atol=1e-12)
        ok &= check(f"T4 cell [row {r}, {c}] matches source ({val_t4} == {val_src})", match)

    # Check gap_below_alpha calculation in T4
    gap_matches = np.allclose(df_t4["gap_below_alpha"], df_t4["alpha"] - df_t4["wrong_skip_bin"], atol=1e-4)
    ok &= check("T4 gap_below_alpha equals (alpha - wrong_skip_bin)", gap_matches)

    # 5. T5_nn_baseline vs nn_baseline.csv
    df_t5 = pd.read_csv(FINAL_DIR / "T5_nn_baseline.csv")
    src_t5 = pd.read_csv(RESULTS_DIR / "nn_baseline.csv")
    common_num_t5 = [c for c in df_t5.select_dtypes(include=[np.number]).columns if c in src_t5.columns]

    for i in range(3):
        r = rng.integers(0, len(df_t5))
        c = rng.choice(common_num_t5)
        val_t5 = df_t5.loc[r, c]
        val_src = src_t5.loc[r, c]
        match = (val_t5 == val_src) or np.isclose(val_t5, val_src, atol=1e-12)
        ok &= check(f"T5 cell [row {r}, {c}] matches source ({val_t5} == {val_src})", match)

    return ok


def test_manifest():
    ok = True
    p = FINAL_DIR / "MANIFEST.md"
    ok &= check("MANIFEST.md exists", p.is_file())
    if not p.is_file():
        return False

    content = p.read_text(encoding="utf-8")
    ok &= check("MANIFEST lists T1_risk_check.csv", "T1_risk_check.csv" in content)
    ok &= check("MANIFEST lists T2_paired_ml_vs_rule.csv", "T2_paired_ml_vs_rule.csv" in content)
    ok &= check("MANIFEST lists T3_paramshift.csv", "T3_paramshift.csv" in content)
    ok &= check("MANIFEST lists T4_calsize.csv", "T4_calsize.csv" in content)
    ok &= check("MANIFEST lists T5_nn_baseline.csv", "T5_nn_baseline.csv" in content)
    ok &= check("MANIFEST lists figures F1..F4", "F1.png" in content and "F4.pdf" in content)
    ok &= check("MANIFEST lists regeneration commands", "python scripts/reproduce_all.py" in content)
    ok &= check("MANIFEST lists Python version", "Python" in content)
    ok &= check("MANIFEST lists random seeds", "Random Seeds" in content or "seed" in content)

    # Check SHA256 checksums in manifest match actual files
    for ds_name in ["kluster_dataset.csv", "kluster_dataset_v2.csv"]:
        ds_file = DATA_DIR / ds_name
        if ds_file.exists():
            h = hashlib.sha256(ds_file.read_bytes()).hexdigest()
            ok &= check(f"MANIFEST contains correct SHA256 for {ds_name}", h in content)

    return ok


def main():
    print("=" * 70)
    print("RUNNING REPRODUCTION DELIVERABLES TEST SUITE")
    print("=" * 70)

    ok = True
    print("\n--- 1. Table Numeric Column NaN Checks ---")
    ok &= test_tables_no_nan()

    print("\n--- 2. Figure Existence and Non-Empty Checks ---")
    ok &= test_figures_exist_and_nonempty()

    print("\n--- 3. Exact Numeric Cell Matching with Source CSVs ---")
    ok &= test_cell_exact_match()

    print("\n--- 4. Manifest Integrity Checks ---")
    ok &= test_manifest()

    print("\n" + "=" * 70)
    if ok:
        print("ALL REPRODUCTION TESTS PASSED")
    else:
        print("SOME REPRODUCTION TESTS FAILED")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

