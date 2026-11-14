# Open Issues and Repository Audit Notes

This document details discrepancies identified during repository verification, weaker claims adopted to remain strictly supported by evidence, and known anomalies between historical documentation drafts and the underlying execution outputs.

---

## 1. Discrepancies in Historical Drafts vs. Output CSVs

### A. T1 Gradient Boosting Risk at Bin 5-8 ($\alpha = 0.05$)
- **Historical Claim in Drafts:** Stated empirical wrong-skip rate of `0.0465`.
- **Actual Verified Value in `results/final/T1_risk_check.csv`:** Mean wrong-skip rate `0.0457` (std err `0.0012`), with `fraction_exceeding_alpha = 0.33`.
- **Comparison & Resolution:** While the mean wrong-skip rate across 100 test splits (`0.0457`) is below $\alpha = 0.05$, risk is controlled on average only. One-third (33%) of individual test splits exhibit empirical risk exceeding $\alpha$, reflecting standard finite-sample variability.

### B. Baseline Rule in Paired Differences (T2)
- **Historical Claim in Drafts:** Paired gains were occasionally quoted as large improvements over the Static rule (e.g. +13.71% in bin 2–4).
- **Actual Verified Pipeline (`results/final/T2_paired_ml_vs_rule.csv`):** The baseline in T2 is explicitly the **Combined rule** (`combined_skip_mean`).
- **Comparison & Resolution:** At $\alpha = 0.05$, the paired difference (Best ML minus Combined rule) is:
  - Bin 2–4: `+0.0015` [95% CI: `-0.0010`, `0.0056`]. The confidence interval spans zero, providing no evidence of ML gain over the combined analytical heuristic on sparse boards.
  - Bin 5–8: `+0.0581` [95% CI: `0.0409`, `0.0752`]. A statistically significant gain of 5.81 percentage points.
  - Bin 9–14: `+0.0379` [95% CI: `0.0127`, `0.0614`]. A statistically significant gain of 3.79 percentage points.

### C. Calibration Dataset Size Range (T4)
- **Historical Claim in Drafts:** Described calibration scaling as extending up to $N_{\text{cal}} = 300$ games and asserted that risk slack uniformly tracks $1/(n+1)$.
- **Actual Verified Pipeline (`results/final/T4_calsize.csv`):** Evaluated at $n_{\text{cal}} \in \{20, 40, 80, 160, 320\}$.
- **Comparison & Resolution:**
  - At $n_{\text{cal}} = 20, \alpha = 0.02$, there are 0 rows in the CSV because the finite-sample conformal quantile requirement $\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil / n_{\text{cal}} > 1$ cannot be satisfied; the method abstains completely from screening (skip rate = 0).
  - Risk slack does not track $1/(n+1)$ across all rules: for the Combined rule at $\alpha = 0.10$ in bin 2–4, the gap below $\alpha$ remains virtually constant at `0.0567` across $n_{\text{cal}} = 20$ to 320 (empirical wrong-skip rate $\approx 0.0433$), because the discrete heuristic cannot produce higher risk regardless of calibration size.

### D. T3 Parameter Shift Risk Values and Subgroup Control
- **Historical Claim in Drafts:** Claimed Experiment A wrong-skip was `0.1776`, and that Experiment B restored risk control across all subgroups.
- **Actual Verified Values in `results/final/T3_paramshift.csv` (at $\alpha = 0.05$):**
  - **Experiment A ($q < 0.8$):** Gradient boosting (+force ratios) achieves wrong-skip `0.0670` (std err `0.0032`), whereas the Distance rule and Critical-distance rule both fail with `0.1720` (std err `0.0017`).
  - **Experiment B (GB + parameters):** Overall wrong-skip is `0.0469` (std err `0.0012`) and the central group ($0.8 \le q \le 1.25$) is `0.0459` (std err `0.0010`). However, risk control is NOT restored across all subgroups: $q < 0.8$ reaches `0.0548` (std err `0.0038`) and $q > 1.25$ reaches `0.0741` (std err `0.0115`), both exceeding $\alpha$.
  - **Sample Size Context:** Across 20 test splits, $q < 0.8$ averages only 13.35 test games and $q > 1.25$ averages only 4.30 test games. These rare subgroups are exploratory rather than confirmatory.
  - **Experiment C (Parameter-aware critical-distance rule):** Yields overall wrong-skip `0.0474`, $q < 0.8$ at `0.0502`, $0.8 \le q \le 1.25$ at `0.0469`, and $q > 1.25$ at `0.0605`.

### E. Table T5 Neural Network Baseline
- **Historical Claim in Drafts:** Quoted pooled skip figures not present in the CSV.
- **Actual Verified Pipeline (`results/final/T5_nn_baseline.csv`):** Reports per-bin metrics and wall-clock training times at $\alpha = 0.05$:
  - Training times: Gradient boosting (+parameters) `8.61` s, Neural network `18.32` s, Logistic (+parameters) `0.647` s.
  - Skip rates:
    - Bin 2–4: GB `0.3126`, NN `0.3128`, Logistic `0.3085`.
    - Bin 5–8: GB `0.6319`, NN `0.6242`, Logistic `0.6001`.
    - Bin 9–14: GB `0.6834`, NN `0.6790`, Logistic `0.6643`.
  - Resolution: All pooled numbers removed; per-bin skip rates and training times reported directly from `T5_nn_baseline.csv`.

### F. Timing Benchmarks
- **Verified Values in `documentation/raw/timing.csv`:**
  - Simulator single-run: `6.78` ms.
  - Simulator convergence-checked run: `20.075` ms (FINAL_EVIDENCE_v3 rounded/reported this as `20.15` ms).
  - Feature extraction: `0.562` ms.
  - Model inference (GB + force ratios): `0.0089` ms.
  - Hardware: AMD Ryzen 7 5800HS with Radeon Graphics (8C/16T), 15.4 GB RAM.

---

## 2. Seed Provenance for `kluster_dataset_v2.csv`

- **Historical Claim:** `data/DATA_CARD.md` stated that `kluster_dataset_v2.csv` was generated with `--seed 42`.
- **Empirical Evidence:** Drawing parameters per `src/kluster/generate.py` via `np.random.default_rng([seed, game_id])` for `game_id` 0..4 yields large discrepancies when `seed = 42` ($\Delta \mu_s \approx 0.34$, $\Delta k_{\text{scale}} \approx 0.74$).
- **Verified Finding:** A brute-force search over seeds in `[0..5000]` plus 2026, 2027, and 42 identified that **`seed = 2027`** reproduces the stored parameters of `game_id` 0..4 with exact machine precision ($\text{diff} \le 8.33 \times 10^{-17}$).
- **Resolution:** Recorded in `README_REPRODUCE.md` and `DATA_CARD.md` that seed 2027 generates the physical parameter distribution stored in `kluster_dataset_v2.csv`.

---

## 3. Numerical Integration Reproducibility Across Python Versions

- **Observation:** Regenerating `kluster_dataset.csv` (300 games, seed 2026) under Python 3.13 / NumPy 2.2 / Numba 0.68 matches all labels (`contact`, `settled`, `timeout`) with 100.00% identity against the original file generated under Python 3.12 / NumPy 1.26 / Numba 0.61. However, floating-point coordinates and velocities exhibit small machine-precision drift ($\sim 10^{-6}$ to $10^{-8}$ mm), resulting in different SHA-256 file hashes.
- **Resolution:** Documented in `CLEANUP_LOG.md` and `README_REPRODUCE.md` that while discrete simulation outcomes are completely reproducible, exact binary byte matching requires the pinned Python 3.12.8 / NumPy 1.26.4 environment.
