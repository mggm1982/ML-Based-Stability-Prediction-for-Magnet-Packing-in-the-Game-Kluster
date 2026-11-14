# Code Walkthrough and Architecture Guide

**Project:** Machine Learning-Based Stability Prediction for Magnet Packing in the Game Kluster  
**Author:** Sankalp Savarn (Enrolment No. 24123038), IIT Roorkee  
**Supervisor:** Prof. Mayank Goswami (mayank.goswami@ph.iitr.ac.in)  


---

## 1. End-to-End Data Flow

The project architecture transforms first-principles physical equations into a surrogate screening pipeline with conformal risk control:

```text
+------------------------+
| src/kluster/           |   Point-dipole force law (3*K/r^4), Coulomb friction (fs, fk),
| simulator.py           |   circular inelastic boundary constraint, semi-implicit Euler.
+-----------+------------+
            |
            v
+------------------------+
| src/kluster/           |   Progressive game play (1 to 14 magnets), rejection sampling
| generate.py            |   of candidate coordinates, dt and dt/2 convergence checks.
+-----------+------------+
            |
            v
+------------------------+
| data/                  |   Tabular dataset of initial board coordinates, polarities,
| kluster_dataset_v2.csv |   friction coefficients (mu_s, mu_k), and physical labels.
+-----------+------------+
            |
            v
+------------------------+
| src/kluster/           |   Engineered spatial distances (d_nn, d_same, d_opp),
| features.py            |   wall-corrected friction force ratios (f_ratio_max, f_ratio_new).
+-----------+------------+
            |
            v
+------------------------+
| Model Training         |   Histogram Gradient Boosting (HGB), Logistic Regression,
| (mondrian.py / nnbase) |   and Multi-Layer Perceptron (MLP) predict contact probability.
+-----------+------------+
            |
            v
+------------------------+
| Conformal Thresholding |   Mondrian Conformal Risk Control (CRC) computes threshold tau_k
| (mondrian.py)          |   per board-size bin k on calibration games: E[Loss] <= alpha.
+-----------+------------+
            |
            v
+------------------------+
| Statistical Evaluation |   Grouped bootstrap evaluation across test games, paired
| (evaluate.py / shift)  |   differences vs. Combined rule, parameter shift (q-groups).
+-----------+------------+
            |
            v
+------------------------+
| results/final/ (T1..T5)|   Verified output tables T1 to T5, reproducible figures F1 to F5,
| & figures/ (F1..F5)    |   and claim ledger numbers cited in the manuscript.
+------------------------+
```

---

## 2. Recommended Reading Order and File Summaries

### Core Physics & Simulation
1. **`src/kluster/simulator.py`**: Implements the fundamental point-dipole force law, Coulomb friction model, and the frictionless circular wall constraint. Functions `_forces` and `_sim` are Numba-compiled for fast semi-implicit Euler integration ($dt = 10^{-5}\text{ s}$), while `label_checked` verifies numerical stability via step-halving.
2. **`tests/test_physics.py`**: The 20-test verification suite asserting dipole force scaling ($1/r^4$), static friction thresholds, non-penetration boundary conditions, and equilibrium relaxation. Always read this first to understand the physical invariances.
3. **`src/kluster/generate.py`**: Simulates complete games of Kluster. For each placement step, it samples candidate positions, records outcomes, and continues the board progression from the final resting coordinates of settled stones.

### Feature Representation & Machine Learning
4. **`src/kluster/features.py`**: Converts raw board configurations into model features without leaking post-placement outcomes. Computes geometric distances (`d_nn`, `d_same`, `d_opp`), local neighbor densities, wall clearance, and friction-relevant force ratios (`f_ratio_max`).
5. **`src/kluster/mondrian.py`**: Implements size-conditional (Mondrian) Conformal Risk Control. Bins board states into sparse (2–4), intermediate (5–8), and crowded (9–14) regimes, calibrating bin-specific thresholds $\tau_k$ on exchangeable calibration games.
6. **`src/kluster/paramshift.py`**: Implements parameter-shift experiments A, B, and C. Evaluates model robustness when friction and dipole force scales vary via ratio $q = (k_{\text{scale}}/\mu_s) / (k_{\text{scale},0}/\mu_{s,0})$ across rare and central subgroups.
7. **`src/kluster/calsize.py`**: Evaluates calibration sample size scaling across $n_{\text{cal}} \in \{20, 40, 80, 160, 320\}$, demonstrating finite-sample abstention when coverage conditions cannot be satisfied.
8. **`src/kluster/nnbase.py`**: Constructs and trains the Multi-Layer Perceptron (MLP 32, 16) neural network baseline, measuring wall-clock training overhead and test skip rates against tree-based models.
9. **`src/kluster/evaluate.py`**: Provides core training, validation tuning, grouped-bootstrap confidence interval estimation, and runtime profiling routines.

### Master Reproduction Runner
10. **`run_all.py`**: The unified entry point. It runs the physics unit tests, loads the verified final CSVs, prints Tables T1–T5, and generates publication figures F1–F5.

---

## 3. Five Commands to Reproduce Everything

```bash
# 1. Set up the tested environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r environment/requirements-tested.txt

# 2. Run the physics unit test suite (asserts 20/20 PASS, exit code 0)
python tests/test_physics.py

# 3. Quick reproduction of all tables (T1..T5) and figures (F1..F5) (< 15 seconds)
python run_all.py --quick

# 4. Optional: Full end-to-end dataset generation and model retraining (~25 minutes)
python run_all.py --full

# 5. Verify byte-for-byte SHA-256 integrity of all final tables
python -c "import hashlib, pathlib; [print(f'{f.name}: {hashlib.sha256(f.read_bytes()).hexdigest()[:16]}... OK') for f in sorted(pathlib.Path('results/final').glob('T*.csv'))]"
```

