# Kluster Magnet-Stability: Project Submission (CWS)

**Student Name:** Sankalp Savarn  
**Enrolment Number:** 24123038  
**Department:** Physics, Indian Institute of Technology Roorkee  
**Supervisor** Prof. Mayank Goswami (mayank.goswami@ph.iitr.ac.in)  

---

## 1. Submission Overview & Compliance Checklist

In accordance with course guidelines (CWS Submission Template), this package provides:
1. **(a) Code & Environment:** Standalone, commented codebase with pinned dependencies (`code/code.zip`, `code/environment/requirements-tested.txt`).
2. **(b) Training Data:** Partitioned reference training split (`data/training data/train_games_v2.csv` [750 games, 101,949 candidate placements] with SHA-256 checksums).
3. **(c) Testing Data:** Disjoint reference evaluation split (`data/testing data/test_games_v2.csv` [225 games, 30,864 candidate placements] with SHA-256 checksums).
4. **(d) Presentation:** Presentation slides (`video presentation/slides.ppt`).
5. **(e) Video Presentation:** Video recording (`video presentation/presentation video.mp4`).
6. **(f) Pre-Print Ready Article:** Fully formatted manuscript in Word (`preprint article/Preprint.docx`), publication PDF (`preprint article/Preprint.pdf`), and BibTeX database (`preprint article/bibliography.bib`).

---

## 2. Project Summary (Overview)
1. This project investigates whether machine learning surrogates can predict contact instability in planar magnet packing for the tabletop game Kluster.
2. We simulate 2D interacting cylindrical magnets modelled as vertical point dipoles subject to Coulomb planar friction and an inelastic circular boundary.
3. A semi-implicit Euler numerical simulator generates ground-truth placement trajectories across 1,500 games (205,856 candidate placements, randomized physical parameters).
4. Hand-engineered features capture local geometry, opposite-polarity dipole distances, and net force-to-friction ratios without exposing trajectory outcomes.
5. HistGradientBoosting and Logistic Regression classifiers are trained to predict contact events and calibrated using size-conditional Conformal Risk Control.
6. Across all board sizes (2 to 14 magnets), size-conditional Conformal Risk Control controls wrong-skip risk ON AVERAGE ONLY across test splits (mean 0.0457 at alpha = 0.05).
7. Paired comparisons against an analytical Combined rule show no ML gain on sparse boards (bin 2-4: +0.0015 [-0.0010, 0.0056]), but moderate gains on crowded boards (+0.0581 in bin 5-8).
8. Under unadapted parameter shift (Exp A, q < 0.8), statistical guarantees collapse (error rate 0.0670 for GB vs 0.1720 for distance rules); parameter-conditioned models restore central control (0.0459 in 0.8 <= q <= 1.25).
9. The combined feature-extraction and ML inference pipeline delivers an empirical wall-clock speed-up of 1.82x over single runs and 2.03x over checked simulation runs.
10. All numerical findings are traced to frozen pipeline outputs, verified against unit physics checks, and accompanied by full reproducibility assets.

---

## 3. Submission Directory Structure
```text
ML-Based Stability Prediction for Magnet Packing in the Game Kluster/
├── README_START_HERE.md            <- Primary submission guide and Drive link template (this file)
├── publication_declaration.txt     <- Signed declaration
├── preprint article/
│   ├── Preprint.docx               <- Formatted academic manuscript (Word docx, editable)
│   ├── Preprint.pdf                <- Compiled pdf file
│   └── bibliography.bib            <- Contains references 
├── code/
│   ├── run_all.py                  <- Master executable runner (--quick and --full modes)
│   ├── CODE_WALKTHROUGH.md         <- Reading order, per-file summaries, and data flow diagram
│   ├── README_REPRODUCE.md         <- Detailed environment setup and execution instructions
│   ├── code.zip                    <- Portable archive of all code, tests, and environment pins
│   ├── figures/                    <- Publication figures F1 to F5 (PNG and PDF)
│   ├── src/kluster/                <- Commented simulator, generator, features, and CRC modules
│   ├── scripts/                    <- Experiment execution and table reproduction runners
│   ├── tests/                      <- Physics unit test suite (test_physics.py, 20 PASS checks)
│   └── environment/                <- Exact dependency specifications (requirements-tested.txt)
├── data/
│   ├── training data/
│   │   ├── train_games_v2.csv      <- Reference training split (750 games, 101,949 rows)
│   │   └── CHECKSUM.txt            <- Cryptographic SHA-256 hash verification
│   └── testing data/
│       ├── test_games_v2.csv       <- Disjoint reference test split (225 games, 30,864 rows)
│       └── CHECKSUM.txt            <- Cryptographic SHA-256 hash verification
├── documentation/
│   ├── DATA_CARD.md                <- Comprehensive dataset card for kluster_dataset_v2.csv
│   ├── limitations.md              <- Physical and statistical claim boundaries
│   └── OPEN_ISSUES.md              <- Reconciled audit notes and draft discrepancy resolutions
└── video presentation/
    ├── slides.pdf                  <- Presentation slides
    └── presentation video.mp4      <- Explanation video
```

---

## 4. Quick Execution (Under 2 Minutes)
From the `code/` folder (or repository root):

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate

# 2. Install tested dependencies
pip install -r environment/requirements-tested.txt

# 3. Run master reproduction runner (runs physics tests, prints T1..T5, regenerates F1..F5)
python run_all.py --quick
```