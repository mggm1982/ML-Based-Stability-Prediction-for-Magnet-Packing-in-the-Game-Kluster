# Data Card: kluster_dataset_v2.csv

## 1. Overview
- **Filename:** `data/kluster_dataset_v2.csv`
- **Description:** Synthetic tabletop magnet placement dataset with physical parameters ($\mu_s, \mu_k, k_{\text{scale}}$) independently randomized per game to simulate manufacturing and surface variations.
- **Generation Command:** `python scripts/make_dataset.py --games 1500 --seed 2027 --random-params --out data/kluster_dataset_v2.csv`
- **Total Rows:** 205,856
- **Total Games:** 1,500 (mean 137.2 candidate rows per game)
- **Unit of Observation:** One row represents one candidate placement of a new magnet on a board that is at rest. Each game continues strictly from the chosen settled candidate's final resting coordinates.
- **Group Variable:** `game_id` (all rows sharing a `game_id` belong to the same sequential game and must be partitioned together during data splitting).

---

## 2. Schema and Feature Definitions

### Board & Candidate Configuration
- `game_id` (int): Sequential identifier for each independent game (0 to 1499).
- `step` (int): Move counter within the game (0 to 12).
- `rw_mm` (float): Radius of the confining boundary in millimetres (discrete choices: 60.0, 70.0, 80.0, 90.0, 100.0 mm).
- `n_total` (int): Total number of magnets on the board including the candidate (2 to 14).
- `x0..x13`, `y0..y13` (float): Initial horizontal positions of magnets in millimetres. The candidate magnet is always placed at index `n_total - 1`. Unused slots are empty/NaN.
- `s0..s13` (float): Polarities ($\pm 1.0$) of magnets.

### Randomized Physical Parameters
- `mu_s` (float): Static friction coefficient, sampled independently per game from $\text{Uniform}(0.15, 0.60)$.
- `mu_k` (float): Kinetic friction coefficient, sampled as $\mu_s \times \text{Uniform}(0.5, 0.8)$.
- `k_scale` (float): Dipole force constant scaling factor, sampled log-uniformly from $\exp(\text{Uniform}(\ln 0.6, \ln 1.6))$.

### Physical Simulation Outcomes
- `label` (str): Ground-truth outcome of the numerical simulation:
  - `contact`: Discs collided (minimum distance $\le 10.5\text{ mm}$).
  - `settled`: System reached static equilibrium with zero motion.
  - `timeout`: Simulation reached $T = 1.0\text{ s}$ without contact or equilibrium.
  - `unresolved`: Trajectory divergence between $\Delta t = 10^{-5}\text{ s}$ and $\Delta t = 5 \times 10^{-6}\text{ s}$.
- `label_dt`, `label_half` (str): Individual integration labels at $\Delta t$ and $\Delta t / 2$.
- `t_event_ms` (float): Time of first contact or settling in milliseconds ($1000.0\text{ ms}$ for timeout).
- `vmax` (float): Maximum speed attained by any disc during the trajectory in m/s.
- `static_flag` (bool): Heuristic baseline flag indicating whether initial net force exceeds static friction.

---

## 3. Label Breakdown (Recomputed)
Across all 205,856 rows:
- `contact`: 110,303 (53.58%)
- `settled`: 95,036 (46.17%)
- `timeout`: 513 (0.25%)
- `unresolved`: 4 (0.00%)
- Total: 205,856 rows (100.00%)

---

## 4. Parameter Distributions (across 1,500 games)
Computed via `df.groupby('game_id').first()`:
- `rw_mm`: Min: 60.0 mm, Median: 80.0 mm, Max: 100.0 mm
- `mu_s`: Min: 0.1504, Median: 0.3746, Max: 0.5999
- `mu_k`: Min: 0.0783, Median: 0.2413, Max: 0.4770
- `k_scale`: Min: 0.6002, Median: 0.9717, Max: 1.5996
- Critical Interaction Distance $r_c = (3 K k_{\text{scale}} / (\mu_s m g))^{1/4}$:
  - Min: 31.91 mm, Median: 40.80 mm, Max: 56.71 mm
- Critical Ratio $q = r_c / r_{c,\text{default}}$ (where $r_{c,\text{default}} \approx 43.033\text{ mm}$):
  - Min: 0.742, Median: 0.948, Max: 1.318

---

## 5. Parameter-Shift Groups & Test-Game Audit
Across 20 repeated splits (seeds 0..19) using `src/kluster/paramshift.py`:
- **Subgroup $0.8 \le q \le 1.25$:** 1,377 games (187,760 rows, 91.80% of games).
  - Mean test games per split: **207.35** (Min: 194, Max: 216; $\ge 30$, **CONFIRMATORY**).
- **Subgroup $q < 0.8$:** 91 games (14,112 rows, 6.07% of games).
  - Mean test games per split: **13.35** (Min: 7, Max: 24; $< 30$, **EXPLORATORY**).
- **Subgroup $q > 1.25$:** 32 games (3,984 rows, 2.13% of games).
  - Mean test games per split: **4.30** (Min: 2, Max: 8; $< 30$, **EXPLORATORY**).

---

## 6. Seed Provenance Note
While initial documentation cited `--seed 42`, parameter reproduction tests confirmed that the physical parameters in `kluster_dataset_v2.csv` match generator seed **2027** across all games to machine precision ($\text{diff} \le 8.33 \times 10^{-17}$).
