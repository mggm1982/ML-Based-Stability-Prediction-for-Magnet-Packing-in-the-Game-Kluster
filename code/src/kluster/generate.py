"""Game-style dataset generator (v1.2).

A "game" starts with one magnet. At every state we sample K_CAND random candidate placements
(random position and polarity) for the next magnet, simulate each, and record the label.
One settled candidate is chosen at random to continue the game, and the next board is the candidate's
FINAL RESTING arrangement (v1.2 fix: v1.1 wrongly continued from the pre-motion coordinates).
All rows from one game share a game_id: this is the GROUP used for splitting (rows within a game are dependent).
"""
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from . import simulator as sm

MAXN = 14
RW_CHOICES_MM = [60, 70, 80, 90, 100]
K_CAND = 16


def sample_candidate(pos, rmax, rng, tries=200):
    """Uniformly sample a valid candidate disc location inside the circular boundary.

    Samples coordinates uniformly over the disc area using polar coordinates
    (r = rmax * sqrt(u), theta ~ U[0, 2*pi]), ensuring the new disc does not
    overlap with any existing magnet on the board.

    Parameters
    ----------
    pos : np.ndarray
        Array of existing magnet positions of shape (n, 2) in metres.
    rmax : float
        Maximum center distance allowed (wall_radius - disc_radius) in metres.
    rng : np.random.Generator
        Independent NumPy pseudo-random number generator.
    tries : int, default=200
        Maximum rejection sampling attempts before aborting.

    Returns
    -------
    np.ndarray or None
        2D coordinate vector [x, y] in metres, or None if no non-overlapping location found.
    """
    for _ in range(tries):
        rr = rmax * np.sqrt(rng.uniform())
        th = rng.uniform(0, 2 * np.pi)
        q = np.array([rr * np.cos(th), rr * np.sin(th)])
        if len(pos) == 0 or np.min(np.linalg.norm(pos - q, axis=1)) > sm.D * (1 + sm.TOL) * 1.02:
            return q
    return None


def play_game(game_id, seed, T=1.0, dt=1e-5, relax=True, wall_balance=True, random_params=False,
              mu_s=None, mu_k=None, k_scale=None):
    """Simulate a full progressive game of Kluster from 1 up to MAXN magnets.

    At each step:
    1. K_CAND candidate placements are sampled and evaluated via the physics simulator.
    2. Step-halving convergence checks (dt and dt/2) are performed to label stability.
    3. If any candidate settles into static equilibrium, one settled outcome is randomly
       selected as the next base board (using its final relaxed positions).
    4. Each evaluation is recorded with initial board features and simulation outcome.

    Parameters
    ----------
    game_id : int
        Unique integer identifier for this game (used as the grouping unit).
    seed : int
        Master random seed; combined with game_id to yield order-independent reproducibility.
    T : float, default=1.0
        Simulation integration horizon in seconds.
    dt : float, default=1e-5
        Base numerical time step in seconds.
    relax : bool, default=True
        Whether to continue subsequent turns from the final resting positions of the settled stone.
    wall_balance : bool, default=True
        Whether wall constraint balances normal force.
    random_params : bool, default=False
        If True, draws game-specific physical friction (mu_s, mu_k) and force scaling (k_scale).
    mu_s, mu_k, k_scale : float, optional
        Explicit physical parameter overrides.

    Returns
    -------
    list of dict
        List of candidate placement records for all steps in the game.
    """
    rng = np.random.default_rng([seed, game_id])          # reproducible, order-independent
    has_custom = (mu_s is not None or mu_k is not None or k_scale is not None)
    if random_params:
        mu_s = float(rng.uniform(0.15, 0.60))
        mu_k = float(mu_s * rng.uniform(0.5, 0.8))
        k_scale = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6))))
    elif has_custom:
        mu_s = sm.MU_S if mu_s is None else float(mu_s)
        mu_k = sm.MU_K if mu_k is None else float(mu_k)
        k_scale = 1.0 if k_scale is None else float(k_scale)
    else:
        mu_s = sm.MU_S
        mu_k = sm.MU_K
        k_scale = 1.0
    rw = rng.choice(RW_CHOICES_MM) / 1000.0
    rmax = rw - sm.R
    pos = sample_candidate(np.empty((0, 2)), rmax, rng)[None, :]
    s = np.array([rng.choice([-1.0, 1.0])])
    rows, step = [], 0
    while len(s) < MAXN:
        settled_opts = []
        for _ in range(K_CAND):
            q = sample_candidate(pos, rmax, rng)
            if q is None:
                continue
            sq = rng.choice([-1.0, 1.0])
            P, S = np.vstack([pos, q]), np.append(s, sq)
            l1, tev, vmax, Pf = sm.simulate_full(P, S, rw, T, dt, mu_s=mu_s, mu_k=mu_k, k_scale=k_scale, wall_balance=wall_balance)
            l2 = sm.simulate(P, S, rw, T, dt / 2, mu_s=mu_s, mu_k=mu_k, k_scale=k_scale, wall_balance=wall_balance)[0]
            lab = l1 if l1 == l2 else 'unresolved'
            n = len(S)
            row = dict(game_id=game_id, step=step, rw_mm=rw * 1000, n_total=n, label=lab,
                       label_dt=l1, label_half=l2, t_event_ms=tev * 1000, vmax=vmax,
                       static_flag=sm.static_baseline(P, S, rw if wall_balance else None, mu_s=mu_s, k_scale=k_scale))
            if random_params or has_custom:
                row['mu_s'] = mu_s
                row['mu_k'] = mu_k
                row['k_scale'] = k_scale
            for i in range(MAXN):
                row[f'x{i}'] = P[i, 0] * 1000 if i < n else np.nan
                row[f'y{i}'] = P[i, 1] * 1000 if i < n else np.nan
                row[f's{i}'] = S[i] if i < n else np.nan
            rows.append(row)
            if lab == 'settled':
                settled_opts.append((Pf if relax else P, S))
        if not settled_opts:
            break
        pos, s = settled_opts[rng.integers(len(settled_opts))]
        step += 1
    return rows


def _chunk(args):
    """Worker helper function to process a batch of game IDs in parallel.

    Parameters
    ----------
    args : tuple
        Tuple of (game_ids_list, seed, [random_params]).

    Returns
    -------
    list of dict
        Aggregated candidate placement records across the batch.
    """
    if len(args) == 3:
        ids, seed, random_params = args
    else:
        ids, seed = args
        random_params = False
    out = []
    for g in ids:
        out += play_game(g, seed, random_params=random_params)
    return out


def generate(n_games, seed=2026, workers=2, random_params=False):
    """Generate a multi-game Kluster simulation dataset utilizing multi-process execution.

    Parameters
    ----------
    n_games : int
        Total number of games to generate.
    seed : int, default=2026
        Master pseudo-random seed.
    workers : int, default=2
        Number of parallel worker processes.
    random_params : bool, default=False
        Whether physical friction and force parameters are randomized per game.

    Returns
    -------
    pd.DataFrame
        Complete DataFrame containing initial placement features, metadata, and outcomes.
    """
    ids = np.array_split(np.arange(n_games), workers * 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_chunk, [(list(map(int, c)), seed, random_params) for c in ids if len(c)]))
    return pd.DataFrame([r for p in parts for r in p])
