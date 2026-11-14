"""Physics + simulator for the Kluster magnet-stability project (version 1.2).

MODEL (all SI units)
- Identical discs, point-dipole approximation, moments fixed vertical (s = +1 up, -1 down).
- Horizontal translation only, start at rest on a flat surface.
- Pair force is central and ~1/r^4: F = 3*K*s_i*s_j / r^4 (positive = repulsive), K = (mu0/4pi) m^2.
- Coulomb friction: static limit FS, kinetic FK.
- Circular frictionless wall, perfectly inelastic. A magnet pressed against the wall is held by the
  wall's normal force, so friction only has to resist the TANGENTIAL part of the magnetic force (v1.2 fix).
- Semi-implicit (Euler-Cromer) integration, fixed step dt.

OUTCOMES
  contact       a pair reached centre distance D*(1+TOL)
  settled       everything at rest and every (wall-corrected) net force <= static friction: equilibrium
  timeout       still moving after time T, no contact seen (finite-time statement only)
  unresolved    NaN, or overshoot deeper than 0.5*D, or dt vs dt/2 disagree
  invalid_start magnets overlap at t = 0

If you edit a parameter below, delete the __pycache__ folder so numba recompiles.
"""
import numpy as np
from numba import njit

# ---------------- parameters (illustrative values: measure real ones later) ----------------
MU0_4PI = 1e-7           # Magnetic constant: mu_0 / (4*pi) in H/m (or N/A^2)
BR = 1.3                 # Remanence (T), standard N42 Neodymium assumption
R, H = 5e-3, 3e-3        # Disc radius (5 mm) and thickness (3 mm) in metres
RHO = 7500.0             # Material density (kg/m^3)
G = 9.81                 # Gravitational acceleration (m/s^2)
MU_S = 0.3               # Nominal static friction coefficient
MU_K = 0.2               # Nominal kinetic friction coefficient
TOL = 0.05               # Contact proximity tolerance: centre distance <= D*(1+TOL)
WALL_EPS = 1e-9          # Radial margin (1 nm): magnets within this count as against the boundary

V = np.pi * R**2 * H                          # Magnet volume (m^3)
M_MOM = BR / (4 * np.pi * 1e-7) * V           # Dipole moment m_mag (A*m^2)
MASS = RHO * V                                # Magnet mass m (kg)
FS = MU_S * MASS * G                          # Static friction limit (N)
FK = MU_K * MASS * G                          # Kinetic friction force (N)
K = MU0_4PI * M_MOM**2                        # Force prefactor K = (mu_0/(4*pi)) * m_mag^2 (N*m^4)
D = 2 * R                                     # Disc diameter (m)

CONTACT, SETTLED, TIMEOUT, UNRESOLVED, INVALID = 0, 1, 2, 3, 4
LABELS = ['contact', 'settled', 'timeout', 'unresolved', 'invalid_start']


def pair_force(r, s):
    """Compute the central dipole-dipole pair force magnitude between two magnets.

    Parameters
    ----------
    r : float or np.ndarray
        Centre-to-centre separation distance between magnets in metres.
    s : float or int
        Relative polarity product s_i * s_j (+1 for parallel / repulsive,
        -1 for antiparallel / attractive).

    Returns
    -------
    float or np.ndarray
        Radial force magnitude in Newtons (+ = repulsive, - = attractive).
    """
    return 3 * s * K / r**4


def critical_distance(mu_s=MU_S, k_scale=1.0):
    """Analytical critical distance r_c where attractive dipole force equals static friction.

    For an isolated antiparallel pair (s_i * s_j = -1), mutual acceleration occurs
    if and only if initial separation r < r_c, where 3*K*k_scale / r_c^4 = mu_s * m * g.

    Parameters
    ----------
    mu_s : float, default=MU_S (0.3)
        Coefficient of static friction.
    k_scale : float, default=1.0
        Scaling multiplier on dipole force constant K.

    Returns
    -------
    float
        Critical separation distance in metres.
    """
    return (3 * K * k_scale / (mu_s * MASS * G))**0.25


@njit(cache=True)
def _forces(px, py, s, n, fx, fy, kk):
    """Compute pairwise dipole-dipole interaction forces on all magnets (Numba compiled).

    Evaluates Newton's third law pair forces F_ij = 3*kk*s_i*s_j * (r_i - r_j) / r^5.

    Parameters
    ----------
    px, py : np.ndarray
        1D arrays of shape (n,) containing magnet x and y coordinates in metres.
    s : np.ndarray
        1D array of shape (n,) containing dipole polarities (+1.0 or -1.0).
    n : int
        Total number of magnets currently on the board.
    fx, fy : np.ndarray
        Pre-allocated 1D arrays of shape (n,) updated in-place with net forces in Newtons.
    kk : float
        Effective force constant K * k_scale.

    Returns
    -------
    float
        Minimum pairwise centre-to-centre distance rmin encountered in metres.
    """
    rmin = 1e30
    for i in range(n):
        fx[i] = 0.0
        fy[i] = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = px[i] - px[j]
            dy = py[i] - py[j]
            r2 = dx * dx + dy * dy
            r = np.sqrt(r2)
            if r < rmin:
                rmin = r
            c = 3.0 * kk * s[i] * s[j] / (r2 * r2 * r)
            fx[i] += c * dx
            fy[i] += c * dy
            fx[j] -= c * dx
            fy[j] -= c * dy
    return rmin


@njit(cache=True)
def _sim(px0, py0, s, wall_radius, T, dt, fs, fk, mass, kk, wall_bal):
    """Core numerical simulation loop using semi-implicit Euler integration (Numba compiled).

    Simulates planar translation of n point dipoles subject to pairwise magnetic forces,
    Coulomb friction (static threshold and kinetic opposing velocity), and an inelastic
    frictionless circular wall constraint.

    Parameters
    ----------
    px0, py0 : np.ndarray
        Initial x and y positions of shape (n,) in metres.
    s : np.ndarray
        Dipole polarities of shape (n,) with values in {-1.0, +1.0}.
    wall_radius : float
        Radius of circular boundary in metres.
    T : float
        Maximum simulation horizon in seconds (default 1.0 s).
    dt : float
        Integration time step in seconds (e.g. 1e-5 s).
    fs : float
        Static friction force threshold in Newtons (mu_s * mass * g).
    fk : float
        Kinetic friction force in Newtons (mu_k * mass * g).
    mass : float
        Individual magnet mass in kg.
    kk : float
        Effective force scaling prefactor (K * k_scale).
    wall_bal : bool
        If True, wall normal force balances outward magnetic force so friction
        only resists tangential motion.

    Returns
    -------
    tuple
        (outcome_code, event_time_s, max_speed_m_s, final_px, final_py)
        outcome_code: 0=CONTACT, 1=SETTLED, 2=TIMEOUT, 3=UNRESOLVED, 4=INVALID
    """
    n = px0.shape[0]
    px = px0.copy()
    py = py0.copy()
    vx = np.zeros(n)
    vy = np.zeros(n)
    fx = np.zeros(n)
    fy = np.zeros(n)
    rmax = wall_radius - R
    nsteps = int(T / dt) + 1
    for step in range(nsteps):
        rmin = _forces(px, py, s, n, fx, fy, kk)
        bad = False
        for i in range(n):
            if not (np.isfinite(fx[i]) and np.isfinite(fy[i])):
                bad = True
        if bad or not np.isfinite(rmin):
            return UNRESOLVED, step * dt, 0.0, px, py
        if rmin <= D * (1.0 + TOL):
            if step == 0:
                return INVALID, 0.0, 0.0, px, py
            vmax = 0.0
            for i in range(n):
                sp = np.sqrt(vx[i]**2 + vy[i]**2)
                if sp > vmax:
                    vmax = sp
            if rmin >= 0.5 * D:
                return CONTACT, step * dt, vmax, px, py
            return UNRESOLVED, step * dt, vmax, px, py
        if wall_bal:
            for i in range(n):
                rr = np.sqrt(px[i]**2 + py[i]**2)
                if rr >= rmax - WALL_EPS and rr > 0.0:
                    nx = px[i] / rr
                    ny = py[i] / rr
                    fn = fx[i] * nx + fy[i] * ny
                    if fn > 0.0:
                        fx[i] -= fn * nx
                        fy[i] -= fn * ny
        all_rest = True
        all_weak = True
        for i in range(n):
            sp = np.sqrt(vx[i]**2 + vy[i]**2)
            fm = np.sqrt(fx[i]**2 + fy[i]**2)
            if sp >= 1e-9:
                all_rest = False
            if fm > fs:
                all_weak = False
        if all_rest and all_weak:
            return SETTLED, step * dt, 0.0, px, py
        for i in range(n):
            sp = np.sqrt(vx[i]**2 + vy[i]**2)
            fm = np.sqrt(fx[i]**2 + fy[i]**2)
            if sp < 1e-9 and fm <= fs:
                vx[i] = 0.0
                vy[i] = 0.0
                continue
            f_net_x = fx[i]
            f_net_y = fy[i]
            if sp >= 1e-9:
                f_net_x -= fk * (vx[i] / sp)
                f_net_y -= fk * (vy[i] / sp)
            nvx = vx[i] + (f_net_x / mass) * dt
            nvy = vy[i] + (f_net_y / mass) * dt
            nsp = np.sqrt(nvx**2 + nvy**2)
            crossed = ((nvx * vx[i] + nvy * vy[i]) < 0.0) or (sp > 0.0 and nsp < 1e-9)
            if crossed and fm <= fs:
                nvx = 0.0
                nvy = 0.0
            vx[i] = nvx
            vy[i] = nvy
        for i in range(n):
            px[i] += vx[i] * dt
            py[i] += vy[i] * dt
            rr = np.sqrt(px[i]**2 + py[i]**2)
            if rr > rmax:
                nx = px[i] / rr
                ny = py[i] / rr
                px[i] = nx * rmax
                py[i] = ny * rmax
                vn = vx[i] * nx + vy[i] * ny
                if vn > 0.0:
                    vx[i] -= vn * nx
                    vy[i] -= vn * ny
    vmax = 0.0
    for i in range(n):
        sp = np.sqrt(vx[i]**2 + vy[i]**2)
        if sp > vmax:
            vmax = sp
    return TIMEOUT, T, vmax, px, py


def simulate_full(pos, s, wall_radius, T=1.0, dt=1e-5, mu_s=MU_S, mu_k=MU_K, k_scale=1.0, wall_balance=True):
    """Execute complete numerical simulation and return trajectory outcomes and resting positions.

    Parameters
    ----------
    pos : np.ndarray or list of (x, y)
        Initial coordinates of shape (n, 2) in metres.
    s : np.ndarray or list of float
        Dipole polarities of length n, values in {-1, +1}.
    wall_radius : float
        Radius of circular boundary in metres.
    T : float, default=1.0
        Simulation duration in seconds.
    dt : float, default=1e-5
        Integration time step in seconds.
    mu_s : float, default=MU_S (0.3)
        Coefficient of static friction.
    mu_k : float, default=MU_K (0.2)
        Coefficient of kinetic friction.
    k_scale : float, default=1.0
        Dipole force scaling multiplier.
    wall_balance : bool, default=True
        Whether boundary normal force relieves friction load against the wall.

    Returns
    -------
    tuple
        (label, event_time_s, max_speed_m_s, final_positions_m)
        label is one of 'contact', 'settled', 'timeout', 'unresolved', 'invalid_start'.
    """
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    code, t, vmax, px, py = _sim(pos[:, 0].copy(), pos[:, 1].copy(), np.asarray(s, dtype=np.float64),
                                 wall_radius, T, dt, mu_s * MASS * G, mu_k * MASS * G, MASS, K * k_scale, wall_balance)
    return LABELS[code], t, vmax, np.column_stack([px, py])


def simulate(pos, s, wall_radius, T=1.0, dt=1e-5, **kw):
    """Lightweight simulation wrapper returning label, event time, and maximum speed.

    Parameters
    ----------
    pos : np.ndarray or list of (x, y)
        Initial positions of shape (n, 2) in metres.
    s : list or np.ndarray
        Dipole moments (+1/-1) of length n.
    wall_radius : float
        Radius of boundary in metres.
    T : float, default=1.0
        Maximum simulation time in seconds.
    dt : float, default=1e-5
        Time step in seconds.
    **kw : dict
        Optional physical parameters (mu_s, mu_k, k_scale, wall_balance).

    Returns
    -------
    tuple
        (label, event_time_s, max_speed_m_s)
    """
    lab, t, vmax, _ = simulate_full(pos, s, wall_radius, T, dt, **kw)
    return lab, t, vmax


def label_checked(pos, s, wall_radius, T=1.0, dt=1e-5, **kw):
    """Convergence check: run integration at dt and dt/2; flag label discrepancies as 'unresolved'.

    Ensures that physical outcomes are robust to numerical integration step-halving.

    Parameters
    ----------
    pos : np.ndarray
        Initial positions of shape (n, 2) in metres.
    s : np.ndarray
        Polarities of length n.
    wall_radius : float
        Boundary radius in metres.
    T : float, default=1.0
        Simulation duration in seconds.
    dt : float, default=1e-5
        Nominal time step in seconds.
    **kw : dict
        Physical parameter overrides.

    Returns
    -------
    str
        Verified outcome label, or 'unresolved' if dt and dt/2 disagree.
    """
    a = simulate(pos, s, wall_radius, T, dt, **kw)[0]
    b = simulate(pos, s, wall_radius, T, dt / 2, **kw)[0]
    return a if a == b else 'unresolved'


def net_forces(pos, s, k_scale=1.0):
    """Vectorized calculation of the net magnetic force vector on each magnet.

    Parameters
    ----------
    pos : np.ndarray
        Coordinates of shape (n, 2) in metres.
    s : np.ndarray
        Polarities of shape (n,).
    k_scale : float, default=1.0
        Force multiplier.

    Returns
    -------
    np.ndarray
        Net force vectors of shape (n, 2) in Newtons.
    """
    pos = np.asarray(pos, float)
    s = np.asarray(s, float)
    d = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, np.inf)
    coef = 3 * K * k_scale * (s[:, None] * s[None, :]) / r**5
    return (coef[:, :, None] * d).sum(axis=1)


def friction_load(force, position, r_max, tol=WALL_EPS):
    """Compute the tangential component of magnetic force that friction must resist for one magnet.

    If a magnet is in contact with the outer boundary (|position| >= r_max - tol) and the net force
    points outward against the wall, the wall normal force cancels the radial component, leaving
    only the tangential component to be resisted by static friction.

    Parameters
    ----------
    force : np.ndarray
        Net magnetic force vector (fx, fy) in Newtons.
    position : np.ndarray
        Planar coordinates (px, py) in metres.
    r_max : float
        Maximum allowable radial position (wall_radius - disc_radius) in metres.
    tol : float, default=WALL_EPS (1e-9)
        Wall contact detection threshold.

    Returns
    -------
    float
        Effective friction load magnitude in Newtons.
    """
    force = np.asarray(force, float)
    position = np.asarray(position, float)
    r = np.linalg.norm(position)
    if r < r_max - tol or r == 0.0:
        return float(np.linalg.norm(force))
    n_hat = position / r
    f_n = float(force @ n_hat)
    if f_n > 0.0:
        force = force - f_n * n_hat
    return float(np.linalg.norm(force))


def effective_forces(pos, s=None, wall_radius=None, k_scale=1.0, mu_s=MU_S):
    """Compute friction-relevant force magnitude on every magnet, applying wall normal correction.

    Parameters
    ----------
    pos : np.ndarray or pd.Series/dict
        Array of shape (n, 2) in metres, or dataset row dict containing coordinates.
    s : np.ndarray, optional
        Polarities of length n.
    wall_radius : float, optional
        Boundary radius in metres.
    k_scale : float, default=1.0
        Force constant multiplier.
    mu_s : float, default=MU_S
        Static friction coefficient.

    Returns
    -------
    np.ndarray
        1D array of effective force magnitudes in Newtons for each magnet.
    """
    if hasattr(pos, '__getitem__') and not isinstance(pos, np.ndarray) and 'n_total' in pos:
        row = pos
        n = int(row['n_total'])
        pos = np.array([[row[f'x{i}'], row[f'y{i}']] for i in range(n)]) / 1000.0
        s = np.array([row[f's{i}'] for i in range(n)])
        if wall_radius is None and 'rw_mm' in row:
            wall_radius = row['rw_mm'] / 1000.0
        if k_scale == 1.0 and 'k_scale' in row and row['k_scale'] == row['k_scale']:
            k_scale = float(row['k_scale'])
    F = net_forces(pos, s, k_scale)
    if wall_radius is None:
        return np.linalg.norm(F, axis=1)
    r_max = wall_radius - R
    return np.array([friction_load(F[i], pos[i], r_max) for i in range(len(F))])


def static_baseline(pos, s=None, wall_radius=None, mu_s=MU_S, k_scale=1.0):
    """Analytical baseline rule: True if any magnet's effective force exceeds static friction at t = 0.

    Parameters
    ----------
    pos : np.ndarray or pd.Series/dict
        Coordinates of shape (n, 2) in metres, or dataset row dictionary.
    s : np.ndarray, optional
        Dipole polarities of length n.
    wall_radius : float, optional
        Boundary radius in metres.
    mu_s : float, default=MU_S
        Coefficient of static friction.
    k_scale : float, default=1.0
        Force constant multiplier.

    Returns
    -------
    bool
        True if static equilibrium is violated at initial state (predicted contact/unstable).
    """
    if hasattr(pos, '__getitem__') and not isinstance(pos, np.ndarray) and 'n_total' in pos:
        row = pos
        n = int(row['n_total'])
        pos = np.array([[row[f'x{i}'], row[f'y{i}']] for i in range(n)]) / 1000.0
        s = np.array([row[f's{i}'] for i in range(n)])
        if wall_radius is None and 'rw_mm' in row:
            wall_radius = row['rw_mm'] / 1000.0
        if mu_s == MU_S and 'mu_s' in row and row['mu_s'] == row['mu_s']:
            mu_s = float(row['mu_s'])
        if k_scale == 1.0 and 'k_scale' in row and row['k_scale'] == row['k_scale']:
            k_scale = float(row['k_scale'])
    return bool((effective_forces(pos, s, wall_radius, k_scale) > mu_s * MASS * G).any())
