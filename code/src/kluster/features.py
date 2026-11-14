"""Feature engineering. One row = one (base state + candidate new magnet).
The NEW magnet is always the last one in the arrangement.

GEOM features use geometry and polarity only (no force calculation).
PHYS features add force ratios (wall-corrected, the same quantities the static-friction baseline uses).
PHYS_PARAMS features append parameter-normalized analytical quantities (rc_mm, d_opp_over_rc).
"""
import numpy as np
import pandas as pd
from . import simulator as sm

MAXN = 14
GEOM = ['n_total', 'rw_mm', 's_new', 'd_nn', 'd_same', 'd_opp', 'n_same_40', 'n_opp_40',
        'n_same_60', 'n_opp_60', 'wall_gap', 'r_min_all', 'mean_r_centre', 'net_pol', 'frac_up']
PHYS = GEOM + ['f_ratio_max', 'f_ratio_new', 'n_over_fs']
PHYS_PARAMS = PHYS + ['rc_mm', 'd_opp_over_rc']


def arrangement(row):
    """Extract magnet coordinates and polarities from a dataset row.

    Parameters
    ----------
    row : pd.Series or dict
        Row containing 'n_total', 'xi', 'yi', and 'si' columns.

    Returns
    -------
    tuple
        (positions, polarities)
        positions: np.ndarray of shape (n, 2) in metres.
        polarities: np.ndarray of shape (n,) with values in {-1.0, +1.0}.
    """
    n = int(row['n_total'])
    P = np.array([[row[f'x{i}'], row[f'y{i}']] for i in range(n)]) / 1000.0
    S = np.array([row[f's{i}'] for i in range(n)])
    return P, S


def featurize(df):
    """Extract physical and geometric features for every candidate placement row.

    Features computed include:
    - Geometric: nearest-neighbor distances, same/opposite polarity separation (d_same, d_opp),
      local neighbor density within 40 mm and 60 mm, distance to boundary (wall_gap).
    - Physical: maximum friction-relevant force ratio (f_ratio_max = F_max / F_s),
      candidate magnet force ratio (f_ratio_new), count of magnets exceeding static friction.
    - Parameter-aware: analytical critical distance rc_mm and normalized ratio d_opp_over_rc.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset DataFrame with candidate placements.

    Returns
    -------
    pd.DataFrame
        Engineered feature DataFrame with rows aligned to the input DataFrame.
    """
    out = []
    for row in df.to_dict('records'):
        P, S = arrangement(row)
        n = len(S)
        new = n - 1
        rw = row['rw_mm'] / 1000.0
        mu_s = float(row['mu_s']) if ('mu_s' in row and pd.notna(row['mu_s'])) else sm.MU_S
        k_scale = float(row['k_scale']) if ('k_scale' in row and pd.notna(row['k_scale'])) else 1.0
        fs = mu_s * sm.MASS * sm.G
        F = sm.effective_forces(P, S, rw, k_scale=k_scale)
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        dn, so, sn = d[new, :new], S[:new], S[new]
        same, opp = dn[so == sn], dn[so != sn]
        d_same_val = same.min() if len(same) else 0.2
        d_opp_val = opp.min() if len(opp) else 0.2
        rc_m = (3.0 * sm.K * k_scale / fs)**0.25
        rc_mm = rc_m * 1000.0
        d_opp_over_rc = d_opp_val / rc_m
        out.append(dict(
            n_total=n, rw_mm=row['rw_mm'], s_new=sn, d_nn=dn.min(),
            d_same=d_same_val, d_opp=d_opp_val,
            n_same_40=(same < 0.04).sum(), n_opp_40=(opp < 0.04).sum(),
            n_same_60=(same < 0.06).sum(), n_opp_60=(opp < 0.06).sum(),
            wall_gap=rw - np.linalg.norm(P[new]), r_min_all=d.min(),
            mean_r_centre=np.linalg.norm(P, axis=1).mean(), net_pol=S.sum(), frac_up=(S > 0).mean(),
            f_ratio_max=F.max() / fs, f_ratio_new=F[new] / fs, n_over_fs=int((F > fs).sum()),
            rc_mm=rc_mm, d_opp_over_rc=d_opp_over_rc))
    return pd.DataFrame(out)
