"""Physics checks. Run:  python tests/test_physics.py   (all lines must print PASS)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from kluster import simulator as sm
from kluster.generate import play_game

def two(r, sa, sb):
    return np.array([[-r/2, 0.0], [r/2, 0.0]]), np.array([sa, sb])

def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    return bool(ok)

ok = True
# 1. dipole moment and friction limit (hand calculation from the notes)
ok &= check("dipole moment m = 0.2437 A m^2", abs(sm.M_MOM-0.24375) < 1e-4)
ok &= check("static friction limit = 5.20 mN", abs(sm.FS-5.2007e-3) < 1e-6)
# 2. pair force at 15 mm (hand: 0.352 N) and 1/r^4 scaling
ok &= check("F(15 mm) = 0.352 N", abs(sm.pair_force(0.015, 1)-0.3521) < 1e-3)
ok &= check("1/r^4 scaling (double r -> F/16)", abs(sm.pair_force(0.03, 1)*16-sm.pair_force(0.015, 1)) < 1e-9)
# 3. critical distance (hand: 43.0 mm)
ok &= check("critical distance = 43.0 mm", abs(sm.critical_distance()*1000-43.03) < 0.05)
# 4. simulator reproduces the critical distance
for r, expect in [(0.040, 'contact'), (0.042, 'contact'), (0.044, 'settled'), (0.050, 'settled')]:
    p, s = two(r, 1, -1)
    ok &= check(f"antiparallel pair at {r*1000:.0f} mm -> {expect}", sm.simulate(p, s, 0.1)[0] == expect)
# 5. parallel pair is repelled, stops, no contact (static rule gives a false alarm here)
p, s = two(0.03, 1, 1)
ok &= check("parallel pair 30 mm in 60 mm wall -> settled", sm.simulate(p, s, 0.06)[0] == 'settled')
ok &= check("static rule flags it anyway (false alarm)", sm.static_baseline(p, s, 0.06))
# 6. invalid start
p, s = two(0.0104, 1, -1)
ok &= check("overlapping start -> invalid_start", sm.simulate(p, s, 0.1)[0] == 'invalid_start')
# 7. time-step convergence
p, s = two(0.042, 1, -1)
ok &= check("dt and dt/2 agree", sm.label_checked(p, s, 0.1) == 'contact')
# 8. Newton's third law
rng = np.random.default_rng(0)
P = rng.uniform(-0.05, 0.05, (6, 2)); S = rng.choice([-1., 1.], 6)
ok &= check("sum of net forces = 0", np.abs(sm.net_forces(P, S).sum(axis=0)).max() < 1e-9)
# 9. wall normal force: friction only resists the tangential part (hand-computed cases, wall at 55 mm)
rmax = 0.055; pos = np.array([rmax, 0.0])
for f, expect, name in [((8e-3, 1e-3), 1.0e-3, "outward force: wall balances the normal part"),
                        ((-8e-3, 1e-3), np.hypot(8e-3, 1e-3), "inward force: wall cannot pull"),
                        ((3e-3, 6e-3), 6.0e-3, "oblique force: only tangential part counts")]:
    ok &= check("wall rule: " + name, abs(sm.friction_load(f, pos, rmax)-expect) < 1e-9)
ok &= check("away from the wall the full force counts", abs(sm.friction_load((8e-3, 1e-3), np.array([0.02, 0.0]), rmax)-np.hypot(8e-3, 1e-3)) < 1e-9)
# 10. a magnet pinned at the wall by an outward force larger than F_s is in equilibrium, not 'timeout'
P = np.array([[0.0, 0.0], [0.040, 0.0]]); S = np.array([1.0, 1.0])          # parallel pair, repulsion pushes #2 outward
lab, t, vmax, Pf = sm.simulate_full(P, S, 0.045+sm.R)                        # wall just outside magnet #2's radius
ok &= check("repelled pair relaxes against the wall and ends 'settled' (not timeout)", lab in ('settled',) and vmax == 0.0)
# 11. settled games continue from RESTING positions: every base board in a game is in equilibrium
rows = play_game(1, 2026)
base_ok = True
import pandas as pd
df = pd.DataFrame(rows)
for step, g in df.groupby('step'):
    if step == 0:
        continue
    r0 = g.iloc[0]
    n = int(r0['n_total']) - 1
    Pb = np.array([[r0[f'x{i}'], r0[f'y{i}']] for i in range(n)])/1000.0
    Sb = np.array([r0[f's{i}'] for i in range(n)])
    if n >= 2 and sm.static_baseline(Pb, Sb, r0['rw_mm']/1000.0):
        base_ok = False
ok &= check("every later base board is at rest (no net force above friction)", base_ok)
print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
sys.exit(0 if ok else 1)
