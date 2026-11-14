"""Generate the synthetic dataset.   python scripts/make_dataset.py --games 300 --workers 2 --seed 2026"""
import sys, pathlib, argparse, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kluster.generate import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "data" / "kluster_dataset.csv"))
    ap.add_argument("--random-params", action="store_true", default=False,
                    help="Randomise physical parameters (mu_s, mu_k, k_scale) per game")
    a = ap.parse_args()
    t = time.time()
    df = generate(a.games, a.seed, a.workers, random_params=a.random_params)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} rows, {df.game_id.nunique()} games, {time.time()-t:.0f} s -> {a.out}")
    print(df.label.value_counts().to_string())
