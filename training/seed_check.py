"""Re-train each algorithm's best configuration with extra seeds (robustness check).

    uv run python -m training.seed_check --algo dqn --seeds 1 2
"""
import argparse
import json

from training.common import MODELS
from training import dqn_training, pg_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    a = ap.parse_args()
    best = json.loads((MODELS / "best_models.json").read_text())[a.algo]["run"]
    if a.algo == "dqn":
        cfg = next(dict(c) for c in dqn_training.SWEEP if c["name"] == best)
    else:
        cfg = next(dict(c) for c in pg_training.SWEEPS[a.algo] if c["name"] == best)
    cfg.pop("name")
    for s in a.seeds:
        if a.algo == "dqn":
            dqn_training.train({**cfg, "seed": s}, f"{best}_seed{s}", save=False)
        else:
            pg_training.train(a.algo, {**cfg, "seed": s}, f"{best}_seed{s}", save=False)


if __name__ == "__main__":
    main()
