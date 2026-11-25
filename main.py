"""Run the best trained agent in the 3D simulation (entry point for markers).

    uv run main.py                     # best overall agent, 3D viewer + verbose terminal
    uv run main.py --algo ppo          # best PPO model (dqn | reinforce | ppo | a2c | best)
    uv run main.py --hard              # unseen harder scenario (earlier floods, less budget)
    uv run main.py --no-render         # terminal only
    uv run main.py --serve             # JSON policy API for a dashboard / mobile client
    uv run main.py --compare           # evaluate all four best models on unseen seeds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from environment import HARD_CONFIG, AidPrepositioningEnv
from environment.custom_env import IPC_NAMES

ROOT = Path(__file__).resolve().parent
BEST = ROOT / "models" / "best_models.json"


def resolve_model(algo: str) -> tuple[str, Path]:
    best = json.loads(BEST.read_text())
    if algo == "best":
        algo = best["overall"]
    return algo, ROOT / best[algo]["path"]


def run_episode(env, act_fn, info_fn, seed: int, verbose: bool = True):
    obs, info = env.reset(seed=seed)
    total, done = 0.0, False
    if verbose:
        print(f"\n=== Episode seed {seed} | budget {env.cfg.budget} | hub stock {env.hub:.2f} loads ===")
    while not done:
        decision = info_fn(obs)
        obs, r, term, trunc, info = env.step(decision["action"])
        total += r
        done = term or trunc
        if verbose:
            ipc = " ".join(f"{c[:4]}:{p}" for c, p in zip(
                [c["name"] for c in env.state_dict_for_render()["counties"]],
                [c["ipc"] for c in env.state_dict_for_render()["counties"]]))
            alt = ", ".join(f"{t['name']} ({t['score']:.2f})" for t in decision["top3"][1:])
            print(f"wk {info['week']:2d} | {decision['action_name']:<46} | r {r:7.3f} | "
                  f"budget {info['budget']:5.1f} | hub {info['hub']:4.1f} | IPC {ipc}")
            print(f"      -> {env.last['outcome']}   [next best: {alt}]")
    if verbose:
        worst = max(c["ipc"] for c in env.state_dict_for_render()["counties"])
        print(f"=== Return {total:.2f} | need covered {100 * info['coverage']:.1f}% | "
              f"worst IPC {worst} ({IPC_NAMES[worst]}) | famine {info['famine']} | "
              f"convoys {info['trucks']} airdrops {info['airdrops']} procurements {info['procurements']} ===")
    return total, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="best", choices=["best", "dqn", "reinforce", "ppo", "a2c"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=10_000, help="evaluation seeds start here (unseen in training)")
    ap.add_argument("--hard", action="store_true", help="harder unseen scenario")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds per simulated week in the viewer")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--serve", action="store_true", help="serve the policy API only")
    ap.add_argument("--compare", action="store_true", help="evaluate all best models on unseen seeds")
    ap.add_argument("--wait", action="store_true", help="wait for Enter before the agent starts")
    args = ap.parse_args()

    from training.common import full_eval, load_policy

    if args.compare:
        best = json.loads(BEST.read_text())
        for a in ["dqn", "reinforce", "ppo", "a2c"]:
            act_fn, _, _ = load_policy(ROOT / best[a]["path"])
            r = full_eval(act_fn)
            print(f"{a:<10} unseen: {r['eval_mean_reward']:7.2f} (famine {r['eval_famine_rate']:.2f}, "
                  f"coverage {r['eval_coverage']:.3f}) | hard: {r['hard_mean_reward']:7.2f} "
                  f"(famine {r['hard_famine_rate']:.2f})")
        return

    algo, path = resolve_model(args.algo)
    act_fn, info_fn, ckpt = load_policy(path)
    print(f"[main] loaded {algo.upper()} model: {path.relative_to(ROOT)}")

    if args.serve:
        from environment.rendering import serve_forever
        serve_forever(args.port, info_fn, meta=dict(agent=f"{algo.upper()} ({path.stem})"))
        return

    config = HARD_CONFIG if args.hard else None
    env = AidPrepositioningEnv(render_mode=None if args.no_render else "human", config=config,
                               viewer_port=args.port, step_delay=args.delay)
    if not args.no_render:
        env.reset(seed=args.seed)              # starts the viewer
        env._viewer.set_policy(info_fn, meta=dict(agent=f"{algo.upper()} ({path.stem})"))
        if args.wait:
            input("[main] viewer ready - press Enter to start the agent... ")
        else:
            print("[main] giving the browser 4 s to connect...")
            time.sleep(4)
    returns = []
    for k in range(args.episodes):
        total, _ = run_episode(env, act_fn, info_fn, seed=args.seed + k)
        returns.append(total)
        if not args.no_render:
            time.sleep(2)
    print(f"\n[main] {algo.upper()} mean return over {args.episodes} episodes: {np.mean(returns):.2f}")
    env.close()


if __name__ == "__main__":
    main()
