"""Aggregate sweep logs into tables, pick best models, and draw report figures.

    uv run python -m training.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment import HARD_CONFIG, AidPrepositioningEnv  # noqa: E402
from environment.custom_env import (N_COUNTIES, greedy_policy, prepositioning_policy,  # noqa: E402
                                    random_policy)
from training.common import EVAL_SEEDS, HARD_SEEDS, LOGS, MODELS, ROOT, evaluate, load_policy  # noqa: E402

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
GRAY = "#8a8a85"
FIG = ROOT / "assets"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e0", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#9a9a94", "figure.dpi": 150, "savefig.bbox": "tight"})

PARAM_COLS = {
    "dqn": ["lr", "gamma", "buffer_size", "batch_size", "eps_end", "eps_frac", "target", "double_dqn", "hidden"],
    "reinforce": ["lr", "gamma", "frames_per_batch", "baseline", "entropy_coef", "normalize_adv", "hidden"],
    "ppo": ["lr", "gamma", "gae_lambda", "clip_eps", "entropy_coef", "epochs", "frames_per_batch", "critic_coef"],
    "a2c": ["lr", "gamma", "gae_lambda", "entropy_coef", "critic_coef", "frames_per_batch", "normalize_adv"],
}


def moving(x, w=100):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        w = max(1, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def converge_episode(returns, w=100):
    """First episode where the 100-ep moving average reaches 90% of its total improvement."""
    ma = moving(returns, w)
    start, final = ma[:50].mean(), ma[-300:].mean()
    if final - start < 1.0:
        return None
    thr = start + 0.9 * (final - start)
    idx = np.argmax(ma >= thr)
    return int(idx + w) if ma[idx] >= thr else None


def load_runs():
    runs = {a: [] for a in ALGOS}
    for a in ALGOS:
        for f in sorted((LOGS / a).glob("*.json")):
            d = json.loads(f.read_text())
            if d["run"].endswith("_default") or d["run"].startswith("tmp") or "_seed" in d["run"]:
                continue
            runs[a].append(d)
    return runs


def sweep_tables(runs):
    tables = {}
    for a in ALGOS:
        rows = []
        for d in runs[a]:
            p, r = d["params"], d["results"]
            row = {"run": d["run"]}
            for c in PARAM_COLS[a]:
                v = p.get(c)
                if a == "dqn" and c == "target":
                    v = f"hard/{p['target_every']}" if v == "hard" else f"soft tau={p['tau']}"
                row[c] = v
            ce = converge_episode(d["episodes"])
            row.update(train_last500=np.mean(d["episodes"][-500:]),
                       eval_reward=r["eval_mean_reward"], eval_std=r["eval_std_reward"],
                       coverage=r["eval_coverage"], famine_rate=r["eval_famine_rate"],
                       hard_reward=r["hard_mean_reward"], hard_famine=r["hard_famine_rate"],
                       episodes=len(d["episodes"]), converge_ep=ce,
                       converge_steps=(float(np.interp(ce, [x["episodes"] for x in d["updates"]],
                                                       [x["frames"] for x in d["updates"]])) if ce else None),
                       minutes=d["wall_time"] / 60)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(LOGS / f"sweep_{a}.csv", index=False)
        tables[a] = df
    return tables


def pick_best(tables, runs):
    best = {}
    for a in ALGOS:
        df = tables[a]
        i = df["eval_reward"].idxmax()
        name = df.loc[i, "run"]
        sub = "dqn" if a == "dqn" else "pg"
        best[a] = dict(run=name, path=f"models/{sub}/{a}_{name}.pt",
                       eval_reward=float(df.loc[i, "eval_reward"]))
    # robustness: best config re-trained with seeds 1 and 2 (training/seed_check.py)
    for a in ALGOS:
        vals = [best[a]["eval_reward"]]
        hard = [float(tables[a].set_index("run").loc[best[a]["run"], "hard_reward"])]
        fam = [float(tables[a].set_index("run").loc[best[a]["run"], "famine_rate"])]
        for f in sorted((LOGS / a).glob(f"{best[a]['run']}_seed*.json")):
            r = json.loads(f.read_text())["results"]
            vals.append(r["eval_mean_reward"]); hard.append(r["hard_mean_reward"]); fam.append(r["eval_famine_rate"])
        best[a].update(seed_rewards=vals, seed_mean=float(np.mean(vals)), seed_std=float(np.std(vals)),
                       seed_hard_mean=float(np.mean(hard)), seed_famine_mean=float(np.mean(fam)))
    # overall = best mean over 3 training seeds (less sensitive to a lucky seed than a single run)
    best["overall"] = max(ALGOS, key=lambda a: best[a]["seed_mean"])
    (MODELS / "best_models.json").write_text(json.dumps(best, indent=2))
    return best


def run_of(runs, a, name):
    return next(d for d in runs[a] if d["run"] == name)


# ----------------------------------------------------------------------------- figures
def fig_cumulative(runs, best, baselines):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharey=True)
    for ax, a in zip(axes.flat, ALGOS):
        d = run_of(runs, a, best[a]["run"])
        ep = np.array(d["episodes"])
        ax.plot(ep, color=COLOR[a], alpha=0.15, lw=0.6)
        ma = moving(ep)
        ax.plot(np.arange(len(ma)) + 100, ma, color=COLOR[a], lw=2, label=f"{LABEL[a]} (100-ep mean)")
        ax.axhline(baselines["expert"]["mean_reward"], color=GRAY, ls="--", lw=1.2, label="expert rule (eval)")
        ax.axhline(baselines["random"]["mean_reward"], color=GRAY, ls=":", lw=1.2, label="random (eval)")
        ax.set_title(f"{LABEL[a]} - best run: {best[a]['run']}", fontsize=9)
        ax.set_xlabel("training episode")
        ax.set_ylabel("episode return")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Cumulative reward per episode, best model of each algorithm (same 166,400-step budget)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cumulative_rewards.png")
    plt.close(fig)


def fig_stability(runs, best):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    d = run_of(runs, "dqn", best["dqn"]["run"])
    u = pd.DataFrame(d["updates"])
    axes[0].plot(u["frames"], u["loss"], color=COLOR["dqn"], alpha=0.25, lw=0.6)
    axes[0].plot(u["frames"], u["loss"].rolling(50, min_periods=1).mean(), color=COLOR["dqn"], lw=2,
                 label="TD loss (Huber), 50-update mean")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("DQN objective (log scale)")
    axes[0].set_title("DQN objective (TD loss)", fontsize=9)
    axes[0].legend(fontsize=7)
    for a in ["reinforce", "ppo", "a2c"]:
        d = run_of(runs, a, best[a]["run"])
        u = pd.DataFrame(d["updates"])
        axes[1].plot(u["frames"], u["entropy"].rolling(10, min_periods=1).mean(), color=COLOR[a], lw=2,
                     label=LABEL[a])
    axes[1].axhline(np.log(14), color=GRAY, ls=":", lw=1, label="uniform policy ln(14)")
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("policy entropy (nats)")
    axes[1].set_title("Policy entropy, policy-gradient methods", fontsize=9)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_objective_entropy.png")
    plt.close(fig)


def fig_convergence(runs, best):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for a in ALGOS:
        d = run_of(runs, a, best[a]["run"])
        ma = moving(d["episodes"])
        x = np.arange(len(ma)) + 100
        ax.plot(x, ma, color=COLOR[a], lw=2, label=LABEL[a])
        ce = converge_episode(d["episodes"])
        if ce:
            ax.scatter([ce], [ma[ce - 100]], color=COLOR[a], s=60, zorder=5, edgecolor="white", linewidth=2)
            ax.annotate(f"{ce}", (ce, ma[ce - 100]), textcoords="offset points", xytext=(4, -12), fontsize=7)
    ax.set_xlabel("training episode")
    ax.set_ylabel("100-episode mean return")
    ax.set_title("Episodes to converge (dot = 90% of total improvement reached)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_convergence.png")
    plt.close(fig)


def fig_sweeps(tables):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, a in zip(axes, ALGOS):
        df = tables[a].sort_values("run")
        idx = [r.split("_", 2)[1] for r in df["run"]]
        ax.bar(idx, df["eval_reward"], color=COLOR[a], width=0.7)
        ax.set_title(LABEL[a], fontsize=9)
        ax.set_xlabel("run #")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("mean return, 100 unseen seeds")
    fig.suptitle("Hyperparameter sweep outcomes (12 runs per algorithm)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sweeps.png")
    plt.close(fig)


def fig_generalization(gen):
    names = list(gen.keys())
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, title in [(axes[0], "mean_reward", "Mean return"), (axes[1], "famine_rate", "Famine rate")]:
        a_vals = [gen[n]["unseen"][key] for n in names]
        b_vals = [gen[n]["hard"][key] for n in names]
        cols = [COLOR.get(n, GRAY) for n in names]
        ax.bar(x - 0.2, a_vals, 0.38, color=cols, label="unseen seeds (default season)")
        ax.bar(x + 0.2, b_vals, 0.38, color=cols, alpha=0.45, hatch="///", edgecolor="white",
               label="unseen harder season")
        ax.set_xticks(x, [LABEL.get(n, n) for n in names], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_generalization.png")
    plt.close(fig)


def behaviour_profile(act_fn, seeds=EVAL_SEEDS[:50]):
    """Share of each action type per week, plus mean onset of flooding."""
    env = AidPrepositioningEnv()
    counts = np.zeros((26, 4))
    for s in seeds:
        o, _ = env.reset(seed=s)
        done = False
        while not done:
            a = act_fn(o)
            w = env.t
            k = 0 if a == 0 else 1 if a <= N_COUNTIES else 2 if a <= 2 * N_COUNTIES else 3
            counts[w, k] += 1
            o, _, te, tr, _ = env.step(a)
            done = te or tr
    return counts


def fig_behaviour(best_act):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    counts = behaviour_profile(best_act)
    share = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    cols = ["#c8c8c2", "#2a78d6", "#eb6834", "#1baf7a"]
    labels = ["hold", "truck convoy", "airdrop", "procure (barge)"]
    bottom = np.zeros(26)
    for k in range(4):
        ax.bar(np.arange(1, 27), share[:, k], bottom=bottom, color=cols[k], label=labels[k],
               edgecolor="white", linewidth=1)
        bottom += share[:, k]
    ax.axvspan(7.5, 16.5, color="#2a78d6", alpha=0.06)
    ax.set_xlim(0.5, 26.5)
    ax.text(11.5, 1.03, "base flood-onset window (weeks 8-16, ±3 per episode)", ha="center", fontsize=7.5)
    ax.set_xlabel("week of lean season")
    ax.set_ylabel("share of decisions")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax.set_title("What the best agent does over the season (50 unseen episodes)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_behaviour.png")
    plt.close(fig)


def fig_env_frame():
    from environment.rendering import render_frame
    env = AidPrepositioningEnv()
    o, _ = env.reset(seed=10_003)
    for _ in range(12):
        o, *_ = env.step(prepositioning_policy(o))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    render_frame(env.state_dict_for_render(), ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_env_2d.png")
    plt.close(fig)


def main():
    runs = load_runs()
    tables = sweep_tables(runs)
    best = pick_best(tables, runs)
    print(json.dumps(best, indent=2))

    rng = np.random.default_rng(0)
    baselines = {
        "random": evaluate(lambda o: random_policy(o, rng)),
        "greedy": evaluate(greedy_policy),
        "expert": evaluate(prepositioning_policy),
    }
    hard_base = {
        "random": evaluate(lambda o: random_policy(o, rng), HARD_SEEDS, HARD_CONFIG),
        "greedy": evaluate(greedy_policy, HARD_SEEDS, HARD_CONFIG),
        "expert": evaluate(prepositioning_policy, HARD_SEEDS, HARD_CONFIG),
    }
    gen = {}
    for a in ALGOS:
        r = tables[a].set_index("run").loc[best[a]["run"]]
        d = run_of(runs, a, best[a]["run"])["results"]
        gen[a] = dict(unseen={k.replace("eval_", ""): v for k, v in d.items() if k.startswith("eval_")},
                      hard={k.replace("hard_", ""): v for k, v in d.items() if k.startswith("hard_")})
        gen[a]["converge_ep"] = r["converge_ep"]
    for n in ["expert", "greedy", "random"]:
        gen[n] = dict(unseen=baselines[n], hard=hard_base[n])
    (LOGS / "generalization.json").write_text(json.dumps(gen, indent=2, default=float))

    fig_cumulative(runs, best, baselines)
    fig_stability(runs, best)
    fig_convergence(runs, best)
    fig_sweeps(tables)
    fig_generalization(gen)
    act, _, _ = load_policy(ROOT / best[best["overall"]]["path"])
    fig_behaviour(act)
    fig_env_frame()
    for a in ALGOS:
        print(f"\n{LABEL[a]}\n", tables[a][["run", "eval_reward", "famine_rate", "hard_reward",
                                            "converge_ep", "converge_steps"]].round(3).to_string(index=False))
    print("\nbaselines unseen:", {k: round(v["mean_reward"], 2) for k, v in baselines.items()})


if __name__ == "__main__":
    main()
