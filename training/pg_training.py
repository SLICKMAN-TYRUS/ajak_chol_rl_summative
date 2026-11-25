"""Policy-gradient training with TorchRL: REINFORCE, PPO and A2C (actor-critic).

Usage:
    uv run python -m training.pg_training --algo ppo                 # one default run
    uv run python -m training.pg_training --algo reinforce --sweep   # 10+ runs
    uv run python -m training.pg_training --algo all --sweep         # every PG sweep
"""

from __future__ import annotations

import argparse

import torch
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.modules import MLP
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.objectives.value import GAE

from training.common import (RunLogger, full_eval, greedy_action_fn, make_actor, make_collector,
                             make_critic, make_env, save_model, verbose)

TOTAL_FRAMES = 166_400  # identical budget for every policy-gradient run

DEFAULTS = {
    "reinforce": dict(lr=1e-3, gamma=0.99, frames_per_batch=520, baseline="value", entropy_coef=0.0,
                      normalize_adv=True, hidden=128, depth=2, critic_lr=1e-3, grad_clip=1.0),
    "ppo": dict(lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01, critic_coef=0.5,
                epochs=4, minibatch=256, frames_per_batch=1040, normalize_adv=True, hidden=128, depth=2,
                grad_clip=0.5),
    "a2c": dict(lr=1e-3, gamma=0.99, gae_lambda=1.0, entropy_coef=0.01, critic_coef=0.5,
                frames_per_batch=208, normalize_adv=False, hidden=128, depth=2, grad_clip=0.5),
}

SWEEPS = {
    "reinforce": [
        dict(name="rf_01_baseline"),
        dict(name="rf_02_lr_low", lr=3e-4),
        dict(name="rf_03_lr_high", lr=3e-3),
        dict(name="rf_04_no_baseline", baseline="none", normalize_adv=False),
        dict(name="rf_05_mean_baseline", baseline="mean", normalize_adv=False),
        dict(name="rf_06_gamma_090", gamma=0.90),
        dict(name="rf_07_small_batch", frames_per_batch=130),
        dict(name="rf_08_large_batch", frames_per_batch=1040),
        dict(name="rf_09_entropy_001", entropy_coef=0.01),
        dict(name="rf_10_entropy_005", entropy_coef=0.05),
        dict(name="rf_11_wide_net", hidden=256),
        dict(name="rf_12_no_norm", normalize_adv=False),
    ],
    "ppo": [
        dict(name="ppo_01_baseline"),
        dict(name="ppo_02_lr_low", lr=3e-4),
        dict(name="ppo_03_lr_high", lr=3e-3),
        dict(name="ppo_04_clip_01", clip_eps=0.1),
        dict(name="ppo_05_clip_03", clip_eps=0.3),
        dict(name="ppo_06_no_entropy", entropy_coef=0.0),
        dict(name="ppo_07_entropy_005", entropy_coef=0.05),
        dict(name="ppo_08_lambda_080", gae_lambda=0.80),
        dict(name="ppo_09_epochs_10", epochs=10),
        dict(name="ppo_10_small_rollout", frames_per_batch=520, minibatch=128),
        dict(name="ppo_11_gamma_090", gamma=0.90),
        dict(name="ppo_12_critic_coef_1", critic_coef=1.0),
    ],
    "a2c": [
        dict(name="a2c_01_baseline"),
        dict(name="a2c_02_lr_low", lr=3e-4),
        dict(name="a2c_03_lr_high", lr=3e-3),
        dict(name="a2c_04_td0", gae_lambda=0.0),
        dict(name="a2c_05_gae_095", gae_lambda=0.95),
        dict(name="a2c_06_no_entropy", entropy_coef=0.0),
        dict(name="a2c_07_entropy_005", entropy_coef=0.05),
        dict(name="a2c_08_critic_coef_1", critic_coef=1.0),
        dict(name="a2c_09_rollout_52", frames_per_batch=52),
        dict(name="a2c_10_rollout_520", frames_per_batch=520),
        dict(name="a2c_11_norm_adv", normalize_adv=True),
        dict(name="a2c_12_gamma_090", gamma=0.90),
    ],
}


def _actor_mlp(actor):
    for m in actor.modules():
        if isinstance(m, MLP):
            return m
    raise RuntimeError("actor MLP not found")


@torch.no_grad()
def _entropy(actor, td):
    return actor.get_dist(td).entropy().mean().item()


def _returns_to_go(data, gamma):
    """Monte-Carlo returns; also a mask removing the trailing unfinished episode."""
    r = data["next", "reward"].squeeze(-1)
    d = data["next", "done"].squeeze(-1).float()
    T = r.shape[0]
    G = torch.zeros_like(r)
    running = 0.0
    for t in reversed(range(T)):
        running = r[t] + gamma * running * (1.0 - d[t])
        G[t] = running
    mask = torch.ones(T, dtype=torch.bool)
    done_idx = torch.nonzero(d > 0).flatten()
    last = int(done_idx[-1]) if len(done_idx) else -1
    mask[last + 1:] = False
    return G, mask


def train(algo: str, params: dict, run_name: str, total_frames: int = TOTAL_FRAMES, save: bool = True):
    p = {**DEFAULTS[algo], **params, "total_frames": total_frames, "seed": params.get("seed", 0)}
    torch.manual_seed(p["seed"])
    env = make_env(seed=p["seed"])
    actor = make_actor(env, p["hidden"], p["depth"])
    critic = make_critic(p["hidden"], p["depth"])

    collector = make_collector(env, actor, p["frames_per_batch"], total_frames,
                               reset_at_each_iter=(algo == "reinforce"))
    log = RunLogger(algo, run_name, p)

    if algo == "reinforce":
        loss_fn = ReinforceLoss(actor, critic, loss_critic_type="l2")
        optim = torch.optim.Adam([{"params": actor.parameters(), "lr": p["lr"]},
                                  {"params": critic.parameters(), "lr": p["critic_lr"]}])
    else:
        gae = GAE(gamma=p["gamma"], lmbda=p["gae_lambda"], value_network=critic, average_gae=False)
        if algo == "ppo":
            loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=p["clip_eps"], entropy_bonus=p["entropy_coef"] > 0,
                                  entropy_coeff=p["entropy_coef"], critic_coeff=p["critic_coef"],
                                  normalize_advantage=p["normalize_adv"], loss_critic_type="l2")
        else:
            loss_fn = A2CLoss(actor, critic, entropy_bonus=p["entropy_coef"] > 0,
                              entropy_coeff=p["entropy_coef"], critic_coeff=p["critic_coef"],
                              loss_critic_type="l2")
        optim = torch.optim.Adam(loss_fn.parameters(), lr=p["lr"])

    frames = 0
    for i, data in enumerate(collector):
        frames += data.numel()
        log.add_episodes(data)
        metrics = {}

        if algo == "reinforce":
            G, mask = _returns_to_go(data, p["gamma"])
            if mask.sum() == 0:
                continue
            td = data[mask].clone()
            g = G[mask]
            with torch.no_grad():
                v = critic(td.clone())["state_value"].squeeze(-1)
            if p["baseline"] == "value":
                adv = g - v
            elif p["baseline"] == "mean":
                adv = g - g.mean()
            else:
                adv = g
            if p["normalize_adv"]:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            td.set("advantage", adv.unsqueeze(-1))
            td.set("value_target", g.unsqueeze(-1))
            out = loss_fn(td)
            ent = actor.get_dist(td).entropy().mean()
            loss = out["loss_actor"] - p["entropy_coef"] * ent
            if p["baseline"] == "value":
                loss = loss + out["loss_value"]
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), p["grad_clip"])
            optim.step()
            metrics = dict(loss_objective=out["loss_actor"].item(), loss_critic=out["loss_value"].item(),
                           entropy=ent.item())
        else:
            with torch.no_grad():
                gae(data)
            flat = data.reshape(-1)
            if algo == "a2c" and p["normalize_adv"]:
                a = flat["advantage"]
                flat.set("advantage", (a - a.mean()) / (a.std() + 1e-8))
            if algo == "ppo":
                rb = TensorDictReplayBuffer(storage=LazyTensorStorage(flat.numel()),
                                            sampler=SamplerWithoutReplacement(), batch_size=p["minibatch"])
                rb.extend(flat)
                acc = {"loss_objective": 0.0, "loss_critic": 0.0, "n": 0}
                for _ in range(p["epochs"]):
                    for _ in range(max(1, flat.numel() // p["minibatch"])):
                        mb = rb.sample()
                        out = loss_fn(mb)
                        loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
                        optim.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), p["grad_clip"])
                        optim.step()
                        acc["loss_objective"] += out["loss_objective"].item()
                        acc["loss_critic"] += out["loss_critic"].item()
                        acc["n"] += 1
                metrics = dict(loss_objective=acc["loss_objective"] / acc["n"],
                               loss_critic=acc["loss_critic"] / acc["n"], entropy=_entropy(actor, flat.clone()))
            else:
                out = loss_fn(flat)
                loss = out["loss_objective"] + out["loss_critic"] + out.get("loss_entropy", 0.0)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), p["grad_clip"])
                optim.step()
                metrics = dict(loss_objective=out["loss_objective"].item(),
                               loss_critic=out["loss_critic"].item(), entropy=_entropy(actor, flat.clone()))

        log.add_update(frames, **metrics)
        every = max(1, 8000 // p["frames_per_batch"])
        if verbose() and i % every == 0:
            print(f"[{algo.upper()} {run_name}] frames {frames:6d} | episodes {len(log.episodes):5d} | "
                  f"avg return(50) {log.recent():7.2f} | entropy {metrics['entropy']:.3f}")
    collector.shutdown()

    net = _actor_mlp(actor)
    results = full_eval(greedy_action_fn(net, "pi"))
    if verbose():
        print(f"[{algo.upper()} {run_name}] EVAL unseen: {results['eval_mean_reward']:.2f} | "
              f"coverage {results['eval_coverage']:.3f} | famine {results['eval_famine_rate']:.2f}")
    log.save(results)
    if save:
        save_model(net, algo, run_name, p, kind="pi")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ppo", choices=["reinforce", "ppo", "a2c", "all"])
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    args = ap.parse_args()
    algos = ["reinforce", "ppo", "a2c"] if args.algo == "all" else [args.algo]
    for algo in algos:
        runs = SWEEPS[algo] if (args.sweep or args.only) else [dict(name=f"{algo}_default")]
        for cfg in runs:
            if args.only and cfg["name"] not in args.only:
                continue
            cfg = dict(cfg)
            name = cfg.pop("name")
            train(algo, cfg, name, total_frames=args.frames)


if __name__ == "__main__":
    main()
