"""DQN (value-based) training with TorchRL.

Usage:
    uv run python -m training.dqn_training                 # default config
    uv run python -m training.dqn_training --sweep         # 10+ hyperparameter runs
"""

from __future__ import annotations

import argparse

import torch
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from tensordict.nn import TensorDictSequential

from training.common import (RunLogger, make_collector, full_eval, greedy_action_fn, make_env, make_q_actor,
                             save_model, verbose)

DEFAULT = dict(lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=128, hidden=128, depth=2,
               eps_start=1.0, eps_end=0.05, eps_frac=0.5, target="hard", target_every=500,
               tau=0.01, double_dqn=False, frames_per_batch=104, updates_per_batch=8,
               init_random_frames=2_080, total_frames=166_400, seed=0)

# 12 meaningfully different configurations (shared + DQN-specific controls)
SWEEP = [
    dict(name="dqn_01_baseline"),
    dict(name="dqn_02_lr_low", lr=1e-4),
    dict(name="dqn_03_lr_high", lr=5e-3),
    dict(name="dqn_04_gamma_090", gamma=0.90),
    dict(name="dqn_05_gamma_0995", gamma=0.995),
    dict(name="dqn_06_small_buffer", buffer_size=2_000),
    dict(name="dqn_07_batch32", batch_size=32),
    dict(name="dqn_08_fast_eps_decay", eps_frac=0.1),
    dict(name="dqn_09_high_eps_floor", eps_end=0.20),
    dict(name="dqn_10_soft_target", target="soft", tau=0.005),
    dict(name="dqn_11_double_dqn", double_dqn=True),
    dict(name="dqn_12_double_soft_wide", double_dqn=True, target="soft", tau=0.01, hidden=256, lr=5e-4),
]


def train(params: dict, run_name: str = "dqn_default", save: bool = True):
    p = {**DEFAULT, **params}
    torch.manual_seed(p["seed"])
    env = make_env(seed=p["seed"])
    qnet = make_q_actor(env, p["hidden"], p["depth"])
    greedy = EGreedyModule(env.action_spec, eps_init=p["eps_start"], eps_end=p["eps_end"],
                           annealing_num_steps=int(p["eps_frac"] * p["total_frames"]))
    explore = TensorDictSequential(qnet, greedy)

    collector = make_collector(env, explore, frames_per_batch=p["frames_per_batch"],
                                  total_frames=p["total_frames"], init_random_frames=p["init_random_frames"])
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(p["buffer_size"]), batch_size=p["batch_size"])
    loss_fn = DQNLoss(qnet, action_space=env.action_spec, delay_value=True, double_dqn=p["double_dqn"],
                      loss_function="smooth_l1")
    loss_fn.make_value_estimator(gamma=p["gamma"])
    updater = (HardUpdate(loss_fn, value_network_update_interval=p["target_every"])
               if p["target"] == "hard" else SoftUpdate(loss_fn, tau=p["tau"]))
    optim = torch.optim.Adam(loss_fn.parameters(), lr=p["lr"])

    log = RunLogger("dqn", run_name, p)
    frames = 0
    for i, data in enumerate(collector):
        frames += data.numel()
        log.add_episodes(data)
        rb.extend(data.reshape(-1))
        greedy.step(data.numel())
        if frames < p["init_random_frames"]:
            continue
        losses, qs = [], []
        for _ in range(p["updates_per_batch"]):
            batch = rb.sample()
            out = loss_fn(batch)
            optim.zero_grad()
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
            optim.step()
            updater.step()
            losses.append(out["loss"].item())
        with torch.no_grad():
            qs = qnet(data.clone())["action_value"].max(-1).values.mean().item()
        log.add_update(frames, loss=sum(losses) / len(losses), q_mean=qs, epsilon=greedy.eps.item())
        if verbose() and i % 50 == 0:
            print(f"[DQN {run_name}] frames {frames:6d} | episodes {len(log.episodes):5d} | "
                  f"avg return(50) {log.recent():7.2f} | loss {log.updates[-1]['loss']:.4f} | "
                  f"eps {greedy.eps.item():.3f}")
    collector.shutdown()

    net = _extract_mlp(qnet)
    results = full_eval(greedy_action_fn(net, "q"))
    if verbose():
        print(f"[DQN {run_name}] EVAL unseen: {results['eval_mean_reward']:.2f} | "
              f"coverage {results['eval_coverage']:.3f} | famine {results['eval_famine_rate']:.2f}")
    log.save(results)
    if save:
        save_model(net, "dqn", run_name, p, kind="q")
    return results


def _extract_mlp(qnet):
    from torchrl.modules import MLP
    for m in qnet.modules():
        if isinstance(m, MLP):
            return m
    raise RuntimeError("MLP not found in Q actor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these sweep names")
    ap.add_argument("--frames", type=int, default=None)
    args = ap.parse_args()
    runs = SWEEP if args.sweep or args.only else [dict(name="dqn_default")]
    for cfg in runs:
        if args.only and cfg["name"] not in args.only:
            continue
        cfg = dict(cfg)
        name = cfg.pop("name")
        if args.frames:
            cfg["total_frames"] = args.frames
        train(cfg, name)


if __name__ == "__main__":
    main()
