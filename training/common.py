"""Shared TorchRL utilities: env factory, networks, evaluation, logging."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import GymWrapper, RewardScaling, RewardSum, StepCounter, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, QValueActor, ValueOperator

from environment.custom_env import HARD_CONFIG, AidPrepositioningEnv

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
EVAL_SEEDS = list(range(10_000, 10_100))      # never used during training
HARD_SEEDS = list(range(20_000, 20_100))

torch.set_num_threads(1)
os.environ.setdefault("RL_WARNINGS", "False")
import logging
import warnings
logging.getLogger("torchrl").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")


REWARD_SCALE = 0.1  # identical for all four algorithms; logged returns stay unscaled


def make_env(seed: int = 0, config: dict | None = None, render_mode: str | None = None, **kw):
    """Wrap the Gymnasium env for TorchRL (integer/categorical actions).

    RewardSum is applied before RewardScaling, so the logged episode_reward is the
    raw environment return while the learners see reward * REWARD_SCALE.
    """
    base = AidPrepositioningEnv(render_mode=render_mode, config=config, **kw)
    env = GymWrapper(base, categorical_action_encoding=True)
    env = TransformedEnv(env, RewardSum())
    env.append_transform(RewardScaling(loc=0.0, scale=REWARD_SCALE))
    env.append_transform(StepCounter())
    env.set_seed(seed)
    return env


def obs_and_actions():
    e = AidPrepositioningEnv()
    return e.observation_space.shape[0], int(e.action_space.n)


def mlp(n_in, n_out, hidden=128, depth=2, act=nn.Tanh):
    return MLP(in_features=n_in, out_features=n_out, num_cells=[hidden] * depth, activation_class=act)


def make_q_actor(env, hidden=128, depth=2):
    n_obs, n_act = obs_and_actions()
    net = mlp(n_obs, n_act, hidden, depth, act=nn.ReLU)
    return QValueActor(net, in_keys=["observation"], spec=env.action_spec)


def make_actor(env, hidden=128, depth=2):
    n_obs, n_act = obs_and_actions()
    net = TensorDictModule(mlp(n_obs, n_act, hidden, depth), in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(net, in_keys=["logits"], out_keys=["action"], spec=env.action_spec,
                              distribution_class=torch.distributions.Categorical,
                              return_log_prob=True)


def make_critic(hidden=128, depth=2):
    n_obs, _ = obs_and_actions()
    return ValueOperator(mlp(n_obs, 1, hidden, depth), in_keys=["observation"])


# ----------------------------------------------------------------------------- #
# Evaluation on unseen seeds (deterministic / greedy actions)
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def greedy_action_fn(policy_net: nn.Module, kind: str):
    """Return obs(np) -> int action using argmax of Q-values or logits."""
    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = policy_net(x)
        return int(out.argmax(-1).item())
    return act


def evaluate(act_fn, seeds=EVAL_SEEDS, config=None) -> dict:
    env = AidPrepositioningEnv(config=config)
    rets, cov, fam, cost, air, failed = [], [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        done, tot = False, 0.0
        while not done:
            obs, r, te, tr, info = env.step(act_fn(obs))
            tot += r
            done = te or tr
        rets.append(tot); cov.append(info["coverage"]); fam.append(float(info["famine"]))
        cost.append(info["cost"]); air.append(info["airdrops"]); failed.append(info["failed"])
    return dict(mean_reward=float(np.mean(rets)), std_reward=float(np.std(rets)),
                coverage=float(np.mean(cov)), famine_rate=float(np.mean(fam)),
                mean_cost=float(np.mean(cost)), airdrops=float(np.mean(air)),
                failed_convoys=float(np.mean(failed)))


def full_eval(act_fn) -> dict:
    a = evaluate(act_fn, EVAL_SEEDS)
    b = evaluate(act_fn, HARD_SEEDS, HARD_CONFIG)
    return {**{f"eval_{k}": v for k, v in a.items()}, **{f"hard_{k}": v for k, v in b.items()}}


class RunLogger:
    """Collects per-episode returns and per-update training metrics, writes JSON."""

    def __init__(self, algo: str, run_name: str, params: dict):
        self.algo, self.run_name, self.params = algo, run_name, params
        self.episodes: list[float] = []
        self.updates: list[dict] = []
        self.t0 = time.time()

    def add_episodes(self, td):
        done = td["next", "done"].squeeze(-1)
        if done.any():
            self.episodes.extend(td["next", "episode_reward"].squeeze(-1)[done].tolist())

    def add_update(self, frames: int, **metrics):
        self.updates.append(dict(frames=frames, episodes=len(self.episodes),
                                 **{k: float(v) for k, v in metrics.items()}))

    def recent(self, n=50):
        return float(np.mean(self.episodes[-n:])) if self.episodes else float("nan")

    def save(self, results: dict):
        out = LOGS / self.algo
        out.mkdir(parents=True, exist_ok=True)
        payload = dict(algo=self.algo, run=self.run_name, params=self.params,
                       episodes=self.episodes, updates=self.updates, results=results,
                       wall_time=time.time() - self.t0)
        with open(out / f"{self.run_name}.json", "w") as f:
            json.dump(payload, f)
        return payload


def save_model(module: nn.Module, algo: str, run_name: str, params: dict, kind: str):
    sub = "dqn" if algo == "dqn" else "pg"
    path = MODELS / sub / f"{algo}_{run_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=module.state_dict(), params=params, algo=algo, kind=kind), path)
    return path


def verbose() -> bool:
    return os.environ.get("RL_QUIET", "0") != "1"


try:  # TorchRL >= 0.15
    from torchrl.collectors import Collector as _Collector
except ImportError:  # older TorchRL
    from torchrl.collectors import SyncDataCollector as _Collector


def make_collector(env, policy, frames_per_batch, total_frames, **kw):
    try:
        return _Collector(env, policy, frames_per_batch=frames_per_batch, total_frames=total_frames,
                          auto_register_policy_transforms=False, **kw)
    except TypeError:
        return _Collector(env, policy, frames_per_batch=frames_per_batch, total_frames=total_frames, **kw)


# ----------------------------------------------------------------------------- #
# Serialization: load a saved model back as a plain obs -> action function
# ----------------------------------------------------------------------------- #
def load_policy(path):
    """Rebuild the saved MLP and return (act_fn, info_fn, checkpoint)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    p = ckpt["params"]
    n_obs, n_act = obs_and_actions()
    act = nn.ReLU if ckpt["kind"] == "q" else nn.Tanh
    net = mlp(n_obs, n_act, p["hidden"], p["depth"], act=act)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    act_fn = greedy_action_fn(net, ckpt["kind"])

    @torch.no_grad()
    def info_fn(obs):
        from environment.custom_env import ACTION_NAMES
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = net(x).squeeze(0)
        scores = torch.softmax(out, -1) if ckpt["kind"] == "pi" else out
        a = int(out.argmax().item())
        top = torch.topk(scores, 3)
        return dict(action=a, action_name=ACTION_NAMES[a], algo=ckpt["algo"],
                    score_type="probability" if ckpt["kind"] == "pi" else "q_value",
                    top3=[dict(action=int(i), name=ACTION_NAMES[int(i)], score=round(float(v), 4))
                          for v, i in zip(top.values, top.indices)])
    return act_fn, info_fn, ckpt
