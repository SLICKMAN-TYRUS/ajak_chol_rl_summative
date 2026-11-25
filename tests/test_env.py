import numpy as np
import pytest

from environment import AidPrepositioningEnv, N_ACTIONS
from environment.custom_env import N_COUNTIES, prepositioning_policy


@pytest.fixture
def env():
    return AidPrepositioningEnv()


def test_spaces(env):
    obs, info = env.reset(seed=1)
    assert obs.shape == (41,) and env.observation_space.contains(obs)
    assert env.action_space.n == N_ACTIONS == 14


def test_seed_determinism(env):
    a, _ = env.reset(seed=5)
    b, _ = env.reset(seed=5)
    assert np.allclose(a, b)


def test_procurement_arrives_after_lead_time(env):
    env.reset(seed=0)
    hub0 = env.hub
    env.step(2 * N_COUNTIES + 1)              # procure
    assert env.pipeline and env.budget < env.cfg.budget
    env.step(0); env.step(0)
    assert env.hub > hub0                     # arrived after 2 weeks


def test_convoy_fails_on_flooded_road(env):
    env.reset(seed=0)
    env.road_open[0] = 0.0
    stock0 = env.stock[0]
    _, r, *_ = env.step(1)
    assert env.ep_stats["failed"] == 1 and env.stock[0] <= stock0 and r < 0


def test_episode_ends_by_horizon_or_famine(env):
    env.reset(seed=3)
    done, steps = False, 0
    while not done:
        _, _, te, tr, info = env.step(0)
        done = te or tr
        steps += 1
    assert steps <= 26 and (info["famine"] or steps == 26)


def test_expert_beats_doing_nothing(env):
    def run(policy):
        tot = []
        for s in range(30):
            o, _ = env.reset(seed=100 + s)
            done, R = False, 0.0
            while not done:
                o, r, te, tr, _ = env.step(policy(o))
                R += r
                done = te or tr
            tot.append(R)
        return np.mean(tot)
    assert run(prepositioning_policy) > run(lambda o: 0) + 5


def test_rgb_render():
    e = AidPrepositioningEnv(render_mode="rgb_array")
    e.reset(seed=0)
    img = e.render()
    assert img.ndim == 3 and img.shape[2] == 3
