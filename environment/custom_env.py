"""AidPrepositioningEnv: humanitarian food-aid pre-positioning in Jonglei, South Sudan.

One logistics coordinator (the agent) at the Bor hub decides, each week of the
26-week lean/flood season, how to use a limited budget: dispatch a truck convoy,
order an airdrop, procure more food (2-week barge lead time) or hold.

Why it is sequential: county stocks, hunger severity, road access, budget and the
procurement pipeline all persist between weeks. Roads close as the floods arrive,
so food that is not pre-positioned early can later only be delivered by expensive
airdrops, and money spent early cannot be spent later.

Units: food is measured in "convoy loads" (1 unit ~ 100 t of mixed food basket).
Populations, distances and flood timings are illustrative, calibrated to the
order of magnitude of Jonglei data (IPC, WFP Logistics Cluster), not real values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ----------------------------------------------------------------------------- #
# Static scenario data
# ----------------------------------------------------------------------------- #
HUB = {"name": "Bor (hub)", "lon": 31.56, "lat": 6.21}

# name, lon, lat, population share, base weekly need (units), base flood-onset week
COUNTIES = [
    ("Twic East", 31.33, 6.88, 0.14, 0.11, 9),
    ("Duk", 31.42, 7.45, 0.13, 0.10, 8),
    ("Ayod", 31.42, 8.12, 0.18, 0.14, 7),
    ("Uror", 32.05, 7.75, 0.16, 0.12, 11),
    ("Akobo", 33.00, 7.78, 0.17, 0.13, 13),
    ("Pibor", 33.13, 6.80, 0.22, 0.16, 15),
]
N_COUNTIES = len(COUNTIES)

ACTION_NAMES = (
    ["Hold (no dispatch this week)"]
    + [f"Truck convoy -> {c[0]}" for c in COUNTIES]
    + [f"Airdrop -> {c[0]}" for c in COUNTIES]
    + ["Procure 6 loads (barge Juba->Bor, 2-week lead)"]
)
N_ACTIONS = len(ACTION_NAMES)  # 14

IPC_NAMES = {1: "Minimal", 2: "Stressed", 3: "Crisis", 4: "Emergency", 5: "Famine"}


@dataclass
class EnvConfig:
    horizon: int = 26                # weeks in the lean season
    budget: float = 40.0             # cost units (1 unit ~ USD 100k)
    truck_load: float = 1.5          # units delivered per convoy
    truck_cost: float = 0.5
    air_load: float = 0.4
    air_cost: float = 2.5
    procure_amount: float = 6.0
    procure_cost: float = 5.5
    procure_lead: int = 2
    county_capacity: float = 2.5     # storage per county
    hub_capacity: float = 10.0
    spoilage: float = 0.01           # weekly loss in county stores
    flood_shift: int = 0             # negative = earlier floods (harder)
    need_scale: float = 1.0
    init_severity_max: float = 0.40
    init_hub_range: tuple = (3.0, 6.0)
    # reward weights
    w_unmet: float = 1.0
    w_severity: float = 0.5
    w_cost: float = 0.05
    p_failed: float = 0.5
    p_invalid: float = 0.3
    p_famine: float = 10.0
    w_final: float = 3.0
    extra: dict = field(default_factory=dict)


HARD_CONFIG = dict(flood_shift=-3, need_scale=1.15, init_severity_max=0.55,
                   budget=35.0, init_hub_range=(2.0, 4.0))


def ipc_phase(severity: float) -> int:
    """Map continuous hunger severity (0..1) to an IPC-like phase 1..5."""
    if severity < 0.2:
        return 1
    if severity < 0.4:
        return 2
    if severity < 0.6:
        return 3
    if severity < 0.8:
        return 4
    return 5


class AidPrepositioningEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 2}

    def __init__(self, render_mode: str | None = None, config: dict | None = None,
                 viewer_port: int = 8765, step_delay: float = 0.8):
        super().__init__()
        self.cfg = EnvConfig(**(config or {}))
        self.render_mode = render_mode
        self.viewer_port = viewer_port
        self.step_delay = step_delay
        self._viewer = None

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.obs_dim = 5 + 6 * N_COUNTIES
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.obs_dim,), dtype=np.float32)

        self.pop = np.array([c[3] for c in COUNTIES], dtype=np.float64)
        self.base_need = np.array([c[4] for c in COUNTIES], dtype=np.float64)
        self.base_onset = np.array([c[5] for c in COUNTIES], dtype=np.float64)

    # ------------------------------------------------------------------ dynamics
    def _weekly_need(self, t: int) -> np.ndarray:
        season = 0.6 + 0.8 * math.sin(math.pi * min(t, self.cfg.horizon) / self.cfg.horizon)
        return self.base_need * season * self.cfg.need_scale

    def _flood_risk(self) -> np.ndarray:
        """Forecast probability that a county road is (or soon will be) cut by floods."""
        z = (self.t - self.onset + 2.0) / 2.0
        risk = 1.0 / (1.0 + np.exp(-z))
        noise = self.np_random.normal(0.0, 0.05, size=N_COUNTIES)
        return np.clip(risk + noise, 0.0, 1.0)

    def _update_roads(self):
        for i in range(N_COUNTIES):
            if self.road_open[i]:
                p_close = 0.02 if self.t < self.onset[i] else 0.45
                if self.np_random.random() < p_close:
                    self.road_open[i] = 0.0
            else:
                p_reopen = 0.15 if self.t < self.onset[i] else 0.04
                if self.np_random.random() < p_reopen:
                    self.road_open[i] = 1.0

    def _obs(self) -> np.ndarray:
        need = self._weekly_need(self.t)
        cover = np.clip(self.stock / np.maximum(need, 1e-6) / 6.0, 0, 1)
        pipe1 = sum(a for (w, a) in self.pipeline if w - self.t <= 1)
        pipe2 = sum(a for (w, a) in self.pipeline)
        g = [self.t / self.cfg.horizon,
             self.hub / self.cfg.hub_capacity,
             self.budget / self.cfg.budget,
             min(pipe1 / 8.0, 1.0),
             min(pipe2 / 8.0, 1.0)]
        per = np.stack([self.stock / self.cfg.county_capacity,
                        cover,
                        self.severity,
                        self.road_open,
                        self.flood_risk,
                        np.clip(need / 0.4, 0, 1)], axis=1).reshape(-1)
        return np.clip(np.concatenate([g, per]).astype(np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        c = self.cfg
        r = self.np_random
        self.t = 0
        self.budget = c.budget
        self.hub = float(r.uniform(*c.init_hub_range))
        self.stock = r.uniform(0.0, 0.8, size=N_COUNTIES)
        self.severity = r.uniform(0.0, c.init_severity_max, size=N_COUNTIES)
        self.onset = self.base_onset + c.flood_shift + r.integers(-3, 4, size=N_COUNTIES)
        self.road_open = np.ones(N_COUNTIES)
        self.pipeline: list[tuple[int, float]] = []
        self.flood_risk = self._flood_risk()
        self.history = []
        self.ep_stats = dict(unmet=0.0, need=0.0, cost=0.0, trucks=0, airdrops=0,
                             procurements=0, failed=0, invalid=0, holds=0, famine=False)
        self.last = dict(action=None, action_name="(start)", reward=0.0, outcome="Episode start")
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    def step(self, action: int):
        c = self.cfg
        action = int(action)
        cost = 0.0
        failed = invalid = False
        outcome = ""

        # 1) arrivals from the procurement pipeline
        arrived = sum(a for (w, a) in self.pipeline if w <= self.t)
        self.pipeline = [(w, a) for (w, a) in self.pipeline if w > self.t]
        self.hub = min(self.hub + arrived, c.hub_capacity)

        # 2) apply the action
        if action == 0:
            self.ep_stats["holds"] += 1
            outcome = "Held resources this week"
        elif 1 <= action <= N_COUNTIES:
            i = action - 1
            name = COUNTIES[i][0]
            load = min(c.truck_load, self.hub, c.county_capacity - self.stock[i])
            if self.budget < c.truck_cost or self.hub <= 0.05 or load <= 0.05:
                invalid = True
                outcome = f"Convoy to {name} not possible (no stock/budget/storage)"
            else:
                cost = c.truck_cost
                self.ep_stats["trucks"] += 1
                if self.road_open[i]:
                    self.hub -= load
                    self.stock[i] += load
                    outcome = f"Convoy delivered {load:.2f} loads to {name}"
                else:
                    failed = True
                    outcome = f"Convoy to {name} turned back: road flooded"
        elif N_COUNTIES < action <= 2 * N_COUNTIES:
            i = action - N_COUNTIES - 1
            name = COUNTIES[i][0]
            load = min(c.air_load, self.hub, c.county_capacity - self.stock[i])
            if self.budget < c.air_cost or self.hub <= 0.05 or load <= 0.05:
                invalid = True
                outcome = f"Airdrop to {name} not possible (no stock/budget/storage)"
            else:
                cost = c.air_cost
                self.hub -= load
                self.stock[i] += load
                self.ep_stats["airdrops"] += 1
                outcome = f"Airdrop delivered {load:.2f} loads to {name}"
        else:  # procure
            if self.budget < c.procure_cost:
                invalid = True
                outcome = "Procurement not possible (budget exhausted)"
            else:
                cost = c.procure_cost
                self.pipeline.append((self.t + c.procure_lead, c.procure_amount))
                self.ep_stats["procurements"] += 1
                outcome = f"Procured {c.procure_amount:.0f} loads, arriving week {self.t + c.procure_lead + 1}"
        self.budget -= cost

        # 3) consumption, hunger dynamics
        need = self._weekly_need(self.t)
        consumed = np.minimum(self.stock, need)
        unmet = need - consumed
        self.stock = (self.stock - consumed) * (1.0 - c.spoilage)
        unmet_frac = unmet / np.maximum(need, 1e-6)
        # severity rises quickly with unmet need, recovers slowly when fully fed
        self.severity = np.clip(self.severity + 0.09 * unmet_frac - 0.05 * (unmet_frac < 0.05), 0.0, 1.0)
        phases = np.array([ipc_phase(s) for s in self.severity])

        # 4) reward
        r_unmet = -c.w_unmet * float(np.sum(self.pop * unmet_frac))
        r_sev = -c.w_severity * float(np.sum(self.pop * self.severity))
        r_cost = -c.w_cost * cost
        r_pen = -(c.p_failed if failed else 0.0) - (c.p_invalid if invalid else 0.0)
        reward = r_unmet + r_sev + r_cost + r_pen

        famine = bool(np.any(phases >= 5))
        self.t += 1
        terminated = famine
        truncated = self.t >= c.horizon
        if famine:
            # penalty scales with the weeks left so that ending early is never attractive
            reward -= c.p_famine + 1.0 * (c.horizon - self.t)
            outcome += " | FAMINE declared - episode ends"
        if terminated or truncated:
            reward += c.w_final * float(np.sum(self.pop * (phases <= 2)))

        # 5) environment evolution for next week
        self._update_roads()
        self.flood_risk = self._flood_risk()

        st = self.ep_stats
        st["unmet"] += float(unmet.sum())
        st["need"] += float(need.sum())
        st["cost"] += cost
        st["failed"] += int(failed)
        st["invalid"] += int(invalid)
        st["famine"] = st["famine"] or famine
        self.last = dict(action=action, action_name=ACTION_NAMES[action], reward=float(reward),
                         outcome=outcome, components=dict(unmet=r_unmet, severity=r_sev,
                                                          cost=r_cost, penalty=r_pen))
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, self._info()

    def _info(self) -> dict:
        st = self.ep_stats
        return dict(week=self.t, budget=self.budget, hub=self.hub,
                    coverage=1.0 - st["unmet"] / max(st["need"], 1e-6),
                    famine=st["famine"], **{k: v for k, v in st.items() if k != "famine"})

    # ------------------------------------------------------------------ rendering
    def state_dict_for_render(self) -> dict:
        need = self._weekly_need(self.t)
        return dict(
            week=int(self.t), horizon=self.cfg.horizon, budget=round(float(self.budget), 3),
            budget0=self.cfg.budget, hub=round(float(self.hub), 3), hub_name=HUB["name"],
            hub_lon=HUB["lon"], hub_lat=HUB["lat"],
            pipeline=[{"week": int(w) + 1, "amount": float(a)} for (w, a) in self.pipeline],
            counties=[dict(name=c[0], lon=c[1], lat=c[2], pop=c[3],
                           stock=round(float(self.stock[i]), 3),
                           need=round(float(need[i]), 3),
                           severity=round(float(self.severity[i]), 3),
                           ipc=ipc_phase(self.severity[i]),
                           road_open=bool(self.road_open[i]),
                           flood_risk=round(float(self.flood_risk[i]), 3))
                      for i, c in enumerate(COUNTIES)],
            last=self.last, stats=self._info(),
        )

    def render(self):
        if self.render_mode == "human":
            from environment.rendering import WebViewer
            if self._viewer is None:
                self._viewer = WebViewer(port=self.viewer_port)
            self._viewer.push(self.state_dict_for_render(), delay=self.step_delay)
            return None
        if self.render_mode == "rgb_array":
            from environment.rendering import render_frame
            return render_frame(self.state_dict_for_render())
        if self.render_mode == "ansi":
            s = self.state_dict_for_render()
            lines = [f"Week {s['week']}/{s['horizon']} | budget {s['budget']:.1f} | hub {s['hub']:.2f}"]
            for cty in s["counties"]:
                lines.append(f"  {cty['name']:<10} stock {cty['stock']:.2f} IPC{cty['ipc']} "
                             f"road {'open' if cty['road_open'] else 'CUT '} risk {cty['flood_risk']:.2f}")
            return "\n".join(lines)
        return None

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ----------------------------------------------------------------------------- #
# Reference policies (for sanity checks and report baselines)
# ----------------------------------------------------------------------------- #
def random_policy(obs, rng):
    return int(rng.integers(N_ACTIONS))


def greedy_policy(obs, rng=None):
    """Hand-written rule: keep hub stocked, send convoys to lowest cover open road,
    airdrop only to severe, cut-off counties."""
    hub, budget, pipe = obs[1] * 10.0, obs[2], obs[4]
    per = obs[5:].reshape(N_COUNTIES, 6)
    stock, cover, sev, road = per[:, 0], per[:, 1], per[:, 2], per[:, 3]
    if hub < 1.5 and pipe == 0 and budget > 0.15:
        return 2 * N_COUNTIES + 1
    open_idx = [i for i in range(N_COUNTIES) if road[i] > 0.5 and stock[i] < 0.6]
    if open_idx and hub > 0.2:
        i = min(open_idx, key=lambda k: cover[k])
        return 1 + i
    cut = [i for i in range(N_COUNTIES) if road[i] < 0.5 and sev[i] > 0.5 and stock[i] < 0.1]
    if cut and hub > 0.2 and budget > 0.2:
        i = max(cut, key=lambda k: sev[k])
        return 1 + N_COUNTIES + i
    return 0


def prepositioning_policy(obs, rng=None):
    """Expert-style rule: pre-position stock where flood risk is rising, then airdrop."""
    hub, budget, pipe = obs[1] * 10.0, obs[2], obs[4]
    per = obs[5:].reshape(N_COUNTIES, 6)
    stock, sev, road, risk = per[:, 0] * 2.5, per[:, 2], per[:, 3], per[:, 4]
    if hub < 2.0 and pipe == 0 and budget > 0.14:
        return 2 * N_COUNTIES + 1
    cand = [i for i in range(N_COUNTIES) if road[i] > 0.5 and stock[i] < 1.6]
    if cand and hub > 0.3:
        return 1 + max(cand, key=lambda k: risk[k] + sev[k] - 0.5 * stock[k])
    cut = [i for i in range(N_COUNTIES) if road[i] < 0.5 and stock[i] < 0.15 and sev[i] > 0.35]
    if cut and hub > 0.2 and budget > 0.07:
        return 1 + N_COUNTIES + max(cut, key=lambda k: sev[k])
    return 0
