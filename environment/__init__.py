"""Custom environment package: food-aid pre-positioning in Jonglei, South Sudan."""
from gymnasium.envs.registration import register

from environment.custom_env import (ACTION_NAMES, HARD_CONFIG, N_ACTIONS,
                                    AidPrepositioningEnv, greedy_policy, random_policy)

register(id="AidPrepositioning-v0", entry_point="environment.custom_env:AidPrepositioningEnv",
         max_episode_steps=26)

__all__ = ["AidPrepositioningEnv", "ACTION_NAMES", "N_ACTIONS", "HARD_CONFIG",
           "greedy_policy", "random_policy"]
