from pdb import set_trace as TT
from typing import Dict

import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.tune import Callback


class StatsCallbacks(DefaultCallbacks):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_callback = {}

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        # path_lengths = []
        # regions = []
        # connectivities = []
        # for env in base_env.get_sub_environments():
        #     regions.append(env.unwrapped._rep_stats['regions'])
        #     connectivities.append(env.unwrapped._rep_stats['connectivity'])
        #     path_lengths.append(env.unwrapped._rep_stats['path-length'])
        
        # episode_stats = {
        #     'regions': np.mean(regions),
        #     'connectivity': np.mean(connectivities),
        #     'path-length': np.mean(path_lengths),
        # }
        env = base_env.get_sub_environments()[env_index]
        episode_stats = env.unwrapped._rep_stats
        

        # stats_list = ['regions', 'connectivity', 'path-length']
        # write to tensorboard file (if enabled)
        # episode.hist_data.update({k: [v] for k, v in episode_stats.items()})
        episode.custom_metrics.update({k: [v] for k, v in episode_stats.items()})

        # episode.hist_data.update({k: [v] for k, v in episode_stats.items() if k in stats_list})
        # episode.custom_metrics.update({k: [v] for k, v in episode_stats.items() if k in stats_list})
