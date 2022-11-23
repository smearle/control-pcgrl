from pdb import set_trace as TT
from typing import Dict

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.tune import Callback


class StatsCallbacks(DefaultCallbacks):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_callback = {}
        self.holey = 'holey' in cfg.problem.name

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        env = base_env.get_sub_environments()[env_index]
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        for k in env.ctrl_metrics:
            episode.hist_data.update({
                f'{k}-trg': None,
            })
        for k in env.metrics:
            episode.hist_data.update({f'{k}-val': None,
        })
        if self.holey:
            episode.hist_data.update({
                'holes_start': None,
                'holes_end': None,
            })

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

        # TODO: log ctrl targets and success rate as heatmap: x is timestep, y is ctrl target, heatmap is success rate

        for k in env.ctrl_metrics:
            # episode.ctrl_metrics = {f'ctrl-{k}': {env.metric_trgs[k]: env.metrics[k]}}
            episode.hist_data.update({
                f'{k}-trg': [env.metric_trgs[k]],  # rllib needs these values to be lists :)
            })
        for k in env.metrics:
            episode.hist_data.update({f'{k}-val': [env.metrics[k]],})

        # episode.hist_data.update({k: [v] for k, v in episode_stats.items() if k in stats_list})
        # episode.custom_metrics.update({k: [v] for k, v in episode_stats.items() if k in stats_list})

        if hasattr(env.unwrapped._prob, '_hole_queue'):
            entrance_coords, exit_coords = env.unwrapped._prob.entrance_coords, env.unwrapped._prob.exit_coords
            if len(entrance_coords.shape) == 1:
                # Then it's 2D.
                episode.hist_data.update({
                    'holes_start': [entrance_coords],
                    'holes_end': [exit_coords],
                })
            else:
                # Just record the foot-room if 3D
                episode.hist_data.update({
                    'holes_start': [tuple(env.unwrapped._prob.entrance_coords[0])],
                    'holes_end': [tuple(env.unwrapped._prob.exit_coords[0])],
                })
