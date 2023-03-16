import copy
import gym
import numpy as np
import torch as th

from control_pcgrl.configs.config import Config
from control_pcgrl.envs.pcgrl_env import PcgrlEnv

class RewardModelWrapper(gym.Wrapper):
    """Wrap a PCGRL env so that when actions are taken, we collect datapoints for training a reward model."""
    def __init__(self, env, cfg):
        self.env = env
        super().__init__(env)
        self.metric_keys = list(env.metrics.keys())
        self.datapoints = []
        self._last_changes = -1

    def step(self, action):
        """Take a step in the environment."""
        ret = self.env.step(action)

        if self.unwrapped._changes == self._last_changes:
            # No changes were made to the map, so don't collect data
            return ret

        metric_values = np.array([self.env.metrics[key] for key in self.metric_keys])

        disc_map = self.get_map()
        n_tiles = self.get_map_dims()[-1]
        onehot_map = np.eye(n_tiles)[disc_map]

        # Collect the datapoint
        self.datapoints.append((onehot_map, metric_values))
        return ret

    def collect_data(self):
        """Collect data from the environment."""
        feats, metrics = zip(*self.datapoints)
        feats = th.from_numpy(np.stack(feats))
        metrics = th.from_numpy(np.stack(metrics))
        self.datapoints = []
        return feats, metrics


def init_reward_model(env: PcgrlEnv):
    """Initialize a reward model and optimizer."""
    r_model_obs_shape = env.get_map_dims()
    r_model_obs_size = np.prod(r_model_obs_shape)
    r_model_out_size = len(env.metrics)
    # Make the reward model a small MLP
    reward_model = th.nn.Sequential(
        th.nn.Linear(r_model_obs_size, 32),
        th.nn.ReLU(),
        th.nn.Linear(32, 32),
        th.nn.ReLU(),
        th.nn.Linear(32, r_model_out_size),
    )
    optimizer = th.optim.Adam(reward_model.parameters(), lr=1e-3)
    return reward_model, optimizer

def train_reward_model(reward_model, optimizer, feats, metrics):
    """Train a reward model on one batch."""
    # Compute the loss
    loss = 0
    feats = feats.view(feats.shape[0], -1)
    pred = reward_model(feats.float())
    loss += th.nn.functional.mse_loss(pred, metrics.float())
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
