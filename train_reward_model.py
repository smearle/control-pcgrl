import os
import hydra
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

import control_pcgrl
from control_pcgrl.configs.config import Config
from control_pcgrl.reward_model_wrappers import init_reward_model, train_reward_model
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config

batch_size = 64
n_train_iters = 10000

@hydra.main(config_path="control_pcgrl/configs", config_name="config")
def main(cfg: Config):
    """Train a model to predict relevant metrics in a PCGRL env. Generate data with random actions 
    (i.e. random map edits).
    """
    validate_config(cfg)

    log_dir = 'logs_reward_model'
    log_dir = os.path.join(hydra.utils.get_original_cwd(), log_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    cfg.train_reward_model = True

    env = make_env(cfg)
    metric_keys = list(env.metrics.keys())
    env.reset()

    model, optimizer = init_reward_model(env)

    writer = SummaryWriter(log_dir=log_dir)

    for i in range(n_train_iters):
        # Collect data
        while len(env.datapoints) < batch_size:
            env.step(env.action_space.sample())
            # print(f"Collected {len(env.datapoints)} datapoints")
        # Train
        feats, metrics = env.collect_data()
        loss = train_reward_model(model, optimizer, feats, metrics)
        writer.add_scalar("Loss", loss, i)
        print(f"Loss: {loss}")
        # Reset
        env.datapoints = []
        env.reset()


if __name__ == "__main__":
    main()