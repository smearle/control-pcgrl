import gymnasium as gym
import hydra
import numpy as np
import os

import ray._private.utils

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from control_pcgrl.configs.config import Config
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config

@hydra.main(config_path="control_pcgrl/configs", config_name="config")
def main(cfg: Config):
    validate_config(cfg)
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        # os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out")
        os.path.join(cfg.log_dir, "demo-out")
    )

    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    env = make_env(cfg=cfg)
    # env = gym.make("CartPole-v1")

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    for eps_id in range(100):
        obs, info = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        terminated = truncated = False
        t = 0
        while not terminated and not truncated:
            action = env.action_space.sample()
            new_obs, rew, terminated, truncated, info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=truncated,
                infos=info,
                new_obs=prep.transform(new_obs),
            )
            obs = new_obs
            prev_action = action
            prev_reward = rew
            t += 1
        writer.write(batch_builder.build_and_reset())


if __name__ == "__main__":
    main()