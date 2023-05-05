import glob
import shutil
import gymnasium as gym
import hydra
import numpy as np
import os

from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter

from control_pcgrl.configs.config import PoDConfig
from control_pcgrl.il.utils import make_pod_env
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.utils import validate_config


TILES_MAP = {
    "g": "door",
    "+": "key",
    "A": "player",
    "1": "bat",
    "2": "spider",
    "3": "scorpion",
    "w": "solid",
    ".": "empty",
}

#TODO: Can get this from environment instead.
INT_MAP = {
    "empty": 0,
    "solid": 1,
    "player": 2,
    "key": 3,
    "door": 4,
    "bat": 5,
    "scorpion": 6,
    "spider": 7,
}

def load_goal_levels(cfg):
    lvl_dir = glob.glob(os.path.join("control_pcgrl", "il", "playable_maps", "*.txt"))
    levels = []

    for f in lvl_dir:
        levels.append(int_arr_from_str_arr(to_2d_array_level(f)))

    return np.array(levels)

# Converts from string[][] to 2d int[][]
def int_arr_from_str_arr(map):
    int_map = []
    for row_idx in range(len(map)):
        new_row = []
        for col_idx in range(len(map[0])):
            new_row.append(INT_MAP[map[row_idx][col_idx]])
        int_map.append(new_row)
    return int_map

# Reads in .txt playable map and converts it to string[][]
def to_2d_array_level(file_name):
    level = []

    with open(file_name, "r") as f:
        rows = f.readlines()
        for row in rows:
            new_row = []
            for char in row:
                if char != "\n":
                    new_row.append(TILES_MAP[char])
            level.append(new_row)

    # Remove the border
    truncated_level = level[1 : len(level) - 1]
    level = []
    for row in truncated_level:
        new_row = row[1 : len(row) - 1]
        level.append(new_row)
    return level

n_train_samples = 1_000_000


@hydra.main(config_path="control_pcgrl/configs", config_name="pod")
def main(cfg: PoDConfig):
    if not validate_config(cfg):
        print("Invalid config!")
        return

    traj_dir = os.path.join(cfg.log_dir, "repair-paths")

    if cfg.overwrite:
        shutil.rmtree(traj_dir, ignore_errors=True)

    goal_levels = load_goal_levels(cfg)

    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(
        # os.path.join(ray._private.utils.get_user_temp_dir(), "demo-out")
        os.path.join(traj_dir)
    )

    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    env = make_pod_env(cfg=cfg)
    # env = gym.make("CartPole-v1")

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    print("The preprocessor is", prep)

    map_id = 0

    n_eps = n_train_samples // env.unwrapped._max_iterations

    for eps_id in range(n_eps):
        # env.queue_goal_map(goal_levels[map_id])

        obs, info = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        terminated = truncated = False
        t = 0
        while not terminated and not truncated:
            # action = env.action_space.sample()
            action = goal_levels[map_id][tuple(env.rep.unwrapped._pos)]

            # repair_action = env.get_repair_action()
            new_obs, rew, terminated, truncated, info = env.step(action)
            batch_builder.add_values(
                # t=t,
                # eps_id=eps_id,
                # agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                # action_prob=1.0,  # put the true action probability here
                # action_logp=0.0,
                rewards=-rew,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                # truncateds=truncated,
                # infos=info,
                # new_obs=dummify_observation(prep.transform(new_obs)),
            )
            obs = new_obs
            prev_action = action
            prev_reward = -rew
            t += 1
        writer.write(batch_builder.build_and_reset())

        map_id = (eps_id + 1) % len(goal_levels)


if __name__ == "__main__":
    main()