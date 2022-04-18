""" 
This file is for debugging path-finding in Minecraft problems, using hand-crafted levels. 

We need to hackishly set the entire level to be equal to the hand-crafted array, then run the path-finding algorithm by calling get_stats().
"""
from pdb import set_trace as TT

import gym
import numpy as np
import gym_pcgrl
from gym_pcgrl.envs.helper import get_string_map

# shape = (height, width, length)
maze = np.array([[0, 0],
                 [0, 0],
                 [0, 1],])



# FIXME: why does this seem to take so god damn long? Path-finding is fast. Just creating the environment itself?

env_name = "binary_ctrl-narrow-v0"
env = gym.make(env_name)
env.adjust_param(render=True)
env._rep._x = env._rep._y = 0
env._rep._map = maze
env.render()
stats = env.unwrapped._prob.get_stats(
    get_string_map(maze, env.unwrapped._prob.get_tile_types()),
)

# Keep rendering for the sake of observation.
while True:
    env.render()
