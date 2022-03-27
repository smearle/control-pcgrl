""" 
This file is for debugging path-finding in Minecraft problems, using hand-crafted levels. 

We need to hackishly set the entire level to be equal to the hand-crafted array, then run the path-finding algorithm by calling get_stats().
"""
import gym
import numpy as np
import gym_pcgrl
from gym_pcgrl.envs.helper_3D import get_string_map

# shape = (height, width, length)
staircase = np.ones((7, 7, 7))
staircase[0:3, 0, 0] = 0
staircase[0:3, 1, 0] = 0
staircase[1:4, 2, 0] = 0
staircase[2:5, 3, 0] = 0

int_map = staircase

env_name = "minecraft_3D_maze_ctrl-cellular3D-v0"
env = gym.make(env_name)
env.adjust_param(render=True)
env.unwrapped._rep._x = env.unwrapped._rep._y = 0
env._rep._map = staircase
env.render()
# FIXME: why does this seem to take so god damn long? Because of debug_path()?
stats = env.unwrapped._prob.get_stats(
    get_string_map(int_map, env.unwrapped._prob.get_tile_types()),
)
env.render()
