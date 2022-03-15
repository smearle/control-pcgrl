import os
from pdb import set_trace as TT

import numpy as np
from PIL import Image

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper_3D import get_range_reward, get_tile_locations, calc_num_regions, get_path_coords, calc_certain_tile, run_dijkstra
from gym_pcgrl.envs.probs.minecraft.mc_render import spawn_3D_maze, spawn_3D_border

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DZeldaProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._length = 7
        self._width = 7
        self._height = 7
        self._prob = {"AIR": 0.5, "DIRT":0.45, "CHEST":0.05}
        self._border_tile = "DIRT"
        self._border_size = (1, 1, 1)

        self._target_path = 20
        self._random_probs = False

        self.path_length = 0
        self.path_coords = []

        self._max_chests = 1

        self.render_path = False 

        self._rewards = {
            "regions": 5,
            "path-length": 1,
            "chests": 3
        }
# NEXT: add platform game or use NCA repre / RL agent to train a Zelda
# and add a easy render 3D pillow option

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["AIR", "DIRT","CHEST"]

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._length = kwargs.get('length', self._length)
        self._target_path = kwargs.get('target_path', self._target_path)
        self._random_probs = kwargs.get('random_probs', self._random_probs)

        self.render_path = kwargs.get('render') or kwargs.get('render_path', self.render_path)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        super().reset(start_stats)
        if self._random_probs:
            self._prob["AIR"] = self._random.random()
            self._prob["DIRT"] = self._random.random()
            self._prob["CHEST"] = 1 - self._prob["AIR"] - self._prob["DIRT"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        self.path = []
        map_stats = {
            "regions": calc_num_regions(map, map_locations, ["AIR"]),
            "path-length": 0,
            "chests": calc_certain_tile(map_locations, ["CHEST"])
        }
        if map_stats["regions"] == 1:
            p_x, p_y, p_z= 0, 0, 0
            if map_stats["chests"] == 1:
                c_x, c_y, c_z = map_locations["CHEST"][0]
                d_x, d_y, d_z = len(map[0][0])-1, len(map[0])-1, len(map)-2
                dikjstra_c, _ = run_dijkstra(
                    p_x, p_y, p_z, map, ["AIR"])
                map_stats["path-length"] += dikjstra_c[c_z][c_y][c_x]
                dikjstra_d, _ = run_dijkstra(
                    c_x, c_y, c_z, map, ["AIR"])
                map_stats["path-length"] += dikjstra_d[d_z][d_y][d_x]
                if self.render_path:
                    pass  # TODO add path rendering

        self.path_length = map_stats["path-length"]
        return map_stats

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf),
            "chests": get_range_reward(new_stats["chests"], old_stats["chests"], 1, 1),
        }
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"] +\
            rewards["chests"] * self._rewards["chests"]

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
        return new_stats["regions"] == 1 and new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            "path-imp": new_stats["path-length"] - self._start_stats["path-length"],
            "chests": new_stats["chests"]
        }

    def render(self, map):
        spawn_3D_border(map, self._border_tile)
        spawn_3D_maze(map, self._border_tile)
        # TODO add path rendering
        return 
