"""
Generate a fully connected top 3D layout where the longest path is greater than a certain threshold.

Paths are measured in terms of the approximate physics of a minecraft player character. The player can move in any of the
four cardinal directions, provided there are two blocks available vertically (for feet and head, let's say). The player
can also move up and down stairs in any of these directions, if the stairs are one block high, and there are three 
vertical blocks available on the lower step (and two vertical blocks available on the taller step).
"""
from pdb import set_trace as TT

import numpy as np
from timeit import default_timer as timer

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper_3D import get_path_coords, get_range_reward, get_tile_locations, calc_num_regions, \
    calc_longest_path, debug_path, run_dijkstra
from gym_pcgrl.envs.probs.minecraft.mc_render import erase_3D_path, spawn_3D_maze, spawn_3D_border, spawn_3D_path


class Minecraft3DmazeProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._length = 7
        self._width = 7
        self._height = 7
        self._prob = {"AIR": 0.0, "DIRT":1.0}
        self._border_tile = "DIRT"
        self._border_size = (1, 1, 1)

        self._target_path = 10
        self._random_probs = False

        self._rewards = {
            "regions": 0,
            "path-length": 5
        }
        self.static_trgs = {"regions": 1, "path-length": np.inf}

        self.path_coords = []
        self.old_path_coords = []
        self.path_length = None
        self.render_path = False

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["AIR", "DIRT"]

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

        self.render_path = kwargs.get('render', self.render_path) or kwargs.get('render_path', self.render_path)
        
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
            self._prob["DIRT"] = 1 - self._prob["AIR"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        self.old_path_coords = self.path_coords
        self.path_coords = []
        # do not fix the positions of entrance and exit (calculating the longest path among 2 random positions) 
        # start_time = timer()
        self.path_length, self.path_coords = calc_longest_path(map, map_locations, ["AIR"], get_path=self.render_path)
        # print(f"minecraft path-finding time: {timer() - start_time}")
        if self.render:
            path_is_valid = debug_path(self.path_coords, map, ["AIR"])
            if not path_is_valid:
                return None
        # # fix the positions of entrance and exit at the bottom and diagonal top, respectively
        # p_x, p_y, p_z = 0, 0, 0
        # dijkstra_p, _ = run_dijkstra(p_x, p_y, p_z, map, ["AIR"])
        # # print("dijkstra map: ", dijkstra_p)
        # d_x, d_y, d_z = len(map[0][0])-1, len(map[0])-1, len(map)-2
        # self.path_length = dijkstra_p.max() if dijkstra_p[d_z][d_y][d_x] < 0 else dijkstra_p[d_z][d_y][d_x]
        # # print("path length: ", self.path_length)

        # if self.render_path:
        #     # TT()
        #     if dijkstra_p[d_z][d_y][d_x] > 0:
        #         self.path_coords = get_path_coords(dijkstra_p, d_x, d_y, d_z)
                # path_debug(path, map, passable_values)
        #     else:
        #         self.path_coords = get_path_coords(dijkstra_p)
                # path_debug(path, map, passable_values)
        #     # print("path coords: ", self.path_coords)

        return {
            "regions": calc_num_regions(map, map_locations, ["AIR"]),
            "path-length": self.path_length,
        }

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
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward
        return rewards["regions"] * self._rewards["regions"] +\
            rewards["path-length"] * self._rewards["path-length"]

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
            "path-imp": new_stats["path-length"] - self._start_stats["path-length"]
        }
    
    def render(self, map, iteration_num, repr_name):
        if iteration_num == 0 or iteration_num == 1:
            spawn_3D_border(map, self._border_tile)

        # if the representation is narrow3D or turtle3D, we don't need to render all the map at each step 
        if repr_name == "narrow3D" or repr_name == "turtle3D":
            # if iteration_num == 0 or iteration_num == 1:      
            spawn_3D_maze(map, self._border_tile)
        else:
            spawn_3D_maze(map, self._border_tile)

        if self.render_path:
            erase_3D_path(self.old_path_coords)
            spawn_3D_path(self.path_coords)
            # time.sleep(0.2)
        return 
