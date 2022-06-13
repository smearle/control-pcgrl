import os
from pdb import set_trace as TT
import time

import numpy as np
from PIL import Image

from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper_3D import (get_range_reward, get_tile_locations, calc_num_regions, get_path_coords,
    calc_certain_tile, run_dijkstra)
from gym_pcgrl.envs.probs.minecraft.mc_render import erase_3D_path, spawn_2D_maze, spawn_3D_maze, spawn_3D_border, spawn_3D_path

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
# class Minecraft3DDungeonProblem(Problem):
#     """
#     The constructor is responsible of initializing all the game parameters
#     """
#     def __init__(self):
#         super().__init__()
#         self._passable = ["AIR", "CHEST", "SKULL", "PUMPKIN"]
#         self._length = 7
#         self._width = 7
#         self._height = 7
#         self._prob = {"AIR": 0.5, "DIRT":0.35, "CHEST":0.05, "SKULL":0.05, "PUMPKIN":0.05}
#         self._prob = {"AIR": 1.5, "DIRT":0., "CHEST":0.0, "SKULL":0.0, "PUMPKIN":0.0}
#         self._border_tile = "DIRT"
#         self._border_size = (1, 1, 1)

#         self._target_path = 20
#         self._random_probs = False

#         self.path_length = 0
#         self.path_coords = []
#         self.old_path_coords = []

#         self.n_jump = 0

#         self._max_chests = 1

#         self._max_enemies = 5
#         self._target_enemy_dist = 4

#         self.render_path = True
#         self._rendered_initial_maze = False
#         n_floors = self._height // 3

#         self._max_nearest_enemy = self._max_path_length // 2

#         self._path_idx = len(self.get_tile_types())

#         # self._reward_weights = {
#         #     "regions": 5,
#         #     "path-length": 1,
#         #     "chests": 3,
#         #     "enemies": 1,
#         #     "nearest-enemy":2
#         # }
        
#         # change floor by stairs require 6 path_length for each floor

# #       self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)

#         # default conditional targets
#         self.static_trgs = {
#             "enemies": (2, self._max_enemies),
#             "regions": 1, 
#             "path-length": 10 * self._max_path_length, 
#             "nearest-enemy": (5, self._max_nearest_enemy),
#             "chests": 1,
#             "n_jump": 3
#         }

#         # boundaries for conditional inputs/targets
#         self.cond_bounds = {
#             # Upper bound: checkerboard
#             "regions": (0, np.ceil(self._width * self._length / 2 * self._height)),
#             # (assume these are stacked in the 3rd dimension aka height)
#             #     10101010
#             #     01010101
#             #     10101010
#             #     01010101
#             #     10101010
#             # FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
#             # and "snake" along that one
#             # Upper bound: zig-zag
#             "path-length": (0, self._max_path_length),
#             "chests": (0, self._width * self._length * self._height - 2 ),
#             "n_jump": (0, self._max_path_length // 2),
#             "nearest-enemy": (0, self._max_nearest_enemy),
#             "enemies": (0, self._width * self._length * self._height - 2),
#             "chests": (0, self._width * self._length * self._height - 2),
#         }

#         self._reward_weights = {
#             "regions": 0, 
#             "path-length": 100, 
#             "chests": 300, 
#             "n_jump": 100,
#             "enemies": 100,
#             "nearest-enemy": 200,
#         }
#         self._ctrl_reward_weights = self._reward_weights

# # NEXT: add a easy render 3D pillow option

#     """
#     Get a list of all the different tile names

#     Returns:
#         string[]: that contains all the tile names
#     """
#     def get_tile_types(self):
#         return ["AIR", "DIRT", "CHEST", "SKULL", "PUMPKIN"]

#     """
#     Adjust the parameters for the current problem

#     Parameters:
#         width (int): change the width of the problem level
#         height (int): change the height of the problem level
#         probs (dict(string, float)): change the probability of each tile
#         intiialization, the names are "empty", "solid"
#         target_path (int): the current path length that the episode turn when it reaches
#         rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
#     """
#     def adjust_param(self, **kwargs):
#         super().adjust_param(**kwargs)
#         self._length = kwargs.get('length', self._length)
#         self._target_path = kwargs.get('target_path', self._target_path)
#         self._random_probs = kwargs.get('random_probs', self._random_probs)

#         self.render_path = kwargs.get('render') or kwargs.get('render_path', self.render_path)

#         rewards = kwargs.get('rewards')
#         if rewards is not None:
#             for t in rewards:
#                 if t in self._reward_weights:
#                     self._reward_weights[t] = rewards[t]

#     """
#     Resets the problem to the initial state and save the start_stats from the starting map.
#     Also, it can be used to change values between different environment resets

#     Parameters:
#         start_stats (dict(string,any)): the first stats of the map
#     """
#     def reset(self, start_stats):
#         self.min_e_path = []
#         self._rendered_initial_maze = False
#         super().reset(start_stats)
#         if self._random_probs:
#             self._prob["AIR"] = self._random.random()
#             self._prob["DIRT"] = self._random.random()
            
#             self._prob["PUMPKIN"] = self._random.random()
#             self._prob["SKULL"] = self._random.random()

#             self._prob["CHEST"] = 1 - self._prob["AIR"] - self._prob["DIRT"] - self._prob["SKULL"] - self._prob["PUMPKIN"]

#     """
#     Get the current stats of the map

#     Returns:
#         dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
#         The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
#     """
#     def get_stats(self, map):
#         map_locations = get_tile_locations(map, self.get_tile_types())

#         self.old_path_coords = self.path_coords

#         self.path_coords = []
#         map_stats = {
#             "regions": calc_num_regions(map, map_locations, ["AIR"]),
#             "path-length": 0,
#             "chests": calc_certain_tile(map_locations, ["CHEST"]),
#             "enemies": calc_certain_tile(map_locations, ["SKULL", "PUMPKIN"]),
#             "nearest-enemy": 0,
#             "n_jump": 0,
#         }
#         if map_stats["regions"] == 1:
#             # entrance is fixed at the bottom of the maze house
#             p_x, p_y, p_z= 0, 0, 0

#             enemies = []
#             enemies.extend(map_locations["SKULL"])
#             enemies.extend(map_locations["PUMPKIN"])
#             if len(enemies) > 0:
#                 dijkstra, _, _ = run_dijkstra(p_x, p_y, p_z, map, self._passable_tiles)
#                 min_dist = self._width * self._height * self._length
#                 for e_x, e_y, e_z in enemies:
#                     if dijkstra[e_z][e_y][e_x] > 0 and dijkstra[e_z][e_y][e_x] < min_dist:
#                         min_dist = dijkstra[e_z][e_y][e_x]
#                 map_stats["nearest-enemy"] = min_dist


#             if map_stats["chests"] == 1:
#                 c_x, c_y, c_z = map_locations["CHEST"][0]
#                 d_x, d_y, d_z = len(map[0][0])-1, len(map[0])-1, len(map)-2

#                 # start point is 0, 0, 0
#                 dijkstra_c, _, jump_map = run_dijkstra(p_x, p_y, p_z, map, self._passable_tiles)
#                 map_stats["path-length"] += dijkstra_c[c_z][c_y][c_x]
#                 map_stats["n_jump"] += jump_map[c_z][c_y][c_x]

#                 # start point is chests
#                 dijkstra_d, _, jump_map = run_dijkstra(c_x, c_y, c_z, map, self._passable_tiles)
#                 map_stats["path-length"] += dijkstra_d[d_z][d_y][d_x]
#                 map_stats["n_jump"] += jump_map[d_z][d_y][d_x]
#                 if self.render_path:
#                     self.path_coords = np.vstack((get_path_coords(dijkstra_c, c_x, c_y, c_z),
#                                                   get_path_coords(dijkstra_d, d_x, d_y, d_z)))

#         self.path_length = map_stats["path-length"]
#         self.n_jump = map_stats["n_jump"]
#         return map_stats

#     """
#     Get the current game reward between two stats

#     Parameters:
#         new_stats (dict(string,any)): the new stats after taking an action
#         old_stats (dict(string,any)): the old stats before taking an action

#     Returns:
#         float: the current reward due to the change between the old map stats and the new map stats
#     """
#     # def get_reward(self, new_stats, old_stats):
#     #     #longer path is rewarded and less number of regions is rewarded
#     #     rewards = {
#     #         "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
#     #         "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf),
#     #         "chests": get_range_reward(new_stats["chests"], old_stats["chests"], 1, 1),
#     #         "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 2, self._max_enemies),
#     #         "nearest-enemy": get_range_reward(new_stats["nearest-enemy"], old_stats["nearest-enemy"], self._target_enemy_dist, np.inf),
#     #     }
#     #     #calculate the total reward
#     #     return rewards["regions"] * self._reward_weights["regions"] +\
#     #         rewards["path-length"] * self._reward_weights["path-length"] +\
#     #         rewards["chests"] * self._reward_weights["chests"] +\
#     #         rewards["enemies"] * self._reward_weights["enemies"] +\
#     #         rewards["nearest-enemy"] * self._reward_weights["nearest-enemy"]

#     """
#     Uses the stats to check if the problem ended (episode_over) which means reached
#     a satisfying quality based on the stats

#     Parameters:
#         new_stats (dict(string,any)): the new stats after taking an action
#         old_stats (dict(string,any)): the old stats before taking an action

#     Returns:
#     #     boolean: True if the level reached satisfying quality based on the stats and False otherwise
#     # """
#     # def get_episode_over(self, new_stats, old_stats):
#     #     return (new_stats["regions"] == 1 and
#     #           new_stats["nearest-enemy"] >= self._target_enemy_dist and
#     #           new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path)

#     """
#     Get any debug information need to be printed

#     Parameters:
#         new_stats (dict(string,any)): the new stats after taking an action
#         old_stats (dict(string,any)): the old stats before taking an action

#     Returns:
#         dict(any,any): is a debug information that can be used to debug what is
#         happening in the problem
#     """
#     def get_debug_info(self, new_stats, old_stats):
#         return {
#             "regions": new_stats["regions"],
#             "path-length": new_stats["path-length"],
#             "path-imp": new_stats["path-length"] - self._start_stats["path-length"],
#             "chests": new_stats["chests"],
#             "enemies": new_stats["enemies"],
#             "nearest-enemy": new_stats["nearest-enemy"],
#             "n_jump": new_stats["n_jump"],
#         }

#     def render(self, map, iteration_num, repr_name, **kwargs):
#         # Render the border if we haven't yet already.
#         if not self._rendered_initial_maze:
#             # spawn_3D_border(map, self._border_tile, entrance_coords=self.entrance_coords, exit_coords=self.exit_coords)
#             # spawn_3D_maze(map)
#             self._rendered_initial_maze = True

#         # render the path
#         old_path_coords = [tuple(coords) for coords in self.old_path_coords]
#         path_to_erase = set(old_path_coords)
#         for (x, y, z) in self.path_coords:
#             if (x, y, z) in path_to_erase:
#                 path_to_erase.remove((x, y, z))
        
#         if self.render_path:
#             # block_dict.update(get_erased_3D_path_blocks(self.old_path_coords))
#             # erase_3D_path(path_to_erase)
                
#             # block_dict.update(get_3D_path_blocks(self.path_coords))
#             spawn_3D_maze(map)
#             spawn_3D_path(self.path_coords)
#             # time.sleep(0.2)
         
#         # if iteration_num == 0:
#         #     spawn_3D_border(map, self._border_tile)
#         # # if the representation is narrow3D or turtle3D, we don't need to render all the map at each step 
#         # if repr_name == "narrow3D" or repr_name == "turtle3D":
#         #     if iteration_num == 0:
#         #         spawn_3D_maze(map, self._border_tile)
#         # else:
#         #     spawn_3D_maze(map, self._border_tile)

#         # if self.render_path:
#         #     spawn_3D_path(self.path_coords)
#         #     # time.sleep(0.2)
#         #     erase_3D_path(self.path_coords)
#         return 
