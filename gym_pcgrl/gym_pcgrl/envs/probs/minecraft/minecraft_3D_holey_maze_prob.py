"""
Generate a fully connected top 3D layout where the longest path is greater than a certain threshold.

Paths are measured in terms of the approximate physics of a minecraft player character. The player can move in any of the
four cardinal directions, provided there are two blocks available vertically (for feet and head, let's say). The player
can also move up and down stairs in any of these directions, if the stairs are one block high, and there are three 
vertical blocks available on the lower step (and two vertical blocks available on the taller step).
"""
from pdb import set_trace as TT
from tkinter import W

import numpy as np
from timeit import default_timer as timer

from gym_pcgrl.envs.helper_3D import get_path_coords, get_range_reward, get_tile_locations, calc_num_regions, \
    calc_longest_path, debug_path, plot_3D_path, remove_stacked_path_tiles, run_dijkstra
from gym_pcgrl.envs.probs.minecraft.mc_render import (erase_3D_path, spawn_3D_bordered_map, spawn_3D_maze, spawn_3D_border, spawn_3D_path, 
    get_3D_maze_blocks, get_3D_path_blocks, get_erased_3D_path_blocks, render_blocks, spawn_base)
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_maze_prob import Minecraft3DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_pb2 import LEAVES, TRAPDOOR
# from gym_pcgrl.test3D import plot_3d_map


class Minecraft3DholeymazeProblem(Minecraft3DmazeProblem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
       
        self.fixed_holes = False

        self._reward_weights.update({
            "regions": 0,
            "path-length": 100,
            "connected-path-length": 120,
            "n_jump": 150,
            # "connectivity": 0,
            # "connectivity": self._width,
            # "connectivity": self._max_path_length,
        })
        self._ctrl_reward_weights = self._reward_weights

        self.static_trgs.update({
            # "connectivity": 1,
            "path-length": 10 * self._max_path_length,
            "connected-path-length": 10 * self._max_path_length,
            "n_jump": 5,
        })

        # boundaries for conditional inputs/targets
        self.cond_bounds.update({
            # "connectivity": (0, 1),
            "path-length": (0, self._max_path_length + 2),
            "connected-path-length": (0, self._max_path_length + 2),
            "n_jump": (0, self._max_path_length // 2),
        })

        dummy_bordered_map = np.zeros((self._height + 2, self._width + 2, self._length+ 2), dtype=np.uint8)
        # # Fill in the non-borders with ones
        # dummy_bordered_map[1:-1, 1:-1, 1:-1] = 1
        # # Fill in the corners with ones
        # dummy_bordered_map[:, 0, 0] = 1
        # dummy_bordered_map[:, 0, -1] = 1
        # dummy_bordered_map[:, -1, 0] = 1
        # dummy_bordered_map[:, -1, -1] = 1
        # # fill in the top and bottom with ones
        # dummy_bordered_map[0, ...] = 1
        # dummy_bordered_map[-1, ...] = 1

        dummy_bordered_map[1:-2, 1:-1, 0] = 1
        dummy_bordered_map[1:-2, 1:-1, -1] = 1
        dummy_bordered_map[1:-2, 0, 1:-1] = 1
        dummy_bordered_map[1:-2, -1, 1:-1] = 1
        self._border_idxs = np.argwhere(dummy_bordered_map == 1)

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self.fixed_holes = kwargs.get('fixed_holes') if 'fixed_holes' in kwargs else self.fixed_holes



    def gen_holes(self):
        """Generate one entrance and one exit hole into/out of the map randomly. Ensure they will not necessarily result
         in trivial paths in/out of the map. E.g., the below are not valid holes:
        0 0    0 x  
        x      x
        x      0
        0      0

        start_xyz[0]: the foot room of the entrance
        start_xyz[0][0]: z  (height)
        start_xyz[0][1]: y  (width)
        start_xyz[0][2]: x  (length)
        """
        # assert the map is not too small
        assert self._height > 2 

        if self.fixed_holes:
            self.start_xyz = np.array(([1, 1, 0], [2, 1, 0]))
            self.end_xyz = np.array(((1, self._width, self._length + 1),
                                     (2, self._width, self._length + 1)))
            return self.start_xyz, self.end_xyz

        else:
            self.start_xyz = np.ones((2, 3), dtype=np.uint8)  
            potential = 26
            idxs = np.random.choice(self._border_idxs.shape[0], size=potential, replace=False)

            # randomly select a hole as the foot room of the entrance
            self.start_xyz[0] = self._border_idxs[idxs[0]]
            # select a coresonding hole as the head room of the entrance
            self.start_xyz[1] = self.start_xyz[0] + np.array([1, 0, 0])

            self.end_xyz = np.ones((2, 3), dtype=np.uint8)
            # select the exit
            # NOTE: Some valid cases are excluded, e.g.:
            #
            #     0
            #     0 0
            #       0
            #
            for i in range(1, potential):
                xyz = self._border_idxs[idxs[i]]
                if np.max((np.abs(self.start_xyz[0] - xyz), np.abs(self.start_xyz[1] - xyz))) != 1: 
                    self.end_xyz[0] = xyz
                    self.end_xyz[1] = xyz + np.array([1, 0, 0])
                    break
        
        return self.start_xyz, self.end_xyz

      

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())

        # for earsing the path of the previous iteration in Minecraft
        # new path coords are updated in the render function
        self.old_path_coords = self.path_coords
        self.old_connected_path_coords = self.connected_path_coords

        # do not fix the positions of entrance and exit (calculating the longest path among 2 random positions) 
        # start_time = timer()
        
        paths, _, jumps = run_dijkstra(self.start_xyz[0][2], self.start_xyz[0][1], self.start_xyz[0][0], map, ["AIR"])
        # connected_path_length = dijkstra_map[self.end_xyz[0][0], self.end_xyz[0][1], self.end_xyz[0][2]]
        end_xyz = tuple(self.end_xyz[0][::-1])  # lol ... why?
        self.connected_path_length = len(paths[end_xyz]) if end_xyz in paths else -1
        self.connected_path_coords = np.array(paths[end_xyz]) if end_xyz in paths else []
        self.connected_path_coords = remove_stacked_path_tiles(self.connected_path_coords)
        self.n_jump = jumps[end_xyz] if end_xyz in jumps else 0
        tiles_paths = [(tile, path) for tile, path in paths.items()]
        max_id = np.argmax(np.array([len(p) for (_,p) in tiles_paths]))
        max_tile, self.path_coords = tiles_paths[max_id]
        self.path_length = len(self.path_coords)
        self.path_coords = remove_stacked_path_tiles(self.path_coords)

        assert not (self.connected_path_length == 0 and len(self.connected_path_coords) > 0)
        # print(f"minecraft path-finding time: {timer() - start_time}")
        
        if self.render:
            path_is_valid = debug_path(self.path_coords, map, ["AIR"])
            connected_path_is_valid = debug_path(self.connected_path_coords, map, ["AIR"])
            if not path_is_valid:
                raise ValueError("The path is not valid, may have some where unstandable for a 2-tile high agent")
            if not connected_path_is_valid:
                raise ValueError("The connected path is not valid, may have some where unstandable"
                                "for a 2-tile high agent")
        # # fix the positions of entrance and exit at the bottom and diagonal top, respectively
        # p_x, p_y, p_z = 0, 0, 0
        # dijkstra_p, _ = run_dijkstra(p_x, p_y, p_z, map, ["AIR"])
        # # print("dijkstra map: ", dijkstra_p)
        # d_x, d_y, d_z = len(map[0][0])-1, len(map[0])-1, len(map)-2
        # self.path_length = dijkstra_p.max() if dijkstra_p[d_z][d_y][d_x] < 0 else dijkstra_p[d_z][d_y][d_x]
        # # print("path length: ", self.path_length)

        # if self.render_path:
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
            "connected-path-length": self.connected_path_length,
            # "path-coords": self.path_coords,
            "n_jump": self.n_jump
        }

    def process_observation(self, observation):
        return super().process_observation(observation, self.connected_path_coords)
        
    """
    This func is handled by the conditional wrapper

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    # def get_reward(self, new_stats, old_stats):
    #     #longer path is rewarded and less number of regions is rewarded
    #     rewards = {
    #         "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
    #         "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf),
    #         "n_jump": get_range_reward(new_stats["n_jump"], old_stats["n_jump"], np.inf, np.inf)
    #     }
    #     #calculate the total reward
    #     return rewards["regions"] * self._reward_weights["regions"] +\
    #         rewards["path-length"] * self._reward_weights["path-length"] +\
    #         rewards["n_jump"] * self._reward_weights["n_jump"]

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    # def get_episode_over(self, new_stats, old_stats):
        # return new_stats["regions"] == 1 and new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path

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
            "n_jump": new_stats["n_jump"]
        }
    
    def render(self, map, iteration_num, repr_name, render_matplotlib=False, **kwargs):
        # NOTE: the agent's action is rendered directly before this function is called.

        # Render the border if we haven't yet already.
        if not self._rendered_initial_maze:
            # spawn_3D_border(map, self._border_tile, start_xyz=self.start_xyz, end_xyz=self.end_xyz)
            spawn_base(map)
            spawn_3D_maze(map)
            # spawn_3D_bordered_map(map)
            self._rendered_initial_maze = True

        # block_dict.update(get_3D_maze_blocks(map))
        # FIXME: these functions which return dictionaries of blocks to be rendered are broken somehow
        # block_dict.update(get_3D_maze_blocks(map))
        # block_dict = {}

        # It would be nice to not have to re-render the whole path at each step, but for now, we do not
        # know if the agent's edit action has disrupted the old path, so we won't delete blocks in the
        # old path that are also in the new path, but we will have to render all blocks in the new path,
        # just in case.
        # old_path_coords = [tuple(coords) for coords in self.old_path_coords]
        # old_connected_path_coords = [tuple(coords) for coords in self.old_connected_path_coords]

        # path_to_erase = set(old_path_coords)
        # connected_path_to_erase = set(old_connected_path_coords)
        
        # path_to_render = []
        # for (x, y, z) in self.path_coords:
        #     if (x, y, z) in path_to_erase:
        #         path_to_erase.remove((x, y, z))
        # for (x, y, z) in self.connected_path_coords:
        #     if (x, y, z) in path_to_erase:
        #         connected_path_to_erase.remove((x, y, z))
            # else:
                # path_to_render.append((x, y, z))
#       print(self.path_coords)
#       print(path_to_render)
#       print(path_to_erase)
#       print(len(self.path_coords))

        if self.render_path:
            # block_dict.update(get_erased_3D_path_blocks(self.old_path_coords))
            # erase_3D_path(path_to_erase)
            # erase_3D_path(connected_path_to_erase)

            # block_dict.update(get_3D_path_blocks(self.path_coords))
            # spawn_3D_path(self.path_coords)
            spawn_3D_maze(map)
            render_path_coords = [tuple(coords) for coords in self.connected_path_coords if map[coords[0], coords[1], coords[2]] == 'AIR']
            spawn_3D_path(render_path_coords, item=TRAPDOOR)
            # spawn_3D_path(self.path_coords, item=LEAVES)
            # time.sleep(0.2)

        # render_blocks(block_dict)

        # plot the path using matplotlib
        if render_matplotlib:
            plot_3D_path(self._length, self._width, self._height, self.path_coords)

        return 

    def get_episode_over(self, new_stats, old_stats):
        """ If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)"""

        return False

    def get_reward(self, new_stats, old_stats):
        return None