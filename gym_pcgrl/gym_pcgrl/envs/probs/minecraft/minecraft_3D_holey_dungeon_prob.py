from pdb import set_trace as TT

from sklearn.utils import check_X_y

from gym_pcgrl.envs.helper_3D import calc_certain_tile, calc_num_regions, get_path_coords, get_tile_locations, plot_3D_path, run_dijkstra
from gym_pcgrl.envs.probs.minecraft.mc_render import erase_3D_path, spawn_3D_border, spawn_3D_bordered_map, spawn_3D_maze, spawn_3D_path
import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3D_dungeon_prob import Minecraft3DDungeonProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_holey_maze_prob import Minecraft3DholeymazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_pb2 import LEAVES

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DholeyDungeonProblem(Minecraft3DDungeonProblem, Minecraft3DholeymazeProblem):
    def __init__(self):
        Minecraft3DholeymazeProblem.__init__(self)
        Minecraft3DDungeonProblem.__init__(self)
       
        # self._reward_weights.update({
        #     "regions": 0,
        #     "path-length": 1,
        #     "n_jump": 1,
        #     # "connectivity": 0,
        #     # "connectivity": self._width,
        #     # "connectivity": self._max_path_length,
        # })

        self.static_trgs.update({
            # "connectivity": 1,
            "path-length": self._max_path_length + 2,
            "n_jump": 5,
        })

        # boundaries for conditional inputs/targets
        self.cond_bounds.update({
            # "connectivity": (0, 1),
            "path-length": (0, self._max_path_length + 2),
            "n_jump": (0, self._max_path_length // 2),
        })

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self.fixed_holes = kwargs.get('fixed_holes') if 'fixed_holes' in kwargs else self.fixed_holes

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())

        self.old_path_coords = self.path_coords

        self.path_coords = []

        map_stats = {
            "regions": calc_num_regions(map, map_locations, ["AIR"]),
            "path-length": 0,
            "chests": calc_certain_tile(map_locations, ["CHEST"]),
            "enemies": calc_certain_tile(map_locations, ["SKULL", "PUMPKIN"]),
            "nearest-enemy": 0,
            "n_jump": 0,
        }
        
        if map_stats["regions"] == 1:
            # entrance is self.start_xyz, a hole on the border(we use the foot room for path finding), in the form of (z, y, x)
            p_z, p_y, p_x = self.start_xyz[0]

            enemies = []
            enemies.extend(map_locations["SKULL"])
            enemies.extend(map_locations["PUMPKIN"])
            if len(enemies) > 0:
                paths_e, _, _ = run_dijkstra(p_x, p_y, p_z, map, self._passable)
                min_dist = self._width * self._height * self._length
                for e_x, e_y, e_z in enemies:  # wtf
                    e_path = paths_e.get((e_x, e_y, e_z), [])
                    e_dist = len(e_path)
                    if e_dist > 0 and e_dist < min_dist:
                        min_dist = e_dist
                        self.min_e_path = e_path
                map_stats["nearest-enemy"] = min_dist


            if map_stats["chests"] == 1:
                c_xyz = map_locations["CHEST"][0]

                # exit is self.end_xyz, a hole on the border(we use the foot room the find the path), in the form of (z, y, x)
                d_xyz = tuple(self.end_xyz[0][::-1])  # lol

                # start point is player
                paths_c, _, jumps_c = run_dijkstra(p_x, p_y, p_z, map, self._passable)
                path_c = paths_c.get(c_xyz, [])
                map_stats["path-length"] += len(path_c)
                map_stats["n_jump"] += jumps_c.get(c_xyz, 0)

                # start point is chests
                paths_d, _, jumps_d = run_dijkstra(*c_xyz, map, self._passable)
                path_d = paths_d.get(d_xyz, [])
                map_stats["path-length"] +=  len(path_d) 
                map_stats["n_jump"] += jumps_d.get(d_xyz, 0)
                # if self.render_path:
                    # self.path_coords = np.vstack((get_path_coords(paths_c, c_x, c_y, c_z),
                                                #   get_path_coords(pathd_d, d_x, d_y, d_z)))
                self.path_coords = path_c + path_d
                # self.path_coords = np.vstack((path_c, path_d))

        self.path_length = map_stats["path-length"]
        self.n_jump = map_stats["n_jump"]
        return map_stats


    # def process_observation(self, observation):
    #     if self.path_coords == []:
    #         return observation
    #     observation['map'][self.path_coords[:, 0], 
    #                         self.path_coords[:, 1], 
    #                         self.path_coords[:, 2]] = self._path_idx
    #     return observation


    def get_debug_info(self, new_stats, old_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            # "path-imp": new_stats["path-length"] - self._start_stats["path-length"],
            "n_jump": new_stats["n_jump"], 
            "chests": new_stats["chests"],
            "enemies": new_stats["enemies"],
            "nearest-enemy": new_stats["nearest-enemy"],
        }


    def render(self, map, iteration_num, repr_name, render_matplotlib=False, **kwargs):
        # NOTE: the agent's action is rendered directly before this function is called.

        # Render the border if we haven't yet already.
        if not self._rendered_initial_maze:
            # spawn_3D_border(map, self._border_tile, start_xyz=self.start_xyz, end_xyz=self.end_xyz)
            # spawn_3D_maze(map)
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
        old_path_coords = [tuple(coords) for coords in self.old_path_coords]
        path_to_erase = set(old_path_coords)
        path_to_render = []
        for (x, y, z) in self.path_coords:
            if (x, y, z) in path_to_erase:
                path_to_erase.remove((x, y, z))
            # else:
                # path_to_render.append((x, y, z))
#       print(self.path_coords)
#       print(path_to_render)
#       print(path_to_erase)
#       print(len(self.path_coords))

        # if self.render_path:
            # block_dict.update(get_erased_3D_path_blocks(self.old_path_coords))
            # erase_3D_path(path_to_erase)

            # block_dict.update(get_3D_path_blocks(self.path_coords))
        spawn_3D_maze(map)
        spawn_3D_path(self.path_coords)
        spawn_3D_path(self.min_e_path, item=LEAVES)
            # time.sleep(0.2)

        # render_blocks(block_dict)

        # plot the path using matplotlib
        if render_matplotlib:
            plot_3D_path(self._length, self._width, self._height, self.path_coords)

        return 
