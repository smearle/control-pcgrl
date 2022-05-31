from gym_pcgrl.envs.helper_3D import plot_3D_path
from gym_pcgrl.envs.probs.minecraft.mc_render import erase_3D_path, spawn_3D_border, spawn_3D_maze, spawn_3D_path
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Dungeon_ctrl_prob import Minecraft3DDungeonCtrlProblem
import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Dungeon_prob import Minecraft3DDungeonProblem

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DholeyDungeonCtrlProblem(Minecraft3DDungeonCtrlProblem):
    def __init__(self):
        super().__init__()
       
        self.fixed_holes = False

        self._reward_weights.update({
            "regions": 0,
            "path-length": 1,
            "connected-path-length": 1.2,
            "n_jump": 1,
            # "connectivity": 0,
            # "connectivity": self._width,
            # "connectivity": self._max_path_length,
        })

        self.static_trgs.update({
            # "connectivity": 1,
            "path-length": self._max_path_length + 2,
            "connected-path-length": self._max_path_length + 2,
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
            self.end_xyz = np.array(((self._height -1 , self._width, self._length + 1),
                                     (self._height, self._width, self._length + 1)))
            return

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
            # I know some cases are excluded, for example:
            #
            #     0
            #     0 0
            #       0
            #
            # This kind of entrance exit pair is actually valid
            # that was considered in the very beginning, but excluding them is just the trade-off you know :)
            for i in range(1, potential):
                xyz = self._border_idxs[idxs[i]]
                if np.max((np.abs(self.start_xyz[0] - xyz), np.abs(self.start_xyz[1] - xyz))) != 1: 
                    self.end_xyz[0] = xyz
                    self.end_xyz[1] = xyz + np.array([1, 0, 0])
                    break
                
        
        return self.start_xyz, self.end_xyz
   
    def process_observation(self, observation):
        if self.connected_path_coords == []:
            return observation
        observation['map'][self.connected_path_coords[:, 0], 
                            self.connected_path_coords[:, 1], 
                            self.connected_path_coords[:, 2]] = self._path_idx
        return observation
    def get_debug_info(self, new_stats, old_stats):
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            "path-imp": new_stats["path-length"] - self._start_stats["path-length"],
            "n_jump": new_stats["n_jump"], 
            
        }
    def render(self, map, iteration_num, repr_name, render_matplotlib=False, **kwargs):
        # NOTE: the agent's action is rendered directly before this function is called.

        # Render the border if we haven't yet already.
        if not self._rendered_initial_maze:
            spawn_3D_border(map, self._border_tile, start_xyz=self._start_xyz, end_xyz=self._end_xyz)
            spawn_3D_maze(map)
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

        if self.render_path:
            # block_dict.update(get_erased_3D_path_blocks(self.old_path_coords))
            erase_3D_path(path_to_erase)

            # block_dict.update(get_3D_path_blocks(self.path_coords))
            spawn_3D_path(self.path_coords)
            # time.sleep(0.2)

        # render_blocks(block_dict)

        # plot the path using matplotlib
        if render_matplotlib:
            plot_3D_path(self._length, self._width, self._height, self.path_coords)

        return 
