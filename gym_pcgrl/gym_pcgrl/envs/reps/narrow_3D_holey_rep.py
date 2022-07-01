from gym_pcgrl.envs.reps.holey_representation_3D import HoleyRepresentation3D
from gym_pcgrl.envs.reps.narrow_3D_rep import Narrow3DRepresentation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict
from gym_pcgrl.envs.helper_3D import gen_random_map
from gym_pcgrl.envs.probs.minecraft.mc_render import edit_3D_maze, edit_bordered_3D_maze, spawn_3D_bordered_map
from pdb import set_trace as TT

"""
The narrow representation where the agent is trying to modify the tile value of a certain
selected position that is selected randomly or sequentially similar to cellular automata
"""
class Narrow3DHoleyRepresentation(HoleyRepresentation3D, Narrow3DRepresentation):
    """
    Get the observation space used by the narrow representation

    Parameters:
        length: the current map length
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Dict: the observation space used by that representation. "pos" Integer
        x,y,z position for the current location. "map" 3D array of tile numbers
    """
    def get_observation_space(self, length, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([1, 1, 1]), high=np.array([length, width, height]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height+2, width+2, length+2))
        })

    def reset(self, *args, **kwargs):
        ret = Narrow3DRepresentation.reset(self, *args, **kwargs)
        HoleyRepresentation3D.reset(self)
        return ret   

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y,z position for the current location. "map" 3D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x+1, self._y+1, self._z+1], dtype=np.uint8),
            # "map": self._map.copy()
            "map": self._bordered_map.copy()
        })

    """
    Update the narrow representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        change += [0,1][self._map[self._z][self._y][self._x] != action]
        self._map[self._z][self._y][self._x] = action
        self._bordered_map[self._z+1][self._y+1][self._x+1] = action
        if self._random_tile:
            # If we've acted on all tiles, but the episode is not over, re-shuffle them and cycle through again.
            if self.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)
            self._x, self._y, self._z = self._act_coords[self.n_step % len(self._act_coords)]
            # self._x = self._random.randint(self._map.shape[2])
            # self._y = self._random.randint(self._map.shape[1])
            # self._z = self._random.randint(self._map.shape[0])

        else:
            self._x += 1
            if self._x >= self._map.shape[2]:
                self._x = 0
                self._y += 1
                if self._y >= self._map.shape[1]:
                    self._y = 0
                    self._z += 1
                    if self._z >= self._map.shape[0]:
                        self._z = 0
        self.n_step += 1
        self._new_coords = [self._x, self._y, self._z]
        return change, [self._x, self._y, self._z]
    
    def render(self, map, **kwargs):
        x, y, z = self._old_coords[0], self._old_coords[1], self._old_coords[2]
        # x, y, z = self._x, self._y, self._z
        edit_bordered_3D_maze(np.array(map), x, y, z, **kwargs)
        # spawn_3D_bordered_map(map)
        self._old_coords = self._new_coords

        return