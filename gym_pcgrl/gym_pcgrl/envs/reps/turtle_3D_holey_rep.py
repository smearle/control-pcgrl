from PIL import Image
from gym import spaces
from gym_pcgrl.envs.reps.turtle_3D_rep import Turtle3DRepresentation
import numpy as np
from collections import OrderedDict
from gym_pcgrl.envs.probs.minecraft.mc_render import edit_3D_maze

"""
The turtle representation where the agent is trying to modify the position of the
turtle or the tile value of its current location similar to turtle graphics.
The difference with narrow representation is the agent now controls the next tile to be modified.
"""
class Turtle3DHoleyRepresentation(Turtle3DRepresentation):
    """
    Get the observation space used by the turtle representation

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
    Update the turtle representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        if action < len(self._dirs):
            self._x += self._dirs[action][0]
            if self._x < 0:
                if self._warp:
                    self._x += self._map.shape[2]
                else:
                    self._x = 0
            if self._x >= self._map.shape[2]:
                if self._warp:
                    self._x -= self._map.shape[2]
                else:
                    self._x = self._map.shape[2] - 1

            self._y += self._dirs[action][1]
            if self._y < 0:
                if self._warp:
                    self._y += self._map.shape[1]
                else:
                    self._y = 0
            if self._y >= self._map.shape[1]:
                if self._warp:
                    self._y -= self._map.shape[1]
                else:
                    self._y = self._map.shape[1] - 1

            self._z += self._dirs[action][2]
            if self._z < 0:
                if self._warp:
                    self._z += self._map.shape[0]
                else:
                    self._z = 0
            if self._z >= self._map.shape[0]:
                if self._warp:
                    self._z -= self._map.shape[0]
                else:
                    self._z = self._map.shape[0] - 1
        else:
            change = [0,1][self._map[self._z][self._y][self._x] != action - len(self._dirs)]
            self._map[self._z][self._y][self._x] = action - len(self._dirs)
            self._bordered_map[self._z+1][self._y+1][self._x+1] = action - len(self._dirs)
        self._new_coords = [self._x, self._y, self._z]
        return change, [self._x, self._y, self._z]
