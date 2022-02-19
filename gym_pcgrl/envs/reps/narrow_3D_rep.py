from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict
from gym_pcgrl.envs.helper_3D import gen_random_map
from gym_pcgrl.envs.probs.minecraft.mc_render import reps_3D_render

"""
The narrow representation where the agent is trying to modify the tile value of a certain
selected position that is selected randomly or sequentially similar to cellular automata
"""
class Narrow3DRepresentation(Representation3D):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()
        self._random_tile = True

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        length (int): the generated map length
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, length, width, height, prob):
        if self._random_start or self._old_map is None:
            self._map = gen_random_map(self._random, length, width, height, prob)
            self._old_map = self._map.copy()
        else:
            self._map = self._old_map.copy()
        self._x = self._random.randint(length)
        self._y = self._random.randint(width)
        self._z = self._random.randint(height)

    """
    Gets the action space used by the narrow representation

    Parameters:
        length: the current map length
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Discrete: the action space used by that narrow representation which
        correspond to which value for each tile type
    """
    def get_action_space(self, length, width, height, num_tiles):
        return spaces.Discrete(num_tiles + 1)

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
            "pos": spaces.Box(low=np.array([0, 0, 0]), high=np.array([length-1, width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width, length))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y,z position for the current location. "map" 3D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y, self._z], dtype=np.uint8),
            "map": self._map.copy()
        })

    """
    Adjust the current used parameters

    Parameters:
        random_start (boolean): if the system will restart with a new map (true) or the previous map (false)
        random_tile (boolean): if the system will move between tiles random (true) or sequentially (false)
    """
    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)
        self._random_tile = kwargs.get('random_tile', self._random_tile)

    """
    Update the narrow representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = 0
        if action > 0:
            change += [0,1][self._map[self._z][self._y][self._x] != action-1]
            self._map[self._z][self._y][self._x] = action-1
        if self._random_tile:
            self._x = self._random.randint(self._map.shape[2])
            self._y = self._random.randint(self._map.shape[1])
            self._z = self._random.randint(self._map.shape[0])

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
        return change, self._x, self._y, self._z

    def render(self, map):
        return reps_3D_render(map, self._x, self._y, self._z)
