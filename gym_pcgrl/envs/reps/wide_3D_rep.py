from gym_pcgrl.envs.reps.representation3D import Representation3D
from PIL import Image
from gym import spaces
import numpy as np
from gym_pcgrl.envs.probs.minecraft.mc_render import reps_3D_render

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""
class Wide3DRepresentation(Representation3D):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self):
        super().__init__()

    """
    Gets the action space used by the wide representation

    Parameters:
        length: the current map length
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        MultiDiscrete: the action space used by that wide representation which
        consists of the x position, y position, z position and the tile value
    """
    def get_action_space(self, length, width, height, num_tiles):
        return spaces.MultiDiscrete([length, width, height, num_tiles])

    """
    Get the observation space used by the wide representation

    Parameters:
        length: the current map length
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Box: the observation space used by that representation. A 3D array of tile numbers
    """
    def get_observation_space(self, length, width, height, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width, length))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 3D array of tile numbers
    """
    def get_observation(self):
        return {
            "map": self._map.copy()
        }

    """
    Update the wide representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        change = [0,1][self._map[action[2]][action[1]][action[0]] != action[3]]
        self._map[action[2]][action[1]][action[0]] = action[3]
        return change, action[0], action[1], action[2]
