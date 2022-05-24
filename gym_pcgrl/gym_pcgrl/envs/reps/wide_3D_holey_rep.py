from pdb import set_trace as TT
from PIL import Image
from gym import spaces
from gym_pcgrl.envs.reps.wide_3D_rep import Wide3DRepresentation
import numpy as np
from gym_pcgrl.envs.probs.minecraft.mc_render import edit_3D_maze

"""
The wide representation where the agent can pick the tile position and tile value at each update.
"""
class Wide3DHoleyRepresentation(Wide3DRepresentation):
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
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height+2, width+2, length+2))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 3D array of tile numbers
    """
    def get_observation(self):
        return {
            # "map": self._map.copy()
            "map": self._bordered_map.copy()
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
        self._bordered_map[action[2]+1][action[1]+1][action[0]+1] = action[3]

        self._new_coords = [action[0], action[1], action[2]]

        return change, [action[0], action[1], action[2]]
