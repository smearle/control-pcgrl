from pdb import set_trace as TT
from gym_pcgrl.envs.probs.minecraft.mc_render import spawn_3D_bordered_map, spawn_3D_maze
from gym_pcgrl.envs.reps.ca_3D_rep import CA3DRepresentation
from gym_pcgrl.envs.reps.holey_representation_3D import HoleyRepresentation3D
from gym_pcgrl.envs.reps.representation_3D import Representation3D
from PIL import Image
from gym import spaces
import numpy as np

"""
The cellular (autamaton-like) representation, where the agent may change all tiles on the map at each step.
"""
class CA3DRepresentationHoley(HoleyRepresentation3D, CA3DRepresentation):
    """
    Get the observation space used by the cellular representation

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

    def get_observation(self):
        return {
            "map": self._bordered_map.copy()
        }

    def reset(self, *args, **kwargs):
        ret = CA3DRepresentation.reset(self, *args, **kwargs)
        HoleyRepresentation3D.reset(self)
        return ret   

    """
    Update the cellular representation with the input action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action, continuous=False):
        ret = super().update(action, continuous)
        self._bordered_map[1:-1, 1:-1, 1:-1] = self._map
        return ret

    def render(self, map):
        spawn_3D_bordered_map(map)
    