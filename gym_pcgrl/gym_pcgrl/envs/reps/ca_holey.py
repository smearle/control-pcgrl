from pdb import set_trace as TT

import cv2
from gym_pcgrl.envs.probs.minecraft.mc_render import spawn_3D_bordered_map, spawn_3D_maze
from gym_pcgrl.envs.reps.ca_rep import CARepresentation
from gym_pcgrl.envs.reps.holey_representation import HoleyRepresentation, StaticBuildRepresentation
from PIL import Image
from gym import spaces
import numpy as np

"""
The cellular (autamaton-like) representation, where the agent may change all tiles on the map at each step.
"""
class CARepresentationHoley(HoleyRepresentation, CARepresentation):
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
    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height+2, width+2))
        })

    def get_observation(self):
        return {
            "map": self._bordered_map.copy()
        }

    def reset(self, *args, **kwargs):
        ret = CARepresentation.reset(self, *args, **kwargs)
        HoleyRepresentation.reset(self)
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
        self._bordered_map[1:-1, 1:-1] = self._map
        return ret


class CARepresentationHoleyStatic(StaticBuildRepresentation, CARepresentationHoley):
    def __init__(self, *args, **kwargs):
        CARepresentationHoley.__init__(self, *args, **kwargs)
        StaticBuildRepresentation.__init__(self)

    def get_observation_space(self, width, height, num_tiles):
        obs_space_0 = CARepresentationHoley.get_observation_space(self, width, height, num_tiles)
        obs_space_1 = StaticBuildRepresentation.get_observation_space(self, width, height, num_tiles)
        obs_space_0.spaces.update(obs_space_1.spaces)
        return obs_space_0
        

    def reset(self, *args, **kwargs):
        ret = CARepresentationHoley.reset(self, *args, **kwargs)
        StaticBuildRepresentation.reset(self)
        return ret

    def update(self, action, **kwargs):
        old_state = self._bordered_map.copy()
        change, pos = CARepresentationHoley.update(self, action, **kwargs)
        new_state = self._bordered_map
        # assert not(np.all(old_state == new_state))
        self._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
        # print(self._bordered_map)
        self._map = self._bordered_map[1:-1, 1:-1]
        # FIXME: below is broken?? (false positives for change detection)
        # change = np.any(old_state != new_state)
        return change, pos

    def render(self, level_image, tile_size, border_size, **kwargs):
        # TODO: check for human/rgb mode
        img = CARepresentationHoley.render(self, level_image, tile_size, border_size, **kwargs)
        img = StaticBuildRepresentation.render(self, img, tile_size, border_size, **kwargs)
        return img