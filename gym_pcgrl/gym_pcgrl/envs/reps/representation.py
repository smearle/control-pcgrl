from abc import ABC
from pdb import set_trace as TT
from typing import List

from gym import spaces
from gym.utils import seeding
import numpy as np
from PIL import Image

from gym_pcgrl.envs import helper
from gym_pcgrl.envs.probs.problem import Problem


"""
The base class of all the representations
"""
class Representation(ABC):
    """
    The base constructor where all the representation variable are defined with default values
    """
    def __init__(self, border_tile_index=1, empty_tile_index=0):
        self._random_start: bool = True
        # self._map: List[List[int]] = None
        self._map: np.ndarray = None
        self._bordered_map: List[List[int]] = None
        self._old_map: List[List[int]] = None
        self._border_tile_index = border_tile_index
        self._empty_tile = empty_tile_index
        self._random_start: bool = True

        self.seed()

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int: the used seed (same as input if not None)
    """
    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    gen_random_map = helper.gen_random_map

    """
    Resets the current representation

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, dims: tuple, prob: Problem):
        self._bordered_map = np.empty(tuple([i + 2 for i in dims[::-1]]), dtype=np.int)
        self._bordered_map.fill(self._border_tile_index)
        if self._random_start or self._old_map is None:
            self._map = type(self).gen_random_map(self._random, dims, prob)
            self._old_map = self._map.copy()
        else:
            self._map = self._old_map.copy()
        self._update_bordered_map()

    """
    Adjust current representation parameter

    Parameters:
        random_start (boolean): if the system will restart with a new map or the previous map
    """
    def adjust_param(self, **kwargs):
        self._random_start = kwargs.get('random_start', self._random_start)

    """
    Gets the action space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ActionSpace: the action space used by that representation
    """
    def get_action_space(self, dims, num_tiles):
        raise NotImplementedError('get_action_space is not implemented')

    """
    Get the observation space used by the representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        ObservationSpace: the observation space used by that representation
    """
    def get_observation_space(self, dims, num_tiles):
        return spaces.Dict({
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=dims)
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. A 3D array of tile numbers
    """

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment
    """
    def get_observation(self):
        return {
            "map": self._map.copy()
        }

    """
    Update the representation with the current action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action):
        return self._update_bordered_map()
        # raise NotImplementedError('update is not implemented')

    """
    Modify the level image with any special modification based on the representation

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size=None):
        return lvl_image

    def _update_bordered_map(self):
        # self._bordered_map[1:-1, 1:-1] = self._map
        self.unwrapped._bordered_map[tuple([slice(1, -1) for _ in range(len(self._map.shape))])] = self.unwrapped._map

    @property
    def unwrapped(self):
        """Completely unwrap this env.

        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self


class EgocentricRepresentation(Representation):
    """Representation in which the generator-agent occupies a particular position, i.e. (x, y) coordinate, on the map
    at each step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Whether the agent begins on a random tile.
        self._random_tile: bool = False
        # An x, y, (z) position
        self._pos: np.ndarray = None

    """
    Resets the current representation where it resets the parent and the current
    turtle location

    Parameters:
        length (int): the generated map length
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, dims, prob):
        super().reset(dims, prob)
        # TODO: Remove this?
        self._new_coords = self._pos
        self._old_coords = self._pos

    def get_observation(self):
        obs = super().get_observation()
        obs.update({
            'pos': np.array(self._pos),
        })
        return obs

    def get_observation_space(self, dims, num_tiles):
        obs_space = super().get_observation_space(dims, num_tiles)
        obs_space.spaces.update({
            "pos": spaces.Box(low=np.array([0 for i in dims]), high=np.array([i-1 for i in dims]), dtype=np.uint8),
        })
        return obs_space

    """
    Modify the level image with a white rectangle around the tile that the egocentric agent is on.

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, border_size):
        y, x = self._pos
        im_arr = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        clr = (255, 255, 255, 255)
        im_arr[(0, 1, -1, -2), :, :] = im_arr[:, (0, 1, -1, -2), :] = clr
        x_graphics = Image.fromarray(im_arr)
        lvl_image.paste(x_graphics, ((x+border_size[0])*tile_size, (y+border_size[1])*tile_size,
                                        (x+border_size[0]+1)*tile_size,(y+border_size[1]+1)*tile_size), x_graphics)
        return lvl_image
