from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.representation import Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
The narrow representation where the agent is trying to modify the tile value of a certain
selected position that is selected randomly or sequentially similar to cellular automata
"""
class NarrowHoleyRepresentation(NarrowRepresentation):
    """
    Get the observation space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Dict: the observation space used by that representation. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation_space(self, width, height, num_tiles):
        return spaces.Dict({
            "pos": spaces.Box(low=np.array([1, 1]), high=np.array([width, height]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height + 2, width + 2))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x+1, self._y+1], dtype=np.uint8),
            "map": self._bordered_map.copy()
            # "map": self._map.copy()
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
        if action > 0:
            change += [0,1][self._map[self._y][self._x] != action-1]
            self._map[self._y][self._x] = action-1
            self._bordered_map[self._y+1][self._x+1] = action-1
        if self._random_tile:
            if self.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)
        self._x, self._y = self._act_coords[self.n_step % len(self._act_coords)]
        self.n_step += 1
        return change, [self._x, self._y]

    """
    Modify the level image with a red rectangle around the tile that is
    going to be modified

    Parameters:
        lvl_image (img): the current level_image without modifications
        tile_size (int): the size of tiles in pixels used in the lvl_image
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level

    Returns:
        img: the modified level image
    """
    def render(self, lvl_image, tile_size, **kwargs):
        x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
        for x in range(tile_size):
            x_graphics.putpixel((0,x),(255,0,0,255))
            x_graphics.putpixel((1,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-2,x),(255,0,0,255))
            x_graphics.putpixel((tile_size-1,x),(255,0,0,255))
        for y in range(tile_size):
            x_graphics.putpixel((y,0),(255,0,0,255))
            x_graphics.putpixel((y,1),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-2),(255,0,0,255))
            x_graphics.putpixel((y,tile_size-1),(255,0,0,255))
        # lvl_image.paste(x_graphics, ((self._x+border_size[0])*tile_size, (self._y+border_size[1])*tile_size,
        #                                 (self._x+border_size[0]+1)*tile_size,(self._y+border_size[1]+1)*tile_size), x_graphics)
        lvl_image.paste(x_graphics, ((self._x+1)*tile_size, (self._y+1)*tile_size,
                                        (self._x+2)*tile_size,(self._y+2)*tile_size), x_graphics)
        return lvl_image
