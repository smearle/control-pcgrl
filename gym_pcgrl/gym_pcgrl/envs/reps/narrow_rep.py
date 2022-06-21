from pdb import set_trace as TT
from gym_pcgrl.envs.reps.representation import EgocentricRepresentation, Representation
from PIL import Image
from gym import spaces
import numpy as np
from collections import OrderedDict

"""
The narrow representation where the agent is trying to modify the tile value of a certain
selected position that is selected randomly or sequentially similar to cellular automata
"""
class NarrowRepresentation(EgocentricRepresentation):
    """
    Initialize all the parameters used by that representation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._act_coords = None
        self.n_step = 0

    """
    Get a list of (y, x) or (z, y, x) coordinates corresponding to coordinates of tiles to be edited by the 
    generator-agent. Dimension ignored.
    """
    def get_act_coords(self):
        act_coords = np.meshgrid(*tuple([np.arange(s) for s in self._map.shape]))
        # Flatten so that we can treat this like a list of coordinates.
        act_coords = np.reshape(np.stack(act_coords, axis=-1), (-1, len(self._map.shape)))
        return act_coords

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, dims, prob):
        super().reset(dims, prob)
        self.n_step = 0
        if self._act_coords is None:
            self._act_coords = self.get_act_coords()
        if self._random_tile:
            np.random.shuffle(self._act_coords)

        # self._x, self._y = self._act_coords[self.n_step]
        self._pos = self._act_coords[self.n_step]

    """
    Gets the action space used by the narrow representation

    Parameters:
        width: the current map width
        height: the current map height
        num_tiles: the total number of the tile values

    Returns:
        Discrete: the action space used by that narrow representation which
        correspond to which value for each tile type
    """
    def get_action_space(self, dims, num_tiles):
        return spaces.Discrete(num_tiles + 1)


    """
    Adjust the current used parameters

    Parameters:
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
            change += [0,1][self._map[tuple(self._pos)] != action-1]
            self._map[tuple(self._pos)] = action-1
        if self._random_tile:
            if self.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)
        self._pos = self._act_coords[self.n_step % len(self._act_coords)]
        self.n_step += 1
        super().update(action)
        return change, self._pos

    # """
    # Modify the level image with a red rectangle around the tile that is
    # going to be modified

    # Parameters:
    #     lvl_image (img): the current level_image without modifications
    #     tile_size (int): the size of tiles in pixels used in the lvl_image
    #     border_size ((int,int)): an offeset in tiles if the borders are not part of the level
        
    # Returns:
    #     img: the modified level image
    # """
    # def render(self, lvl_image, tile_size, border_size):
    #     x_graphics = Image.new("RGBA", (tile_size,tile_size), (0,0,0,0))
    #     clr = (0, 255, 0, 255)
    #     for x in range(tile_size):
    #         x_graphics.putpixel((0,x),clr)
    #         x_graphics.putpixel((1,x),clr)
    #         x_graphics.putpixel((tile_size-2,x),clr)
    #         x_graphics.putpixel((tile_size-1,x),clr)
    #     for y in range(tile_size):
    #         x_graphics.putpixel((y,0),clr)
    #         x_graphics.putpixel((y,1),clr)
    #         x_graphics.putpixel((y,tile_size-2),clr)
    #         x_graphics.putpixel((y,tile_size-1),clr)
    #     lvl_image.paste(x_graphics, ((self._x+border_size[0])*tile_size, (self._y+border_size[1])*tile_size,
    #                                     (self._x+border_size[0]+1)*tile_size,(self._y+border_size[1]+1)*tile_size), x_graphics)
    #     return lvl_image