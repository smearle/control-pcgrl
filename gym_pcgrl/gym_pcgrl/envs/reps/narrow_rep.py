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
        self._random_tile = False
        self._act_coords = None
        self.n_step = 0

    """
    Get a list of (x, y) coordinates corresponding to coordinates of tiles to be edited by the generator-agent.
    """
    def get_act_coords(self):
        act_coords = np.meshgrid(np.arange(self._map.shape[1]), np.arange(self._map.shape[0]))
        act_coords = np.reshape(np.stack(act_coords, axis=-1), (-1, 2))
        return act_coords

    """
    Resets the current representation where it resets the parent and the current
    modified location

    Parameters:
        width (int): the generated map width
        height (int): the generated map height
        prob (dict(int,float)): the probability distribution of each tile value
    """
    def reset(self, width, height, prob):
        super().reset(width, height, prob)
        self.n_step = 0
        if self._act_coords is None:
            self._act_coords = self.get_act_coords()
        if self._random_tile:
            np.random.shuffle(self._act_coords)

        self._x, self._y = self._act_coords[self.n_step]

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
    def get_action_space(self, width, height, num_tiles):
        return spaces.Discrete(num_tiles + 1)

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
            "pos": spaces.Box(low=np.array([0, 0]), high=np.array([width-1, height-1]), dtype=np.uint8),
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=(height, width))
        })

    """
    Get the current representation observation object at the current moment

    Returns:
        observation: the current observation at the current moment. "pos" Integer
        x,y position for the current location. "map" 2D array of tile numbers
    """
    def get_observation(self):
        return OrderedDict({
            "pos": np.array([self._x, self._y], dtype=np.uint8),
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
            change += [0,1][self._map[self._y][self._x] != action-1]
            self._map[self._y][self._x] = action-1
        if self._random_tile:
#           self._x = self._random.randint(self._map.shape[1])
#           self._y = self._random.randint(self._map.shape[0])
            if self.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)
        self._x, self._y = self._act_coords[self.n_step % len(self._act_coords)]
        # else:
        #     self._x += 1
        #     if self._x >= self._map.shape[1]:
        #         self._x = 0
        #         self._y += 1
        #         if self._y >= self._map.shape[0]:
        #             self._y = 0
        self.n_step += 1
        return change, [self._x, self._y]

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