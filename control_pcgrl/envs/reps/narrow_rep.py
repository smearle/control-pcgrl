from pdb import set_trace as TT
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.reps.representation import EgocentricRepresentation, Representation
from PIL import Image
from gymnasium import spaces
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
        self._positions = []

    """
    Get a list of (y:height, x:width) or (z:height, y:width, x:length) coordinates corresponding to coordinates of tiles to be edited by the 
    generator-agent. Dimension ignored.
    """
    def get_act_coords(self):
        # act_coords = np.argwhere(np.ones(self._map.shape))
        act_coords = super().get_valid_agent_coords()
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
        self._positions = [self._pos]

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
        # FIXME: For backward compatibility only!
        # return spaces.Discrete(num_tiles + 1)
        return spaces.Discrete(num_tiles)


    """
    Adjust the current used parameters

    Parameters:
        random_tile (boolean): if the system will move between tiles random (true) or sequentially (false)
    """
    def adjust_param(self, cfg: Config):
        super().adjust_param(cfg=cfg)
        self._act_coords = self.get_act_coords()

    """
    Update the narrow representation with the input action
    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        boolean: True if the action change the map, False if nothing changed
    """
    def update(self, action, **kwargs):
        #FIXME: Use the `pos` provided as argument
        change = 0
        # if action > 0:
        change += [0,1][self._map[tuple(self._pos)] != action]
        self._map[tuple(self._pos)] = action
        if self._random_tile:
            if self.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)
        self._pos = self._act_coords[self.n_step % len(self._act_coords)]
        self._positions = [self._pos]  # In case cfg.show_agents is True in single-player setting.
        self.n_step += 1
        super().update(action)
        return change, self._pos
    
    def update_state(self, action):        # ZJ: why do we need this?
        return self.update(action)

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


    def get_pos_at_step(self, step):
        return self._act_coords[step % len(self._act_coords)]