from abc import ABC
from pathlib import Path
from pdb import set_trace as TT
from gym.utils import seeding
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

GVGAI_SPRITES = False
PROB_DIR = str(Path(__file__).parent)  # for convenience when loading sprite .pngs

"""
The base class for all the problems that can be handled by the interface

map in prob are list of strings
"""
class Problem(ABC):
    """
    Constructor for the problem that initialize all the basic parameters
    """
    def __init__(self):
        self._width = 9
        self._height = 9
        tiles = self.get_tile_types()

        # FIXME: assumption, will overrule a similar declaration by the child.
        self._empty_tile = tiles[0]

        self._prob = {}
        for tile in range(len(tiles)):
            self._prob.update({tile: 1.0 / len(tiles)})

        self._border_size = (1, 1)
        self._border_tile = tiles[0]
        if GVGAI_SPRITES:
            self._tile_size = 24
            self.GVGAI_SPRITES = True
        else:
            self._tile_size = 16
            self.GVGAI_SPRITES = False
        self._graphics = None
        self.render_path = False
        self.path_to_erase = set({})  # FIXME: only 3D really needs this.

    # def gen_holes(self):
        # return None

    def is_continuous(self):
        return False

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

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        self._start_stats = start_stats

    """
    Get a list of all the different tile names

    Returns:
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        raise NotImplementedError('get_tile_types is not implemented')

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are the same as the tile types from get_tile_types
    """
    def adjust_param(self, **kwargs):
        self._width, self._height = kwargs.get('width', self._width), kwargs.get('height', self._height)
        prob = kwargs.get('probs')
        if prob is not None:
            for t in prob:
                if t in self._prob:
                    self._prob[t] = prob[t]
        if self.render_path:
            # Make room for displaying path length
            self._border_size = (1, 2)
        else:
            self._border_size = (1, 1)

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations
    """
    def get_stats(self, map, **kwargs):
        raise NotImplementedError('get_graphics is not implemented')

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    # def get_reward(self, new_stats, old_stats):
    #     raise NotImplementedError('get_reward is not implemented')

    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    # def get_episode_over(self, new_stats, old_stats):
    #     raise NotImplementedError('get_graphics is not implemented')

    """
    Get any debug information need to be printed

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        dict(any,any): is a debug information that can be used to debug what is
        happening in the problem
    """
    def get_debug_info(self, new_stats, old_stats):
        raise NotImplementedError('get_debug_info is not implemented')

    def process_observation(self, observation):
        return observation

    def get_observable_tile_types(self):
        return self.get_tile_types()

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the problem
        graphics or default grey scale colors
    """
    def render(self, map, render_path=None):
        tiles = self.get_tile_types()
        if self._graphics == None:
            self._graphics = {}
            for i in range(len(tiles)):
                color = (i*255/len(tiles),i*255/len(tiles),i*255/len(tiles),255)
                self._graphics[tiles[i]] = Image.new("RGBA",(self._tile_size,self._tile_size),color)
            if render_path:
                self._graphics["path"] = Image.new("RGBA", (self._tile_size, self._tile_size), color)

        full_width = len(map[0])+2*self._border_size[0]
        full_height = len(map)+2*self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,255))
        # Background floor everywhere
        for y in range(full_height):
            for x in range(full_width):
                lvl_image.paste(self._graphics['empty'], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        # Borders
        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))
        # Map tiles
        for y in range(len(map)):
            for x in range(len(map[y])):
                tile_image = self._graphics[map[y][x]]
                lvl_image.paste(self._graphics[map[y][x]], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size), mask=tile_image)

        # Path, if applicable
        if render_path is not None and self.render_path:
            tile_graphics = self._graphics["path"]
            for (y, x) in render_path:
                lvl_image.paste(tile_graphics, ((x + self._border_size[0]) * self._tile_size, (y + self._border_size[1]) * self._tile_size, (x + self._border_size[0] + 1) * self._tile_size, (y + self._border_size[1] + 1) * self._tile_size), mask=tile_graphics)
            draw = ImageDraw.Draw(lvl_image)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font_size = 32
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                try:
                    font = ImageFont.truetype("LiberationMono-Regular.ttf", font_size)
                except OSError:
                    font = ImageFont.truetype("SFNSMono.ttf", 32)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text(((full_width - 1) * self._tile_size / 2, 0),"{}".format(self.path_length),(255,255,255),font=font)
        return lvl_image

    def get_episode_over(self, new_stats, old_stats):
        """ If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)"""

        return False

    def get_reward(self, new_stats, old_stats):
        return None


class Problem3D(Problem): pass