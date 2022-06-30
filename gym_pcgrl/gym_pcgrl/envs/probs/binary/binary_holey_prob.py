from gym_pcgrl.envs.probs.holey_prob import HoleyProblem
import numpy as np
import os
from pdb import set_trace as TT

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from gym_pcgrl.envs.probs.problem import PROB_DIR, Problem
from gym_pcgrl.envs.helper import get_path_coords, get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path, run_dijkstra
from gym_pcgrl.envs.probs.binary.binary_prob import BinaryProblem

class BinaryHoleyProblem(HoleyProblem, BinaryProblem):
    def __init__(self):
        BinaryProblem.__init__(self)
        HoleyProblem.__init__(self)

        self._reward_weights = {
            "regions": 100,
            "path-length": 0,
            "connected-path-length": 100,
            # "connectivity": 0,
            # "connectivity": self._width,
            # "connectivity": self._max_path_length,
        }

        self.static_trgs.update({
            # "connectivity": 1,
            "path-length": self._max_path_length + 2,
            "connected-path-length": self._max_path_length + 2,
        })

        # boundaries for conditional inputs/targets
        self.cond_bounds.update({
            # "connectivity": (0, 1),
            "path-length": (0, self._max_path_length + 2),
            "connected-path-length": (0, self._max_path_length + 2),
        })
       
        dummy_bordered_map = np.zeros((self._width + 2, self._height + 2), dtype=np.uint8)
        # Fill in the borders with ones
        dummy_bordered_map[0, 1:-1] = dummy_bordered_map[-1, 1:-1] = 1
        dummy_bordered_map[1:-1, 0] = dummy_bordered_map[1:-1, -1] = 1
        self._border_idxs = np.argwhere(dummy_bordered_map == 1)


    def adjust_param(self, **kwargs):
        super(BinaryProblem, self).adjust_param(**kwargs)
        self.fixed_holes = kwargs.get('fixed_holes') if 'fixed_holes' in kwargs else self.fixed_holes


    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map, lenient_paths=False):
        map_locations = get_tile_locations(map, self.get_tile_types())
        # self.path_length, self.path_coords = calc_longest_path(map, map_locations, ["empty"], get_path=self.render_path)
        dijkstra, _ = run_dijkstra(self.entrance_coords[1], self.entrance_coords[0], map, ["empty"])
        connected_path_length = dijkstra[self.exit_coords[0], self.exit_coords[1]]

        max_start_path = np.max(dijkstra)
        self.path_length = max_start_path

        # Give a consolation prize if start and end are NOT connected.
        if connected_path_length == -1:
            # connectivity_bonus = 0
            self.connected_path_length = 0
            self.connected_path_coords = []

        # Otherwise (holes are connected), give a bonus (to guarantee we beat the loser above), plus the actual path length.
        else:
            # connectivity_bonus = 1
            self.connected_path_length = connected_path_length
            self.connected_path_coords = get_path_coords(dijkstra, init_coords=(self.exit_coords[0], self.exit_coords[1]))

        # FIXME: This is a hack to prevent weird path coord list of [[0,0]]
        if max_start_path < 1:
            self.path_coords = []
        else:
            maxcoord = np.argwhere(dijkstra == max_start_path)[0]
            #                                                             y           x
            self.path_coords = get_path_coords(dijkstra, init_coords=(maxcoord[0], maxcoord[1]))

            assert not (self.connected_path_length == 0 and len(self.connected_path_coords) > 0)

        # print("Connected path length:", self.connected_path_length)
        # print("connected_path_coords:", self.connected_path_coords)
        return {
            "regions": calc_num_regions(map, map_locations, ["empty"]),
            "path-length": self.path_length,
            "connected-path-length": self.connected_path_length,
            # "connectivity": connectivity_bonus,
            # "path-coords": self.path_coords,
        }

    def process_observation(self, observation):
        if self.connected_path_coords == []:
            return observation
        observation['map'][self.connected_path_coords[:, 0], self.connected_path_coords[:, 1]] = self._path_idx
        return observation

    """
    Get the current game reward between two stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        float: the current reward due to the change between the old map stats and the new map stats
    """
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], 125, 125),
            "connected-path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], 125, 125),
            # "connectivity": get_range_reward(new_stats["connectivity"], old_stats["connectivity"], 1, 1),
        }
        #calculate the total reward
        return rewards["regions"] * self._reward_weights["regions"] +\
            rewards["path-length"] * self._reward_weights["path-length"] +\
            rewards["connected-path-length"] * self._reward_weights["connected-path-length"]
            # rewards["connectivity"] * self._reward_weights["connectivity"]


    """
    Uses the stats to check if the problem ended (episode_over) which means reached
    a satisfying quality based on the stats

    Parameters:
        new_stats (dict(string,any)): the new stats after taking an action
        old_stats (dict(string,any)): the old stats before taking an action

    Returns:
        boolean: True if the level reached satisfying quality based on the stats and False otherwise
    """
    def get_episode_over(self, new_stats, old_stats):
#       return new_stats["regions"] == 1 and new_stats["path-length"] - self._start_stats["path-length"] >= self._target_path
        return new_stats["regions"] == 1 and new_stats["path-length"] == self._max_path_length # and \
            # new_stats["connectivity"] == 1

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
        return {
            "regions": new_stats["regions"],
            "path-length": new_stats["path-length"],
            "connected-path-length": new_stats["connected-path-length"],
            # "path-imp": new_stats["path-length"] - self._start_stats["path-length"]
            # "connectivity": new_stats["connectivity"],
        }


    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the problem
        graphics or default grey scale colors
    """
    def render(self, map, render_path=None):
        if self._graphics == None:
            if self.GVGAI_SPRITES:
                self._graphics = {
                    "empty": Image.open(os.path.dirname(__file__) + "/sprites/oryx/floor3.png").convert('RGBA'),
                    "solid": Image.open(os.path.dirname(__file__) + "/sprites/oryx/wall3.png").convert('RGBA'),
                    "path" : Image.open(os.path.dirname(__file__) + "/sprites/newset/snowmanchest.png").convert('RGBA'),
                }
            else:
                self._graphics = {
                    "empty": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
                    "solid": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA'),
                    "path" : Image.open(PROB_DIR + "/common/path_g.png").convert('RGBA'),
                    "c_path" : Image.open(os.path.dirname(__file__) + "/binary/path_b.png").convert('RGBA'),
                }
        render_path=self.path_coords
        render_cnct_path=self.connected_path_coords
        # render_connected_path=self.connected_path_coords
        # render_path=self.connected_path_coords


        ### modified render function from Problem class below ###

        # full_width = len(map[0])+2*self._border_size[0]
        full_width = len(map[0])
        # full_height = len(map)+2*self._border_size[1]
        full_height = len(map)
        lvl_image = Image.new("RGBA", (full_width*self._tile_size, full_height*self._tile_size), (0,0,0,255))
        # Background floor everywhere
        for y in range(full_height):
            for x in range(full_width):
                lvl_image.paste(self._graphics['empty'], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        # # Borders
        # for y in range(full_height):
        #     for x in range(self._border_size[0]):
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        #         lvl_image.paste(self._graphics[self._border_tile], ((full_width-x-1)*self._tile_size, y*self._tile_size, (full_width-x)*self._tile_size, (y+1)*self._tile_size))
        # for x in range(full_width):
        #     for y in range(self._border_size[1]):
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size))
        #         lvl_image.paste(self._graphics[self._border_tile], (x*self._tile_size, (full_height-y-1)*self._tile_size, (x+1)*self._tile_size, (full_height-y)*self._tile_size))

        # Map tiles
        for y in range(len(map)):
            for x in range(len(map[y])):
                tile_image = self._graphics[map[y][x]]
                # lvl_image.paste(self._graphics[map[y][x]], ((x+self._border_size[0])*self._tile_size, (y+self._border_size[1])*self._tile_size, (x+self._border_size[0]+1)*self._tile_size, (y+self._border_size[1]+1)*self._tile_size), mask=tile_image)
                lvl_image.paste(self._graphics[map[y][x]], (x*self._tile_size, y*self._tile_size, (x+1)*self._tile_size, (y+1)*self._tile_size), mask=tile_image)

        # Path, if applicable
        # if render_path is not None and self.render_path:
        tile_graphics = self._graphics["path"]
        for (y, x) in render_path:
            # lvl_image.paste(tile_graphics, ((x + self._border_size[0]) * self._tile_size, (y + self._border_size[1]) * self._tile_size, (x + self._border_size[0] + 1) * self._tile_size, (y + self._border_size[1] + 1) * self._tile_size), mask=tile_graphics)
            lvl_image.paste(tile_graphics, (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size), mask=tile_graphics)
        tile_graphics = self._graphics["c_path"]
        for (y, x) in render_cnct_path:
            lvl_image.paste(tile_graphics, (x * self._tile_size, y * self._tile_size, (x + 1) * self._tile_size, (y + 1) * self._tile_size), mask=tile_graphics)
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