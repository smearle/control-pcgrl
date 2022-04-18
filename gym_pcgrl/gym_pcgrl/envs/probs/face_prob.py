import os
from functools import reduce

import numpy as np
from operator import mul
from PIL import Image, ImageDraw, ImageFont
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path
from pdb import set_trace as TT


"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class FaceProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()


        self._width = 32
        self._height = 32
#       font_size = 32
#       try:
#           font = ImageFont.truetype("arial.ttf", font_size)
#       except OSError:
#           try:
#               font = ImageFont.truetype("LiberationMono-Regular.ttf", font_size)
#           except OSError:
#               font = ImageFont.truetype("SFNSMono.ttf", font_size)
#       trg_image = Image.new(mode="RGB", size=(16, 16))
#       draw = ImageDraw.Draw(trg_image)
#       draw.text((1, 1), "A", font=font, fill=(255, 0, 0))
#       trg_image.save("trg_img.png")
#       self.face_np = np.array(trg_image)


        with Image.open("gym_pcgrl/envs/probs/face/lena.jpeg") as im:
#           im.show()
            im = im.resize((self._width, self._height))
            self.face_np = np.array(im)
#           im.show()
        im.save('face_trg.png')
#       self.face_np = self.face_np.transpose(2, 0, 1)

#       self._prob = {"empty": 0.5, "solid":0.5}
        self._border_tile = "solid"

        self._target_path = 20
        self._random_probs = True

        self._reward_weights = {
            "face_1": 1,
        }

        # default conditional targets
        self.static_trgs = {
            "face_1": 0,
        }
        # boundaries for conditional inputs/targets
        self.cond_bounds = {
            "face_1": (0, 1),
        }

        self._reward_weights = {"face_1": 1}

    """
    Get a list of all the different tile names

    Returns:`
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ['r','g','b']

    def is_continuous(self):
        return True

    """
    Adjust the parameters for the current problem

    Parameters:
        width (int): change the width of the problem level
        height (int): change the height of the problem level
        probs (dict(string, float)): change the probability of each tile
        intiialization, the names are "empty", "solid"
        target_path (int): the current path length that the episode turn when it reaches
        rewards (dict(string,float)): the weights of each reward change between the new_stats and old_stats
    """
    def adjust_param(self, **kwargs):
        self.render_path = kwargs.get('render', self.render_path) or kwargs.get('render_path', self.render_path)
        super().adjust_param(**kwargs)

        self._target_path = kwargs.get('target_path', self._target_path)
        self._random_probs = kwargs.get('random_probs', self._random_probs)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._reward_weights:
                    self._reward_weights[t] = rewards[t]

    """
    Resets the problem to the initial state and save the start_stats from the starting map.
    Also, it can be used to change values between different environment resets

    Parameters:
        start_stats (dict(string,any)): the first stats of the map
    """
    def reset(self, start_stats):
        super().reset(start_stats)

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "path-length": the longest path across the map
    """
    def get_stats(self, map, lenient_paths=False):
#       map_locations = get_tile_locations(map, self.get_tile_types())
#       self.path_length, self.path_coords = calc_longest_path(map, map_locations, ["empty"], get_path=self.render_path)
 #      return {
 #          "regions": calc_num_regions(map, map_locations, ["empty"]),
 #          "path-length": self.path_length,
 #      }
        stats = {
            "face_1": np.sum(np.abs(self.face_np.transpose(2, 0, 1)/255 - map)) / reduce(mul, map.shape),
        }
        return stats

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
            "face_1": get_range_reward(new_stats["face_1"], old_stats["face_1"], 1, 1),
        }
        #calculate the total reward
        return rewards["face_1"] * self._reward_weights["face_1"]

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
        return new_stats["face_1"] == 1

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
            "face_1": new_stats["face_1"] - self._start_stats["face_1"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        # FIXME: this seems maaaaad inefficient no?
        map = map.transpose(1, 2, 0)
#       map = self.face_np.transpose(1, 2, 0)
        return Image.fromarray((map*255).astype(np.uint8), 'RGB')
#   def render(self, map):
#       if self._graphics == None:
#           if self.GVGAI_SPRITES:
#               self._graphics = {
#                   "empty": Image.open(os.path.dirname(__file__) + "/sprites/oryx/floor3.png").convert('RGBA'),
#                   "solid": Image.open(os.path.dirname(__file__) + "/sprites/oryx/wall3.png").convert('RGBA'),
#                   "path" : Image.open(os.path.dirname(__file__) + "/sprites/newset/snowmanchest.png").convert('RGBA'),
#               }
#           else:
#               self._graphics = {
#                   "empty": Image.open(os.path.dirname(__file__) + "/binary/empty.png").convert('RGBA'),
#                   "solid": Image.open(os.path.dirname(__file__) + "/binary/solid.png").convert('RGBA'),
#                   "path" : Image.open(os.path.dirname(__file__) + "/binary/path_g.png").convert('RGBA'),
#               }
#       return super().render(map, render_path=self.path_coords)
