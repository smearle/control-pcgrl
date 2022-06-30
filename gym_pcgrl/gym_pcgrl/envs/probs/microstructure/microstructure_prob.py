import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import PROB_DIR, Problem
from gym_pcgrl.envs.helper import calc_tortuosity, get_range_reward, get_tile_locations, calc_num_regions, calc_longest_path


"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class MicroStructureProblem(Problem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 64
        self._height = 64
        self._prob = {"empty": 0.5, "solid":0.5}
        self._border_tile = "solid"

        self._random_probs = True

        self._reward_weights = {
            # nth moment?
#           "short_circuit_current",  # max amount of current-per-unit-area when applied voltage is zero
#           "fill_factor",  # max amount of power
            "tortuosity": 1,
        }
        

        self._max_path_length = np.ceil(self._width / 2) * (self._height) + np.floor(self._height/2)
        self._max_tortuosity = self._max_path_length / 2
        self._target_path = 20
        self.render_path = False
        self.path_coords = []
        self.path_length = None

        self._max_path_length = np.ceil(self._width / 2) * (self._height) + np.floor(self._height / 2)
        #       self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)

        # default conditional targets
        self.static_trgs = {
            "tortuosity": self._max_tortuosity,
        }

        # Upper and lower bounds for the values of conditional metrics.
        self.cond_bounds = {
            "path-length": (0, self._max_path_length),
            "tortuosity": (0, self._max_tortuosity),
        }


    """
    Get a list of all the different tile names

    Returns:`
        string[]: that contains all the tile names
    """
    def get_tile_types(self):
        return ["empty", "solid"]

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
        if self._random_probs:
            self._prob["empty"] = self._random.random()
            self._prob["solid"] = 1 - self._prob["empty"]

    """
    Get the current stats of the map

    Returns:
        dict(string,any): stats of the current map to be used in the reward, episode_over, debug_info calculations.
        The used status are "reigons": number of connected empty tiles, "F_abs": the longest path across the map
    """
    def get_stats(self, map, lenient_paths=False):
        map_locations = get_tile_locations(map, self.get_tile_types())
        # self.path_length, self.path_coords = calc_longest_path(map, map_locations, ["empty"], get_path=self.render_path)
        self.tortuosity, self.path_length, self.path_coords = calc_tortuosity(map, map_locations, ["empty"], get_path=self.render_path)
        m=np.array(map)
        emptiness= (m=='empty').sum()/m.size
        emptiness+= 1e-04
        return {
            # "regions": calc_num_regions(map, map_locations, ["empty"]),
            "path-length": self.path_length,
            # "tortuosity": self.path_length/emptiness,
            "tortuosity": self.tortuosity,
        }

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
        rewards = {}
        #calculate the total reward
        return 0

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
#       return new_stats["regions"] == 1 and new_stats["F_abs"] - self._start_stats["F_abs"] >= self._target_path
        return False

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
            "F_abs": new_stats["F_abs"],
            "path-imp": new_stats["F_abs"] - self._start_stats["F_abs"]
        }

    """
    Get an image on how the map will look like for a specific map

    Parameters:
        map (string[][]): the current game map

    Returns:
        Image: a pillow image on how the map will look like using the binary graphics
    """
    def render(self, map):
        if self._graphics == None:
            if self.GVGAI_SPRITES:
                self._graphics = {
                    "empty": Image.open(os.path.dirname(__file__) + "/sprites/oryx/floor3.png").convert('RGBA'),
                    "solid": Image.open(os.path.dirname(__file__) + "/sprites/oryx/wall3.png").convert('RGBA'),
                    "path" : Image.open(os.path.dirname(__file__) + "/sprites/newset/snowmanchest.png").convert('RGBA'),
                }
            else:
                self._graphics = {
                    "empty": Image.open(PROB_DIR + "/common/empty.png").convert('RGBA'),
                    "solid": Image.open(PROB_DIR + "/common/binary/solid.png").convert('RGBA'),
                    "path" : Image.open(PROB_DIR + "/common/path_g.png").convert('RGBA'),
                }
        return super().render(map, render_path=self.path_coords)


# def get_two_spatial(int_map, env):
#     int_map = np.expand_dims(int_map, axis=0)
#     data = PrimitiveTransformer(n_state=2, min_=0.0, max_=1.0).transform(int_map)
# #   plot_microstructures(
# #       data[0, :, :, 0],
# #       data[0, :, :, 1],
# #       titles=['First phase with ones', 'Second phase with ones'],
# #       cmap='gray',
# #       colorbar=False
# #   )

#     auto_correlation = TwoPointCorrelation(
#         periodic_boundary=True,
#         cutoff=25,
#         correlations=[(0, 0)]
#     ).transform(data)
#     mean_2nd_moment = np.asarray(auto_correlation.mean()).item()

#     return mean_2nd_moment