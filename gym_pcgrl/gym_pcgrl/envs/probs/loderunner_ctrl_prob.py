from pdb import set_trace as TT
import numpy as np

from gym_pcgrl.envs.helper import (
    calc_certain_tile,
    calc_num_regions,
    get_range_reward,
    get_tile_locations,
    run_dijkstra,
    get_path_coords,
)
from gym_pcgrl.envs.probs.loderunner_prob import LoderunnerProblem


class LoderunnerCtrlProblem(LoderunnerProblem):
    def __init__(self):
        super(LoderunnerCtrlProblem, self).__init__()
        # TODO: Do not assume it's a square
        # Twice the optimal zig-zag minus one for the end-point at which the player turns around
        self._max_path_length = (np.ceil(self._width / 2) * (self._height) + np.floor(self._height / 2)) * 2 - 1
#       self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)
        # like "_reward_weights" but for use with ParamRew
        self._reward_weights = self._reward_weights

        self.static_trgs = {
            "player": 1,
            "enemies": 2,
            "gold": (1, 10),
            "win": 1,
            "path-length": self._max_path_length,
        }
        # conditional inputs/targets ( just a default we don't use in the ParamRew wrapper)
        self.cond_trgs = self.static_trgs

        max_n_tile = self._height * self._width
        # boundaries for conditional inputs/targets
        self.cond_bounds = {
            "player": (0, max_n_tile),
            "enemies": (0, max_n_tile),
#           "gold": (0, max_n_tile),
            "gold": (0, 10),
            "win": (0, 1),
            "path-length": (0, self._max_path_length),
        }

    # We do these things in the ParamRew wrapper
    def get_episode_over(self, new_stats, old_stats):
        return False

    def get_reward(self, new_stats, old_stats):
        return None
