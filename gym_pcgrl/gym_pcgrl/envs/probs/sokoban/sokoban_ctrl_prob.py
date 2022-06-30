from pdb import set_trace as T
from collections import OrderedDict

import numpy as np

from gym_pcgrl.envs.probs.sokoban.sokoban_prob import SokobanProblem


class SokobanCtrlProblem(SokobanProblem):
    def __init__(self):
        super(SokobanCtrlProblem, self).__init__()
        self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)
        # like _reward_weights but for use with ParamRew
        #       self._reward_weights = self._reward_weights
        self._reward_weights = {
            "player": 3,
            "crate": 1,
#           "target": 1,
            "regions": 5,
            "ratio": 2,
            "dist-win": 0.0,
            "sol-length": 1,
        }
        self._ctrl_reward_weights = self._reward_weights

        self.static_trgs = {
            "player": 1,
            "crate": (2, self._max_crates),
#           "target": (1, self._max_crates),
            "regions": 1,
            "ratio": 0,
            "dist-win": 0,
            "sol-length": self._max_path_length,
        }

        # boundaries for conditional inputs/targets
        self.cond_bounds = {
            "player": (1, self._width * self._height),
            "crate": (
                1,
                self._width * self._height / 2 - max(self._width, self._height),
            ),
#           "target": (1, self._width * self._height),
            "ratio": (0, self._width * self._height),
            "dist-win": (0, self._width * self._height * (self._width + self._height)),
            "sol-length": (0, 2 * self._max_path_length),
            "regions": (0, self._width * self._height / 2),
        }

    # We do these things in the ParamRew wrapper
    def get_episode_over(self, new_stats, old_stats):
        return False

    def get_reward(self, new_stats, old_stats):
        return None

    def get_stats(self, map):
        stats = super().get_stats(map)
        stats["sol-length"] = len(stats["solution"])
        stats["ratio"] = abs(stats["crate"] - stats["target"])
        #       if stats['dist-win'] == self._width * self._height * (self._width + self._height):
        #           stats['dist-win'] = 0

        return stats
