import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_prob import Minecraft3DmazeProblem

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DmazeCtrlProblem(Minecraft3DmazeProblem):
    def __init__(self):
        super().__init__()
        self._max_path_length = (np.ceil(
            self._width / 2) * (self._length) + np.floor(self._length/2)) * np.floor(self._height/2) + np.floor(self._height/2)

#       self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)

        # default conditional targets
        self.static_trgs = {"regions": 1, "path-length": self._max_path_length}

        # boundaries for conditional inputs/targets
        self.cond_bounds = {
            # Upper bound: checkerboard
            "regions": (0, np.ceil(self._width * self._length / 2 * self._height)),
            #     10101010
            #     01010101
            #     10101010
            #     01010101
            #     10101010
            # FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
            # and "snake" along that one
            # Upper bound: zig-zag
            "path-length": (0, self._max_path_length),
            #   11111111
            #   00000001
            #   11111111
            #   10000000
            #   11111111
        }

        self.weights = {"regions": 1, "path-length": 1}

    # We do these things in the ParamRew wrapper (note that max change and iterations

    def get_episode_over(self, new_stats, old_stats):
        """ If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)"""

        return False

    def get_reward(self, new_stats, old_stats):
        return None
