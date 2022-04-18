import numpy as np

from gym_pcgrl.envs.probs.minecraft.minecraft_3D_maze_prob import Minecraft3DmazeProblem

"""
Generate a fully connected top down layout where the longest path is greater than a certain threshold
"""
class Minecraft3DmazeCtrlProblem(Minecraft3DmazeProblem):
    def __init__(self):
        super().__init__()
        n_floors = self._height // 3

        # Max path length involves having a zig-zag pattern on each floor, connected by a set of stairs.
        max_path_per_floor = np.ceil(self._width / 2) * (self._length) + np.floor(self._length/2)
        self._max_path_length = n_floors * max_path_per_floor

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
        }

        self._reward_weights = {"regions": 1, "path-length": 1}

    # We do these things in the ParamRew wrapper (note that max change and iterations

    def get_episode_over(self, new_stats, old_stats):
        """ If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)"""

        return False

    def get_reward(self, new_stats, old_stats):
        return None
