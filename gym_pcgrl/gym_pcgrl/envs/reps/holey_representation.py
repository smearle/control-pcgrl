from pdb import set_trace as TT
from typing import List

import numpy as np

from gym.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map
from gym_pcgrl.envs.reps.representation import Representation


class HoleyRepresentation(Representation):
    def reset(self, *args, **kwargs):
        # ret = super().reset(*args, **kwargs)
        self.dig_holes(self.start_xy, self.end_xy)

    def set_holes(self, start_xy, end_xy):
        self.start_xy, self.end_xy = start_xy, end_xy

    def dig_holes(self, start_xy, end_xy):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self._bordered_map[start_xy[0], start_xy[1]] = self._empty_tile
        self._bordered_map[end_xy[0], end_xy[1]] = self._empty_tile
        