from pdb import set_trace as TT
from typing import List

import numpy as np

from gym.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map
from gym_pcgrl.envs.reps.representation import Representation


class HoleyRepresentation(Representation):
    def reset(self, *args, **kwargs):
        # ret = super().reset(*args, **kwargs)
        self.dig_holes(self.entrance_coords, self.exit_coords)

    def set_holes(self, entrance_coords, exit_coords):
        self.entrance_coords, self.exit_coords = entrance_coords, exit_coords

    def dig_holes(self, entrance_coords, exit_coords):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self._bordered_map[entrance_coords[0], entrance_coords[1]] = self._empty_tile
        self._bordered_map[exit_coords[0], exit_coords[1]] = self._empty_tile
        