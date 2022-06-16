from pdb import set_trace as TT
from typing import List
from gym_pcgrl.envs.reps.representation_3D import Representation3D

import numpy as np

from gym.utils import seeding
from gym_pcgrl.envs.helper import gen_random_map


class HoleyRepresentation3D(Representation3D):
    def reset(self, *args, **kwargs):
        self.dig_holes(self.entrance_coords, self.exit_coords)

    def set_holes(self, entrance_coords, exit_coords):
        self.entrance_coords, self.exit_coords = entrance_coords, exit_coords

    def dig_holes(self, s, e):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self._bordered_map[s[0][0]][s[0][1]][s[0][2]] = self._bordered_map[s[1][0]][s[1][1]][s[1][2]] = self._empty_tile
        self._bordered_map[e[0][0]][e[0][1]][e[0][2]] = self._bordered_map[e[1][0]][e[1][1]][e[1][2]] = self._empty_tile