from pdb import set_trace as T
import os
import numpy as np
from PIL import Image
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_num_regions, calc_certain_tile, run_dijkstra

from gym_pcgrl.envs.probs.zelda.zelda_ctrl_prob import ZeldaCtrlProblem

"""
Generate a fully connected GVGAI zelda level where there is only a player and a door.
"""

class MiniZeldaProblem(ZeldaCtrlProblem):
    """
    The constructor is responsible of initializing all the game parameters
    """
    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5


        
