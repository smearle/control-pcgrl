import copy
from pdb import set_trace as TT
from re import S
import time
from gym_pcgrl.envs.pcgrl_env_3D import PcgrlEnv3D
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper_3D import get_int_prob, get_string_map
from gym_pcgrl.envs.reps.representation import Representation
from gym_pcgrl.envs.reps.wrappers import HoleyRepresentation
import numpy as np
import gym
from gym import spaces
import PIL
import collections

"""
The PCGRL GYM Environment with borders added into obervations. 2 holes are digged on the borders to make entrance and 
exit.
"""
class PcgrlHoleyEnv3D(PcgrlEnv3D):
    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow", **kwargs):
        super(PcgrlHoleyEnv3D, self).__init__(prob, rep, **kwargs)
        self._rep: HoleyRepresentation = REPRESENTATIONS[rep](
            border_tile_index = self.get_border_tile(),
            empty_tile_index = self.get_empty_tile(),
            )
        self.is_holey = True
    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self):
        self._rep.set_holes(*self._prob.gen_holes())
        return super().reset()

    def get_empty_tile(self) -> int:
        return self._prob.get_tile_types().index(self._prob._empty_tile)

    def _get_rep_map(self):
        return self._rep.unwrapped._bordered_map


    # def render(self, mode='human'):
        # super().render(mode)
        # Render the agent's edit action.
        # self._rep.render(get_string_map(
            # self._get_rep_map(), self._prob.get_tile_types()))

        # Render the resulting path.
        # self._prob.render(get_string_map(
            # self._get_rep_map(), self._prob.get_tile_types()), self._iteration, self._repr_name)

        # print(get_string_map(
        #     self._rep._map, self._prob.get_tile_types()))
        # return

