import copy
from pdb import set_trace as TT
from re import S
import time
from gym_pcgrl.envs.pcgrl_ctrl_env import PcgrlCtrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
from gym_pcgrl.envs.reps.holey_representation import HoleyRepresentation
from gym_pcgrl.envs.reps.representation import Representation
import numpy as np
import gym
from gym import spaces
import PIL
import collections

"""
The PCGRL GYM Environment with borders added into obervations. 2 holes are digged on the borders to make entrance and 
exit.
"""
class PcgrlHoleyEnv(PcgrlCtrlEnv):
    """
    The type of supported rendering
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    """
    Constructor for the interface.

    Parameters:
        prob (string): the current problem. This name has to be defined in PROBLEMS
        constant in gym_pcgrl.envs.probs.__init__.py file
        rep (string): the current representation. This name has to be defined in REPRESENTATIONS
        constant in gym_pcgrl.envs.reps.__init__.py
    """
    def __init__(self, prob="binary", rep="narrow", **kwargs):
        super(PcgrlHoleyEnv, self).__init__(prob, rep, **kwargs)
        self._rep: HoleyRepresentation = REPRESENTATIONS[rep](
            border_tile_index = self.get_border_tile(),
            empty_tile_index = self.get_empty_tile(),
            )
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
        return self._rep._bordered_map

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        tile_size = 16
        img = self._prob.render(self.get_string_map(self._rep._bordered_map, self._prob.get_tile_types(), continuous=self._prob.is_continuous()))
        img = self._rep.render(img, self._prob._tile_size).convert("RGB")

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering

            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)

            return self.viewer.isopen

    """
    Close the environment
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
