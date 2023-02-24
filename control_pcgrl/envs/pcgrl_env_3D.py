import collections
from pdb import set_trace as TT
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.pcgrl_ctrl_env import PcgrlCtrlEnv
from control_pcgrl.envs.probs import PROBLEMS
from control_pcgrl.envs.probs.problem import Problem
from control_pcgrl.envs.reps import REPRESENTATIONS
from control_pcgrl.envs.helper_3D import get_int_prob, get_string_map
from control_pcgrl.envs.reps.representation import Representation
import numpy as np
from gym import spaces

RENDER_OPENGL = 0
RENDER_MINECRAFT = 1

RENDER_MODE = RENDER_MINECRAFT
# RENDER_MODE = RENDER_OPENGL

if RENDER_MODE == RENDER_OPENGL:
    from control_pcgrl.envs.probs.minecraft.gl_render import Scene

"""
The 3D PCGRL GYM Environment
"""
class PcgrlEnv3D(PcgrlCtrlEnv):
    def __init__(self, cfg: Config, prob="minecraft_3D_maze", rep="narrow3D"):
        super().__init__(cfg, prob, rep)
        self.get_string_map = get_string_map
        self.gl_scene = None
        self.is_holey = False
    
    def render(self, mode='human'):
        if RENDER_MODE == RENDER_OPENGL:
            self.render_opengl(self._get_rep_map())
            return
        elif RENDER_MODE == RENDER_MINECRAFT:
            # Render the agent's edit action.
           
            # NOTE: can't call rep render twice (or agent's current position becomes invalid on second call...)
            self._rep.render(get_string_map(
                self._get_rep_map(), self._prob.get_tile_types()))

            # self._rep.render(get_string_map(
            #     self._get_rep_map(), self._prob.get_tile_types()), offset=(-10, 0, 0))

            # Render the resulting path.
            self._prob.render(get_string_map(
                self._get_rep_map(), self._prob.get_tile_types()), self._iteration, self._repr_name)
            # print(get_string_map(
            #     self._rep._map, self._prob.get_tile_types()))
            return

    def render_opengl(self, rep_map):
        if self.gl_scene is None:
            self.gl_scene = Scene()
        self.gl_scene.render(rep_map, paths=[self._prob.path_coords], bordered=self.is_holey)
