import collections
from pdb import set_trace as TT
from gym_pcgrl.envs.pcgrl_ctrl_env import PcgrlCtrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper_3D import get_int_prob, get_string_map
from gym_pcgrl.envs.reps.representation import Representation
import numpy as np
from gym import spaces

RENDER_OPENGL = 0
RENDER_MINECRAFT = 1

RENDER_MODE = RENDER_MINECRAFT

if RENDER_MODE == RENDER_OPENGL:
    from gym_pcgrl.envs.probs.minecraft.gl_render import Scene

"""
The 3D PCGRL GYM Environment
"""
class PcgrlEnv3D(PcgrlCtrlEnv):
    def __init__(self, prob="minecraft_3D_maze", rep="narrow3D", **kwargs):
        super().__init__(prob, rep, **kwargs)
        self.get_string_map = get_string_map
        self.gl_scene = None
        self.is_holey = False
#         self._prob: Problem = PROBLEMS[prob]()
#         self._rep: Representation = REPRESENTATIONS[rep]()

#         self._repr_name = rep

#         self._rep_stats = None
#         self.metrics = {}
#         # print('problem static trgs: {}'.format(self._prob.static_trgs))
#         for k in {**self._prob.static_trgs}:
#             self.metrics[k] = None
#         # print('env metrics: {}'.format(self.metrics))
#         self._iteration = 0
#         self._changes = 0
        
#         self._change_percentage = 0.2
#         # NOTE: allow overfitting: can take as many steps as there are tiles in the maps, can change every tile on the map
#         # self._max_changes = np.inf

#         self._max_changes = max(int(self._change_percentage * self._prob._length * self._prob._width * self._prob._height), 1)
# #       self._max_changes = max(
# #           int(0.2 * self._prob._length * self._prob._width * self._prob._height), 1)
#         self._max_iterations = self._prob._length * self._prob._width * self._prob._height + 1
# #       self._max_iterations = self._max_changes * \
# #           self._prob._length * self._prob._width * self._prob._height
#         self._heatmap = np.zeros(
#             (self._prob._height, self._prob._width, self._prob._length))

#         self.action_space = self._rep.get_action_space(
#             self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
#         self.observation_space = self._rep.get_observation_space(
#             self._prob._length, self._prob._width, self._prob._height, self.get_num_observable_tiles())
#         self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(
#             self._prob._height, self._prob._width, self._prob._length))

       
#         self._reward_weights = self._prob._reward_weights
#         self.cond_bounds = self._prob.cond_bounds
#         self.compute_stats = False

#         self.static_trgs = self._prob.static_trgs
#         self.width = self._prob._width

    def get_map_dims(self):
        return (self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())

    def get_observable_map_dims(self):
        return (self._prob._length, self._prob._width, self._prob._height, self.get_num_observable_tiles())

    
    # def reset(self):
    #     self._changes = 0
    #     self._iteration = 0
    #     self._rep.reset(self._prob._length, self._prob._width, self._prob._height, get_int_prob(
    #                                                 self._prob._prob, self._prob.get_tile_types()))
    #     self._rep_stats = self._prob.get_stats(
    #         get_string_map(self._get_rep_map(), self._prob.get_tile_types()))
    #     # Check for invalid path
    #     if self._rep_stats is None:
    #         self.render()
    #         raise Exception("Pathfinding bug: invalid path.")
    #     self._prob.reset(self._rep_stats)
    #     self._heatmap = np.zeros(
    #         (self._prob._height, self._prob._width, self._prob._length))

    #     observation = self._rep.get_observation()
    #     observation["heatmap"] = self._heatmap.copy()
    #     return observation
        
#     def adjust_param(self, **kwargs):
#         self._change_percentage = kwargs['change_percentage']
#         if self._change_percentage is not None:
#             # percentage = min(1, max(0, self._change_percentage))
#             # Allow, e.g., 200% change
#             percentage = max(0, self._change_percentage)

#         # if 'change_percentage' in kwargs:
#         #     percentage = min(1, max(0, kwargs.get('change_percentage')))
#             self._max_changes = max(
#                 int(percentage * self._prob._length * self._prob._width * self._prob._height), 1)
# #       self._max_iterations = self._max_changes * \
# #           self._prob._length * self._prob._width * self._prob._height
#         self.compute_stats = kwargs.get('compute_stats') if 'compute_stats' in kwargs else self.compute_stats
#         self._prob.adjust_param(**kwargs)
#         self._rep.adjust_param(**kwargs)
#         self.action_space = self._rep.get_action_space(
#             self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
#         self.observation_space = self._rep.get_observation_space(
#             self._prob._length, self._prob._width, self._prob._height, self.get_num_observable_tiles())
#         self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(
#             self._prob._height, self._prob._width, self._prob._length))

    
    def render(self, mode='human'):
        if RENDER_MODE == RENDER_OPENGL:
            self.render_opengl(self._get_rep_map())
            return
        elif RENDER_MODE == RENDER_MINECRAFT:
            # Render the agent's edit action.
            self._rep.render(get_string_map(
                self._get_rep_map(), self._prob.get_tile_types()))

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
