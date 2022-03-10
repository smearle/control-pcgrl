from pdb import set_trace as TT
import imp
from gym_pcgrl.envs.pcgrl_ctrl_env import PcgrlCtrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper_3D import get_int_prob, get_string_map
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
import numpy as np
import gym
from gym import spaces
import PIL

"""
The 3D PCGRL GYM Environment
"""
class PcgrlEnv3D(PcgrlCtrlEnv):
    def __init__(self, prob="minecraft_3D_maze", rep="narrow3D"):
        self._prob = PROBLEMS[prob]()
        self._rep = REPRESENTATIONS[rep]()
        self._rep_stats = None
        self._iteration = 0
        self._changes = 0
        
        # NOTE: allow overfitting: can take as many steps as there are tiles in the maps, can change every tile on the map
        self._max_changes = np.inf
#       self._max_changes = max(
#           int(0.2 * self._prob._length * self._prob._width * self._prob._height), 1)
        self._max_iterations = self._prob._length * self._prob._width * self._prob._height
#       self._max_iterations = self._max_changes * \
#           self._prob._length * self._prob._width * self._prob._height
        self._heatmap = np.zeros(
            (self._prob._height, self._prob._width, self._prob._length))
        
        self.seed()
        self.viewer = None

        self.action_space = self._rep.get_action_space(
            self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(
            self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(
            self._prob._height, self._prob._width, self._prob._length))

        self.metrics = {}
        print('problem static trgs: {}'.format(self._prob.static_trgs))
        for k in {**self._prob.static_trgs}:
            self.metrics[k] = None
        print('env metrics: {}'.format(self.metrics))
        self.weights = self._prob.weights
        self.cond_bounds = self._prob.cond_bounds
        self.static_trgs = self._prob.static_trgs
        self.width = self._prob._width
    
    def reset(self):
        self._changes = 0
        self._iteration = 0
        self._rep.reset(self._prob._length, self._prob._width, self._prob._height, get_int_prob(
                                                    self._prob._prob, self._prob.get_tile_types()))
        self._rep_stats = self._prob.get_stats(
            get_string_map(self._rep._map, self._prob.get_tile_types()))
        self._prob.reset(self._rep_stats)
        self._heatmap = np.zeros(
            (self._prob._height, self._prob._width, self._prob._length))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        return observation
        
    def adjust_param(self, **kwargs):
        if 'change_percentage' in kwargs:
            percentage = min(1, max(0, kwargs.get('change_percentage')))
            self._max_changes = max(
                int(percentage * self._prob._length * self._prob._width * self._prob._height), 1)
#       self._max_iterations = self._max_changes * \
#           self._prob._length * self._prob._width * self._prob._height
        self._prob.adjust_param(**kwargs)
        self._rep.adjust_param(**kwargs)
        self.action_space = self._rep.get_action_space(
            self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(
            self._prob._length, self._prob._width, self._prob._height, self.get_num_tiles())
        self.observation_space.spaces['heatmap'] = spaces.Box(low=0, high=self._max_changes, dtype=np.uint8, shape=(
            self._prob._height, self._prob._width, self._prob._length))


    def step(self, action):
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action
        change, x, y, z= self._rep.update(action)
        if change > 0:
            self._changes += change
            self._heatmap[z][y][x] += 1.0
            self._rep_stats = self._prob.get_stats(
                get_string_map(self._rep._map, self._prob.get_tile_types()))
        # calculate the values
        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()
        reward = self._prob.get_reward(self._rep_stats, old_stats)

        # NOTE: not ending the episode if we reach targets in our metrics of interest for now
#       done = self._prob.get_episode_over(self._rep_stats, old_stats) or \
#           self._changes >= self._max_changes or \
#           self._iteration >= self._max_iterations
        done = self._iteration >= self._max_iterations

        info = self._prob.get_debug_info(self._rep_stats, old_stats)
        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes
        #return the values
        return observation, reward, done, info
    
    def render(self, mode='human'):
        self._prob.render(get_string_map(
            self._rep._map, self._prob.get_tile_types()))
        self._rep.render(get_string_map(
            self._rep._map, self._prob.get_tile_types()))
        return

