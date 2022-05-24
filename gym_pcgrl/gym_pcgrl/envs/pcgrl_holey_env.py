import copy
from pdb import set_trace as TT
from re import S
import time
from gym_pcgrl.envs.pcgrl_ctrl_env import PcgrlCtrlEnv
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.reps import REPRESENTATIONS
from gym_pcgrl.envs.helper import get_int_prob, get_string_map
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
        self._rep: Representation = REPRESENTATIONS[rep](
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
        self._changes = 0
        self._iteration = 0
        self._rep.reset(
            self._prob._width, 
            self._prob._height, 
            get_int_prob(self._prob._prob, self._prob.get_tile_types())
        )
        continuous = False if not hasattr(self._prob, 'get_continuous') else self._prob.get_continuous()
        self._rep.dig_holes(*self._prob.gen_holes())
        self._rep_stats = self._prob.get_stats(self.get_string_map(self._rep._bordered_map, self._prob.get_tile_types(), continuous=continuous))
        self.metrics = self._rep_stats
        self._prob.reset(self._rep_stats)

        self._heatmap = np.zeros((self._prob._height, self._prob._width))

        observation = self._rep.get_observation()
        observation["heatmap"] = self._heatmap.copy()

        return observation

    def get_empty_tile(self) -> int:
        return self._prob.get_tile_types().index(self._prob._empty_tile)


    """
    Advance the environment using a specific action

    Parameters:
        action: an action that is used to advance the environment (same as action space)

    Returns:
        observation: the current observation after applying the action
        float: the reward that happened because of applying that action
        boolean: if the problem eneded (episode is over)
        dictionary: debug information that might be useful to understand what's happening
    """
    def step(self, action):
        """Step the environment.

        Args:
            action (_type_): The actions to be taken by the generator agent.
            compute_stats (bool, optional): Compute self._rep_stats even when we don't need them for (sparse) reward. 
                for visualizing, e.g., path-length during inference. Defaults to False.

        Returns:
            _type_: _description_
        """
       #print('action in pcgrl_env: {}'.format(action))
        self._iteration += 1
        #save copy of the old stats to calculate the reward
        old_stats = self._rep_stats
        # update the current state to the new state based on the taken action

        change, map_coords = self._rep.update(action)

        if change > 0:
            self._changes += change

            # Not using heamap, would need to do this differently for 2/3D envs
            # self._heatmap[*map_coords[::-1]] += 1.0

            # self._rep_stats = self._prob.get_stats(get_string_map(self._rep._map, self._prob.get_tile_types()))
            # self.metrics = self._rep_stats

        # Get the agent's observation of the map
        observation = self._rep.get_observation()

        # observation["heatmap"] = self._heatmap.copy()

        # NOTE: in control-pcgr, the ParamRew wrapper now handles rewards for all environments (even when not training a
        # "controllable" RL agent). Just need to specify the metrics of interest and their targets in __init__.
        reward = None
        # reward = self._prob.get_reward(self._rep_stats, old_stats)

        # TODO: actually we do want to allow max_change_percentage to terminate the episode!
        # NOTE: not ending the episode if we reach targets in our metrics of interest for now.
        # done = self._prob.get_episode_over(self._rep_stats,old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        done = self._iteration >= self._max_iterations

        # Only get level stats at the end of the level, for sparse, loss-based reward.
        # Uncomment the below to use dense rewards (also need to modify the ParamRew wrapper).
        if change > 0:
        # if done or self.compute_stats:
            self._rep_stats = self._prob.get_stats(self.get_string_map(self._rep._bordered_map, self._prob.get_tile_types()))

            if self._rep_stats is None:
                raise Exception("self._rep_stats in pcgrl_env.py is None, what happened? Maybe you should check your path finding"
                "function in your problem class.")

            info = self._prob.get_debug_info(self._rep_stats, old_stats)

            # Log episode infos for sb2/tensorboard (in our callback) when done.
            # if done:
                # info['episode'] = copy.copy(info)

        else:
            info = {}


        info["iterations"] = self._iteration
        info["changes"] = self._changes
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes

        return observation, reward, done, info


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
