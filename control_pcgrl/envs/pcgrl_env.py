import collections
import json
from pdb import set_trace as TT
import PIL

import gym
from gym import spaces
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.probs.holey_prob import HoleyProblem
import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.annotations import override

from control_pcgrl.envs.reps.wrappers import wrap_rep, MultiAgentWrapper
from control_pcgrl.envs.probs import PROBLEMS
from control_pcgrl.envs.probs.problem import Problem, Problem3D
from control_pcgrl.envs.reps import REPRESENTATIONS
from control_pcgrl.envs.helper import get_int_prob, get_string_map
from control_pcgrl.envs.reps.representation import Representation

"""
The PCGRL GYM Environment
"""
class PcgrlEnv(gym.Env):
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
    def __init__(self, cfg: Config, prob="binary", rep="narrow"):
        self._has_been_assigned_map = False  # TODO: Factor this out into a ... wrapper?

        # Whether we need to load a new evaluation map.
        self.switch_env = False
        self.cur_map_idx = 0

        self._repr_name = rep
        # Attach this function to the env, since it will be different for, e.g., 3D environments.
        self.get_string_map = get_string_map

        self._prob: Problem = PROBLEMS[prob](cfg=cfg)
        self._prob.init_tile_int_dict()
        self._rep_cls = REPRESENTATIONS[rep]
        self._rep: Representation = self._rep_cls()
        self._rep_is_wrapped: bool = False
        self._rep_stats = None
        self.metrics = {}
        # print('problem metrics trgs: {}'.format(self._prob.static_trgs))
        for k in self._prob.static_trgs:
            self.metrics[k] = None
        # print('env metrics: {}'.format(self.metrics))
        self._iteration = 0
        self._changes = 0
        self.width = self._prob._width
        self._change_percentage = 0.2

        # Should we compute stats each step/reset? If not, assume some external process is computing terminal reward.
        # TODO: adapt evolution to toggle a `terminal_reward` setting in the env.
        self._get_stats_on_step = True

        # Effectively a dummy variable if `change_percentage` is later set to `None`.
        self._max_changes = max(int(self._change_percentage * np.prod(self.get_map_dims()[:-1])), 1)

        # self._max_iterations = self._max_changes * self._prob._width * self._prob._height
        # self._max_iterations = self._prob._width * self._prob._height + 1
        self._max_iterations = 700
        self._heatmap = np.zeros(self.get_map_dims()[:-1])

        self.seed()
        self.viewer = None

        map_dims = self.get_map_dims()
        self.action_space = self._rep.get_action_space(map_dims[:-1], map_dims[-1])
        obs_map_dims = self.get_observable_map_dims()
        self.observation_space = self._rep.get_observation_space(obs_map_dims[:-1], obs_map_dims[-1])
        # self.observation_space.spaces['heatmap'] = spaces.Box(
        #     low=0, high=self._max_changes, dtype=np.uint8, shape=self.get_map_dims()[:-1])

        # For use with gym-city ParamRew wrapper, for dynamically shifting reward targets
        
        self.metric_trgs = collections.OrderedDict(self._prob.static_trgs)  # FIXME: redundant??

        # Normalize reward weights w.r.t. bounds of each metric.
        self._reward_weights = {
            k: v * 1 / (self._prob.cond_bounds[k][1] - self._prob.cond_bounds[k][0]) \
                for k, v in self._prob._reward_weights.items()
        }
        self._ctrl_reward_weights = {
            k: v * 1 / (self._prob.cond_bounds[k][1] - self._prob.cond_bounds[k][0]) \
                for k, v in self._prob._ctrl_reward_weights.items()
        }

#       self.param_bounds = self._prob.cond_bounds
        self.compute_stats = False

    @override(TaskSettableEnv)
    def sample_tasks(self, n_maps):
        """Implement this to sample n random tasks."""
        return [np.random.randint(self._n_maps, n_maps)]

    @override(TaskSettableEnv)
    def get_task(self):
        """Implement this to get the current task (curriculum level)."""
        return self.cur_map_idx

    @override(TaskSettableEnv)
    def set_task(self, map_idx):
        """Implement this to set the task (curriculum level) for this env."""
        if map_idx is not None:
            self.cur_map_idx = map_idx
            self.switch_env = True

    def get_rep(self):       # ZJ: why do we need this?
        return self._rep

    def get_map(self):      # ZJ: why not use self._get_rep_map()?
        return self._rep._map

    def get_map_dims(self):
        return (self._prob._width, self._prob._height, self.get_num_tiles())

    def get_observable_map_dims(self):
        return (self._prob._width, self._prob._height, self.get_num_observable_tiles())

    def configure(self, cfg: Config):  # , max_step=300):
        map_shape = cfg.problem.map_shape
        # What is this garbage??
        # ZJ: I think map_shape comes as a list? why do we need to check if it is a string?
        if isinstance(map_shape, str):
            map_shape = json.loads(map_shape)
        
        self._prob._width = map_shape[0]
        self._prob._height = map_shape[1]
        self.width = map_shape[0]  #UGH 
#       self._prob.max_step = max_step


#   def get_param_bounds(self):
#       return self.param_bounds

#   def set_param_bounds(self, bounds):
#       #TODO
#       return len(bounds)

    def set_params(self, trgs):
        for k, v in trgs.items():
            self.metric_trgs[k] = v

    def display_metric_trgs(self):
        pass

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with random start.

    Parameters:
        seed (int): the starting seed, if it is None a random seed number is used.

    Returns:
        int[]: An array of 1 element (the used seed)
    """
    def seed(self, seed=None):
        seed = self._rep.seed(seed)
        self._prob.seed(seed)

        return [seed]

    def get_spaces(self):
        return self.observation_space.spaces, self.action_space

    """
    Resets the environment to the start state

    Returns:
        Observation: the current starting observation have structure defined by
        the Observation Space
    """
    def reset(self):
        self._changes = 0
        self._iteration = 0
        # avoid default probabilities with normal distribution if we seed manually
        if hasattr(self._prob, '_random'):
            probs = self._prob._random.random(size=len(self._prob.get_tile_types()))
            self._prob._prob = {tile: prob for tile, prob in zip(self._prob.get_tile_types(), probs)}
        if self.switch_env:
            self._rep.reset(self.get_map_dims()[:-1], get_int_prob(self._prob._prob, self._prob.get_tile_types()),
                next_map=self._prob.eval_maps[self.cur_map_idx])
            self.switch_env = False
        else:
            self._rep.reset(self.get_map_dims()[:-1], get_int_prob(self._prob._prob, self._prob.get_tile_types()))
        # continuous = False if not hasattr(self._prob, 'get_continuous') else self._prob.get_continuous()
        if self._get_stats_on_step:
            self._rep_stats = self._prob.get_stats(self.get_string_map(self._get_rep_map(), self._prob.get_tile_types()))  #, continuous=continuous))
        self.metrics = self._rep_stats
        self._prob.reset(self._rep_stats)
        self._prob._prob = probs
        self._heatmap = np.zeros(self.get_map_dims()[:-1])

        if issubclass(type(self._rep), MultiAgentWrapper):
            observation = self._rep.get_observation(all_agents=True)
        else:
            observation = self._rep.get_observation() # all_agents parameter does not exist for representations without MultiAgentWrapper

        # observation["heatmap"] = self._heatmap.copy()

        return observation

    """
    Get the border tile that can be used for padding

    Returns:
        int: the tile number that can be used for padding
    """
    def get_border_tile(self) -> int:
        return self._prob.get_tile_types().index(self._prob._border_tile)

    """
    Get the number of different type of tiles that are allowed in the observation

    Returns:
        int: the number of different tiles
    """
    def get_num_tiles(self):
        return len(self._prob.get_tile_types())

    def get_num_observable_tiles(self):
        return len(self._prob.get_observable_tile_types())

    """
    Adjust the used parameters by the problem or representation

    Parameters:
        change_percentage (float): a value between 0 and 1 that determine the
        percentage of tiles the algorithm is allowed to modify. Having small
        values encourage the agent to learn to react to the input screen.
        **kwargs (dict(string,any)): the defined parameters depend on the used
        representation and the used problem
    """
    def adjust_param(self, cfg: Config):
        # TODO: Factor this out into a ... wrapper?
        self.evaluation_env = cfg.evaluation_env
        if self.evaluation_env:
            self.num_eval_envs = cfg.num_eval_envs

        _prob_cls = type(self._prob)
        static_build = cfg.static_prob is not None

        # multi = cfg.multi is not None

        # Wrap the representation if we haven't already.
        if not self._rep_is_wrapped:
            self._rep = wrap_rep(self._rep, _prob_cls, self.get_map_dims(), static_build=static_build, 
                                 cfg=cfg)
            self._rep_is_wrapped = True

        # self.compute_stats = kwargs.get('compute_stats') if 'compute_stats' in kwargs else self.compute_stats
        # self.compute_stats = cfg.compute_stats

        # self._change_percentage = kwargs['change_percentage'] if 'change_percentage' in kwargs else self._change_percentage
        self._change_percentage = cfg.change_percentage

        if self._change_percentage is not None:
            percentage = min(1, max(0, self._change_percentage))
            self._max_changes = max(int(percentage * np.prod(self.get_map_dims()[:-1])), 1)
        # self._max_iterations = self._max_changes * self._prob._width * self._prob._height

        # if cfg.model.name is not None and 'Decoder' in cfg.model.name: 
            # Pretty much for debugging, no config for this model at present
            # self._max_iterations = 1
        # else:
        max_board_scans = cfg.max_board_scans
        self._max_iterations = np.prod(self.get_map_dims()[:-1]) * max_board_scans + 1
        self._prob.adjust_param(cfg=cfg)
        self._rep.adjust_param(cfg=cfg)
        self.action_space = self._rep.get_action_space(self.get_map_dims()[:-1], self.get_num_tiles())
        self.observation_space = self._rep.get_observation_space(
            self.get_map_dims()[:-1], self.get_num_observable_tiles())
        # self.observation_space.spaces['heatmap'] = spaces.Box(
        #     low=0, high=self._max_changes, dtype=np.uint8, shape=self.get_map_dims()[:-1])

    def get_agent_position(self):
        return self._rep._positions

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
        observation = self._prob.process_observation(observation)

        # observation["heatmap"] = self._heatmap.copy()

        # NOTE: in control-pcgrl, the ParamRew wrapper now handles rewards for all environments (even when not training a
        # "controllable" RL agent). Just need to specify the metrics of interest and their targets in __init__.
        reward = None
        # reward = self._prob.get_reward(self._rep_stats, old_stats)

        # NOTE: not ending the episode when we reach targets in our metrics of interest for now.
        # done = self._prob.get_episode_over(self._rep_stats,old_stats) or self._changes >= self._max_changes or self._iteration >= self._max_iterations
        done = self._iteration > self._max_iterations
        if self._change_percentage is not None:
            done = done or self._changes > self._max_changes

        # Only get level stats at the end of the level, for sparse, loss-based reward.
        # Uncomment the below to use dense rewards (also need to modify the ParamRew wrapper).
        if change > 0:
        # if done:
        # if done or self.compute_stats:

            # old_path_coords = set([tuple(e) for e in self._prob.old_path_coords])
            # last_build_coords = tuple(self._rep._new_coords)
            # if last_build_coords in old_path_coords:
            #     old_path_coords.remove(last_build_coords)
            #     self._prob.path_to_erase = old_path_coords
            self._rep_stats = self._prob.get_stats(self.get_string_map(self._get_rep_map(), self._prob.get_tile_types()))

            if self._rep_stats is None:
                raise Exception("self._rep_stats in pcgrl_env.py is None, what happened? Maybe you should check your path finding"
                "function in your problem class.")

            info = self._prob.get_debug_info(self._rep_stats, old_stats)

        else:
            info = {}

        info["iterations"] = self._iteration
        info["changes"] = self._changes                    # noop is removed now, so this is the same as iteration
        info["max_iterations"] = self._max_iterations
        info["max_changes"] = self._max_changes

        return observation, reward, done, info

    def _get_rep_map(self):
        return self._rep.unwrapped._map

    """
    Render the current state of the environment

    Parameters:
        mode (string): the value has to be defined in render.modes in metadata

    Returns:
        img or boolean: img for rgb_array rendering and boolean for human rendering
    """
    def render(self, mode='human'):
        img: PIL.Image = self._prob.render(self.get_string_map(
            self._get_rep_map(), self._prob.get_tile_types(), continuous=self._prob.is_continuous()))
        # Transpose image
        # img = img.transpose(PIL.Image.TRANSPOSE)
        img = self._rep.render(img, self._prob._tile_size, self._prob._border_size).convert("RGB")

        if mode == 'rgb_array':
            return np.array(img)
        elif mode == 'image':
            return img
        elif mode == 'human':

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
                # self.viewer = GUI()

            if not hasattr(img, 'shape'):
                img = np.array(img)
            self.viewer.imshow(img)

            return self.viewer.isopen

    """
    Close the environment rendering window.
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
