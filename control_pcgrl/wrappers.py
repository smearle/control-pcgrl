import math
from pdb import set_trace as TT
from typing import Iterable

import gym
from gym import spaces
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.pcgrl_env import PcgrlEnv
from control_pcgrl.envs.probs.problem import Problem3D
import numpy as np
from ray.rllib import MultiAgentEnv


# clean the input action
def get_action(a):
    if isinstance(a, int):
        return a
    return a.item() if a.shape == [1] else a


class AuxTiles(gym.Wrapper):
    """Let the generator write to and observe additional, "invisible" channels."""
    def __init__(self, game, n_aux_tiles, cfg: Config):
        self.n_aux_tiles = n_aux_tiles
        self.env = game
        super().__init__(self.env)
        map_obs_space = self.env.observation_space.spaces['map']
        self.env.observation_space.spaces['aux'] =spaces.Box(
            shape=(*map_obs_space.shape[:-1], n_aux_tiles),
            low = 0,
            high = 1,
            dtype=np.float32,
        )
        self.action_space = spaces.Dict(
            action=self.env.action_space, 
            aux=spaces.Box(low=0, high=1, shape=(n_aux_tiles,), dtype=np.float32),
        )

    def reset(self):
        obs = self.env.reset()
        self._pos = obs['pos']
        aux = np.zeros((*self.env.observation_space.spaces['map'].shape[:-1], self.n_aux_tiles), dtype=np.float32)
        obs['aux'] = aux
        self.aux_map = aux
        return obs

    def step(self, action):
        self._write_to_aux(self._pos, action['aux'])
        obs, reward, done, info = self.env.step(action['action'])
        self._pos = obs['pos']
        obs['aux'] = self.aux_map
        return obs, reward, done, info

    def _write_to_aux(self, pos, aux):
        self.aux_map[tuple(pos)] = aux


class TransformObs(gym.Wrapper):
    """Lil' hack to transform nested observation dicts when dealing with multi-agent environments."""
    def __init__(self, cfg: Config):
        super().__init__(self.env)
        n_agents = cfg.multiagent.n_agents
        if n_agents != 0:
            self.transform = self._transform_multiagent
        else:
            self.transform = self._transform

    def _transform_multiagent(self, obs):
        # Assuming these are agent keys
        for k in obs:
            obs[k] = self._transform(obs[k])
        return obs


class ToImage(TransformObs):
    """
    Return a Box instead of dictionary by stacking different similar objects

    Can be stacked as Last Layer
    """
    def __init__(self, game, names, cfg: Config):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(cfg)
        super().__init__(cfg=cfg)
        self.shape = None
        depth = 0
        max_value = 0

        for n in names:
            assert (
                n in self.env.observation_space.spaces.keys()
            ), "This wrapper only works if your observation_space is spaces.Dict with the input names."

            if self.shape is None:
                self.shape = self.env.observation_space.spaces[n].shape
            new_shape = self.env.observation_space.spaces[n].shape
            if (issubclass(type(self.unwrapped._prob), Problem3D) and len(new_shape) == 3 
                    or len(new_shape) <= 2):  # here we assume the problem is 2D if not 3D
                depth += 1
                n_cat_dims = len(new_shape)
            else:
                depth += new_shape[-1]
                n_cat_dims = len(new_shape) - 1
            assert (
                # self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1]
                np.all([self.shape[i] == new_shape[i] for i in range(n_cat_dims)])
            ), "This wrapper only works when all objects have same width, height, length..."

            if self.env.observation_space[n].high.max() > max_value:
                max_value = self.env.observation_space[n].high.max()
        self.names = names

        self.show_agents = cfg.show_agents
        self.n_agents = cfg.multiagent.n_agents

        self.observation_space = spaces.Box(
            low=0, high=max_value if self.show_agents else max(max_value, cfg.multiagent.n_agents), shape=(*self.shape[:-1], depth)
        )


    def step(self, action, **kwargs):
        if not isinstance(action, dict):
            action = get_action(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def _transform(self, obs):
        final = np.empty([])

        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(*self.shape[:-1], -1)
            else:
                final = np.append(
                    final, obs[n].reshape(*self.shape[:-1], -1), axis=-1
                )
        return final

class ToImageCA(ToImage):
    def __init__(self, game, name, cfg: Config):
        super().__init__(game, name, cfg)

    # FIXME: kwargs in step feels bad and wrong
    def step(self, action, **kwargs):
        # action = action.reshape((self.dim-1, self.w, self.h))  # Assuming observable path(?)
        action = action.reshape((self.dim, self.w, self.h))
        # action = np.argmax(action, axis=0)
        # obs, reward, done, info = self.env.step(action[:self.dim-1], **kwargs)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info


class OneHotEncoding(TransformObs):
    """
    Transform any object in the dictionary to one hot encoding
    can be stacked
    """
    def __init__(self, game, name, cfg: Config, padded: bool = False):
        """
        Args:
            padded (bool): if True, the observation we are receiving from the wrapper below us has `0s` to represent 
                padded tiles.
        """
        self.padded = padded
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(cfg)
        super().__init__(cfg=cfg)

        assert (
            name in self.env.observation_space.spaces.keys()
        ), "This wrapper only works for representations thave have a {} key".format(
            name
        )
        self.name = name

        self.observation_space = gym.spaces.Dict({})

        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space.spaces[self.name].shape

        # The number of tiles we can place
        self.dim = (
            self.observation_space.spaces[self.name].high.max()
            - self.observation_space.spaces[self.name].low.min()
            + 1 # ??? did we need this ??? I think so, to represent out-of-bounds tiles.
        )

        new_shape.extend(shape)
        # if len(new_shape) > 2:
        #     new_shape[-1] += self.dim - 1
        # else:
        new_shape.append(self.dim)
        #import pdb; pdb.set_trace()
        # self.show_agents = kwargs.get('show_agents', False)
        self.show_agents = cfg.show_agents
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=1 if not self.show_agents else max(1, cfg.multiagent.n_agents), shape=new_shape, dtype=np.uint8
        )

    def step(self, action, **kwargs):
        # action = get_action(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def _transform(self, obs):
        named_obs = obs[self.name]
        if self.show_agents: # if agent positions are a part of the observation, then the map observation will already have an extra dimension
            old = named_obs[:, :, 0]
        else:
            old = named_obs

        if self.padded:
            # Replace out-of-bounds values with all-zeros (i.e. slice off the ``OOB'' channel).
            new = np.eye(self.dim + 1)[old]
            new = new[..., 1:]
        else:
            new = np.eye(self.dim)[old]

        # add the agent positions back into the observation
        if self.show_agents:
            new = np.concatenate((new, named_obs[:, :, -1][:, :, None]), axis=-1)

        obs[self.name] = new
        #import pdb; pdb.set_trace()

        return obs

    def get_one_hot_map(self):
        obs = {'map': self.env._rep._map}
        return self.transform(obs)


class ActionMap(gym.Wrapper):
    """
    Transform the action input space to a 3D map of values where the argmax value will be applied. Can be stacked.
    """
    def __init__(self, game, cfg: Config, bordered_observation=False):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(cfg)
        gym.Wrapper.__init__(self, self.env)

        assert (
            "map" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a map key"
        self.old_obs = None
        self.one_hot = len(self.env.observation_space.spaces["map"].shape) > 2
        w, h, dim = 0, 0, 0

        if self.one_hot:
            h, w, dim = self.env.observation_space.spaces["map"].shape
        else:
            h, w = self.env.observation_space.spaces["map"].shape
            dim = self.env.observation_space.spaces["map"].high.max()

        # it's getting the h and w from the observation space, which is different from the action space
        if bordered_observation:
            h -= 2
            w -= 2
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
        # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h * w * self.dim)

    def reset(self):
        self.old_obs = self.env.reset()

        return self.old_obs

    def step(self, action, **kwargs):
        # y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))

        if "pos" in self.old_obs:
            o_x, o_y = self.old_obs["pos"]

            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v, **kwargs)
            else:
                o_v = self.old_obs["map"][o_y][o_x]

                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v, **kwargs)
        else:
            obs, reward, done, info = self.env.step([x, y, v], **kwargs)
        self.old_obs = obs

        return obs, reward, done, info


class CAMap(ActionMap):
    def __init__(self, *args, cfg: Config):
        super().__init__(*args, cfg)
        # self.action_space = spaces.MultiDiscrete([self.dim] * self.h * self.w)
        self.action_space = spaces.Box(0, 1, shape=(self.dim * self.w * self.h,))

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)


class Cropped(TransformObs):
    """
    Crops and centers the view around the agent and replace the map with cropped version
    The crop size can be larger than the actual view, it just pads the outside
    This wrapper only works on games with a position coordinate can be stacked
    """
    def __init__(self, game, obs_window: Iterable, pad_value: int, name: str, cfg: Config):
        self.obs_window = np.array(obs_window)
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(cfg)
        super().__init__(cfg=cfg)

        assert (
            "pos" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works for representations that have a position"
        assert (
            name in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a {} key".format(name)
        assert (
            len(self.env.observation_space.spaces[name].shape) in [2, 3]
        ), "This wrapper only works on 2D or 3D arrays."
        self.name = name
        self.show_agents = cfg.show_agents
        map_shape = np.array(cfg.task.map_shape)
        self.shape = map_shape
        pad_r = np.floor(self.obs_window / 2)
        self.pad = np.stack((pad_r, pad_r), axis=1).astype(np.int8)

        if self.show_agents:
            self.shape.append(2) # add extra two channels for the positions
        #self.pad = crop_shape // 2
        # self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})

        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max() + 1  # 0s correspond to out-of-bounds tiles
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=high_value if not self.show_agents else max(high_value, cfg.multiagent.n_agents), shape=tuple(self.obs_window), dtype=np.uint8
        )

    def step(self, action, **kwargs):
        # action = get_action(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def _transform(self, obs):
        # Incrementing all tile indices by 1 to avoid 0s (out-of-bounds).
        map = obs[self.name] + 1
        # x, y = obs["pos"]
        pos = obs['pos']

        # If crop shape is greater than map shape, pad the map
        # View Padding
        padded = np.pad(map, self.pad, constant_values=0)  # Denote out-of-bounds tiles as 0.

        # Compensate for the bottom-left padding.
        # View Centering
        # cropped = padded[tuple([slice(p, p + self.shape[i]) for i, p in enumerate(pos)])]

        # Crop the map
        cropped = padded[tuple([slice(p + self.pad[i][0] - self.obs_window[i] // 2, 
                                    p + self.pad[i][0] + math.ceil(self.obs_window[i] / 2)) for i, p in enumerate(pos)])]
        assert np.all(cropped.shape == self.obs_window)

        # if show positions is turned on: add an extra channel that shows agent positions
        # NOTE: Wide representaion cannot use this, since positions are not stored in representation
        if self.show_agents:
            #import pdb; pdb.set_trace()
            #map_expanded = map[:, :, None]
            agent_positions = self.unwrapped.get_agent_position()
            agent_positions_map = np.zeros(map.shape)
            for i, pos in enumerate(agent_positions):
                agent_positions_map[tuple(pos)] = i + 1
            #agent_positions_map[agent_positions[:, 0], agent_positions[:, 1]] = 1
            # view padding
            padded_positions = np.pad(agent_positions_map, self.pad, constant_values=0)

            # view centering
            cropped_positions = padded_positions[tuple([slice(p, p + self.shape[i]) for i, p in enumerate(pos)])]

            cropped = np.concatenate((cropped[:, :, None], cropped_positions[:, :, None]), axis=-1).astype(np.uint8)

        #import pdb; pdb.set_trace()
        obs[self.name] = cropped

        return obs


################################################################################
#   Final used wrappers for the experiments
################################################################################
class CroppedImagePCGRLWrapper(gym.Wrapper):
    """
    The wrappers we use for narrow and turtle experiments
    """
    def __init__(self, game, cfg: Config):
        # static_prob = kwargs.get('static_prob')
        static_prob = cfg.static_prob
        # obs_size = kwargs.get('observation_size')
        # crop_size = obs_size if obs_size is not None else crop_size
        env: PcgrlEnv = gym.make(game, cfg=cfg)
        env.adjust_param(cfg)

        # Keys of (box) observation spaces to be concatenated (channel-wise)
        flat_indices = ["map"]
        flat_indices += ["static_builds"] if static_prob is not None else []

        # Cropping map, etc. to the correct crop_size
        for k in flat_indices:
            env = Cropped(
                game=env, obs_window=cfg.task.obs_window, pad_value=env.get_border_tile(), name=k, 
                cfg=cfg,
            )
            
        # Transform the map to a one hot encoding
        # for k in flat_indices:
        env = OneHotEncoding(env, 'map', padded=True, cfg=cfg)

        if cfg.n_aux_tiles > 0:
            flat_indices += ["aux"]
            env = AuxTiles(env, n_aux_tiles=cfg.n_aux_tiles, cfg=cfg)
        

        # Final Wrapper has to be ToImage or ToFljat
        env = ToImage(env, flat_indices, cfg=cfg)

        self.env = env
        gym.Wrapper.__init__(self, self.env)


# class Cropped3DImagePCGRLWrapper(gym.Wrapper):
#     def __init__(self, game, crop_size, n_aux_tiles, **kwargs):
#         self.pcgrl_env = gym.make(game)
#         self.pcgrl_env.adjust_param(**kwargs)
#         # Cropping the map to the correct crop_size
#         env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map', **kwargs)
#         env = OneHotEncoding(env, 'map', padded=True, **kwargs)
        
#         # Now we one hot encode the observation for all probs including the binary
#         # Indices for flatting
#         flat_indices = ['map']
#         # Final Wrapper has to be ToImage or ToFlat
#         if n_aux_tiles > 0:
#             flat_indices += ["aux"]
#             env = AuxTiles(env, n_aux_tiles=n_aux_tiles, **kwargs)
#         self.env = ToImage(env, flat_indices, **kwargs)
#         gym.Wrapper.__init__(self, self.env)


"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""
class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, cfg: Config):
        self.pcgrl_env = gym.make(game)

        if "micropolis" in game.lower():
            self.pcgrl_env = SimCityWrapper(self.pcgrl_env)
            self.env = self.pcgrl_env
        elif "RCT" in game:
            self.pcgrl_env = RCTWrapper(self.pcgrl_env)
            self.env = self.pcgrl_env
        else:
            self.pcgrl_env.adjust_param(cfg=cfg)
            # Indices for flattening
            flat_indices = ["map"]
            env = self.pcgrl_env

            # Add the action map wrapper
            env = ActionMap(env, cfg=cfg)
            # Transform to one hot encoding if not binary

            # if "RCT" not in game and "Micropolis" not in game:
            env = OneHotEncoding(env, "map", padded=False, cfg=cfg)
            # Final Wrapper has to be ToImage or ToFlat
            self.env = ToImage(env, flat_indices, cfg=cfg)
        gym.Wrapper.__init__(self, self.env)

# This precedes the ParamRew wrapper so we only worry about the map as observation
class CAactionWrapper(gym.Wrapper):
    def __init__(self, game, cfg: Config):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(cfg)
        # Indices for flattening
        flat_indices = ['map']
        env = self.pcgrl_env
        # Add the action map wrapper
        env = CAMap(env, cfg)
        # Transform to one hot encoding if not binary
        # if 'binary' not in game:
            # ) or ('minecraft_2Dmaze' not in game)
        env = OneHotEncoding(env, 'map', padded=False, cfg=cfg)
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImageCA(env, flat_indices, cfg=cfg)
        gym.Wrapper.__init__(self, self.env)


class ActionMap3DImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, cfg: Config):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(cfg)
        # Indices for flattening
        flat_indices = ['map']
        env = self.pcgrl_env
        width = env._prob._width
        height = env._prob._height
        length = env._prob._length
        self.n_ca_tick = 0
        # Add the action map wrapper
        #       env = ActionMap(env)
        # Transform to one hot encoding if not binary

        # we need the observation to be one-hot, so we can reliably separate map from control observations for NCA skip connection
        env = OneHotEncoding(env, 'map', padded=False)
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)
        # NOTE: check this insanity out so cool
        # self.action_space = self.pcgrl_env.action_space = gym.spaces.MultiDiscrete([self.env.dim] * width * height * length)
        self.action_space = self.unwrapped.action_space = gym.spaces.Discrete(self.env.dim * width * height * length)
        #       self.action_space = self.pcgrl_env.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_tile_types* width* height,))
        self.last_action = None
        self.INFER = cfg.infer

    def step(self, action, **kwargs):
        """
        :param action: (int) the unravelled index of the action. We will re-ravel to get spatial (x, y, z) coordinates, 
                      and action type.
        """
        action = np.unravel_index(action, (self.observation_space.shape))

        return super().step(action, **kwargs)

    def reset(self):
        self.last_action = None
        self.n_ca_tick = 0
        #       if self.pcgrl_env.unwrapped._rep._map is None:
        #           # get map dimensions (sloppy)
        #           super().reset()
        #       # load up our initial state (empty)
        #       self.env.unwrapped._rep._random_start = True
        #       init_state = np.zeros(self.unwrapped._rep._map.shape).astype(np.uint8)
        #       self.unwrapped._rep._old_map = init_state
        obs = self.env.reset()
        #       self.pcgrl_env._map = self.env._rep._map

        #       self.render()
        #       obs = self.env.get_one_hot_map()

        return obs


class SimCityWrapper(gym.Wrapper):
    def __init__(self, game, cfg: Config):
        self.env = game
        self.env.configure(map_width=16)
        super(SimCityWrapper, self).__init__(self.env)

    #       self.observation_space = gym.spaces.Dict({
    #               'map': self.observation_space,
    #               })

    #       self.action_space = self.unwrapped.action_space = gym.spaces.MultiDiscrete((self.map_width, self.map_width, self.n_tools))

    def step(self, action, **kwargs):
        obs, rew, done, info = super().step(action, **kwargs)
        #       obs = {'map': np.array(obs).transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs, rew, done, info

    def reset(self):
        obs = super().reset()
        #       obs = {'map': obs.transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs

    def adjust_param(self, cfg: Config):
        return

    def get_border_tile(self):
        return 0


class RCTWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.env = game
        self.env.configure()
        super(RCTWrapper, self).__init__(self.env)
        #       self.observation_space = gym.spaces.Dict({
        #               'map': self.observation_space,
        #               })
        self.unwrapped.static_trgs = self.unwrapped.metric_trgs
        self.unwrapped.cond_bounds = self.param_bounds

    def step(self, action, **kwargs):
        action = np.array(action)
        obs, rew, done, info = super().step(action, **kwargs)
        #       obs = {'map': np.array(obs).transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs, rew, done, info

    def reset(self):
        obs = super().reset()
        #       obs = {'map': obs.transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs


# TODO

def disable_passive_env_checker(env):
    # remove the passive environment checker wrapper from the env attribute of an env
    # base case -> the environment is not a wrapper
    if not hasattr(env, 'env'):
        return env

    root = env
    prev = env 
    while hasattr(prev, 'env'):
        next_ = prev.env
        if isinstance(next_, gym.wrappers.env_checker.PassiveEnvChecker):
            prev.env = next_.env
        prev = next_
            
    return root

"""
gym wrappers do not allow for consistent seeding
add a seed method to each wrapper
"""
#def seedify(env):
#    def seed(self, s):
#        print(self)
#        return self.env.seed(s)
#    
#    root = env
#    curr = env
#    while hasattr(curr, 'env'):
#        type(curr).seed = seed
#        curr = curr.env
#    return root


class MultiAgentWrapper(gym.Wrapper, MultiAgentEnv):
    def __init__(self, game, cfg: Config):
        # multiagent_args = kwargs.get('multiagent')
        self.env = disable_passive_env_checker(game) # DISABLE GYM PASSIVE ENVIRONMENT CHECKER
        gym.Wrapper.__init__(self, self.env)
        MultiAgentEnv.__init__(self.env)
        self.n_agents = cfg.multiagent.n_agents
        # try:
            # self.n_agents = multiagent_args.get('n_agents', 2)
        # except AttributeError:
        #     self.n_agents = json.loads(multiagent_args.replace('\'', '\"'))['n_agents']
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Dict({})
        for i in range(self.n_agents):
            self.observation_space.spaces[f'agent_{i}'] = self.env.observation_space
            self.action_space.spaces[f'agent_{i}'] = self.env.action_space
        # otherwise gym utils throws an error???
        self.unwrapped.observation_space = self.observation_space
        self.unwrapped.action_space = self.action_space

    def reset(self):
        obs = super().reset()
        return obs

    def seed(self, s):
        return self.unwrapped.seed(s)

    def step(self, action):
        # print(f"Step:")
        # print(f"Action: {action}")
        obs, rew, done, info = {}, {}, {}, {}

        for k, v in action.items():
            self.unwrapped._rep.set_active_agent(k)
            obs_k, rew[k], done[k], info[k] = super().step(action={k: v})
            obs.update(obs_k)
        done['__all__'] = np.all(list(done.values()))

        return obs, rew, done, info


class GroupedEnvironmentWrapper(MultiAgentEnv):
    def __init__(self, env, cfg: Config):
        #import pdb; pdb.set_trace()
        MultiAgentEnv.__init__(self)
        #gym.Wrapper.__init__(self, env.env)
        self.env = env
        self.groups = self.env.groups
        self.agent_id_to_group = self.env.agent_id_to_group
        self._unwrapped = self.env.env.unwrapped
        #self.thing = 5
        #super().__init__(env) # inherit the attributes of the base environment
        #self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.ctrl_metrics = self.env.env.ctrl_metrics
        self.metrics = self.env.env.metrics

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def _ungroup_items(self, items):
        return self.env._ungroup_items(items)
    
    def _group_items(self, items):
        return self.env._group_items(items)



