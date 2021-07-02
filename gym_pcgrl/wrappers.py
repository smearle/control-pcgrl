import gym
import gym_pcgrl
from stable_baselines.bench import Monitor
#from gym_city import envs
#from envs import MicropolisMonitor

import numpy as np

import gym

try:
    import gym_city
except ImportError:
    print(
        "gym-city module not installed, cannot use SimCity RL environment. You can install from  source at: https://github.com/smearle/gym-city"
    )
try:
    import micro_rct
except ImportError:
    print(
        "micro-rct module not installed, cannot use micro-RollerCoaster Tycoon RL environment. You can install from  source at: https://github.com/smearle/micro-rct"
    )


# clean the input action
def get_action(a):
    return a.item() if hasattr(a, "item") else a


# unwrap all the environments and get the PcgrlEnv
def get_pcgrl_env(env):
    return (
        env
        if "PcgrlEnv" in str(type(env))
        or "PcgrlCtrlEnv" in str(type(env))
        or "Micropolis" in str(type(env))
        or "RCT" in str(type(env))
        else get_pcgrl_env(env.env)
    )


class MaxStep(gym.Wrapper):
    """
    Wrapper that resets environment only after a certain number of steps.
    """

    def __init__(self, game, max_step):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game

        # get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        self.max_step = max_step
        self.unwrapped.max_step = max_step
        self.n_step = 0
        gym.Wrapper.__init__(self, self.env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.n_step += 1

        if self.n_step == self.max_step:
            done = True
        # else:
        #    done = False

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.n_step = 0

        return obs



class Wrapper(gym.Wrapper):
    def __init__(self, game, filename='./', **kwargs):
        pass


"""
Does not intervene.
"""
class Full(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.env = gym.make(game)
        self.env.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
        # ignore position information
        self.observation_space = self.env.observation_space

"""
Return a Box instead of dictionary by stacking different similar objects

Can be stacked as Last Layer
"""


class ToImage(gym.Wrapper):
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)
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
            depth += 1 if len(new_shape) <= 2 else new_shape[2]
            assert (
                self.shape[0] == new_shape[0] and self.shape[1] == new_shape[1]
            ), "This wrapper only works when all objects have same width and height"

            if self.env.observation_space.spaces[n].high.max() > max_value:
                max_value = self.env.observation_space.spaces[n].high.max()
        self.names = names

        self.observation_space = gym.spaces.Box(
            low=0, high=max_value, shape=(self.shape[0], self.shape[1], depth)
        )

    def step(self, action):
        print('action {}'.format(action.shape))
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        final = np.empty([])

        for n in self.names:
            if len(final.shape) == 0:
                final = obs[n].reshape(self.shape[0], self.shape[1], -1)
            else:
                final = np.append(
                    final, obs[n].reshape(self.shape[0], self.shape[1], -1), axis=2
                )

        return final


class OneHotEncoding(gym.Wrapper):
    """
    Transform any object in the dictionary to one hot encoding
    can be stacked
    """

    def __init__(self, game, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        print(self.env.observation_space, self.env.observation_space.spaces.keys())
        assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a {} key'.format(name)
        self.name = name

        self.observation_space = gym.spaces.Dict({})

        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        new_shape = []
        shape = self.env.observation_space.spaces[self.name].shape
        self.dim = (
            self.observation_space.spaces[self.name].high.max()
            - self.observation_space.spaces[self.name].low.min()
            + 1
        )

        for v in shape:
            new_shape.append(v)
        new_shape.append(self.dim)
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.uint8
        )

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        old = obs[self.name]
        obs[self.name] = np.eye(self.dim)[old]

        return obs


"""
Transform the action input space to a 3D map of values where the argmax value will be applied

can be stacked
"""


class ActionMap(gym.Wrapper):
    def __init__(self, game, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert (
            "map" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a map key"
        self.old_obs = None
        print(self.env.observation_space)
        self.one_hot = len(self.env.observation_space.spaces["map"].shape) > 2
        w, h, dim = 0, 0, 0

        if self.one_hot:
            h, w, dim = self.env.observation_space.spaces["map"].shape
        else:
            h, w = self.env.observation_space.spaces["map"].shape
            dim = self.env.observation_space.spaces["map"].high.max()
        self.h = self.unwrapped.h = h
        self.w = self.unwrapped.w = w
        self.dim = self.unwrapped.dim = self.env.get_num_tiles()
        # self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(h,w,dim))
        self.action_space = gym.spaces.Discrete(h * w * self.dim)

    def reset(self):
        self.old_obs = self.env.reset()

        return self.old_obs

    def step(self, action):
        # y, x, v = np.unravel_index(np.argmax(action), action.shape)
        y, x, v = np.unravel_index(action, (self.h, self.w, self.dim))

        if "pos" in self.old_obs:
            o_x, o_y = self.old_obs["pos"]

            if o_x == x and o_y == y:
                obs, reward, done, info = self.env.step(v)
            else:
                o_v = self.old_obs["map"][o_y][o_x]

                if self.one_hot:
                    o_v = o_v.argmax()
                obs, reward, done, info = self.env.step(o_v)
        else:
            obs, reward, done, info = self.env.step([x, y, v])
        self.old_obs = obs

        return obs, reward, done, info


"""
Crops and centers the view around the agent and replace the map with cropped version
The crop size can be larger than the actual view, it just pads the outside
This wrapper only works on games with a position coordinate

can be stacked
"""


class Cropped(gym.Wrapper):
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        get_pcgrl_env(self.env).adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

        assert (
            "pos" in self.env.observation_space.spaces.keys()
        ), "This wrapper only works for representations thave have a position"
        assert (
            name in self.env.observation_space.spaces.keys()
        ), "This wrapper only works if you have a {} key".format(name)
        assert (
            len(self.env.observation_space.spaces[name].shape) == 2
        ), "This wrapper only works on 2D arrays."
        self.name = name
        self.size = crop_size
        self.pad = crop_size // 2
        self.pad_value = pad_value

        self.observation_space = gym.spaces.Dict({})

        for (k, s) in self.env.observation_space.spaces.items():
            self.observation_space.spaces[k] = s
        high_value = self.observation_space[self.name].high.max()
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=high_value, shape=(crop_size, crop_size), dtype=np.uint8
        )

    def step(self, action):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        map = obs[self.name]
        x, y = obs["pos"]

        # View Centering
        padded = np.pad(map, self.pad, constant_values=self.pad_value)
        cropped = padded[y : y + self.size, x : x + self.size]
        obs[self.name] = cropped

        return obs


################################################################################
#   Final used wrappers for the experiments
################################################################################
"""
The wrappers we use for narrow and turtle experiments
"""


class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, crop_size, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Cropping the map to the correct crop_size
        env = Cropped(self.pcgrl_env, crop_size, self.pcgrl_env.get_border_tile(), 'map', **kwargs)
        # Transform to one hot encoding if not binary
        if 'binary' not in game:
            env = OneHotEncoding(env, 'map')
        # Indices for flatting
        flat_indices = ['map']
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


"""
Similar to the previous wrapper but the input now is the index in a 3D map (height, width, num_tiles) of the highest value
Used for wide experiments
"""


class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)

        if "micropolis" in game.lower():
            self.pcgrl_env = SimCityWrapper(self.pcgrl_env)
            self.env = self.pcgrl_env
        elif "RCT" in game:
            self.pcgrl_env = RCTWrapper(self.pcgrl_env)
            self.env = self.pcgrl_env
        else:
            self.pcgrl_env.adjust_param(**kwargs)
            # Indices for flatting
            flat_indices = ["map"]
            env = self.pcgrl_env
            # Add the action map wrapper
            env = ActionMap(env)
            # Transform to one hot encoding if not binary

            if "binary" not in game and "RCT" not in game and "Micropolis" not in game:
                env = OneHotEncoding(env, "map")
            # Final Wrapper has to be ToImage or ToFlat
            self.env = ToImage(env, flat_indices)
        gym.Wrapper.__init__(self, self.env)


class SimCityWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.env = game
        self.env.configure(map_width=16)
        super(SimCityWrapper, self).__init__(self.env)

    #       self.observation_space = gym.spaces.Dict({
    #               'map': self.observation_space,
    #               })

    #       self.action_space = self.unwrapped.action_space = gym.spaces.MultiDiscrete((self.map_width, self.map_width, self.n_tools))

    def step(self, action):
        obs, rew, done, info = super().step(action)
        #       obs = {'map': np.array(obs).transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs, rew, done, info

    def reset(self):
        obs = super().reset()
        #       obs = {'map': obs.transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs

    def adjust_param(self, **kwargs):
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

    def step(self, action):
        action = np.array(action)
        obs, rew, done, info = super().step(action)
        #       obs = {'map': np.array(obs).transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs, rew, done, info

    def reset(self):
        obs = super().reset()
        #       obs = {'map': obs.transpose(1, 2, 0)}
        obs = obs.transpose(1, 2, 0)

        return obs
