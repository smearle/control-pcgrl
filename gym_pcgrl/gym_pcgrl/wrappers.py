from pdb import set_trace as TT

import gym
from gym import spaces
import numpy as np

# try:
#     import gym_city
# except ImportError:
#     print(
#         "gym-city module not installed, cannot use SimCity RL environment. You can install from  source at: https://github.com/smearle/gym-city"
#     )
# try:
#     import micro_rct
# except ImportError:
#     print(
#         "micro-rct module not installed, cannot use micro-RollerCoaster Tycoon RL environment. You can install from  source at: https://github.com/smearle/micro-rct"
#     )


# clean the input action
def get_action(a):
    if isinstance(a, int):
        return a
    return a.item() if a.shape == [1] else a



# class MaxStep(gym.Wrapper):
#     """
#     Wrapper that resets environment only after a certain number of steps.
#     """

#     def __init__(self, game, max_step):
#         if isinstance(game, str):
#             self.env = gym.make(game)
#         else:
#             self.env = game

#         # get_pcgrl_env(self.env).adjust_param(**kwargs)
#         gym.Wrapper.__init__(self, self.env)

#         self.max_step = max_step
#         self.unwrapped.max_step = max_step
#         self.n_step = 0
#         gym.Wrapper.__init__(self, self.env)

#     def step(self, action, **kwargs):
#         obs, reward, done, info = self.env.step(action, **kwargs)
#         self.n_step += 1

#         if self.n_step == self.max_step:
#             done = True
#         # else:
#         #    done = False

#         return obs, reward, done, info

#     def reset(self):
#         obs = self.env.reset()
#         self.n_step = 0

#         return obs


class AuxTiles(gym.Wrapper):
    """Let the generator write to and observe additional, "invisible" channels."""
    def __init__(self, game, n_aux_tiles, **kwargs):
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
        self.pos = obs['pos']
        aux = np.zeros((*self.env.observation_space.spaces['map'].shape[:-1], self.n_aux_tiles), dtype=np.float32)
        obs['aux'] = aux
        self.aux_map = aux
        return obs

    def step(self, action):
        self._write_to_aux(self.pos, action['aux'])
        obs, reward, done, info = self.env.step(action['action'])
        self.pos = obs['pos']
        obs['aux'] = self.aux_map
        return obs, reward, done, info

    def _write_to_aux(self, pos, aux):
        self.aux_map[tuple(pos)] = aux


# class AuxTiles3D(AuxTiles): pass
    # def _write_to_aux(self, pos, aux):
        # self.aux_map[pos[0], pos[1], pos[2]] = aux


class ToImage(gym.Wrapper):
    """
    Return a Box instead of dictionary by stacking different similar objects

    Can be stacked as Last Layer
    """
    def __init__(self, game, names, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(**kwargs)
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
            if len(new_shape) <= 2:
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

        self.observation_space = spaces.Box(
            low=0, high=max_value, shape=(*self.shape[:-1], depth)
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

    def transform(self, obs):
        final = np.empty([])

        for n in self.names:
#           if len(self.env.observation_space.spaces[n].shape) == 3:
            if len(final.shape) == 0:
                final = obs[n].reshape(*self.shape[:-1], -1)
            else:
                final = np.append(
                    final, obs[n].reshape(*self.shape[:-1], -1), axis=-1
                )
#           else:
#               if len(final.shape) == 0:
#                   final = obs[n].reshape(self.shape[0], self.shape[1], self.shape[2], -1)
#               else:
#                   final = np.append(
#                       final, obs[n].reshape(self.shape[0], self.shape[1], self.shape[2], -1), axis=2
#                   )

        return final

class ToImageCA(ToImage):
    def __init__(self, game, name, **kwargs):
        super().__init__(game, name, **kwargs)

    def step(self, action, **kwargs):
        # action = action.reshape((self.dim-1, self.w, self.h))  # Assuming observable path(?)
        action = action.reshape((self.dim, self.w, self.h))
        # action = np.argmax(action, axis=0)
        # obs, reward, done, info = self.env.step(action[:self.dim-1], **kwargs)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info


class OneHotEncoding(gym.Wrapper):
    """
    Transform any object in the dictionary to one hot encoding
    can be stacked
    """
    def __init__(self, game, name, padded: bool = False, **kwargs):
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
        self.env.unwrapped.adjust_param(**kwargs)
        gym.Wrapper.__init__(self, self.env)

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

    def step(self, action, **kwargs):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        old = obs[self.name]
        if self.padded:
            # Replace out-of-bounds values with all-zeros.
            new = np.eye(self.dim + 1)[old]
            new = new[..., 1:]

        else:
            new = np.eye(self.dim)[old]

        obs[self.name] = new

        return obs

    def get_one_hot_map(self):
        obs = {'map': self.env._rep._map}
        return self.transform(obs)


class ActionMap(gym.Wrapper):
    """
    Transform the action input space to a 3D map of values where the argmax value will be applied. Can be stacked.
    """
    def __init__(self, game, bordered_observation=False, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(**kwargs)
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.action_space = spaces.MultiDiscrete([self.dim] * self.h * self.w)
        self.action_space = spaces.Box(0, 1, shape=(self.dim * self.w * self.h,))

    def step(self, action, **kwargs):
        return self.env.step(action, **kwargs)


class Cropped(gym.Wrapper):
    """
    Crops and centers the view around the agent and replace the map with cropped version
    The crop size can be larger than the actual view, it just pads the outside
    This wrapper only works on games with a position coordinate can be stacked
    """
    def __init__(self, game, crop_size, pad_value, name, **kwargs):
        if isinstance(game, str):
            self.env = gym.make(game)
        else:
            self.env = game
        self.env.unwrapped.adjust_param(**kwargs)
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
        high_value = self.observation_space[self.name].high.max() + 1  # 0s correspond to out-of-bounds tiles
        self.observation_space.spaces[self.name] = gym.spaces.Box(
            low=0, high=high_value, shape=tuple([crop_size for _ in self.get_map_dims()[:-1]]), dtype=np.uint8
        )

    def step(self, action, **kwargs):
        action = get_action(action)
        obs, reward, done, info = self.env.step(action, **kwargs)
        obs = self.transform(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self.transform(obs)

        return obs

    def transform(self, obs):
        # Incrementing all tile indices by 1 to avoid 0s (out-of-bounds).
        map = obs[self.name] + 1
        # x, y = obs["pos"]
        pos = obs['pos']

        # View Centering
        # padded = np.pad(map, self.pad, constant_values=self.pad_value)
        padded = np.pad(map, self.pad, constant_values=0)  # Denote out-of-bounds tiles as 0.
        # cropped = padded[x : x + self.size, y : y + self.size]
        cropped = padded[tuple([slice(i, i + self.size) for i in pos])]
        obs[self.name] = cropped

        return obs

# class Cropped3D(Cropped):
#     def __init__(self, game, crop_size, pad_value, name, **kwargs):
#         if isinstance(game, str):
#             self.env = gym.make(game)
#         else:
#             self.env = game
#         self.env.unwrapped.adjust_param(**kwargs)
#         gym.Wrapper.__init__(self, self.env)

#         assert 'pos' in self.env.observation_space.spaces.keys(), 'This wrapper only works for representations thave have a position'
#         assert name in self.env.observation_space.spaces.keys(), 'This wrapper only works if you have a {} key'.format(name)
#         assert len(self.env.observation_space.spaces[name].shape) == 3, "This wrapper only works on 2D arrays."
#         self.name = name
#         self.size = crop_size
#         self.pad = crop_size//2
#         self.pad_value = pad_value

#         self.observation_space = gym.spaces.Dict({})
#         for (k,s) in self.env.observation_space.spaces.items():
#             self.observation_space.spaces[k] = s
#         high_value = self.observation_space[self.name].high.max() + 1  # 0s correspond to out-of-bounds tiles
#         self.observation_space.spaces[self.name] = gym.spaces.Box(low=0, high=high_value, shape=(crop_size, crop_size, crop_size), dtype=np.uint8)

#     def transform(self, obs):
#         map = obs[self.name]
#         x, y, z = obs['pos']

#         #View Centering
#         # padded = np.pad(map, self.pad, constant_values=self.pad_value)
#         padded = np.pad(map, self.pad, constant_values=0)  # Denote out-of-bounds tiles as 0.
#         cropped = padded[z:z+self.size, y:y+self.size, x:x+self.size]
#         obs[self.name] = cropped
#         return obs


################################################################################
#   Final used wrappers for the experiments
################################################################################
class CroppedImagePCGRLWrapper(gym.Wrapper):
    """
    The wrappers we use for narrow and turtle experiments
    """
    def __init__(self, game, crop_size, n_aux_tiles, **kwargs):
        static_prob = kwargs.get('static_prob')
        env = gym.make(game)
        env.adjust_param(**kwargs)

        # Keys of (box) observation spaces to be concatenated (channel-wise)
        flat_indices = ["map"]
        flat_indices += ["static_builds"] if static_prob is not None else []

        # Cropping map, etc. to the correct crop_size
        for k in flat_indices:
            env = Cropped(
                game=env, crop_size=crop_size, pad_value=env.get_border_tile(), name=k, 
                **kwargs,
            )

        # Transform the map to a one hot encoding
        # for k in flat_indices:
        env = OneHotEncoding(env, 'map', padded=True, **kwargs)

        if n_aux_tiles > 0:
            flat_indices += ["aux"]
            env = AuxTiles(env, n_aux_tiles=n_aux_tiles, **kwargs)
        

        # Final Wrapper has to be ToImage or ToFljat
        env = ToImage(env, flat_indices, **kwargs)

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
            # Indices for flattening
            flat_indices = ["map"]
            env = self.pcgrl_env

            # Add the action map wrapper
            env = ActionMap(env, **kwargs)
            # Transform to one hot encoding if not binary

            # if "RCT" not in game and "Micropolis" not in game:
            env = OneHotEncoding(env, "map", padded=False, **kwargs)
            # Final Wrapper has to be ToImage or ToFlat
            self.env = ToImage(env, flat_indices, **kwargs)
        gym.Wrapper.__init__(self, self.env)

# This precedes the ParamRew wrapper so we only worry about the map as observation
class CAactionWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
        # Indices for flattening
        flat_indices = ['map']
        env = self.pcgrl_env
        # Add the action map wrapper
        env = CAMap(env, **kwargs)
        # Transform to one hot encoding if not binary
        # if 'binary' not in game:
            # ) or ('minecraft_2Dmaze' not in game)
        env = OneHotEncoding(env, 'map', padded=False, **kwargs)
        # Final Wrapper has to be ToImage or ToFlat
        self.env = ToImageCA(env, flat_indices, **kwargs)
        gym.Wrapper.__init__(self, self.env)


class ActionMap3DImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, game, **kwargs):
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)
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
        self.INFER = kwargs.get('infer')

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
    def __init__(self, game, **kwargs):
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
