from abc import ABC
from copy import deepcopy
from collections import OrderedDict
from inspect import isclass
import logging
import math
from pdb import set_trace as TT

from gym import spaces
import gym
from control_pcgrl.envs import helper_3D
from control_pcgrl.envs.probs.holey_prob import HoleyProblem
from control_pcgrl.envs.probs.minecraft.mc_render import spawn_3D_maze
from control_pcgrl.envs.probs.minecraft.minecraft_3D_rain import Minecraft3Drain
from control_pcgrl.envs.probs.problem import Problem, Problem3D
import numpy as np
from PIL import Image

from control_pcgrl.envs.helper_3D import gen_random_map as gen_random_map_3D
from control_pcgrl.envs.reps.ca_rep import CARepresentation
from control_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from control_pcgrl.envs.reps.representation import EgocentricRepresentation, Representation
from control_pcgrl.envs.reps.turtle_rep import TurtleRepresentation


# class RepresentationWrapper(Representation):
class RepresentationWrapper():
    def __init__(self, rep: Representation, **kwargs):
        self.rep = rep
        # TODO: implement below so that they all point to the same object
        # self._map = self.rep._map
        # self._bordered_map = self.rep._bordered_map  # Doing this results in self._borderd_map != self.rep._bordered_map
        # self._random_start = self.rep._random_start

    def _set_pos(self, pos):
        self.rep._pos = pos

    def adjust_param(self, **kwargs):
        return self.rep.adjust_param(**kwargs)

    def update(self, *args, **kwargs):
        return self.rep.update(*args, **kwargs)

    def get_observation(self, *args, **kwargs):
        return self.rep.get_observation(*args, **kwargs)

    def get_observation_space(self, *args, **kwargs):
        return self.rep.get_observation_space(*args, **kwargs)

    def get_action_space(self, *args, **kwargs):
        return self.rep.get_action_space(*args, **kwargs)

    def reset(self, *args, **kwargs):
        ret = self.rep.reset(*args, **kwargs)
        return ret

    def render(self, *args, **kwargs):
        return self.rep.render(*args, **kwargs)

    def _update_bordered_map(self):
        return self.rep._update_bordered_map()

    def __repr__(self):
        return str(self)

    def __getattr__(self, name):
        # Removing this check causes errors when serializing this object with pickle. E.g. when using ray for parallel
        # environments. Variables that start with underscore will need to be unwrapped manually.
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.rep, name)

    # @property
    # def spec(self):
    #     return self.rep.spec

    @classmethod
    def class_name(cls):
        return cls.__name__

    # def close(self):
        # return self.rep.close()

    def seed(self, seed=None):
        return self.rep.seed(seed)

    def __str__(self):
        return "<{}{}>".format(type(self).__name__, self.rep)

    @property
    def unwrapped(self):
        return self.rep.unwrapped


# class Representation3DABC(Representation):


# class Representation3D(rep_cls, Representation3DABC):
class Representation3D(RepresentationWrapper):
    """
    The base class of all the 3D representations

    map in repr are np.array of numbers
    """
    _dirs = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0),(0,0,-1),(0,0,1)]
    _gen_random_map = helper_3D.gen_random_map

    # def _update_bordered_map(self):
        # self._bordered_map[1:-1, 1:-1, 1:-1] = self._map

    def render(self, map, mode='human', **kwargs):
        # TODO: Check if we are Egocentric. If so, render the agent edit. Otherwise, render the whole map (assume cellular)
        spawn_3D_maze(map)
        # return self.rep.render(mode, **kwargs)
        # pass
        


class HoleyRepresentation(RepresentationWrapper):
    def set_holes(self, entrance_coords, exit_coords):
        self.entrance_coords, self.exit_coords = entrance_coords, exit_coords

    def dig_holes(self, entrance_coords, exit_coords):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self.unwrapped._bordered_map[entrance_coords[0], entrance_coords[1]] = self.unwrapped._empty_tile
        self.unwrapped._bordered_map[exit_coords[0], exit_coords[1]] = self.unwrapped._empty_tile


    def update(self, action):
        ret = super().update(action)
        return ret

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        self.dig_holes(self.entrance_coords, self.exit_coords)
        return ret

    def get_observation(self):
        obs: dict = super().get_observation()
        obs.update(
            {'map': self.unwrapped._bordered_map.copy(),}
        )
        if 'pos' in obs:
            obs['pos'] += 1  # support variable border sizes?
        return obs

    def get_observation_space(self, dims, num_tiles):
        obs_space = super().get_observation_space(dims, num_tiles)
        map_shape = tuple([i + 2 for i in obs_space['map'].shape])
        obs_space.spaces.update({
            "map": spaces.Box(low=0, high=num_tiles-1, dtype=np.uint8, shape=map_shape)
        })
        if "pos" in obs_space.spaces:
            old_pos_space = obs_space.spaces["pos"]
            obs_space.spaces.update({
                "pos": spaces.Box(low=old_pos_space.low + 1, high=old_pos_space.high + 1, \
                    dtype=old_pos_space.dtype, shape=old_pos_space.shape)
            })
        return obs_space


class HoleyRepresentation3D(HoleyRepresentation):
    """A 3D variant of the holey representation. Holes on the border of the map are 2 tiles high, to support the
    size of the player in our Minecraft-inspired 3D problems."""

    def dig_holes(self, s, e):
        # TODO: Represent start/end differently to accommodate one-way paths.
        self.unwrapped._bordered_map[s[0][0]][s[0][1]][s[0][2]] = self.unwrapped._bordered_map[s[1][0]][s[1][1]][s[1][2]] = self.unwrapped._empty_tile
        self.unwrapped._bordered_map[e[0][0]][e[0][1]][e[0][2]] = self.unwrapped._bordered_map[e[1][0]][e[1][1]][e[1][2]] = self.unwrapped._empty_tile


class StaticBuildRepresentation(RepresentationWrapper):
    def __init__(self, rep, **kwargs):
        super().__init__(rep, **kwargs)
        self.prob_static = 0.0
        self.window = None

    def adjust_param(self, **kwargs):
        self.prob_static = kwargs.get('static_prob')
        self.n_aux_tiles = kwargs.get('n_aux_tiles')
        return super().adjust_param(**kwargs)

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)
        # Uniformly sample a probability of static builds from within the range [0, self.prob_static]
        prob_static = self.unwrapped._random.random() * self.prob_static
        # TODO: take into account validity constraints on number of certain tiles
        self.static_builds = (self.unwrapped._random.random(self.unwrapped._bordered_map.shape) < prob_static).astype(np.uint8)
        # Borders are always static
        self.static_builds[(0, -1), :] = 1
        self.static_builds[:, (0, -1)] = 1

        # Remove any action coordinates that correspond to static tiles (unless we have aux chans, in which case 
        # we'll let the agent leave messages for itself on those channels, even on static tiles.)
        # NOTE: We only have `_act_coords` for narrow representation. Can we make this cleaner?
        if hasattr(self, '_act_coords') and self.n_aux_tiles == 0:
            self._act_coords = self._act_coords[np.where(
                self.static_builds[self._act_coords[:, 0], self._act_coords[:, 1]] == 0)] 
        return ret

    def get_observation_space(self, dims, num_tiles):
        obs_space = super().get_observation_space(dims, num_tiles)
        obs_space.spaces.update({
            'static_builds': spaces.Box(low=0, high=1, dtype=np.uint8, shape=dims)
        })
        return obs_space

    def get_observation(self):
        obs = super().get_observation()
        obs.update({
            'static_builds': self.static_builds,
        })
        return obs

    def render(self, lvl_image, tile_size, border_size=None):
        lvl_image = super().render(lvl_image, tile_size, border_size)
        im_arr = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
        clr = (255, 0, 0, 255)
        im_arr[(0, 1, -1, -2), :, :] = im_arr[:, (0, 1, -1, -2), :] = clr
        x_graphics = Image.fromarray(im_arr)

        for (y, x) in np.argwhere(self.static_builds[1:-1, 1:-1] == 1):
            y, x = y + 1, x + 1  # ignoring the border
            lvl_image.paste(x_graphics, ((x+border_size[0]-1)*tile_size, (y+border_size[1]-1)*tile_size,
                                            (x+border_size[0])*tile_size,(y+border_size[1])*tile_size), x_graphics)

        # if not hasattr(self, 'window'):
            # self.window = cv2.namedWindow('static builds', cv2.WINDOW_NORMAL)
            # cv2.resize('static builds', 100, 800)
            # cv2.waitKey(1)
        # im = self.static_builds.copy()
        # cv2.imshow('static builds', im * 255)
        # cv2.waitKey(1)

        return lvl_image

    # update = {
        # CARepresentationHoley: update_ca_holey,
    # }[rep_cls]

    def update(self, action, **kwargs):
        old_state = self.unwrapped._bordered_map.copy()
        change, pos = super().update(action, **kwargs)
        new_state = self.unwrapped._bordered_map
        # assert not(np.all(old_state == new_state))
        self.unwrapped._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
        # print(self._bordered_map)
        self.unwrapped._map = self.unwrapped._bordered_map[
            tuple([slice(1, -1) for _ in range(len(self.unwrapped._map.shape))])]
        change = np.any(old_state != new_state)
        return change, pos


class RainRepresentation(RepresentationWrapper):
    def get_action_space(self, dims, num_tiles):
        # Need no-op because raining sand/acid will always change map (if column is not empty).
        return spaces.Discrete(num_tiles + 1)

    # TODO:
    def update(self, action, **kwargs):
        # FIXME: Assuming a narrow representation!
        change, pos = super().update(action, **kwargs)
        if change:
            self.unwrapped._map[pos[0], pos[1]] = self.unwrapped._empty_tile
        return change, pos

    def render(self, map, mode='human', **kwargs):
        # TODO: just place a sand block at the top
        spawn_3D_maze(map)


class MultiActionRepresentation(RepresentationWrapper):
    '''
    A wrapper that makes the action space change multiple tiles at each time step. Maybe useful for all representations 
    (for 2D, 3D, narrow, turtle, wide, ca, ...).
    NOW JUST FOR EGOCENTRIC REPRESENTATIONS.
    '''
    def _set_inner_padding(self, action_size):
        """These are like buffers. The agent should not be centered on these buffers because it will act on them anyway 
        when at either edge of the map. 
        For any odd action patch, these are equal (e.g., for 3, they are both 1). For, e.g. 4, they are 1 and 2.
        We define a left/right (bottom/top, close/far) pair for each map dimension."""
        self.inner_l_pads = np.floor((action_size - 1) / 2).astype(int)
        self.inner_r_pads = np.ceil((action_size - 1) / 2).astype(int)

    def __init__(self, rep, map_dims, **kwargs):
        super().__init__(rep, **kwargs)
        self.action_size = np.array(kwargs.get('action_size'))       # if we arrive here, there must be an action_size in kwargs
        self._set_inner_padding(self.action_size)
        self.map_size = map_dims                            # map_dims is a tuple (height, width, n_tiles) in 2D
        self.map_dim = len(map_dims[:-1])                        # 2 for 2D, 3 for 3D
        self.strides = np.ones(len(self.map_size[:-1]), dtype=np.int32) * 3   # strides are just 3 for each dimension now

        # We should not set this here. This is defined in the underlying representation class. In this underlying class,
        # it is initialized on `reset`.
        # self._act_coords = None

        # Check the action size is the same dimension as the map
        assert self.map_dim == len(self.action_size), \
            f"Action size ({len(self.action_size)}) should be the same dimension as the map size ({self.map_dim})"
        # Check whether we have a valid action size and stride
        for i in range(self.map_dim):
            logging.warning(f"Not validating your action size ({self.action_size}) and stride ({self.strides}, w.r.t." +
                " the map size ({self.map_size}). If these are mismatches, the agent may not be able to edit the bottom" +
                    " right/far edges of the map.")
            # FIXME: below assertion is thrown whenever stride = 1 and action_size > 1. But these are valid settings.
            # assert self.map_size[i] - self.action_size[i] + self.strides[i] == self.map_size[i] * self.strides[i], \
            #     "Please make sure that the action size and stride are valid for the map size."
        
        # NOTE: This function will not be called by the object we are wrapping. (Like it would be if we
        #   inherited from it instead.) So we'll be gross, and overwrite this function in the wrapped class manually.
        self.unwrapped.get_act_coords = self.get_act_coords
    
    def get_action_space(self, *args, **kwargs):
        # the tiles inside the action are not neccearily the same
        action_space = []
        for i in range(math.prod(self.action_size)):
            action_space.append(self.map_size[-1])
        return spaces.MultiDiscrete(action_space)
    
    # This gets overwritten in the wrapped class in `__init__` above.
    def get_act_coords(self):
        '''
        Get the coordinates of the action space. Regards the top left corner's coordinate (the smallest coords in the 
        action block) as the coordinate of current action. 

        The formula of calculating the size of 2d convolutional layer is:
        (W-F+2P)/S + 1
        where W is the width of input (the map size here), F is the width of filter (action_size here), 
        P is the padding (0 here), S is the stride. To get the same size of input and output, we have:
        (W-F)/S + 1 = W
        => W - F + S = W * S for each dimension
        '''
        coords = []
        for i in range(self.map_dim):
            coords.append(np.arange(self.inner_l_pads[i], self.map_size[i] - self.inner_r_pads[i], self.strides[i]))
        act_coords = np.array(np.meshgrid(*coords)).T.reshape(-1, self.map_dim)  # tobe checked! copilot writes this but looks good
        act_coords = np.flip(act_coords, axis=1)  # E.g., in 2D, scan horizontally first.
        return act_coords
            
    
    def update(self, action, **kwargs):
        '''
        Update the map according to the action, the action is a vector of size action_size

        In previous narrow_multi representation, the action is also a vector (MultiDiscrete). However the action 
        outside the map will be discarded (do I understand it right?). This will make the entries of the action space 
        sometimes crucial (inside the map) and sometimes trivial (outside the map). This is not good for RL. (copilot agree 
        this is not good)
        '''
        # unravel the action from a vector to a matrix (of size action_size)
        action = action.reshape(self.action_size)

        old_state = self.unwrapped._map.copy()

        # replace the map at self._pos with the action TODO: is there any better way to make it dimension independent? (sam: yes. Slices!) (copilot: yes, np.take_along_axis)(zehua:is this right?) (copilot:I think so)
        _pos = self.unwrapped._pos  # Why is _pos private again? (copilot: I don't know) Ok thanks copilot. (copilot: you're welcome)

        # Let's center the action patch around _pos. If the agent's observation is centered
        # around _pos, then we want the action patch to be centered around _pos as well.
        # The inner padding tells us how many tiles can be acted on to the left/right of _pos.
        top_left = _pos - self.inner_l_pads
        bottom_right = _pos + self.inner_r_pads

        slices = [slice(top_left[i], bottom_right[i] + 1) for i in range(self.map_dim)]
        # (zehua: yes use slices!)
        # not tested: map[tuple(starmap(slice, zip(top_left, bottom_right)))] = action
        # Or this(more similar to sam's code but single line): map[tuple(slice(*indexes) for indexes in zip(top_left, bottom_right))] = action

        ### Some checks for safety (could comment these out later). ###
        # Check that the action patch is within the map.
        assert np.all(top_left >= 0), \
            f"Action patch is outside the map. Top left corner: {top_left}"
        assert np.all(bottom_right < self.map_size[:-1]), \
            f"Action patch is outside the map. Bottom right corner: {bottom_right}"
        ################################################################

        self.unwrapped._map[tuple(slices)] = action
        # if self.map_dim == 2:
        #     self.unwrapped._map[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = action
        # elif self.map_dim == 3:
        #     self.unwrapped._map[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1] = action


        new_state = self.unwrapped._map
        if self.unwrapped._random_tile:
            if self.unwrapped.n_step == len(self._act_coords):
                np.random.shuffle(self._act_coords)

        # self._set_pos(self.unwrapped._act_coords[self.n_step % len(self.unwrapped._act_coords)])
        self._set_pos(self.get_pos_at_step(self.n_step))
        self.unwrapped.n_step += 1

        self.unwrapped._bordered_map[tuple([slice(1, -1) for _ in range(len(self.unwrapped._map.shape))])] = self.unwrapped._map

        change = np.any(old_state != new_state)

        return change, self.unwrapped._pos

    def render(self, lvl_image, tile_size=16, border_size=None):
        y, x = self.get_pos()
        # This is a little image with our border in it
        im_arr = np.zeros((tile_size * self.action_size[0], tile_size * self.action_size[1], 4), dtype=np.uint8)
        # Grey color
        clr = np.array([128, 128, 128, 255], dtype=np.uint8)
        # Two pixels on each side for column
        im_arr[(0, 1, -1, -2), :, :] = clr
        # Two pixels on each side for row
        im_arr[:, (0, 1, -1, -2), :] = clr
        x_graphics = Image.fromarray(im_arr)
        # Paste our border image into the level image at the agent's position
        lvl_image.paste(x_graphics, (
            # Left corner of the image we're pasting in
            (x+border_size[0]-self.inner_l_pads[0])*tile_size, (y+border_size[1]-self.inner_l_pads[1])*tile_size,
            # Right corner
            (x+border_size[0]+self.inner_r_pads[0]+1)*tile_size, (y+border_size[1]+self.inner_r_pads[1]+1)*tile_size), x_graphics)
        return super().render(lvl_image, tile_size, border_size)


class MultiAgentRepresentation(RepresentationWrapper):
    agent_colors = [
        (255, 255, 255, 255),
        (0, 255, 0, 255),
        (255, 0, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 0, 255),
        (255, 0, 255, 255),
        (0, 255, 255, 255),
    ]
    def __init__(self, rep, **kwargs):
        self.n_agents = kwargs['multiagent']['n_agents']
        # create a single representation for each agent
        # all representations share maps
        self._rep = rep
        self.reps = {f'agent_{i}': deepcopy(rep) for i in range(self.n_agents)}
        self._active_agent = None
        super().__init__(rep, **kwargs)

    def get_rep_map(self):
        return self.reps['agent_0']._map

    def reset(self, dims, prob, **kwargs):
        self._active_agent = None
        shared_map = None
        for agent, r in self.reps.items():
            r.reset(dims, prob, **kwargs)
            if shared_map is None:
                shared_map = r._map
            else:
                r._map = shared_map
            # default to random initialization
            import pdb; pdb.set_trace()
            r._pos = [int(r._random.random() * i) for i in dims]

        super().reset(dims, prob, **kwargs)
        self.unwrapped._map = shared_map

        # FIXME: specific to turtle
        #self._positions = np.floor(np.random.random((self.n_agents, len(dims))) * (np.array(dims))).astype(int)

    #def update(self, action):
    #    change = False

    #    # FIXME: mostly specific to turtle
    #    # for i, pos_0 in enumerate(self._positions):
    #    for k, v in action.items():
    #        i = int(k.split('_')[-1])
    #        pos_0 = self._positions[i]
    #        change_i, pos = self.update_pos(action[f'agent_{i}'], pos_0)
    #        change = change or change_i
    #        self._positions[i] = pos

    #    return change, self._positions

    def update(self, action):
        change = False
        for k, v in action.items():
            change_i, new_pos = self.reps[k].update(v)
            #i = int(k.split('_')[-1])
            #self.rep._pos = self._positions[i]
            #change_i, new_pos = self.rep.update(v)
            change = change or change_i
            self.reps[k]._pos = new_pos
            for r in self.reps:
                r._map = self.reps[k]._map
            self._map = self.reps[k]._map
            #self._positions[i] = new_pos
        return change, self._positions

    def render(self, lvl_image, tile_size=16, border_size=None):

        for (y, x), clr in zip(self._positions, self.agent_colors):
            im_arr = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)

            im_arr[(0, 1, -1, -2), :, :] = im_arr[:, (0, 1, -1, -2), :] = clr
            x_graphics = Image.fromarray(im_arr)
            lvl_image.paste(x_graphics, ((x+border_size[0])*tile_size, (y+border_size[1])*tile_size,
                                            (x+border_size[0]+1)*tile_size,(y+border_size[1]+1)*tile_size), x_graphics)
        return lvl_image

    def get_positions(self):
        return [r._pos for _, r in self.reps.items()]

    def get_observation(self, *args, **kwargs):
        # Note that this returns a dummy/meaningless position that never changes...
        base_obs = super().get_observation(*args, **kwargs)

        agent_name = self._active_agent
        multiagent_obs = {}
        if agent_name is None:
            for agent, r in self.reps.items():
                multiagent_obs[agent] = r.get_observation(*args, **kwargs)
            #for i in range(self.n_agents):
            #    obs_i = base_obs.copy()
            #    obs_i['pos'] = self._positions[i]
            #    multiagent_obs[f'agent_{i}'] = obs_i
            return multiagent_obs
        else:
            multiagent_obs[agent_name] = self.reps[agent_name].get_observation(*args, **kwargs)
            #multiagent_obs[agent_name] = base_obs
            #multiagent_obs[agent_name]['pos'] = self._positions[int(agent_name.split('_')[-1])]
            return multiagent_obs
            # base_obs['pos'] = self._positions[int(agent_name.split('_')[-1])]
            # return base_obs

    def set_active_agent(self, agent_name):
        self._active_agent = agent_name

def wrap_rep(rep: Representation, prob_cls: Problem, map_dims: tuple, static_build = False, multi = False, **kwargs):
    """Should only happen once!"""
    if multi:
        rep = MultiActionRepresentation(rep, map_dims, **kwargs)

    if static_build:
        # rep_cls = StaticBuildRepresentation(rep_cls)
        rep = StaticBuildRepresentation(rep, **kwargs)


    # FIXME: this is a hack to make sure that rep_cls is a class name but not an object
    # rep_cls = rep_cls if isclass(rep_cls) else type(rep_cls)
    # if issubclass(prob_cls, Minecraft3Drain):
        # rep = RainRepresentation(rep)
    if issubclass(prob_cls, Problem3D):
        rep = Representation3D(rep, **kwargs)
        # rep_cls = wrap_3D(rep_cls)
        # if issubclass(rep_cls, EgocentricRepresentation):
            # rep_cls = EgocentricRepresentation3D()
        # else:
            # rep_cls = Representation3D(rep_cls)
    
    # FIXME: this is a hack to make sure that rep_cls is a class name but not an object
    # rep_cls = rep_cls if isclass(rep_cls) else type(rep_cls)
    # if issubclass(prob_cls, HoleyProblem) and not issubclass(type(rep), HoleyRepresentation):
    if issubclass(prob_cls, HoleyProblem):

        if issubclass(prob_cls, Problem3D):
            rep = HoleyRepresentation3D(rep, **kwargs)
        
        else:
            rep = HoleyRepresentation(rep, **kwargs)

    if kwargs.get("multiagent")['n_agents'] != 0:
        #if not issubclass(type(rep), TurtleRepresentation):
        #    raise NotImplementedError("Multiagent only works with TurtleRepresentation currently")
        rep = MultiAgentRepresentation(rep, **kwargs)

    return rep
    
    


# def update_ca_holey(self, action, **kwargs):
#     old_state = self._bordered_map.copy()
#     change, pos = CARepresentationHoley.update(self, action, **kwargs)
#     new_state = self._bordered_map
#     # assert not(np.all(old_state == new_state))
#     self._bordered_map = np.where(self.static_builds < 1, new_state, old_state)
#     # print(self._bordered_map)
#     self._map = self._bordered_map[1:-1, 1:-1]
#     change = np.any(old_state != new_state)
#     return change, pos


# def update_narrow_holey(self, action, **kwargs):
#     change = 0
#     if action > 0:
#         change += [0,1][self._map[self._y][self._x] != action-1]
#         self._map[self._y][self._x] = action-1
#         self._bordered_map[self._y+1][self._x+1] = action-1
#     if self._random_tile:
#         if self.n_step == len(self._act_coords):
#             np.random.shuffle(self._act_coords)
#     self._x, self._y = self._act_coords[self.n_step % len(self._act_coords)]
#     self.n_step += 1
#     return change, [self._x, self._y]