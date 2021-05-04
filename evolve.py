import argparse
import json
import sys
import os
import pickle
import time
from pdb import set_trace as T
from random import randint
# import cv2
from typing import Tuple
from datetime import datetime
from pathlib import Path

import matplotlib
import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import envs
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter, OptimizingEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from torch import ByteTensor, Tensor, nn
from torch.nn import Conv2d, Linear, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
# Use for .py file
from tqdm import tqdm
import scipy
#from example_play_call import random_player
#gvgai_path = '/home/sme/GVGAI_GYM/'
#sys.path.insert(0,gvgai_path)
#from play import play

import ray


import gym_pcgrl
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

# Use for notebook
# from tqdm.notebook import tqdm

# Use print to confirm access to local pcgrl gym
# print([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point])
"""
/// Required Environment ///
conda create -n ribs-pt python=3.7
pip install scipy==1.2.0  # must use this version with GVGAI_GYM
conda install -c conda-forge notebook
conda install pytorch torchvision torchaudio -c pytorch
conda install tensorboard
pip install 'ribs[all]' gym~=0.17.0 Box2D~=2.3.10 tqdm
git clone https://github.com/amidos2006/gym-pcgrl.git
cd gym-pcgrl  # Must run in project root folder for access to pcgrl modules

/// Instructions ///
To start TensorBoard run the following command:
$ tensorboard --logdir=runs

Then go to:
http://localhost:6006

/// Resources ///

Sam's example code:
https://github.com/smearle/gol-cmame/blob/master/gol_cmame.py

PCGRL Repo:
https://github.com/amidos2006/gym-pcgrl

Neural CA Paper:
https://arxiv.org/pdf/2009.01398.pdf

RIBS examples:
https://docs.pyribs.org/en/stable/tutorials/lunar_lander.html
"""

def id_action(action, **kwargs):
    # the argmax along tile_type dimension is performed inside the representation's update function
    skip = False
    return action, skip

def wide_action(action, **kwargs):
    # only consider tiles where the generator suggests something different than the existing tile
    int_map = kwargs.get('int_map')
    act_mask = action.argmax(axis=0) != int_map
    n_new_builds = np.sum(act_mask)
    act_mask = act_mask.reshape((1, *act_mask.shape))
    act_mask = np.repeat(act_mask, kwargs.get('n_tiles'), axis=0)
#   action = action * act_mask
    action = np.where(act_mask == False, action.min() - 10, action)
    coords = np.unravel_index(action.argmax(), action.shape)
    if n_new_builds > 0:
        assert act_mask[coords[0], coords[1], coords[2]] == 1
    coords = coords[2], coords[1], coords[0]
#   assert int_map[coords[0], coords[1]] != coords[2]
    skip = False
    return coords, skip

def narrow_action(action, **kwargs):
    x = kwargs.get('x')
    y = kwargs.get('y')
    act = action[:, y, x].argmax()
    if act == 0:
        skip = True
    else:
        skip = False
    return act, skip

def turtle_action(action, **kwargs):
    x = kwargs.get('x')
    y = kwargs.get('y')
    n_dirs = kwargs.get('n_dirs')
    act = action[:, y, x].argmax()
    # moving is counted as a skip, so lack of change does not end episode
    if act < n_dirs:
        skip = True
    else:
        skip = False
    return act, skip

def flat_to_box(action, **kwargs):
    int_map = kwargs.get('int_map')
    n_tiles = kwargs.get('n_tiles')
    action = action.reshape((n_tiles, *int_map.shape))
    skip = False
    return action, skip


def flat_to_wide(action, **kwargs):
    int_map = kwargs.get('int_map')
    w = int_map.shape[0]
    h = int_map.shape[1]
    n_tiles = kwargs.get('n_tiles')
    assert len(action) == int_map.shape[0] + int_map.shape[1] + n_tiles
    action = (
        action[:w].argmax(),
        action[w:w+h].argmax(),
        action[w+h:].argmax(),
    )
    skip = False
    return action, skip

def flat_to_narrow(action, **kwargs):
    act = action.argmax()
    if act == 0:
        skip = True
    else:
        skip = False
    return act, skip

def flat_to_turtle(action, **kwargs):
    n_dirs = kwargs.get('n_dirs')
    act = action.argmax()
    if act < n_dirs:
        skip = True
    else:
        skip = False
    return act, skip

preprocess_action_funcs = {
    'NCA':
        {
            'cellular': id_action,
            'wide': wide_action,
            'narrow': narrow_action,
            'turtle': turtle_action,
        },
    'CNN':
        {
            # will try to build this logic into the model
            'cellular': flat_to_box,
            'wide': flat_to_wide,
            'narrow': flat_to_narrow,
            'turtle': flat_to_turtle,
        }
}

def id_observation(obs, **kwargs):
    return obs

def local_observation(obs, **kwargs):
    x = kwargs.get('x')
    y = kwargs.get('y')
    local_obs = np.zeros((1, obs.shape[1], obs.shape[2]))
    # What is up with these y x lol
    local_obs[0, y, x] = 1
    np.concatenate((obs, local_obs), axis=0)
    return obs

preprocess_observation_funcs = {
    'NCA':
        {
            'cellular': id_observation,
            'wide': id_observation,
            'narrow': local_observation,
            'turtle': local_observation,
        },
    'CNN':
        {
            'cellular': id_observation,
            'wide': id_observation,
            'narrow': local_observation,
            'turtle': local_observation,
        }
}

class FlexArchive(GridArchive):
    def __init__(self, *args, **kwargs):
        self.score_hists = {}
        super().__init__(*args, **kwargs)

    def update_elite(self, obj, bcs):
        index = self._get_index(np.array(bcs))
        self.update_elite_idx(index, obj)

    def update_elite_idx(self, index, obj):
        if index not in self.score_hists:
            self.score_hists[index] = []
        score_hists = self.score_hists[index]
        score_hists.append(obj)
        obj = np.mean(score_hists)
        self._solutions[index][2] = obj
        self._objective_values[index] = obj

        while len(score_hists) > 500:
            score_hists.pop(0)

    def add(self, solution, objective_value, behavior_values):
        index = self._get_index(behavior_values)

        if index in self.score_hists:
            self.score_hists[index] = [objective_value]

        return super().add(solution, objective_value, behavior_values)

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


# TODO: Use the GPU!

#   if CUDA:
#       m.cuda()
#       m.to('cuda:0')


class GeneratorNN(nn.Module):
    ''' A neural cellular automata-type NN to generate levels or wide-representation action distributions.'''
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
#       if 'zelda' in PROBLEM:
        n_hid_1 = 32
        n_hid_2 = 16
#       if 'binary' in PROBLEM:
#           n_hid_1 = 32
#           n_hid_2 = 16
#       if PROBLEM == 'mini':
#           #TODO: make this smaller maybe?
#           n_hid_1 = 32
#           n_hid_2 = 16

        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with torch.no_grad():
            x = self.l1(x)
            x = torch.nn.functional.relu(x)
            x = self.l2(x)
            x = torch.nn.functional.relu(x)
            x = self.l3(x)
            x = torch.sigmoid(x)
            if not CA_ACTION:
                x = torch.stack([
                    unravel_index(x[i].argmax(), x[i].shape)
                    for i in range(x.shape[0])
                ])
        # axis 0 is batch
        # axis 0,0 is the 0 or 1 tile
        # axis 0,1 is the x value
        # axis 0,2 is the y value
        return x


class GeneratorNNDense(nn.Module):
    ''' A neural cellular automata-type NN to generate levels or wide-representation action distributions.'''
    def __init__(self, n_in_chans, n_actions, observation_shape, n_flat_actions):
        super().__init__()
        n_hid_1 = 16
        # Hack af. Pad the input to make it have root 2? idk, bad
        sq_i = 2
        assert observation_shape[-1] == observation_shape[-2]
        while sq_i < observation_shape[-1]:
            sq_i = sq_i ** 2
        pad_0 = sq_i - observation_shape[-1]
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, pad_0+1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 3, 2, 1, bias=True)
        self.flatten = torch.nn.Flatten()
#       n_flat = self.flatten(self.l3(self.l2(self.l1(torch.zeros(size=observation_shape))))).shape[-1]
        n_flat = n_hid_1
        self.d1 = Linear(n_flat, n_flat_actions)
#       self.d2 = Linear(16, n_flat_actions)
        self.layers = [self.l1, self.l2, self.d1]
        self.apply(init_weights)

    def forward(self, x):
        with torch.no_grad():
            x = self.l1(x)
            x = torch.nn.functional.relu(x)
            for i in range(int(np.log2(x.shape[2])) + 1):
                x = self.l2(x)
                x = torch.nn.functional.relu(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = torch.tanh(x)
#           x = self.d2(x)
#           x = torch.sigmoid(x)
            if not CA_ACTION:
                x = torch.stack([
                    unravel_index(x[i].argmax(), x[i].shape)
                    for i in range(x.shape[0])
                ])
        return x

class PlayerNN(nn.Module):
    def __init__(self, n_tile_types, n_actions=4):
        super().__init__()
        self.n_tile_types = n_tile_types
        assert 'zelda' in PROBLEM
        self.l1 = Conv2d(n_tile_types, 16, 3, 1, 0, bias=True)
        self.l2 = Conv2d(16, 16, 3, 2, 1, bias=True)
        self.l3 = Conv2d(16, n_actions, 3, 1, 1, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_play_weights)
        self.flatten = torch.nn.Flatten()
        self.net_reward = 0
        self.n_episodes = 0

    def forward(self, x):
        x = torch.Tensor(get_one_hot_map(x, self.n_tile_types))
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = torch.relu(self.l1(x))
            for i in range(int(np.log2(x.shape[2])) + 1):
#           for i in range(1):
                x = torch.relu(self.l2(x))
            x = torch.relu(self.l3(x))

#           x = x.argmax(1)
#           x = x[0]
            x = x.flatten()
            x = torch.softmax(x, axis=0)
           #x = [x.argmax().item()]
            act_ids = np.arange(x.shape[0])
            probs = x.detach().numpy()
            x = np.random.choice(act_ids, 1, p=probs)

        return x

    def assign_reward(self, rew):
        self.net_reward += rew
        self.n_episodes += 1

    def reset(self):
        self.net_reward = 0
        self.n_episodes = 0

    def get_reward(self):
        mean_rew = self.net_reward / self.n_episodes
        return mean_rew




def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)


def init_play_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, gain=0)
        m.bias.data.fill_(0.00)

    if type(m) == torch.nn.Conv2d:
#       torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.weight, 0)



def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
    """
    Use to get flat vector of weights from PyTorch model
    """
    init_params = []
    for lyr in nn.layers:
        init_params.append(lyr.weight.view(-1).numpy())
        init_params.append(lyr.bias.view(-1).numpy())
    init_params = np.hstack(init_params)
    print('number of initial NN parameters: {}'.format(init_params.shape))

    return init_params


def set_weights(nn, weights):
    with torch.no_grad():
        n_el = 0

        for layer in nn.layers:
            l_weights = weights[n_el:n_el + layer.weight.numel()]
            n_el += layer.weight.numel()
            l_weights = l_weights.reshape(layer.weight.shape)
            layer.weight = torch.nn.Parameter(torch.Tensor(l_weights))
            layer.weight.requires_grad = False
            b_weights = weights[n_el:n_el + layer.bias.numel()]
            n_el += layer.bias.numel()
            b_weights = b_weights.reshape(layer.bias.shape)
            layer.bias = torch.nn.Parameter(torch.Tensor(b_weights))
            layer.bias.requires_grad = False

    return nn

def get_one_hot_map(int_map, n_tile_types):
    obs = (np.arange(n_tile_types) == int_map[...,None]-1).astype(int)
    obs = obs.transpose(2, 0, 1)

    return obs


"""
Behavior Characteristics Functions
"""

def get_entropy(int_map, env):
    """
    Function to calculate entropy of levels represented by integers
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns the entropy of the level normalized roughly to a range of 0.0 to 1.0
    """
    #FIXME: make this robust to different action spaces
    n_classes = len(env._prob._prob)
    max_val = -(1 / n_classes)*np.log(1 / n_classes)* n_classes
    total = len(int_map.flatten())
    entropy = 0.0
    for tile in range(n_classes):
        p = (tile == int_map.flatten()).astype(int).sum() / total
        if p != 0:
            entropy -= p*np.log(p)
    return entropy / max_val


def get_counts(int_map, env):
    """
    Function to calculate the tile counts for all possible tiles
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a python list with tile counts for each tile normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width * env._prob._height  # for example 14*14=196
    return [np.sum(int_map.flatten() == tile)/max_val for tile in range(len(env._prob._prob))]


def get_emptiness(int_map, env):
    """
    Function to calculate how empty the level is
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns an emptiness value normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width * env._prob._height  # for example 14*14=196
    return np.sum(int_map.flatten() == 0) / max_val

def get_hor_sym(int_map, env):
    """
    Function to get the horizontal symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width*env._prob._height/2  # for example 14*14/2=98
    m = 0
    if int(int_map.shape[0])%2==0:
        m = np.sum((int_map[:int(int_map.shape[0]/2)] == np.flip(int_map[int(int_map.shape[0]/2):],0)).astype(int))
        m = m/max_val
    else:
        m = np.sum((int_map[:int(int_map.shape[0]/2)] == np.flip(int_map[int(int_map.shape[0]/2)+1:],0)).astype(int))
        m = m/max_val
    return m

def get_ver_sym(int_map, env):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width*env._prob._height/2  # for example 14*14/2=98
    m = 0
    if int(int_map.shape[1])%2==0:
        m = np.sum((int_map[:,:int(int_map.shape[1]/2)] == np.flip(int_map[:,int(int_map.shape[1]/2):],1)).astype(int))
        m = m/max_val
    else:
        m = np.sum((int_map[:,:int(int_map.shape[1]/2)] == np.flip(int_map[:,int(int_map.shape[1]/2)+1:],1)).astype(int))
        m = m/max_val
    return m

# SYMMETRY
def get_sym(int_map, env):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    result = (get_ver_sym(int_map, env) + get_hor_sym(int_map, env))/2.0
    return result

# CO-OCCURRANCE
def get_co(int_map, env):
    max_val = env._prob._width*env._prob._height*4
    result = (np.sum((np.roll(int_map, 1, axis=0) == int_map).astype(int))+
    np.sum((np.roll(int_map, -1, axis=0) == int_map).astype(int))+
    np.sum((np.roll(int_map, 1, axis=1) == int_map).astype(int))+
    np.sum((np.roll(int_map, -1, axis=1) == int_map).astype(int)))
    return result/max_val

def get_regions(stats):
    return stats['regions']

def get_path_length(stats):
    return stats['path-length']

# TODO: call this once to return the releveant get_bc function, then call this after each eval, so that we don't have to repeatedly compare strings
def get_bc(bc_name, int_map, stats, env):
    if bc_name in stats.keys():
        return stats[bc_name]
    elif bc_name == 'co-occurance':
        return get_co(int_map, env)
    elif bc_name == 'symmetry':
        return get_sym(int_map, env)
    elif bc_name == 'symmetry-vertical':
        return get_ver_sym(int_map, env)
    elif bc_name == 'symmetry-horizontal':
        return get_hor_sym(int_map, env)
    elif bc_name == 'emptiness':
        return get_emptiness(int_map, env)
    elif bc_name == 'entropy':
        return get_entropy(int_map, env)
    elif bc_name == 'NONE':
        return 0
    else:  
        print('The BC {} is not recognized.'.format(bc_name))
        raise Exception
        return 0.0

class PlayerLeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_i = 0

    def forward(self, obs):
        return [0]

class RandomPlayer(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space
        self.act_i = 0

    def forward(self, obs):
        return [self.action_space.sample()]

class PlayerRight(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_i = 0

    def forward(self, obs):
        return [1]

def log_archive(archive, name, itr, start_time, level_json=None):
    # TensorBoard Logging.
    df = archive.as_pandas(include_solutions=False)
    elapsed_time = time.time() - start_time
    writer.add_scalar('{} ArchiveSize'.format(name), len(df), itr)
    writer.add_scalar('{} score/mean'.format(name), df['objective'].mean(), itr)
    writer.add_scalar('{} score/max'.format(name), df['objective'].max(), itr)
    writer.add_scalar('{} score/min'.format(name), df['objective'].min(), itr)

    # Change: log mean, max, and min for all stats
    if level_json:
        stats = ['batch_reward','variance','diversity','targets']
        # level_json = {'level': final_levels.tolist(),'batch_reward':[batch_reward] * len(final_levels.tolist()), 'variance': [variance_penalty] * len(final_levels.tolist()), 'diversity':[diversity_bonus] * len(final_levels.tolist()),'targets':trg.tolist(), **bc_dict}
        for stat in stats:
            writer.add_scalar('Training {}/min'.format(stat), np.min(level_json[stat]), itr)
            writer.add_scalar('Training {}/mean'.format(stat), np.mean(level_json[stat]), itr)
            writer.add_scalar('Training {}/max'.format(stat), np.max(level_json[stat]), itr)
 
    # Logging.
    if itr % 1 == 0:
        print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
        print(f"  - {name} Archive Size: {len(df)}")
        print(f"  - {name} Max Score: {df['objective'].max()}")
        print(f"  - {name} Mean Score: {df['objective'].mean()}")
        print(f"  - {name} Min Score: {df['objective'].min()}")

N_PLAYER_STEPS = 100

def play_level(env, level, player):
    env._rep._old_map = level
    env._rep._random_start = False
    p_obs = env.reset()
    if not env.is_playable():
        return 0, None
    # TODO: check if env is playable!
    env.set_active_agent(1)
    if RENDER:
        env.render()
    net_p_rew = 0
    action_hist = []
    for p_i in range(N_PLAYER_STEPS):
        action = player(p_obs['map'])
        if isinstance(action, torch.Tensor):
            # TODO: this logic belongs with the model
            player_coords = env._prob.player.coords
            action = np.array(action)[player_coords[0], player_coords[1]]
        elif isinstance(action, list) or isinstance(action, np.ndarray):
            assert len(action) == 1
            action = action[-1]
        else:
            raise Exception
        action_hist.append(action)
        p_obs, p_rew, p_done, p_info = env.step(action)
        if RENDER:
            env.render()
       #net_p_rew += p_rew
        net_p_rew = p_rew
        if p_done:
            break
#   player.assign_reward(net_p_rew)
    action_freqs = np.bincount(action_hist, minlength=len(env.player_actions))
    action_entropy = scipy.stats.entropy(action_freqs)
    local_action_entropy = np.mean(
            [scipy.stats.entropy(np.bincount(action_hist[i:i+10], minlength=len(env.player_actions))) for i in np.arange(0, len(action_hist) - 10, 6)])
    local_action_entropy = np.nan_to_num(local_action_entropy)
    return net_p_rew, [action_entropy, local_action_entropy]

@ray.remote
def multi_evo(env, model, model_w, n_tile_types, init_states, bc_names, static_targets, seed, player_1, player_2):
    set_weights(model, model_w)
    level_json, obj, bcs = simulate(
        env=env,
        model=model,
        n_tile_types=n_tile_types,
        init_states=init_states,
        bc_names=bc_names,
        static_targets=static_targets,
        seed=seed,
        player_1=player_1,
        player_2=player_2)
    return level_json, obj, bcs

@ray.remote
def multi_play_evo(env, gen_model, player_1_w, n_tile_types, init_states, play_bc_names, static_targets, seed, player_1, player_2, playable_levels):
    set_weights(player_1, player_1_w)
    obj, bcs = player_simulate(
        env=env,
        n_tile_types=n_tile_types,
        play_bc_names=play_bc_names,
        seed=seed,
        player_1=player_1,
        playable_levels=playable_levels)
    return obj, bcs

def gen_playable_levels(env, gen_model, init_states, n_tile_types):
    ''' To get only the playable levels of a given generator, so that we can run player evaluations on them more quickly.'''
    final_levels = []
    for int_map in init_states:
        obs = get_one_hot_map(int_map, n_tile_types)
        if RENDER:
            env.render()
        done = False
        n_step = 0
        last_int_map = None
        while not done:
            int_tensor = torch.unsqueeze(torch.Tensor(obs), 0)
            action = gen_model(int_tensor)[0].numpy()
            obs = action
            int_map = action.argmax(axis=0)
            env._rep._map = int_map
            done = (int_map == last_int_map).all() or n_step >= N_STEPS
            if INFER:
                time.sleep(1/30)
            if done:
                env._rep._old_map = int_map
                env._rep._random_start = False
                _ = env.reset()
                if env.is_playable():
                    final_levels.append(int_map)
            n_step += 1
    return final_levels

def player_simulate(env, n_tile_types, play_bc_names, player_1, playable_levels, seed=None):
    n_evals = 10
    net_reward = 0
    bcs = []
    for int_map in playable_levels * n_evals:
        if INFER:
#           env.render()
            input('ready player 1')
        p_1_rew, p_bcs = play_level(env, int_map, player_1)
        bcs.append(p_bcs)
        if INFER:
            print('p_1 reward: ', p_1_rew)
        net_reward += p_1_rew

    reward = net_reward / len(playable_levels * n_evals)
    bcs = [np.mean([bcs[j][i] for j in range(len(bcs))]) for i in range(len(bcs[0]))]
    return reward, bcs



def simulate(env, model, n_tile_types, init_states, bc_names, static_targets, seed=None, player_1=None, player_2=None):
    """
    Function to run a single trajectory and return results.

    Args:
        env (gym.Env): A copy of the binary-wide-v0 environment.
        model (np.ndarray): The array of weights for the policy.
        seed (int): The seed for the environment.
        player_sim (bool): Are we collecting obj and bcs for the player, rather than the generator?
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        path_length (float): The path length of the final solution.
        regions (float): The number of distinct regions of the final solution.
    """
    if seed is not None:
        env.seed(seed)

    if PLAY_LEVEL:
        assert player_1 is not None
        assert player_2 is not None
    # Allow us to manually set the level-map on reset (using the "_old_map" attribute)
    # Actually we have found a more efficient workaround for now.
#   env._rep._random_start = False
#   if n_episode == 0 and False:
#       env._rep._old_map = init_state
#       obs = env.reset()
#       int_map = obs['map']
    n_init_states = init_states.shape[0]
    width = init_states.shape[1]
    height = init_states.shape[2]
    bcs = np.empty(shape=(len(bc_names), n_init_states))
    if SAVE_LEVELS:
        trg = np.empty(shape=(n_init_states))
    final_levels = np.empty(shape=init_states.shape, dtype=np.uint8)
    batch_reward = 0
    batch_time_penalty = 0
    batch_targets_penalty = 0
    batch_play_bonus = 0

    for (n_episode, init_state) in enumerate(init_states):
        # NOTE: Sneaky hack. We don't need initial stats. Never even reset. Heh. Be careful!!
        env._rep._map = init_state.copy()
        # Only applies to narrow and turtle. Better than using reset, but ugly, and not optimal
        # TODO: wrap the env instead
        env._rep._x = np.random.randint(env._prob._width)
        env._rep._y = np.random.randint(env._prob._height)
        int_map = init_state
        obs = get_one_hot_map(int_map, n_tile_types)
        if RENDER:
            env.render()
            if INFER:
#               time.sleep(10/30)
#               input()
                pass
        done = False

        n_step = 0
        while not done:
#           in_tensor = torch.unsqueeze(
#               torch.unsqueeze(torch.tensor(np.float32(obs['map'])), 0), 0)
            in_tensor = torch.unsqueeze(torch.Tensor(obs), 0)
            action = model(in_tensor)[0].numpy()
            # There is probably a better way to do this, so we are not passing unnecessary kwargs, depending on representation
            action, skip = preprocess_action(
                action,
                int_map=env._rep._map,
                x=env._rep._x,
                y=env._rep._y,
                n_dirs=N_DIRS,
                n_tiles=n_tile_types)
            change, x, y = env._rep.update(action)
            int_map = env._rep._map
            obs = get_one_hot_map(env._rep.get_observation()['map'], n_tile_types)
            preprocess_observation(
                obs,
                x=env._rep._x,
                y=env._rep._y)
#           int_map = action.argmax(axis=0)
#           obs = get_one_hot_map(int_map, n_tile_types)
#           env._rep._map = int_map
            done = not (change or skip) or n_step >= N_STEPS
           #done = n_step >= N_STEPS
            if INFER:
                time.sleep(1/30)
            if done:
                final_levels[n_episode] = int_map
                stats = env._prob.get_stats(
                    get_string_map(int_map,
                                   env._prob.get_tile_types()))


                # get BCs
                # Resume here. Use new BC function.
                for i in range(len(bc_names)):
                    bc_name = bc_names[i]
                    bcs[i, n_episode] = get_bc(bc_name, int_map, stats, env)


                # TODO: reward calculation should depend on self.reward_names
                # ad hoc reward: shorter episodes are better?
                time_penalty = n_step
                batch_time_penalty -= time_penalty

                # we want to hit each of our static targets exactly, penalize for anything else.
                # for ranges, we take our least distance to any element in the range
                targets_penalty = 0
                for k in static_targets:
                    if k in bc_names:
                        continue
                    if isinstance(static_targets[k], tuple):
                        # take the smalles distance from current value to any point in range
                        targets_penalty += abs(np.arange(*static_targets[k]) - stats[k]).min()
                    else:
                        targets_penalty += abs(static_targets[k] - stats[k])
                #                   targets_penalty = np.sum([abs(static_targets[k] - stats[k]) if not isinstance(static_targets[k], tuple) else abs(np.arange(*static_targets[k]) - stats[k]).min() for k in static_targets])
                batch_targets_penalty -= targets_penalty
                if SAVE_LEVELS:
                    trg[n_episode] = -targets_penalty



                if PLAY_LEVEL:
                    if INFER:
                        env.render()
                        input('ready player 1')
                    p_1_rew, p_bcs = play_level(env, int_map, player_1)
                    if INFER:
                        print('p_1 reward: ', p_1_rew)
                        input('ready player 2')
                    p_2_rew, p_bcs = play_level(env, int_map, player_2)
                    if INFER:
                        print('p_2 reward: ', p_2_rew)

                    max_regret = env._prob.max_reward - env._prob.min_reward
                    # add this in case we get worst possible regret (don't want to punish a playable map)
                    batch_play_bonus += max_regret + p_1_rew - p_2_rew



            if RENDER:
                env.render()
            if done and INFER:
                time.sleep(5/30)
                print('stats: {}\n\ntime_penalty: {}\n targets_penalty: {}'.format(stats, time_penalty, targets_penalty))
            last_int_map = int_map
            n_step += 1
    final_bcs = [bcs[i].mean() for i in range(bcs.shape[0])]
    batch_targets_penalty = 10 * batch_targets_penalty / N_INIT_STATES
   #batch_targets_penalty = batch_targets_penalty / N_INIT_STATES
    batch_reward += batch_targets_penalty
    if PLAY_LEVEL:
        batch_reward += batch_play_bonus / N_INIT_STATES
        time_penalty, targets_penalty, variance_penalty, diversity_bonus = None, None, None, None
    else:
#       batch_time_penalty = batch_time_penalty / N_INIT_STATES
        if N_INIT_STATES > 1 and (batch_targets_penalty == 0 or not CASCADE_REWARD):
            # Variance penalty is the negative average (per-BC) standard deviation from the mean BC vector.
            variance_penalty = - np.sum([bcs[i].std()
                              for i in range(bcs.shape[0])]) / bcs.shape[0]
            # Diversity bonus. We want minimal variance along BCS *and* diversity in terms of the map.
            # Sum pairwise hamming distances between all generated maps.
            diversity_bonus = np.sum([np.sum(final_levels[j] != final_levels[k]) if j != k else 0 for k in range(N_INIT_STATES) for j in range(N_INIT_STATES)]) / (N_INIT_STATES * N_INIT_STATES - 1) 
            # ad hoc scaling :/
            diversity_bonus = 10 * diversity_bonus / (width * height)
            batch_reward = batch_reward + max(0, variance_penalty + diversity_bonus)
        else:
            variance_penalty = None
            diversity_bonus = None

    if not INFER:
        if SAVE_LEVELS:
            bc_dict = {}
            for i in range(len(bc_names)):
                bc_name = bc_names[i]
                bc_dict[bc_name] = bcs[i, :].tolist()
            level_json = {'level': final_levels.tolist(),'batch_reward':[batch_reward] * len(final_levels.tolist()), 'variance': [variance_penalty] * len(final_levels.tolist()), 'diversity':[diversity_bonus] * len(final_levels.tolist()),'targets':trg.tolist(), **bc_dict}
        else:
            level_json = None
        return level_json, batch_reward, final_bcs
    else:
        return batch_reward, final_bcs, (time_penalty, batch_targets_penalty, variance_penalty, diversity_bonus)




class EvoPCGRL():
    def __init__(self):
        self.init_env()
        assert self.env.observation_space['map'].low[0, 0] == 0
        # get number of tile types from environment's observation space
        # here we assume that all (x, y) locations in the observation space have the same upper/lower bound
        self.n_tile_types = self.env.observation_space['map'].high[0, 0] + 1
        self.width = self.env.observation_space['map'].low.shape[0]
        self.height = self.env.observation_space['map'].low.shape[1]

        #FIXME why not?
        # self.width = self.env._prob._width

        # TODO: make reward a command line argument?
        # TODO: multi-objective compatibility?
        self.bc_names = BCS

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        self.bc_bounds = self.env._prob.cond_bounds
        self.bc_bounds.update({
                'co-occurance': (0.0, 1.0),
                'symmetry': (0.0, 1.0),
                'symmetry-vertical': (0.0, 1.0),
                'symmetry-horizontal': (0.0, 1.0),
                'emptiness': (0.0, 1.0),
                'entropy': (0.0, 1.0),
            })

        self.static_targets = self.env._prob.static_trgs

        if RANDOM_INIT_LEVELS:
            # If we are constantly providing new random seeds to generators, we will need to regularly re-evaluate elites
            gen_archive_cls = FlexArchive
        else:
            gen_archive_cls = GridArchive

        if PLAY_LEVEL:
            self.play_bc_names = ['action_entropy', 'local_action_entropy']
            self.play_bc_bounds = {
                    'action_entropy': (0, 4),
                    'local_action_entropy': (0, 4),
                }
            self.gen_archive = gen_archive_cls(
                [100 for _ in self.bc_names],
               #[1],
               #[(-1, 1)],
                [self.bc_bounds[bc_name] for bc_name in self.bc_names],
            )
            self.play_archive = FlexArchive(
                # minimum of: 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
               #[min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                [100 for _ in self.play_bc_names],
                # min/max for each BC
                [self.play_bc_bounds[bc_name] for bc_name in self.play_bc_names],
            )
        else:
            if CMAES:
                # Restrict the archive to 1 cell so that we are effectively doing CMAES. BCs should be ignored.
                self.gen_archive = gen_archive_cls(
                    [1, 1], [(0, 1), (0, 1)],
                )
            else:
                self.gen_archive = gen_archive_cls(
                    # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                   #[min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                    [100 for _ in self.bc_names],
                    # min/max for each BC
                    [self.bc_bounds[bc_name] for bc_name in self.bc_names],
                )


        reps_to_out_chans = {
            'cellular': self.n_tile_types,
            'wide': self.n_tile_types,
            'narrow': self.n_tile_types + 1,
            'turtle': self.n_tile_types + N_DIRS,
        }

        reps_to_in_chans = {
            'cellular': self.n_tile_types,
            'wide': self.n_tile_types,
            'narrow': self.n_tile_types + 1,
            'turtle': self.n_tile_types + 1,
        }

        n_out_chans = reps_to_out_chans[REPRESENTATION]
        n_in_chans = reps_to_in_chans[REPRESENTATION]
        if MODEL == "NCA":
            self.gen_model = GeneratorNN(n_in_chans=self.n_tile_types, n_actions=n_out_chans)
        elif MODEL == "CNN":
            # Adding n_tile_types as a dimension here. Why would this ever be the env's observation space though? Should be one-hot by default?
            observation_shape = (1, self.n_tile_types, *self.env.observation_space['map'].shape)
            if isinstance(self.env.action_space, gym.spaces.Box):
                action_shape = self.env.action_space.shape
                assert len(action_shape) == 3
                n_flat_actions = action_shape[0] * action_shape[1] * action_shape[2]
            elif isinstance(self.env.action_space, gym.spaces.MultiDiscrete):
                nvec = self.env.action_space.nvec
                assert len(nvec) == 3
                n_flat_actions = nvec[0] + nvec[1] + nvec[2]
            elif isinstance(self.env.action_space, gym.spaces.Discrete):
                n_flat_actions = self.env.action_space.n
            else:
                raise NotImplementedError("I don't know how to handle this action space: {}".format(type(self.env.action_space)))
            self.gen_model = GeneratorNNDense(n_in_chans=self.n_tile_types, n_actions=n_out_chans,
                                              observation_shape=observation_shape,
                                              n_flat_actions=n_flat_actions)
        set_nograd(self.gen_model)
        initial_w = get_init_weights(self.gen_model)
        assert len(initial_w.shape) == 1
        self.n_generator_weights = initial_w.shape[0]
        self.n_player_weights = 0
        # TODO: different initial weights per emitter as in pyribs lunar lander relanded example?


        if PLAY_LEVEL:
            gen_emitters = [
    #           ImprovementEmitter(
                OptimizingEmitter(
                    self.gen_archive,
                    initial_w.flatten(),
                    # TODO: play with initial step size?
                    1,  # Initial step size.
                    batch_size=30,
                ) for _ in range(5)  # Create 5 separate emitters.
            ]
            # Concatenate designer and player weights
            self.play_model = PlayerNN(self.n_tile_types, n_actions=len(self.env.player_actions))
            set_nograd(self.play_model)
            initial_play_w = get_init_weights(self.play_model)
            assert len(initial_play_w.shape) == 1
            self.n_player_weights = initial_play_w.shape[0]
            play_emitters = [
                OptimizingEmitter(
                    self.play_archive,
                    initial_play_w.flatten(),
                    # NOTE: Big step size, shit otherwise
                    1,  # Initial step size.
                    batch_size=30,
                ) for _ in range(5)  # Create 5 separate emitters.
            ]
            self.play_optimizer = Optimizer(self.play_archive, play_emitters)
        else:
            if CMAES:
                # The optimizing emitter will prioritize fitness over exploration of behavior space
                emitter_type = OptimizingEmitter
            else:
                emitter_type = ImprovementEmitter
            gen_emitters = [
                emitter_type(
                    self.gen_archive,
                    initial_w.flatten(),
                    # TODO: play with initial step size?
                    1,  # Initial step size.
                    batch_size=30,
                ) for _ in range(5)  # Create 5 separate emitters.
            ]
        self.gen_optimizer = Optimizer(self.gen_archive, gen_emitters)

        # These are the initial maps which will act as seeds to our NCA models
        if N_INIT_STATES == 0:
            # special square patch
            self.init_states = np.zeros(shape=(1, self.width, self.height))
            self.init_states[0, 5:-5, 5:-5] = 1
        else:
            self.init_states = np.random.randint(0, self.n_tile_types, (N_INIT_STATES, self.width, self.height))

        self.start_time = time.time()
        self.total_itrs = N_GENERATIONS
        self.n_itr = 1
        if PLAY_LEVEL:
            self.player_1 = PlayerNN(self.n_tile_types)
            self.player_2 = RandomPlayer(self.env.player_action_space)
        else:
            self.player_1 = None
            self.player_2 = None
        # This directory might already exist if a previous experiment failed before the first proper checkpoint/save
        if not os.path.isdir(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        # Save the command line arguments with which we launched
        with open(os.path.join(SAVE_PATH, 'settings.json'), 'w', encoding='utf-8') as f:
            json.dump(arg_dict, f, ensure_ascii=False, indent=4)

    def evolve(self):

        net_p_itr = 0
        for itr in tqdm(range(self.n_itr, self.total_itrs + 1)):
            # Request models from the optimizer.
            gen_sols = self.gen_optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            if RANDOM_INIT_LEVELS:
                init_states = np.random.randint(0, self.n_tile_types,
                                                size=self.init_states.shape)
            else:
                init_states = self.init_states
            if THREADS:
                futures = [multi_evo.remote(self.env,
                                            self.gen_model,
                                            model_w,
                                            self.n_tile_types,
                                            init_states,
                                            self.bc_names,
                                            self.static_targets,
                                            seed,
                                            player_1=self.player_1,
                                            player_2=self.player_2)
                           for model_w in gen_sols]
                results = ray.get(futures)
                for result in results:
                    level_json, m_obj, m_bcs = result
                    if SAVE_LEVELS:
                        df = pd.DataFrame(level_json)
                        df = df[df['targets'] == 0]
                        if len(df) > 0:
                            df.to_csv(os.path.join(SAVE_PATH, "levels.csv"), mode='a', header=False, index=False)
                    objs.append(m_obj)
                    bcs.append([*m_bcs])
            else:
                for model_w in gen_sols:
                    set_weights(self.gen_model, model_w)
                    level_json, m_obj, m_bcs = simulate(
                        env=self.env,
                        model=self.gen_model,
                        n_tile_types=self.n_tile_types,
                        init_states=init_states,
                        bc_names=self.bc_names,
                        static_targets=self.static_targets,
                        seed=seed,
                        player_1=self.player_1,
                        player_2=self.player_2,
                    )
                    if SAVE_LEVELS:
                        # Save levels to disc
                        df = pd.DataFrame(level_json)
                        df = df[df['targets'] == 0]
                        if len(df) > 0:
                            df.to_csv(os.path.join(SAVE_PATH, "levels.csv"), mode='a', header=False, index=False)
                    objs.append(m_obj)
                    bcs.append(m_bcs)

            # Send the results back to the optimizer.
            self.gen_optimizer.tell(objs, bcs)

            # Re-evaluate elite generators
            if RANDOM_INIT_LEVELS:
                df = self.gen_archive.as_pandas()
                high_performing = df.sort_values("objective", ascending=False)
                elite_models = np.array(high_performing.loc[:, "solution_0":])
                if THREADS:
                    futures = [multi_evo.remote(
                        self.env,
                        self.gen_model,
                        elite_models[i],
                        self.n_tile_types,
                        init_states,
                        self.bc_names,
                        self.static_targets,
                        seed,
                        player_1=self.player_1,
                        player_2=self.player_2) for i in range(min(len(elite_models), 150)
                    )]
                    results = ray.get(futures)
                    for result in results:
                        level_json, el_obj, el_bcs = result
                        if SAVE_LEVELS:
                            # Save levels to disc
                            df = pd.DataFrame(level_json)
                            df = df[df['targets'] == 0]
                            if len(df) > 0:
                                df.to_csv(os.path.join(SAVE_PATH, "levels.csv"), mode='a', header=False, index=False)
                        self.gen_archive.update_elite(el_obj, el_bcs)

                else:
                    # 150 to match number of new-model evaluations
                    for elite_i in range(min(len(elite_models), 150)):
                        gen_model_weights = elite_models[elite_i]
                        set_weights(self.gen_model, gen_model_weights)

                        level_json, el_obj, el_bcs = simulate(env=self.env,
                                                        model=self.gen_model,
                                                        n_tile_types=self.n_tile_types,
                                                        init_states=init_states,
                                                        bc_names=self.bc_names,
                                                        static_targets=self.static_targets,
                                                        seed=seed,
                                                        player_1=self.player_1,
                                                        player_2=self.player_2)

                        self.gen_archive.update_elite(el_obj, el_bcs)


            log_archive(self.gen_archive, 'Generator', itr, self.start_time, level_json)

            # FIXME: implement these
#           self.play_bc_names = ['action_entropy', 'action_entropy_local']
            if PLAY_LEVEL:
               #elite_model_w = self.gen_archive.get_random_elite()[0]
                df = self.gen_archive.as_pandas()
                high_performing = df.sort_values("objective", ascending=False)
                models = np.array(high_performing.loc[:, "solution_0":])
                np.random.shuffle(models)
                playable_levels = []
                for m_i in range(len(models)):
                    elite_model_w = models[m_i]
                    set_weights(self.gen_model, elite_model_w)
                    playable_levels += gen_playable_levels(self.env, self.gen_model, self.init_states, self.n_tile_types)
                    if len(playable_levels) >= 50:
                        break
                if len(playable_levels) >= 10:
                    play_start_time = time.time()
                    self.playable_levels = playable_levels
                    for p_itr in tqdm(range(1, 2)): 
                        net_p_itr += 1
                        play_sols = self.play_optimizer.ask()
                        objs, bcs = [], []
                        if THREADS:
                            futures = [multi_play_evo.remote(self.env, self.gen_model, player_w, self.n_tile_types, init_states, self.play_bc_names, self.static_targets, seed, player_1=self.player_1, player_2=self.player_2, playable_levels=playable_levels) for player_w in play_sols]
                            results = ray.get(futures)
                            for result in results:
                                m_obj, m_bcs = result
                                objs.append(m_obj)
                                bcs.append([*m_bcs])
                        else:
                            play_i = 0
                            for play_w in play_sols:
                                play_i += 1
                                set_weights(self.play_model, play_w)
                                m_obj, m_bcs = player_simulate(
                                    env=self.env,
                                    n_tile_types=self.n_tile_types,
                                    play_bc_names=self.play_bc_names,
                                    seed=seed,
                                    player_1=self.player_1,
                                    playable_levels=playable_levels,
                                )
                                objs.append(m_obj)
                                bcs.append(m_bcs)
                        self.play_optimizer.tell(objs, bcs)

                        #TODO: parallelize me
                        df = self.play_archive.as_pandas()
                        high_performing = df.sort_values("objective", ascending=False)
                        elite_models = np.array(high_performing.loc[:, "solution_0":])
                        for elite_i in range(10):
                            play_model_weights = elite_models[elite_i]
                            init_nn = set_weights(self.play_model, play_model_weights)

                            obj, bcs = player_simulate(self.env, self.n_tile_types, self.play_bc_names, init_nn, playable_levels=playable_levels)

                            self.play_archive.update_elite(obj, bcs)

                           #    m_objs.append(obj)
                           #bc_a = get_bcs(init_nn)
                           #obj = np.mean(m_objs)
                           #objs.append(obj)
                           #bcs.append([bc_a])
                        log_archive(self.play_archive, 'Player', p_itr, play_start_time)

                        if net_p_itr > 0 and net_p_itr % SAVE_INTERVAL == 0:
                            self.save()

                        df = self.play_archive.as_pandas()
                        high_performing = df.sort_values("objective", ascending=False)
                        elite_scores = np.array(high_performing.loc[:, "objective"])

                        if np.array(elite_scores).max() >= self.env._prob.max_reward:
                            break

                    # TODO: assuming an archive of one here! Make it more general, like above for generators
                    set_weights(self.play_model, self.play_archive.get_random_elite()[0])


            # Save checkpoint
            if itr % SAVE_INTERVAL == 0:
                self.save()

            self.n_itr += 1

    def save(self):
        global ENV
        ENV = self.env
        self.env = None
        pickle.dump(self, open(os.path.join(SAVE_PATH, "evolver.pkl"), 'wb'))
        self.env = ENV

    def init_env(self):
        '''Initialize the PCGRL level-generation RL environment and extract any useful info from it.'''

        env_name = '{}-{}-v0'.format(PROBLEM, REPRESENTATION)
        self.env = gym.make(env_name)
        if CMAES:
            # Give a little wiggle room from targets, to allow for some diversity
            if "binary" in PROBLEM:
                path_trg = self.env._prob.static_trgs['path-length']
                self.env._prob.static_trgs.update({'path-length': (path_trg - 20, path_trg)})
            elif "zelda" in PROBLEM:
                path_trg = self.env._prob.static_trgs['path-length']
                self.env._prob.static_trgs.update({'path-length': (path_trg - 40, path_trg)})
            elif "sokoban" in PROBLEM:
                sol_trg = self.env._prob.static_trgs['sol-length']
                self.env._prob.static_trgs.update({'sol-length': (sol_trg - 10, sol_trg)})
            elif "smb" in PROBLEM:
                pass
            else:
                raise NotImplemented


        global N_DIRS
        if hasattr(self.env._rep, '_dirs'):
            N_DIRS = len(self.env._rep._dirs)
        else:
            N_DIRS = 0

        global N_STEPS
        if N_STEPS is None:
            max_ca_steps = 10
            max_changes = self.env._prob._width * self.env._prob._height
            reps_to_steps = {
                'cellular': max_ca_steps,
                'wide': max_changes,
                'narrow': max_changes,
                'turtle': max_changes * 2,  # So that it can move around to each tile I guess
            }
            N_STEPS = reps_to_steps[REPRESENTATION]

    def visualize(self):
        archive = self.gen_archive
        # # Visualize Result
        plt.figure(figsize=(8, 6))
#       grid_archive_heatmap(archive, vmin=self.reward_bounds[self.reward_names[0]][0], vmax=self.reward_bounds[self.reward_names[0]][1])
#       if PROBLEM == 'binary':
#           vmin = -20
#           vmax = 20
#       elif PROBLEM == 'zelda':
#           vmin = -20
#           vmax = 20
#       grid_archive_heatmap(archive, vmin=vmin, vmax=vmax)
        df_obj = archive.as_pandas()['objective']
        vmin = np.floor(df_obj.min())
        vmax = np.ceil(df_obj.max())
        grid_archive_heatmap(archive, vmin=vmin, vmax=vmax)

#       plt.gca().invert_yaxis()  # Makes more sense if larger BC_1's are on top.
        plt.xlabel(self.bc_names[0])
        plt.ylabel(self.bc_names[1])
        plt.savefig('{}.png'.format(SAVE_PATH))
        if SHOW_VIS:
           plt.show()

        # Print table of results
        df = archive.as_pandas()
        # high_performing = df[df["objective"] > 200].sort_values("objective", ascending=False)
        print(df)

    def infer(self):
        assert INFER
        self.init_env()
        archive = self.gen_archive
        df = archive.as_pandas()
#       high_performing = df[df["behavior_1"] > 50].sort_values("behavior_1", ascending=False)
        if 'binary' in PROBLEM:
#           high_performing = df.sort_values("behavior_1", ascending=False)
            high_performing = df.sort_values("objective", ascending=False)
        if 'zelda' in PROBLEM:
            # path lenth
#           high_performing = df.sort_values("behavior_1", ascending=False)
            # nearest enemies
#           high_performing = df.sort_values("behavior_0", ascending=False)
            high_performing = df.sort_values("objective", ascending=False)
        else:
            high_performing = df.sort_values("objective", ascending=False)
        rows = high_performing
        models = np.array(rows.loc[:, "solution_0":])
        bcs_0 = np.array(rows.loc[:, "behavior_0"])
        bcs_1 = np.array(rows.loc[:, "behavior_1"])
        objs = np.array(rows.loc[:, "objective"])

        if RENDER_LEVELS:
            global RENDER
            global N_INIT_STATES
            RENDER = False
            N_INIT_STATES = 1

            if 'smb' in PROBLEM:
                d = 4
                figw, figh = 32, 4
            else:
                d = 6  # number of rows and columns
                figw, figh = self.env._prob._width, self.env._prob._height
            fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figw, figh))

            df_g = df.sort_values(by=['behavior_0', 'behavior_1'], ascending=False)

            df_g['row'] = np.floor(np.linspace(0, d, len(df_g), endpoint=False)).astype(int)

            for row_num in range(d):
                row = df_g[df_g['row']==row_num]
                row = row.sort_values(by=['behavior_1'], ascending=True)
                row['col'] = np.arange(0,len(row), dtype=int)
                idx = np.floor(np.linspace(0,len(row)-1,d)).astype(int)
                row = row[row['col'].isin(idx)]
                row = row.drop(['row','col'], axis=1)
                grid_models = np.array(row.loc[:,'solution_0':])
                for col_num in range(len(row)):
                    model = grid_models[col_num]
                    axs[row_num,col_num].set_axis_off()

                    # initialize weights
                    init_nn = set_weights(self.gen_model, model)

                    # run simulation, but only on the first level-seed
                    _, _, (time_penalty, targets_penalty, variance_penalty, diversity_bonus) = simulate(self.env, init_nn,
                                    self.n_tile_types, self.init_states[0:1], self.bc_names, self.static_targets, seed=None)
                    # Get image
                    img = self.env.render(mode='rgb_array')
                    axs[row_num,col_num].imshow(img, aspect='auto')
            fig.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.tight_layout()
            fig.savefig(os.path.join(SAVE_PATH, 'levelGrid_{}-bin.png'.format(d)), dpi=300)

        if PLAY_LEVEL:
            player_simulate(self.env, self.n_tile_types, self.play_bc_names, self.play_model, playable_levels=self.playable_levels, seed=None)
        i = 0
        if EVALUATE:
            RENDER = False
            # Iterate through our archive of trained elites, evaluating them and storing stats about them.
            # Borrowing logic from grid_archive_heatmap from pyribs.

            # Retrieve data from archive
            lower_bounds = archive.lower_bounds
            upper_bounds = archive.upper_bounds
            x_dim, y_dim = archive.dims
            x_bounds = np.linspace(lower_bounds[0], upper_bounds[0], x_dim + 1)
            y_bounds = np.linspace(lower_bounds[1], upper_bounds[1], y_dim + 1)

            # Color for each cell in the heatmap
            playability_scores = np.full((y_dim, x_dim), np.nan)
            diversity_scores = np.full((y_dim, x_dim), np.nan)
            reliability_scores = np.full((y_dim, x_dim), np.nan)

            idxs_0 = np.array(rows.loc[:, "index_0"])
            idxs_1 = np.array(rows.loc[:, "index_1"])

            if RANDOM_INIT_LEVELS:
                # Effectively doing inference on a (presumed) held-out set of levels
                N_EVAL_STATES = 100
                init_states = np.random.randint(0, self.n_tile_types, size=(N_EVAL_STATES, *self.init_states.shape[1:]))
            else:
                init_states = self.init_states

            if THREADS:
                raise NotImplementedError
            else:
                while i < len(models):
                    # iterate through all models and record stats, on either training seeds or new ones (to test generalization)
                    model = models[i]
                    id_0 = idxs_0[i]
                    id_1 = idxs_1[i]


                    # TODO: Parallelize me
                    init_nn = set_weights(self.gen_model, model)
                    _, _, (time_penalty, targets_penalty, variance_penalty, diversity_bonus) = \
                        simulate(self.env,
                                init_nn,
                                self.n_tile_types,
                                init_states,
                                self.bc_names,
                                self.static_targets,
                                seed=None,
                                player_1 = self.player_1,
                                player_2 = self.player_2)
                    playability_scores[id_0, id_1] = targets_penalty
                    if diversity_bonus is not None:
                        diversity_scores[id_0, id_1] = diversity_bonus
                    if variance_penalty is not None:
                        reliability_scores[id_0, id_1] = variance_penalty

                def plot_score_heatmap(scores, score_name, cmap_str="magma"):
                    ax = plt.gca()
                    ax.set_xlim(lower_bounds[0], upper_bounds[0])
                    ax.set_ylim(lower_bounds[0], upper_bounds[0])
                    vmin = np.min(scores)
                    vmax = np.max(scores)
                    t = ax.pcolormesh(x_bounds, y_bounds, scores, cmap=matplotlib.cm.get_cmap(cmap_str), vmin=vmin, vmax=vmax)
                    ax.figure.colorbar(t, ax=ax, pad=0.1)
                    if SHOW_VIS:
                        plt.show()
                    f_name = score_name
                    if RANDOM_INIT_LEVELS:
                        f_name = f_name + '_randSeeds'
                    f_name += '.png'
                    plt.savefig(os.path.join(SAVE_PATH, f_name))
                    plt.close()

                plot_score_heatmap(playability_scores, 'playability')
                plot_score_heatmap(diversity_scores, 'diversity')
                plot_score_heatmap(reliability_scores, 'reliability')
                stats = {
                    'playability': np.mean(playability_scores),
                    'diversity': np.mean(diversity_scores),
                    'reliability': np.mean(reliability_scores),
                }
                f_name = 'stats'
                if RANDOM_INIT_LEVELS:
                    f_name = f_name + '_randSeeds'
                f_name += '.json'
                with open(os.path.join(SAVE_PATH, f_name), 'w', encodign='utf-8') as f:
                    json.dump(stats, f, ensure_ascii=False, indent=4)
                return


        while True:
#           model = self.archive.get_random_elite()[0]
#           model = models[np.random.randint(len(models))]
            model = models[i]
            init_nn = set_weights(self.gen_model, model)
            if RANDOM_INIT_LEVELS:
                init_states = np.random.randint(0, self.n_tile_types, size=self.init_states.shape)
            else:
                init_states = self.init_states
            _, _, (time_penalty, targets_penalty, variance_penalty, diversity_bonus) = simulate(self.env, init_nn,
                            self.n_tile_types, init_states, self.bc_names, self.static_targets, seed=None, player_1=self.player_1, player_2=self.player_2)
            input("Mean behavior characteristics:\n\t{}: {}\n\t{}: {}\nMean reward:\n\tTotal: {}\n\ttime: {}\n\ttargets: {}\n\tvariance: {}\n\tdiversity: {}\nPress any key for next generator...".format(
                self.bc_names[0], bcs_0[i], self.bc_names[1], bcs_1[i], objs[i], time_penalty, targets_penalty, variance_penalty, diversity_bonus))
            i += 1

            if i == len(models):
                i = 0

if __name__ == '__main__':
    """
    Set Parameters
    """
    seed = 420
    CA_ACTION = True

    opts = argparse.ArgumentParser(
        description='Evolving Neural Cellular Automata for PCGRL')
    opts.add_argument(
        '-p',
        '--problem',
        help='Which game to generate levels for (PCGRL "problem").',
        default='binary_ctrl',
    )
    opts.add_argument(
        '-e',
        '--exp_name',
        help='Name of the experiment, for save files.',
        default='test_0',
    )
    opts.add_argument(
        '-ng',
        '--n_generations',
        type=int,
        help='Number of generations for which to run evolution.',
        default=10000,
    )
    opts.add_argument(
        '-nis',
        '--n_init_states',
        help='The number of initial states on which to evaluate our models. 0 for a single fixed map with a square of wall in the centre.',
        type=int,
        default=10,
    )
    opts.add_argument(
        '-ns',
        '--n_steps',
        help='Maximum number of steps in each generation episode.',
        type=int,
        default=None,
    )
    opts.add_argument(
        '-bcs',
        '--behavior_characteristics',
        nargs='+',
        help='A list of strings corresponding to the behavior characteristics that will act as the dimensions for our grid of elites during evolution.',
        default=['NONE'],
    )
    opts.add_argument(
        '-r',
        '--render',
        help='Render the environment.',
        action='store_true',
    )
    opts.add_argument(
        '-i',
        '--infer',
        help='Run inference with evolved archive of individuals.',
        action='store_true',
    )
    opts.add_argument(
        '-v',
        '--visualize',
        help='Visualize heatmap of the archive of individuals.',
        action='store_true',
    )
    opts.add_argument(
        '--show_vis',
        help='Render visualizations in matplotlib rather than saving them to png.',
        action='store_true',
    )
    opts.add_argument(
        '-g',
        '--render_levels',
        help='Save grid of levels to png.',
        action='store_true',
    )
    opts.add_argument(
        '-m',
        '--multi_thread',
        help='Use multi-thread evolution process.',
        action='store_true',
    )
    opts.add_argument(
        '--play_level',
        help="Use a playing agent to evaluate generated levels.",
        action='store_true',
    )
    opts.add_argument(
        '-ev',
        '--evaluate',
        help='Run evaluation on the archive of elites.',
        action='store_true',
    )
    opts.add_argument(
        '-s',
        '--save_levels',
        help='Save all levels to a csv.',
        action='store_true',
    )
    opts.add_argument(
        '-fix',
        '--fixed_init_levels',
        help='Use a fixed set of random levels throughout evolution, rather than providing the generators with new random initial levels during evaluation.',
        action='store_true',
    )
    opts.add_argument(
        '-cr',
        '--cascade_reward',
        help='Incorporate diversity/variance bonus/penalty into fitness only if targets are met perfectly (rather than always incorporating them).',
        action='store_true',
    )
    opts.add_argument(
        '-rep',
        '--representation',
        help='The interface between generator-agent and environment. cellular: agent acts as cellular automaton, observing and'
             ' supplying entire next stats. wide: agent observes entire stats, and changes any one tile. narrow: agent '
             'observes state and target tile, and selects built at target tile. turtle: agent selects build at current '
             'tile or navigates to adjacent tile.',
        default='cellular',
    )
    opts.add_argument(
        '-la',
        '--load_args',
        help='Rather than having the above args supplied by the command-line, load them from a settings.json file. (Of '
             'course, the value of this arg in the json will have no effect.)',
        type=int,
        default=None,
    )
    opts.add_argument(
        '--model',
        help='Which neural network architecture to use for the generator. NCA: just conv layers. CNN: Some conv layers, then a dense layer.',
        default='NCA',
    )


    opts = opts.parse_args()
    arg_dict = vars(opts)
    if opts.load_args is not None:
        with open('configs/evo/settings_{}.json'.format(opts.load_args)) as f:
            new_arg_dict = json.load(f)
            arg_dict.update(new_arg_dict)
    global INFER
    global EVO_DIR
    global CUDA
    global RENDER
    global PROBLEM
    global SHOW_VIS
    global VISUALIZE
    global N_STEPS
    global N_GENERATIONS
    global N_INIT_STATES
    global N_INFER_STEPS
    global BCS
    global RENDER_LEVELS
    global THREADS
    global PLAY_LEVEL
    global CMAES
    global EVALUATE
    global SAVE_LEVELS
    global RANDOM_INIT_LEVELS
    global CASCADE_REWARD
    global REPRESENTATION
    global MODEL
    global preprocess_action
    MODEL = arg_dict['model']
    REPRESENTATION = arg_dict['representation']
    CASCADE_REWARD = arg_dict['cascade_reward']
    RANDOM_INIT_LEVELS = not arg_dict['fixed_init_levels']
    CMAES = arg_dict['behavior_characteristics'] == ["NONE"]
    EVALUATE = arg_dict['evaluate']
    PLAY_LEVEL = arg_dict['play_level']
    BCS = arg_dict['behavior_characteristics']
    N_GENERATIONS = arg_dict['n_generations']
    N_INIT_STATES = arg_dict['n_init_states']
    N_STEPS = arg_dict['n_steps']

    SHOW_VIS = arg_dict['show_vis']
    PROBLEM = arg_dict['problem']
    CUDA = False
    VISUALIZE = arg_dict['visualize']
    INFER = arg_dict['infer']
    N_INFER_STEPS = N_STEPS
#   N_INFER_STEPS = 100
    RENDER_LEVELS = arg_dict['render_levels']
    THREADS = arg_dict['multi_thread']
    SAVE_INTERVAL = 100
    preprocess_action = preprocess_action_funcs[MODEL][REPRESENTATION]
    preprocess_observation = preprocess_observation_funcs[MODEL][REPRESENTATION]



    if THREADS:
        ray.init()
    SAVE_LEVELS = arg_dict['save_levels']

#   exp_name = 'EvoPCGRL_{}-{}_{}_{}-batch_{}-step_{}'.format(PROBLEM, REPRESENTATION, BCS, N_INIT_STATES, N_STEPS, arg_dict['exp_name'])
    exp_name = 'EvoPCGRL_{}-{}_{}_{}_{}-batch_{}'.format(PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES, arg_dict['exp_name'])
    if CASCADE_REWARD:
        exp_name += '_cascRew'
    if not RANDOM_INIT_LEVELS:
        exp_name += '_fixLvls'
    SAVE_PATH = os.path.join('evo_runs', exp_name)

    # Create TensorBoard Log Directory if does not exist
    LOG_NAME = './runs/' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '-' + exp_name
    writer = SummaryWriter(LOG_NAME)

    try:
        evolver = pickle.load(open(os.path.join(SAVE_PATH, "evolver.pkl"), 'rb'))
        print('Loaded save file at {}'.format(SAVE_PATH))
        if VISUALIZE:
            evolver.visualize()
        if INFER:
            global RENDER
            RENDER = True
            N_STEPS = N_INFER_STEPS
            evolver.infer()
        if not (INFER or VISUALIZE):
            # then we train
            RENDER = arg_dict['render']
            evolver.init_env()
            evolver.total_itrs = arg_dict['n_generations']
            evolver.evolve()
    except FileNotFoundError as e:
        if not INFER:
            RENDER = arg_dict['render']
            print(
                "Failed loading from an existing save-file. Evolving from scratch. The error was: {}"
                .format(e))
            evolver = EvoPCGRL()
            evolver.evolve()
        else:
            print(
                "Loading from an existing save-file failed. Cannot run inference. The error was: {}"
                .format(e))
