import argparse
import os
import pickle
import time
from pdb import set_trace as T
from random import randint
# import cv2
from typing import Tuple
from datetime import datetime
from pathlib import Path

import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import envs
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from torch import ByteTensor, Tensor, nn
from torch.nn import Conv2d, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
# Use for .py file
from tqdm import tqdm

import gym_pcgrl
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

# Use for notebook
# from tqdm.notebook import tqdm

# Use print to confirm access to local pcgrl gym
# print([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point])
"""
/// Required Environment ///
conda create -n ribs-pt python=3.7
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


#   if CUDA:
#       m.cuda()
#       m.to('cuda:0')


class NNGoL(nn.Module):
    def __init__(self, n_tile_types):
        super().__init__()
        if PROBLEM == 'zelda':
            n_hid_1 = 32
            n_hid_2 = 16
        if PROBLEM == 'binary':
            n_hid_1 = 32
            n_hid_2 = 16

        self.l1 = Conv2d(n_tile_types, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_2, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_2, n_tile_types, 1, 1, 0, bias=True)
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


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        torch.nn.init.orthogonal_(m.weight)


def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
    """
    Use to get dimension of weights from PyTorch
    """
    init_weights = []
    for lyr in nn.layers:
        init_weights.append(lyr.weight.view(-1).numpy())
        init_weights.append(lyr.bias.view(-1).numpy())
    init_weights = np.hstack(init_weights)

    return init_weights


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
    n_classes = env.action_space.nvec[2]
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
    max_val = env.action_space.nvec[0]*env.action_space.nvec[1]  # for example 14*14=196
    return [np.sum(int_map.flatten() == tile)/max_val for tile in range(env.action_space.nvec[2])]


def get_emptiness(int_map, env):
    """
    Function to calculate how empty the level is
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns an emptiness value normalized to a range of 0.0 to 1.0
    """
    max_val = env.action_space.nvec[0]*env.action_space.nvec[1]  # for example 14*14=196
    return np.sum(int_map.flatten() == 0)/max_val

def get_hor_sym(int_map, env):
    """
    Function to get the horizontal symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env.action_space.nvec[0]*env.action_space.nvec[1]/2  # for example 14*14/2=98
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
    max_val = env.action_space.nvec[0]*env.action_space.nvec[1]/2  # for example 14*14/2=98
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
    max_val = env.action_space.nvec[0]*env.action_space.nvec[1]*4
    result = (np.sum((np.roll(int_map, 1, axis=0) == int_map).astype(int))+
    np.sum((np.roll(int_map, -1, axis=0) == int_map).astype(int))+
    np.sum((np.roll(int_map, 1, axis=1) == int_map).astype(int))+
    np.sum((np.roll(int_map, -1, axis=1) == int_map).astype(int)))
    return result/max_val

def get_regions(stats):
    return stats['regions']

def get_path_length(stats):
    return stats['path-length']

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
    else:  
        print('The BC {} is not recognized.'.format(bc_name))
        return 0.0



def simulate(env, model, n_tile_types, init_states, bc_names, static_targets, seed=None):
    """
    Function to run a single trajectory and return results.

    Args:
        env (gym.Env): A copy of the binary-wide-v0 environment.
        model (np.ndarray): The array of weights for the policy.
        seed (int): The seed for the environment.
    Returns:
        total_reward (float): The reward accrued by the lander throughout its
            trajectory.
        path_length (float): The path length of the final solution.
        regions (float): The number of distinct regions of the final solution.
    """
    if seed is not None:
        env.seed(seed)

    # Allow us to manually set the level-map on reset (using the "_old_map" attribute)
#   env._rep._random_start = False
#   if n_episode == 0 and False:
#       env._rep._old_map = init_state
#       obs = env.reset()
#       int_map = obs['map']
    n_init_states = init_states.shape[0]
    width = init_states.shape[1]
    height = init_states.shape[2]
    bcs = np.empty(shape=(len(bc_names), n_init_states))
    final_levels = np.empty(shape=init_states.shape, dtype=np.uint8)
    batch_reward = 0
    batch_time_penalty = 0
    batch_targets_penalty = 0
    for (n_episode, init_state) in enumerate(init_states):
        # NOTE: Sneaky hack. We don't need initial stats. Never even reset. Heh. Be careful!!
        env._rep._map = init_state
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
        last_int_map = None
        while not done:
#           in_tensor = torch.unsqueeze(
#               torch.unsqueeze(torch.tensor(np.float32(obs['map'])), 0), 0)
            in_tensor = torch.unsqueeze(torch.Tensor(obs), 0)
            action = model(in_tensor)[0].numpy()
            # The standard single-build action (why throw away all that information at the NCA's output, though? Not sure if this really works).
            if not CA_ACTION:
                pass
#               action = np.array([action[1], action[2], action[0]])
#               # two identical actions means that we are caught in a loop, assuming we sample actions deterministically from the NN (we do)
#               done = (action == last_int_map).all() or n_step >= 100
#               if done:
#                   bcs[0, n_episode] = env._rep_stats['path-length']
#                   bcs[1, n_episode] = env._rep_stats['regions']
#                   reward = 0
#               else:
#                   obs, reward, _, info = env.step(action)

            # Hack implementation of a cellular automata-like action. We only need to get stats at the end of the episode!
            else:
                obs = action
                int_map = action.argmax(axis=0)
                env._rep._map = int_map
                reward = 0
                done = (int_map == last_int_map).all() or n_step >= N_STEPS
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

                    # we want to hit each of our static targets exactly, penalize for anything else
                    # for ranges, we take our least distance to any element in the range
                    targets_penalty = np.sum([abs(static_targets[k] - stats[k]) if not isinstance(static_targets[k], tuple) else abs(np.arange(*static_targets[k]) - stats[k]).min() for k in static_targets])
                    batch_targets_penalty -= targets_penalty

            if RENDER:
                env.render()
            if done and INFER:
                time.sleep(5/30)
                print('stats: {}\n\ntime_penalty: {}\n targets_penalty: {}'.format(stats, time_penalty, targets_penalty))
            last_int_map = int_map
            n_step += 1
    final_bcs = [bcs[i].mean() for i in range(bcs.shape[0])]
    if N_INIT_STATES > 0:
        batch_targets_penalty = 10 * batch_targets_penalty / N_INIT_STATES
        batch_time_penalty = batch_time_penalty / N_INIT_STATES
    batch_reward += batch_targets_penalty
    if N_INIT_STATES > 1:
        # Variance penalty is the negative average (per-BC) standard deviation from the mean BC vector.
        variance_penalty = - np.sum([bcs[i].std()
                          for i in range(bcs.shape[0])]) / bcs.shape[0]
        # Diversity bonus. We want minimal variance along BCS *and* diversity in terms of the map.
        # Sum pairwise hamming distances between all generated maps.
        diversity_bonus = np.sum([np.sum(final_levels[j] != final_levels[k]) if j != k else 0 for k in range(N_INIT_STATES) for j in range(N_INIT_STATES)]) / (N_INIT_STATES * N_INIT_STATES - 1) 
        # ad hoc scaling :/
        diversity_bonus = 20 * diversity_bonus / (width * height)
        batch_reward = batch_reward + variance_penalty + diversity_bonus
    else:
        variance_penalty = None
        diversity_bonus = None

    if not INFER:
        return batch_reward, final_bcs
    else:
        return batch_reward, final_bcs, (time_penalty, targets_penalty, variance_penalty, diversity_bonus)



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

        # TODO: maybe make these command-line arguments?
        # TODO: multi-objective compatibility?
        if PROBLEM in ('binary'):
            # pass
            # self.bc_names = ['regions', 'path-length']
            self.bc_names = BCS
#           self.reward_names = ['variance']
        elif PROBLEM in ('zelda'):
#           pass
            # self.bc_names = ['nearest-enemy', 'path-length']#, 'n_walls']
#           self.reward_names = ['static_targets']
            self.bc_names = BCS

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        if PROBLEM == 'binary':
            self.bc_bounds = {
                # Upper bound: checkerboard
                'regions':
                (0, self.width * np.ceil(self.height / 2)),

                #     10101010
                #     01010101
                #     10101010
                #     01010101 #     10101010

                # FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
                # and "snake" along that one
                # Upper bound: zig-zag
                'path-length': (0, np.ceil(self.width / 2 + 1) *
                                (self.height)),

                #     11111111
                #     00000001
                #     11111111
                #     10000000
                #     11111111
                'co-occurance': (0.0, 1.0),
                'symmetry': (0.0, 1.0),
                'symmetry-vertical': (0.0, 1.0),
                'symmetry-horizontal': (0.0, 1.0),
                'emptiness': (0.0, 1.0),
                'entropy': (0.0, 1.0),
            }
#           self.reward_bounds = {
#               'variance': (-50, 0),
#               }
            self.static_targets = {}
        elif PROBLEM == 'zelda':
            self.bc_bounds = {
                #TODO: adapt this zelda path! ???
                # Upper bound: zig-zag
                'path-length': (0, np.ceil(self.width / 2 + 1) *
                                (self.height)),

                #     11111111
                #     00000001
                #     11111111
                #     10000000
                #     11111111


               #'nearest-enemy': (0, max(self.width, self.height)),
               #WTF
                'nearest-enemy': (0, np.ceil(self.width / 2 + 1) *
                                (self.height)),
                'co-occurance': (0.0, 1.0),
                'symmetry': (0.0, 1.0),
                'symmetry-vertical': (0.0, 1.0),
                'symmetry-horizontal': (0.0, 1.0),
                'emptiness': (0.0, 1.0),
                'entropy': (0.0, 1.0),
            }

            # metrics we always want to work toward
            self.static_targets = {
                'player': 1,
                'key': 1,
                'door': 1,
                'regions': 1,
                'enemies': (2, 5),
            }
#           self.reward_bounds = {
#               'variance': (-50, 0),
#               'static_targets': (-20, 0),
#               }


        else:
            raise Exception('{} problem is not supported'.format(PROBLEM))

        self.archive = GridArchive(
            # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
           #[min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
            [100 for _ in self.bc_names],
            # min/max for each BC
            [self.bc_bounds[bc_name] for bc_name in self.bc_names],
        )

        self.model = NNGoL(self.n_tile_types)
        set_nograd(self.model)
        initial_w = get_init_weights(self.model)
        emitters = [
            ImprovementEmitter(
                self.archive,
                initial_w.flatten(),
                # NOTE: Big step size, shit otherwise
                1,  # Initial step size.
                batch_size=30,
            ) for _ in range(5)  # Create 5 separate emitters.
        ]

        self.optimizer = Optimizer(self.archive, emitters)

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

    def evolve(self):

        for itr in tqdm(range(self.n_itr, self.total_itrs + 1)):
            # Request models from the optimizer.
            sols = self.optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            for model_w in sols:
                set_weights(self.model, model_w)
                obj, (bc_0, bc_1) = simulate(
                    env=self.env,
                    model=self.model,
                    n_tile_types=self.n_tile_types,
                    init_states=self.init_states,
                    bc_names=self.bc_names,
                    static_targets=self.static_targets,
                    seed=seed)
                objs.append(obj)
                bcs.append([bc_0, bc_1])

            # Send the results back to the optimizer.
            self.optimizer.tell(objs, bcs)

            # TensorBoard Logging.
            df = self.archive.as_pandas(include_solutions=False)
            elapsed_time = time.time() - self.start_time
            writer.add_scalar('ArchiveSize', elapsed_time, itr)
            writer.add_scalar('score/mean', df['objective'].mean(), itr)
            writer.add_scalar('score/max', df['objective'].max(), itr)
            writer.add_scalar('score/min', df['objective'].min(), itr)

            # Logging.
            if itr % 1 == 0:
                print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
                print(f"  - Archive Size: {len(df)}")
                print(f"  - Max Score: {df['objective'].max()}")
                print(f"  - Mean Score: {df['objective'].mean()}")
                print(f"  - Min Score: {df['objective'].min()}")
            # Save checkpoint
            if itr % 1 == 0:
                global ENV
                ENV = self.env
                self.env = None
                pickle.dump(self, open(SAVE_PATH, 'wb'))
                self.env = ENV
            self.n_itr += 1

    def init_env(self):
        env_name = '{}-wide-v0'.format(PROBLEM)
        self.env = gym.make(env_name)

    def visualize(self):
        archive = self.archive
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

#       plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
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
        archive = self.archive
        df = archive.as_pandas()
#       high_performing = df[df["behavior_1"] > 50].sort_values("behavior_1", ascending=False)
        if PROBLEM == 'binary':
            high_performing = df.sort_values("behavior_1", ascending=False)
#           high_performing = df.sort_values("objective", ascending=False)
        if PROBLEM == 'zelda':
            # path lenth
            high_performing = df.sort_values("behavior_1", ascending=False)
            # nearest enemies
#           high_performing = df.sort_values("behavior_0", ascending=False)
#           high_performing = df.sort_values("objective", ascending=False)
        rows = high_performing
        models = np.array(rows.loc[:, "solution_0":])
        bcs_0 = np.array(rows.loc[:, "behavior_0"])
        bcs_1 = np.array(rows.loc[:, "behavior_1"])
        objs = np.array(rows.loc[:, "objective"])
        i = 0

        if GRID:

            d = 6  # dimension of rows and columns
            figw, figh = 16.0, 16.0
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
                    init_nn = set_weights(self.model, model)

                    # run simulation
                    _, _, (time_penalty, targets_penalty, variance_penalty, diversity_bonus) = simulate(self.env, init_nn,
                                    self.n_tile_types, self.init_states, self.bc_names, self.static_targets, seed=None)
                    # Get image
                    img = self.env.render(mode='rgb_array')
                    axs[row_num,col_num].imshow(img, aspect='auto')
            fig.subplots_adjust(hspace=0.01, wspace=0.01)
            fig.savefig('evo_runs/test_grid.png', dpi=300)

        while True:
#           model = self.archive.get_random_elite()[0]
#           model = models[np.random.randint(len(models))]
            model = models[i]
            init_nn = set_weights(self.model, model)
#           init_states = (np.random.random(
#               size=(1, 1, MAP_WIDTH, MAP_WIDTH)) < 0.2).astype(np.uint8)
            _, _, (time_penalty, targets_penalty, variance_penalty, diversity_bonus) = simulate(self.env, init_nn,
                            self.n_tile_types, self.init_states, self.bc_names, self.static_targets, seed=None)
            input("Mean behavior characteristics:\n\t{}: {}\n\t{}: {}\nMean reward:\n\tTotal: {}\n\ttime: {}\n\ttargets: {}\n\tvariance: {}\n\tdiversity: {}\nPress any key for next generator...".format(
                self.bc_names[0], bcs_0[i], self.bc_names[1], bcs_1[i], objs[i], time_penalty, targets_penalty, variance_penalty, diversity_bonus))
            i += 1

            if i == len(models):
                i = 0

if __name__ == '__main__':
    """
    Set Parameters
    """
    seed = 1339
    CA_ACTION = True

    opts = argparse.ArgumentParser(
        description='Evolving Neural Cellular Automata for PCGRL')
    opts.add_argument(
        '-p',
        '--problem',
        help='Which game to generate levels for (PCGRL "problem").',
        default='binary',
    )
    opts.add_argument(
        '-e',
        '--exp_name',
        help='Name of the experiment, for save files.',
        default='0',
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
        default=10,
    )
    opts.add_argument(
        '-bcs',
        '--behavior_characteristics',
        nargs='+',
        help='A list of strings corresponding to the behavior characteristics that will act as the dimensions for our grid of elites during evolution.',
        default=['regions','path-length'],
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
        '--save_grid',
        help='Save grid of levels to png.',
        action='store_true',
    )

    opts = opts.parse_args()
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
    global GRID
    BCS = opts.behavior_characteristics
    N_GENERATIONS = opts.n_generations
    N_INIT_STATES = opts.n_init_states
    N_STEPS = opts.n_steps
    SHOW_VIS = opts.show_vis
    PROBLEM = opts.problem
    CUDA = False
    VISUALIZE = opts.visualize
    INFER = opts.infer
    N_INFER_STEPS = N_STEPS
#   N_INFER_STEPS = 100
    GRID = opts.save_grid

    exp_name = 'EvoPCGRL_{}_{}-batch_{}-step_{}'.format(PROBLEM, N_INIT_STATES, N_STEPS, opts.exp_name)
    SAVE_PATH = os.path.join('evo_runs', exp_name)

    # Create TensorBoard Log Directory if does not exist
    LOG_NAME = './runs/' + datetime.now().strftime("%Y%m%d-%H%M%S")+ '-' + exp_name
    writer = SummaryWriter(LOG_NAME)

    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
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
            RENDER = opts.render
            evolver.init_env()
            evolver.total_itrs = opts.n_generations
            evolver.evolve()
    except FileNotFoundError as e:
        if not INFER:
            RENDER = opts.render
            print(
                "Failed loading from an existing save-file. Evolving from scratch. The error was: {}"
                .format(e))
            evolver = EvoPCGRL()
            evolver.evolve()
        else:
            print(
                "Loading from an existing save-file failed. Cannot run inference. The error was: {}"
                .format(e))
