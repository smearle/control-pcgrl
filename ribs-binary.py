import argparse
import os
import pickle
import time
from pdb import set_trace as T
from random import randint
import cv2
from typing import Tuple

import gym
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
pip install 'ribs[all]' gym~=0.17.0 Box2D~=2.3.10 tqdm
git clone https://github.com/amidos2006/gym-pcgrl.git
cd gym-pcgrl  # Must run in project root folder for access to pcgrl modules
"""
"""
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
        n_hid_1 = 15
        n_hid_2 = 10
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


def simulate(env, model, n_tile_types, init_states, bc_names, static_trgs, seed=None):
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
    env._rep._random_start = False
    n_init_states = init_states.shape[0]
    bcs = np.empty(shape=(len(bc_names), n_init_states))
    total_reward = 0
    for (n_episode, init_state) in enumerate(init_states):
        env._rep._old_map = init_state
        obs = env.reset()
        int_map = obs['map']
        obs = get_one_hot_map(int_map, n_tile_types)
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
                action = np.array([action[1], action[2], action[0]])
                # two identical actions means that we are caught in a loop, assuming we sample actions deterministically from the NN (we do)
                done = (action == last_int_map).all() or n_step >= 100
                if done:
                    bcs[0, n_episode] = env._rep_stats['path-length']
                    bcs[1, n_episode] = env._rep_stats['regions']
                    reward = 0
                else:
                    obs, reward, _, info = env.step(action)
            # Hack implementation of a cellular automata-like action. We only need to get stats at the end of the episode!
            else:
                obs = action
                int_map = action.argmax(axis=0)
                env._rep._map = int_map
#               obs = {'map': action}
                reward = 0
                done = (int_map == last_int_map).all() or n_step >= 100
                if done:
                    stats = env._prob.get_stats(
                        get_string_map(env._rep._map,
                                       env._prob.get_tile_types()))
                    # get BCs
                    for i in range(len(bcs)):
                        bc_name = bc_names[i]
                        bcs[i, n_episode] = stats[bc_name]
                    # ad hoc reward: shorter episodes are better
                    # reward = -n_step

                    # we want to hit each of our static targets exactly, penalize for anything else
                    total_reward -= np.sum([abs(static_trgs[k] - stats[k]) for k in static_trgs])
            if RENDER:
                env.render()
            last_int_map = int_map
            n_step += 1
    final_bcs = [bcs[i].mean() for i in range(bcs.shape[0])]
    # Reward is the average (per-BC) standard deviation from the mean BC vector.
    # TODO: We want minimal variance along BCS *and* diversity in terms of the map. 
    # Could implement this by summing pairwise hamming distances between all generated maps
#   variance_penalty = -np.sum([bcs[i].std()
#                     for i in range(bcs.shape[0])]) / bcs.shape[0]
#   reward = total_reward + variance_penalty
    reward = total_reward

    return reward, final_bcs


class EvoPCGRL():
    def __init__(self):
        self.init_env()
        # get number of tile types from environment's observation space
        # here we assume that all (x, y) locations in the observation space have the same upper/lower bound
        self.n_tile_types = self.env.observation_space['map'].high[0, 0] + 1
        assert self.env.observation_space['map'].low[0, 0] == 0

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        self.bc_bounds = {
            # Upper bound: checkerboard
            'regions':
            (0, self.env._prob._width * np.ceil(self.env._prob._height / 2)),

            #     10101010
            #     01010101
            #     10101010
            #     01010101 #     10101010

            # FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
            # and "snake" along that one
            # Upper bound: zig-zag
            'path-length': (0, np.ceil(self.env._prob._width / 2 + 1) *
                            (self.env._prob._height)),

            #     11111111
            #     00000001
            #     11111111
            #     10000000
            #     11111111
        }
        self.static_targets = {}
        if PROBLEM == 'zelda':
            # TODO: add some fun zelda-specific BCs
            self.bc_bounds.update({})
            # metrics we always want to work toward
            self.static_targets.update({
                'player': 1,
                'key': 1,
                'door': 1,
            })

        if PROBLEM in ('binary', 'zelda'):
            self.bc_names = ['regions', 'path-length']
        else:
            raise Exception('{} problem is not supported'.format(PROBLEM))

        self.archive = GridArchive(
            #           [100, 100],  # 100 bins in each dimension.
            [100 for _ in self.bc_names],
            #           [(1.0, 196.0), (1.0, 196.0)],  # path length and num rooms
            [self.bc_bounds[bc_name] for bc_name in self.bc_names],
        )

        self.model = NNGoL(self.n_tile_types)
        set_nograd(self.model)
        initial_w = get_init_weights(self.model)
        emitters = [
            ImprovementEmitter(
                self.archive,
                initial_w.flatten(),
                # FIXME: does this need to be so damn big?
                1,  # Initial step size.
                batch_size=30,
            ) for _ in range(5)  # Create 5 separate emitters.
        ]

        self.optimizer = Optimizer(self.archive, emitters)

        # This is the initial map which will act as a seed to our NCAs
#       self.init_states = np.random.randint(0, self.n_tile_types, (10, 14, 14))
        self.init_states = np.zeros(shape=(1, 14, 14))
        self.init_states[0, 5:-5, 5:-5] = 1

        self.start_time = time.time()
        self.total_itrs = 1000
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
                    bc_names=list(self.bc_bounds.keys()),
                    static_trgs=self.static_targets,
                    seed=seed)
                objs.append(obj)
                bcs.append([bc_0, bc_1])

            # Send the results back to the optimizer.
            self.optimizer.tell(objs, bcs)

            # Logging.
            if itr % 1 == 0:
                df = self.archive.as_pandas(include_solutions=False)
                elapsed_time = time.time() - self.start_time
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
        grid_archive_heatmap(archive, vmin=-300, vmax=300)
#       plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        plt.xlabel(self.bc_names[0])
        plt.ylabel(self.bc_names[1])
       #plt.show()
        plt.savefig('{}.png'.format(SAVE_PATH))

        # Print table of results
        df = archive.as_pandas()
        # high_performing = df[df["objective"] > 200].sort_values("objective", ascending=False)
        print(df)

    def infer(self):
        self.init_env()
        archive = self.archive
        df = archive.as_pandas()
        high_performing = df[df["behavior_1"] > 50].sort_values("behavior_1", ascending=False)
        rows = high_performing
        models = np.array(rows.loc[:, "solution_0":])
        bcs_0 = np.array(rows.loc[:, "behavior_0"])
        bcs_1 = np.array(rows.loc[:, "behavior_1"])
        i = 0

        while True:
#           model = self.archive.get_random_elite()[0]
            model = models[np.random.randint(len(models))]
            model = models[i]
            init_nn = set_weights(self.model, model)
#           init_states = (np.random.random(
#               size=(1, 1, MAP_WIDTH, MAP_WIDTH)) < 0.2).astype(np.uint8)
            _, _ = simulate(self.env, init_nn,
                            self.n_tile_types, self.init_states, self.bc_names, self.static_targets, seed=None)
            input("Mean behavior characteristics of generator:\n{}: {}\n{}: {}\nPress any key for next generator...".format(
                self.bc_names[0], bcs_0[i], self.bc_names[1], bcs_1[i]))
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
        '-i',
        '--infer',
        help='Run inference with evolved archive of individuals.',
        action='store_true',
    )
    opts.add_argument(
        '-e',
        '--exp_name',
        help='Name of the experiment, for save files.',
        default='0',
    )
    opts.add_argument(
        '-r',
        '--render',
        help='Render the environment.',
        action='store_true',
    )
    opts.add_argument(
        '-v',
        '--visualize',
        help='Visualize heatmap of the archive of individuals.',
        action='store_true',
    )
    opts = opts.parse_args()
    global INFER
    global EVO_DIR
    global CUDA
    global RENDER
    global PROBLEM
    PROBLEM = opts.problem
    CUDA = False
    VISUALIZE = opts.visualize
    INFER = opts.infer
    exp_name = 'EvoPCGRL_{}_{}'.format(PROBLEM, opts.exp_name)
    SAVE_PATH = os.path.join('evo_runs', exp_name)

    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        if VISUALIZE:
            evolver.visualize()
        elif INFER:
            global RENDER
            RENDER = True
            evolver.infer()
        else:
            RENDER = opts.render
            evolver.init_env()
            evolver.evolve()
    except FileNotFoundError as e:
        print(
            "Loading from an existing save-file failed. Evolving from scratch. The error was: {}"
            .format(e))
        evolver = EvoPCGRL()
        evolver.evolve()
