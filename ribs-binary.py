import argparse
import os
import pickle
import time
from pdb import set_trace as T
from random import randint
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
    def __init__(self):
        super().__init__()
        self.m = 5
        self.l1 = Conv2d(1, 2 * self.m, 3, 1, 1, bias=True)
        self.l2 = Conv2d(2 * self.m, self.m, 1, 1, 0, bias=True)
        self.l3 = Conv2d(self.m, 2, 1, 1, 0, bias=True)
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


def simulate(env, model, init_states, seed=None):
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

    total_reward = 0.0
    path_length = 0.0
    regions = 0.0
    # Allow us to manually set the level-map on reset (using the "_old_map" attribute)
    env._rep._random_start = False
    n_init_states = init_states.shape[0]
    # TODO: make bc collection general for any choice of bcs for some env
    bcs_0 = np.empty(n_init_states)
    bcs_1 = np.empty(n_init_states)
    for (n_episode, init_state) in enumerate(init_states):
        env._rep._old_map = init_state
        obs = env.reset()
        done = False

        n_step = 0
        last_action = None
        while not done:
            in_tensor = torch.unsqueeze(
                torch.unsqueeze(torch.tensor(np.float32(obs['map'])), 0), 0)
            action = model(in_tensor)[0].numpy()
            # Hack implementation of a cellular automata-like action. We only need to get stats at the end of the episode!
            if not CA_ACTION:
                action = np.array([action[1], action[2], action[0]])
                # two identical actions means that we are caught in a loop, assuming we sample actions deterministically from the NN
                done = (action == last_action).all() or n_step >= 100
                if done:
                    path_length = env._rep_stats['path-length']
                    regions = env._rep_stats['regions']
                    reward = 0
                else:
                    obs, reward, _, info = env.step(action)
            # The standard single-build action.
            else:
                action = action.argmax(axis=0)
                env._rep._map = action
                obs = {'map': action}
                reward = 0
                done = (action == last_action).all() or n_step >= 100
                if done:
                    stats = env._prob.get_stats(
                        get_string_map(env._rep._map, env._prob.get_tile_types()))
                    path_length = stats['path-length']
                    regions = stats['regions']
                    # ad hoc reward
                    # reward = -n_step
            bcs_0[n_episode] = path_length
            bcs_1[n_episode] = regions
            if RENDER:
                env.render()
            last_action = action
#           total_reward += reward
            n_step += 1
    bc_0 = bcs_0.mean()
    bc_1 = bcs_1.mean()
    reward = -(bcs_0.std() + bcs_1.std())


    return reward, bc_0, bc_1


class EvoPCGRL():
    def __init__(self):
        env_name = '{}-wide-v0'.format(PROBLEM)
        self.env = gym.make(env_name)

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        self.bc_bounds = {
            # Upper bound: checkerboard
            'regions': (0, self.env._prob._width * np.ceil(self.env._prob._height / 2)),

            #     10101010
            #     01010101
            #     10101010
            #     01010101
            #     10101010

            # FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
            # and "snake" along that one
            # Upper bound: zig-zag
            'path-length': (0, np.ceil(self.env._prob._width / 2 + 1) * (self.env._prob._height)),

            #     11111111
            #     00000001
            #     11111111
            #     10000000
            #     11111111
        }

        if PROBLEM == 'binary':
            self.bc_names = ['regions', 'path-length']
        else:
            raise Exception('{} problem is not supported'.format(PROBLEM))

        self.archive = GridArchive(
#           [100, 100],  # 100 bins in each dimension.
            [100 for _ in self.bc_names],
#           [(1.0, 196.0), (1.0, 196.0)],  # path length and num rooms
            [self.bc_bounds[bc_name] for bc_name in self.bc_names],
        )

        self.model = NNGoL()
        set_nograd(self.model)
        initial_w = get_init_weights(self.model)
        emitters = [
            ImprovementEmitter(
                self.archive,
                initial_w.flatten(),
                #FIXME: does this need to be so damn big?
                1,  # Initial step size.
                batch_size=30,
            ) for _ in range(5)  # Create 5 separate emitters.
        ]

        self.optimizer = Optimizer(self.archive, emitters)

        # This is the initial map which will act as a seed to our NCAs
        self.init_states = np.random.randint(0, 2, (10, 14, 14))
#       self.init_state = np.zeros((14, 14))
#       self.init_state[5:-5, 5:-5] = 1

        self.start_time = time.time()
        self.total_itrs = 10000
        # total_itrs = 500
        self.n_itr = 1

    def evolve(self):

        for itr in tqdm(range(self.n_itr, self.total_itrs + 1)):
            # Request models from the optimizer.
            sols = self.optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            for model_w in sols:
                set_weights(self.model, model_w)
                obj, path_length, regions = simulate(self.env, self.model,
                                                     self.init_states, seed)
                objs.append(obj)
                bcs.append([path_length, regions])

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

    def restore(self):
        self.env = gym.make("binary-wide-v0")

    def visualize(self):
        archive = self.archive
        # # Visualize Result
        # plt.figure(figsize=(8, 6))
        # grid_archive_heatmap(archive, vmin=-300, vmax=300)
        # plt.gca().invert_yaxis()  # Makes more sense if larger velocities are on top.
        # plt.ylabel("Impact y-velocity")
        # plt.xlabel("Impact x-position")

        # Print table of results
        df = archive.as_pandas()
        # high_performing = df[df["objective"] > 200].sort_values("objective", ascending=False)
        print(df)


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
        default='EvoPCGRL_0',
    )
    opts.add_argument(
        '-r',
        '--render',
        help='Render the environment.',
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
    RENDER = opts.render
    INFER = opts.infer
    exp_name = '{}'.format(opts.exp_name)
    SAVE_PATH = os.path.join('evo_runs', exp_name)

    try:
        evolver = pickle.load(open(SAVE_PATH, 'rb'))
        if INFER:
            evolver.infer()
        evolver.restore()
        evolver.evolve()
    except FileNotFoundError as e:
        print(
            "Loading from an existing save-file failed. Evolving from scratch. The error was: {}"
            .format(e))
        evolver = EvoPCGRL()
        evolver.evolve()
