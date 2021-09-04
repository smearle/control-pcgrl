import argparse
import gc
import json
import os
import pickle
import pprint
import sys
import time
from datetime import datetime
from timeit import default_timer as timer
from pathlib import Path
from pdb import set_trace as TT
from random import randint

import cv2
from typing import Tuple

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import ray
import scipy
import torch as th
import torch.nn.functional as F
from gym import envs
from numba import njit
from qdpy.phenotype import Fitness
from ribs.archives import GridArchive
from ribs.archives._add_status import AddStatus
from ribs.emitters import (
    GradientImprovementEmitter,
    ImprovementEmitter,
    OptimizingEmitter,
)
from ribs.emitters.opt import CMAEvolutionStrategy
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from torch import ByteTensor, Tensor, nn
from torch.nn import Conv2d, CrossEntropyLoss, Linear
from torch.utils.tensorboard import SummaryWriter
import deap
import deap.tools
import deap.algorithms
import qdpy
from qdpy import algorithms, containers, benchmarks, plots, tools
from deap.base import Toolbox
import graphviz
import warnings
import copy


# Use for .py file
from tqdm import tqdm

import gym_pcgrl
from evo_args import get_args
from gym_pcgrl.envs.helper import get_int_prob, get_string_map

# from example_play_call import random_player
# gvgai_path = '/home/sme/GVGAI_GYM/'
# sys.path.insert(0,gvgai_path)
# from play import play

# Use for notebook
# from tqdm.notebook import tqdm

# Use print to confirm access to local pcgrl gym
# print([env.id for env in envs.registry.all() if "gym_pcgrl" in env.entry_point])
"""
/// Required Environment ///
conda create -n ribs-pt python=3.7
pip install scipy==1.2.0  # must use this version with GVGAI_GYM
conda install -c conda-forge notebook
conda install pytorch torchvision torchaudio -c pyth
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
TARGETS_PENALTY_WEIGHT = 10

def draw_net(config: object, genome: object, view: object = False, filename: object = None, node_names: object = None, show_disabled: object = True,
             prune_unused: object = False,
             node_colors: object = None, fmt: object = 'svg') -> object:
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add(cg.key)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

def save_level_frames(level_frames, model_name):

    renders_dir = os.path.join(SAVE_PATH, "renders")
    if not os.path.isdir(renders_dir):
        os.mkdir(renders_dir)
    model_dir = os.path.join(renders_dir, "model_{}".format(model_name))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    for j, im in enumerate(level_frames):
        im.save(
            os.path.join(
                model_dir, "frame_{:0>4d}.png".format(j)
            )
        )


def get_qd_score(archive, env, bc_names):
    if ALGO == 'ME':
        TT()  # gotta add the max loss here
        qd_score = archive.qd_score()
    else:
        df = archive.as_pandas(include_solutions=False)
        max_loss = env.get_max_loss(ctrl_metrics=bc_names)
        max_loss = max_loss * TARGETS_PENALTY_WEIGHT
        qd_score = (df['objective'] + max_loss).sum()
        return qd_score


def save_train_stats(objs, archive, env, bc_names, itr=None):
    train_time_stats = {
            "qd_score": get_qd_score(archive, env, bc_names),
            "objective": get_stats(objs),
            }

    if itr is not None:
        save_path = os.path.join(SAVE_PATH, "checkpoint_{}".format(itr))
    else:
        save_path = SAVE_PATH
    json.dump(
        train_time_stats,
        open(os.path.join(save_path, "train_time_stats.json"), "w"),
        indent=4,
    )


def get_stats(stats):
    """Take 1D numpy array of data and return some fun facts in the form of a dictionary."""

    return {
        "mean": np.nanmean(stats),
        "std": np.nanstd(stats),
        "max": np.nanmax(stats),
        "min": np.nanmin(stats),
    }


def save_grid(csv_name="levels", d=4):
    if CMAES:
        # TODO: implement me

        return
    # save grid using csv file
    # get path to CSV
    levels_path = os.path.join(SAVE_PATH, csv_name + ".csv")
    # get env name
    env_name = "{}-{}-v0".format(PROBLEM, REPRESENTATION)
    # create env
    env = gym.make(env_name)
    map_width = env._prob._width

    df = pd.read_csv(levels_path, header=0, skipinitialspace=True)
    #   .rename(
#       index=str,
#       header=0,
#       columns={
#           0: "level",
#           1: "batch_reward",
#           2: "variance",
#           3: "diversity",
#           4: "targets",
#       },
#   )

    bc_names = []
    for i in range(5, 7):  # assume 2 BCs
        bc_names.append(df.columns[i])
    # look for the most valid levels
    targets_thresh = 0.0
    og_df = df
    df = og_df[og_df['targets'] == targets_thresh]
    last_len = len(df)
    while len(df) < d**2 and targets_thresh <= df['targets'].max():
        last_len = len(df)
        targets_thresh += 1.0
        df = og_df[og_df['targets'] <= targets_thresh]
    # d = 6  # dimension of rows and columns
    figw, figh = 16.0, 16.0
    fig = plt.figure()
    fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figw, figh))

    df_g = df.sort_values(by=bc_names, ascending=False)

    df_g["row"] = np.floor(np.linspace(0, d, len(df_g), endpoint=False)).astype(int)

    for row_num in range(d):
        row = df_g[df_g["row"] == row_num]
        row = row.sort_values(by=[bc_names[1]], ascending=True)
        row["col"] = np.arange(0, len(row), dtype=int)
        idx = np.floor(np.linspace(0, len(row) - 1, d)).astype(int)
        row = row[row["col"].isin(idx)]
        row = row.drop(["row", "col"], axis=1)
        # grid_models = np.array(row.loc[:,'solution_0':])
        grid_models = row["level"].tolist()

        for col_num in range(len(row)):
            axs[row_num, col_num].set_axis_off()
            level = np.zeros((map_width, map_width), dtype=int)

            for i, l_rows in enumerate(grid_models[col_num].split("], [")):
                for j, l_col in enumerate(l_rows.split(",")):
                    level[i, j] = int(
                        l_col.replace("[", "").replace("]", "").replace(" ", "")
                    )

            # Set map
            env._rep._x = env._rep._y = 0
            env._rep._map = level
            img = env.render(mode="rgb_array")
#           axs[row_num, col_num].imshow(img, aspect="auto")
            axs[-col_num-1, -row_num-1].imshow(img, aspect="auto")

    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    levels_png_path = os.path.join(SAVE_PATH, "{}_grid.png".format(csv_name))
    fontsize = 32
    fig.text(0.5, 0.01, bc_names[0], ha='center', va='center',fontsize=fontsize)
    fig.text(0.01, 0.5, bc_names[1], ha='center', va='center', rotation='vertical', fontsize=fontsize)
    plt.tight_layout(rect=[0.025, 0.025, 1, 1])
    fig.savefig(levels_png_path, dpi=300)
    plt.close()


def auto_garbage_collect(pct=80.0):
    if psutil.virtual_memory().percent >= pct:
        gc.collect()

def tran_action(action, **kwargs):
    skip = False
    # return action, skip
    return action.swapaxes(1, 2), skip

# usually, if action does not turn out to change the map, then the episode is terminated
# the skip boolean tells us whether, for some representation-specific reason, the agent has chosen not to act, but
# without ending the episode
@njit
def id_action(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    # the argmax along tile_type dimension is performed inside the representation's update function
    skip = False

    return action, skip


# @njit
def wide_action(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    # only consider tiles where the generator suggests something different than the existing tile
    act_mask = action.argmax(axis=0) != int_map
    n_new_builds = np.sum(act_mask)
    act_mask = act_mask.reshape((1, *act_mask.shape))
    #   action = action * act_mask
    action = np.where(act_mask == False, action.min() - 10, action)
    coords = np.unravel_index(action.argmax(), action.shape)

    if n_new_builds > 0:
        assert act_mask[0, coords[1], coords[2]] == 1
    coords = coords[2], coords[1], coords[0]
    #   assert int_map[coords[0], coords[1]] != coords[2]
    skip = False

    return coords, skip


@njit
def narrow_action(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    act = action[:, y, x].argmax()

    if act == 0:
        skip = True
    else:
        skip = False

    return act, skip


@njit
def turtle_action(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    act = action[:, y, x].argmax()
    # moving is counted as a skip, so lack of change does not end episode

    if act < n_dirs:
        skip = True
    else:
        skip = False

    return act, skip


@njit
def flat_to_box(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    action = action.reshape((n_tiles, *int_map.shape))
    skip = False

    return action, skip


@njit
def flat_to_wide(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    w = int_map.shape[0]
    h = int_map.shape[1]
    assert len(action) == int_map.shape[0] + int_map.shape[1] + n_tiles
    action = (action[:w].argmax(), action[w : w + h].argmax(), action[w + h :].argmax())
    skip = False

    return action, skip


@njit
def flat_to_narrow(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    act = action.argmax()

    if act == 0:
        skip = True
    else:
        skip = False

    return act, skip


@njit
def flat_to_turtle(action, int_map=None, n_tiles=None, x=None, y=None, n_dirs=None):
    act = action.argmax()

    if act < n_dirs:
        skip = True
    else:
        skip = False

    return act, skip


preprocess_action_funcs = {
    "NCA": {
        "cellular": id_action,
        "wide": wide_action,
        "narrow": narrow_action,
        "turtle": turtle_action,
    },
    "CPPN": {
        "cellular": tran_action,
    },
    "CNN": {
        # will try to build this logic into the model
        "cellular": flat_to_box,
        "wide": flat_to_wide,
        "narrow": flat_to_narrow,
        "turtle": flat_to_turtle,
    },
}


def id_observation(obs, **kwargs):
    return obs


def local_observation(obs, **kwargs):
    x = kwargs.get("x")
    y = kwargs.get("y")
    local_obs = np.zeros((1, obs.shape[1], obs.shape[2]))
    # What is up with these y x lol
    local_obs[0, y, x] = 1
    np.concatenate((obs, local_obs), axis=0)

    return obs


preprocess_observation_funcs = {
    "NCA": {
        "cellular": id_observation,
        "wide": id_observation,
        "narrow": local_observation,
        "turtle": local_observation,
    },
    "CNN": {
        "cellular": id_observation,
        "wide": id_observation,
        "narrow": local_observation,
        "turtle": local_observation,
    },
}

@njit
def archive_init_states(init_states_archive, init_states, index):
    init_states_archive[index] = init_states


# @njit
def get_init_states(init_states_archive, index):
    return init_states_archive[index]



def mate_individuals(ind_0, ind_1):
    return ind_0.mate(ind_1)

def mutate_individual(ind):
    ind.mutate()
    return (ind,)

class MEOptimizer():
    def __init__(self, grid, ind_cls, batch_size, ind_cls_args, start_time=None, stats=None):
        self.batch_size = batch_size
        self.grid = grid
        self.inds = []
        self.stats=stats
        for _ in range(batch_size):
            self.inds.append(ind_cls(**ind_cls_args))
        toolbox = Toolbox()
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("mutate", mutate_individual)
        toolbox.register("mate", mate_individuals)
        toolbox.register("select", tools.sel_random)

        self.cxpb = 0
        self.mutpb = 1.0
        self.toolbox = toolbox
        if start_time == None:
            self.start_time = timer()
        self.logbook = deap.tools.Logbook()
        self.logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + ["elapsed"]
        self.i = 0


    def tell(self, objective_values, behavior_values):
        # Update individuals' stats with results of last batch of simulations
#       [(ind.fitness.setValues(obj), ind.fitness.features.setValues(bc)) for
#        (ind, obj, bc) in zip(self.inds, objective_values, behavior_values)]
        for (ind, obj, bc) in zip(self.inds, objective_values, behavior_values):
            ind.fitness.setValues([obj])
            ind.features.setValues(bc)
        # Replace the current population by the offspring
        nb_updated = self.grid.update(self.inds, issue_warning=True, ignore_exceptions=False)
        # Compile stats and update logs
        record = self.stats.compile(self.grid) if self.stats else {}
        self.logbook.record(iteration=self.i, containerSize=self.grid.size_str(), evals=len(self.inds), nbUpdated=nb_updated, elapsed=timer()-self.start_time, **record)
        self.i += 1
        print(self.logbook.stream)

    def ask(self):

        if len(self.grid) == 0:
            # Return the initial batch
            return self.inds

        elif len(self.grid) < self.batch_size:
            # If few elites, supplement the population with individuals from the last generation
            np.random.shuffle(self.inds)
            breedable = self.grid.items + self.inds[:-len(self.grid)]

        else:
            breedable = self.grid

        # Select the next batch individuals
        batch = [self.toolbox.select(breedable) for i in range(self.batch_size)]

        ## Vary the pool of individuals
        self.inds = deap.algorithms.varAnd(batch, self.toolbox, self.cxpb, self.mutpb)

        return self.inds



class InitStatesArchive(GridArchive):
    """Save (some of) the initial states upon which the elites were evaluated when added to the archive, so that we can
    reproduce their behavior at evaluation time (and compare it to generalization to other seeds)."""

    def __init__(self, bin_sizes, bin_bounds, n_init_states, map_w, map_h, **kwargs):
        super(InitStatesArchive, self).__init__(bin_sizes, bin_bounds, **kwargs)
        self.init_states_archive = np.empty(
            shape=(*bin_sizes, n_init_states, map_w, map_h)
        )

    def set_init_states(self, init_states):
        self.init_states = init_states

    def add(self, solution, objective_value, behavior_values, meta, index=None):
        status, dtype_improvement = super().add(
            solution, objective_value, behavior_values
        )

        # NOTE: for now we won't delete these when popping an elite for re-evaluation

        if status != AddStatus.NOT_ADDED:
            if index is None:
                index = self.get_index(behavior_values)
            archive_init_states(self.init_states_archive, self.init_states, index)

        return status, dtype_improvement


class MEGrid(containers.Grid):
    def __init__(self, bin_sizes, bin_bounds):
        super(MEGrid, self).__init__(shape=bin_sizes, max_items_per_bin=1,
                                     features_domain=bin_bounds,
                                     fitness_domain=((-np.inf, np.inf),),
                                     )

class MEInitStatesArchive(MEGrid):
    """Save (some of) the initial states upon which the elites were evaluated when added to the archive, so that we can
    reproduce their behavior at evaluation time (and compare it to generalization to other seeds)."""

    def __init__(self, bin_sizes, bin_bounds, n_init_states, map_w, map_h, **kwargs):
        super(MEInitStatesArchive, self).__init__(bin_sizes, bin_bounds, **kwargs)
        self.init_states_archive = np.empty(
            shape=(*bin_sizes, n_init_states, map_w, map_h)
        )

    def set_init_states(self, init_states):
        self.init_states = init_states

    def add(self, item):
        index = super(MEInitStatesArchive, self).add(item)

        if index is not None:
            idx = self.index_grid(item.features)
            archive_init_states(self.init_states_archive, self.init_states, idx)

        return index



class FlexArchive(InitStatesArchive):
    """ Subclassing a pyribs archive class to do some funky stuff."""

    def __init__(self, *args, **kwargs):
        self.n_evals = {}
        #       self.obj_hist = {}
        #       self.bc_hist = {}
        super().__init__(*args, **kwargs)
        #       # "index of indices", so we can remove them from _occupied_indices when removing
        #       self._index_ranks = {}
        self._occupied_indices = set()

    def _add_occupied_index(self, index):
        #       rank = len(self._occupied_indices)
        #       self._index_ranks[index] = rank  # the index of the index in _occupied_indices

        return super()._add_occupied_index(index)

    def _remove_occupied_index(self, index):
        self._occupied_indices.remove(index)
        self._occupied_indices_cols = tuple(
            [self._occupied_indices[i][j] for i in range(len(self._occupied_indices))]
            for j in range(len(self._storage_dims))
        )

    def pop_elite(self, obj, bcs, old_bcs):
        """
        Need to call update_elite after this!
        """
        # Remove it, update it
        old_idx = self.get_index(np.array(old_bcs))
        self._remove_occupied_index(old_idx)

        #       rank = self._index_ranks.pop(old_idx)
        #       self._occupied_indices.pop(rank)
        #       [self._occupied_indices_cols[i].pop(rank) for i in range(len(self._storage_dims))]
        n_evals = self.n_evals.pop(old_idx)
        old_obj = self._objective_values[old_idx]
        mean_obj = (old_obj * n_evals + obj) / (n_evals + 1)
        mean_bcs = np.array(
            [
                (old_bcs[i] * n_evals + bcs[i]) / (n_evals + 1)
                for i in range(len(old_bcs))
            ]
        )
        #       obj_hist = self.obj_hist.pop(old_idx)
        #       obj_hist.append(obj)
        #       mean_obj = np.mean(obj_hist)
        #       bc_hist = self.bc_hist.pop(old_idx)
        #       bc_hist.append(bcs)
        #       bc_hist_np = np.asarray(bc_hist)
        #       mean_bcs = bc_hist_np.mean(axis=0)
        self._objective_values[old_idx] = np.nan
        self._behavior_values[old_idx] = np.nan
        self._occupied[old_idx] = False
        solution = self._solutions[old_idx].copy()
        self._solutions[old_idx] = np.nan
        self._metadata[old_idx] = np.nan
        #       while len(obj_hist) > 100:
        #           obj_hist = obj_hist[-100:]
        #       while len(bc_hist) > 100:
        #           bc_hist = bc_hist[-100:]

        return solution, mean_obj, mean_bcs, n_evals

    def update_elite(self, solution, mean_obj, mean_bcs, n_evals):
        """
        obj: objective score from new evaluations
        bcs: behavior characteristics from new evaluations
        old_bcs: previous behavior characteristics, for getting the individuals index in the archive
        """

        # Add it back

        self.add(solution, mean_obj, mean_bcs, None, n_evals=n_evals)

    #       self._occupied[new_idx] = True
    #       self._behavior_values[new_idx] = mean_bcs
    #       # FIXME: how to remove old index from occupied_indices :((( Hopefully this does not fuxk us too hard
    #       # (it fucks get_random_elite and checking emptiness... but that's ok for now)
    #       self._occupied_indices.append(new_idx)  # this doesn't really do anything then
    # self._add_occupied_index(new_idx)
    #       self.bc_hists[new_idx] = bc_hists
    #       self.obj_hists[new_idx] = obj_hists

    def add(self, solution, objective_value, behavior_values, meta, n_evals=0):

        index = self.get_index(behavior_values)

        status, dtype_improvement = super().add(
            solution, objective_value, behavior_values, meta, index
        )

        if not status == AddStatus.NOT_ADDED:
            if n_evals == 0:
                self.n_evals[index] = 1
            else:
                self.n_evals[index] = min(n_evals + 1, 100)

        return status, dtype_improvement


def unravel_index(
    indices: th.LongTensor, shape: Tuple[int, ...]
) -> th.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `th` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = th.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = th.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)


# TODO: Use the GPU!

#   if CUDA:
#       m.cuda()
#       m.to('cuda:0')

class ResettableNN(nn.Module):

    def reset(self):
        pass


def gauss(x, mean=0, std=1):
    return th.exp((-(x - mean) ** 2)/(2 * std ** 2))


class MixActiv(nn.Module):
    def __init__(self):
        super().__init__()
        self.activations = (th.sin, th.tanh, gauss, th.relu)
        self.n_activs = len(self.activations)


    def forward(self, x):
        n_chan = x.shape[1]
        chans_per_activ = n_chan / self.n_activs
        chan_i = 0
        xs = []
        for i, activ in enumerate(self.activations):
            xs.append(activ(x[:, int(chan_i):int(chan_i + chans_per_activ), :, :]))
            chan_i += chans_per_activ
        x = th.cat(xs, axis=1)
        return x


class AuxNCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions, n_aux=3):
        super().__init__()
        self.n_hid_1 = n_hid_1 = 32
        self.n_aux = n_aux
        self.l1 = Conv2d(n_in_chans + self.n_aux, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions + self.n_aux, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        self.last_aux = None
        if RENDER:
            cv2.namedWindow("Auxiliary NCA")

    def forward(self, x):
        with th.no_grad():
            if self.last_aux is None:
                self.last_aux = th.zeros(size=(1, self.n_aux, *x.shape[-2:]))
            x_in = th.cat([x, self.last_aux], axis=1)
            x = self.l1(x_in)
            x = th.nn.functional.relu(x)
            x = self.l2(x)
            x = th.nn.functional.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)
            self.last_aux = x[:,-self.n_aux:,:,:]
            x = x[:, :-self.n_aux,:,:]
            if RENDER:
#               im = self.last_aux[0].cpu().numpy().transpose(1,2,0)
                aux = self.last_aux[0].cpu().numpy()
                aux = aux / aux.max()
                im = np.expand_dims(np.vstack(aux), axis=0)
                im = im.transpose(1, 2, 0)
                cv2.imshow("Auxiliary NCA", im)
                cv2.waitKey(1)

        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False

    def reset(self, init_aux=None):
        self.last_aux = None


class DoneAuxNCA(AuxNCA):
    def __init__(self, n_in_chans, n_actions, n_aux=3):
        # Add an extra auxiliary ("done") channel after the others
        n_aux += 1
        super().__init__(n_in_chans, n_actions, n_aux=n_aux)
        done_kernel_size = 3
        self.l_done = Conv2d(1, 1, 7, stride=999)

    def forward(self, x):
        with th.no_grad():
            x, done = super().forward(x)
            # retrieve local activation from done channel
            done_x = th.sigmoid(self.l_done(x[:,-1:,:,:])).flatten() - 0.5
            done = (done_x > 0).item()

        return x, done

    def reset(self, init_aux=None):
        self.last_aux = None


class GeneratorNN(ResettableNN):
#class NCA(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, **kwargs):
        super().__init__()
        n_hid_1 = 32
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.nn.functional.relu(x)
            x = self.l2(x)
            x = th.nn.functional.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)


        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False

class MixNCA(ResettableNN):
    def __init__(self, *args, **kwargs):
        super(MixNCA, self).__init__(*args, **kwargs)
        self.mix_activ = MixActiv()

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = self.mix_activ(x)
            x = self.l2(x)
            x = self.mix_activ(x)
            x = self.l3(x)
            x = th.sigmoid(x)


class CoordNCA(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions.
    With coordinates as additional input, like a CPPN."""

    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid_1 = 28
        #       n_hid_2 = 16

        self.l1 = Conv2d(n_in_chans + 2, n_hid_1, 3, 1, 1, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            coords = get_coord_grid(x, normalize=True)
            x = th.hstack((coords, x))
            x = self.l1(x)
            x = th.nn.functional.relu(x)
            x = self.l2(x)
            x = th.nn.functional.relu(x)
            x = self.l3(x)
            x = th.sigmoid(x)

        # axis 0 is batch
        # axis 1 is the tile-type (one-hot)
        # axis 0,1 is the x value
        # axis 0,2 is the y value

        return x, False


from pytorch_neat.cppn import create_cppn, Leaf
import neat
from neat.genome import DefaultGenome

def get_coord_grid(x, normalize=False):
    width = x.shape[-2]
    height = x.shape[-1]
    X = th.arange(width)
    Y = th.arange(height)
    if normalize:
        X = X / width
        Y = Y / height
    else:
        X = X / 1
        Y = Y / 1
    X, Y = th.meshgrid(X, Y)
    x = th.stack((X, Y)).unsqueeze(0)

    return x


#class ReluCPPN(ResettableNN):
class FeedForwardCPPN(nn.Module):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        x = get_coord_grid(x, normalize=True)
        with th.no_grad():
            x = th.relu(self.l1(x))
            x = th.relu(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenReluCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True)
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.relu(self.l1(x))
            x = th.relu(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class SinCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        x = get_coord_grid(x, normalize=True) * 2
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenSinCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class MixCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        self.mix_activ = MixActiv()


    def forward(self, x):
        x = get_coord_grid(x, normalize=True) * 2
        with th.no_grad():
            x = self.mix_activ(self.l1(x))
            x = self.mix_activ(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class GenMixCPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2+n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        self.mix_activ = MixActiv()


    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = self.mix_activ(self.l1(x))
            x = self.mix_activ(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class FixedGenCPPN(ResettableNN):
    """A fixed-topology CPPN that takes additional channels of noisey input to prompts its output.
    Like a CoordNCA but without the repeated passes and with 1x1 rather than 3x3 kernels."""
    # TODO: Maybe try this with 3x3 conv, just to cover our bases?
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid = 64
        self.l1 = Conv2d(2 + n_in_chans, n_hid, kernel_size=1)
        self.l2 = Conv2d(n_hid, n_hid, kernel_size=1)
        self.l3 = Conv2d(n_hid, n_actions, kernel_size=1)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)

    def forward(self, x):
        coord_x = get_coord_grid(x, normalize=True) * 2
        x = th.cat((x, coord_x), axis=1)
        with th.no_grad():
            x = th.sin(self.l1(x))
            x = th.sin(self.l2(x))
            x = th.sigmoid(self.l3(x))

        return x, True


class CPPN(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        neat_config_path = 'config_cppn'
        self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                              neat_config_path)
        self.n_actions = n_actions
        self.neat_config.genome_config.num_outputs = n_actions
        self.neat_config.genome_config.num_hidden = 2
        self.genome = DefaultGenome(0)
        self.genome.configure_new(self.neat_config.genome_config)
        self.input_names = ['x_in', 'y_in']
        self.output_names = ['tile_{}'.format(i) for i in range(n_actions)]
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def mate(self, ind_1, fit_0, fit_1):
        self.genome.fitness = fit_0
        ind_1.genome.fitness = fit_1
        return self.genome.configure_crossover(self.genome, ind_1.genome, self.neat_config.genome_config)

    def mutate(self):
#       print(self.input_names, self.neat_config.genome_config.input_keys, self.genome.nodes)
        self.genome.mutate(self.neat_config.genome_config)
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def draw_net(self):
        draw_net(self.neat_config, self.genome,  view=True, filename='cppn')

    def forward(self, x):
        X = th.arange(x.shape[-2])
        Y = th.arange(x.shape[-1])
        X, Y = th.meshgrid(X/X.max(), Y/Y.max())
        tile_probs = [self.cppn[i](x_in=X, y_in=Y) for i in range(self.n_actions)]
        multi_hot = th.stack(tile_probs, axis=0)
        multi_hot = multi_hot.unsqueeze(0)
        return multi_hot, True


class CPPNCA(ResettableNN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__()
        n_hid_1 = 32
        with th.no_grad():
            self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 1, bias=True)
            self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
            self.l3 = Conv2d(n_hid_1, n_actions, 1, 1, 0, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_weights)
        n_nca_params = sum(p.numel() for p in self.parameters())
        self.cppn_body = GenCPPN(n_in_chans, n_actions)
        self.normal = th.distributions.multivariate_normal.MultivariateNormal(th.zeros(1), th.eye(1))

    def mate(self):
        raise NotImplementedError

    def mutate(self):
        self.cppn_body.mutate()

        with th.no_grad():
            for layer in self.layers:
                dw = self.normal.sample(layer.weight.shape)
                layer.weight = th.nn.Parameter(layer.weight + dw.squeeze(-1))
                db = self.normal.sample(layer.bias.shape)
                layer.bias = th.nn.Parameter(layer.bias + db.squeeze(-1))

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.nn.functional.relu(x)
            x = self.l2(x)
            x = th.nn.functional.relu(x)
            x = th.sigmoid(x)
        x, _ = self.cppn_body(x)
        return x, False


class GenCPPN(CPPN):
    def __init__(self, n_in_chans, n_actions):
        super().__init__(n_in_chans, n_actions)
        neat_config_path = 'config_cppn'
        self.neat_config = neat.config.Config(DefaultGenome, neat.reproduction.DefaultReproduction,
                                              neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                              neat_config_path)
        self.n_actions = n_actions
        self.neat_config.genome_config.num_outputs = n_actions
        self.genome = DefaultGenome(0)
        self.input_names = ['x_in', 'y_in'] + ['tile_{}_in'.format(i) for i in range(n_actions)]
        n_inputs = len(self.input_names)
        self.output_names = ['tile_{}_out'.format(i) for i in range(n_actions)]
        self.neat_config.genome_config.input_keys = (-1*np.arange(n_inputs) - 1).tolist()
        self.neat_config.genome_config.num_inputs = n_inputs
        self.neat_config.genome_config.num_hidden = 2
        self.genome.configure_new(self.neat_config.genome_config)
        self.cppn = create_cppn(self.genome, self.neat_config, self.input_names, self.output_names)

    def forward(self, x):
        X = th.arange(x.shape[-2])
        Y = th.arange(x.shape[-1])
        X, Y = th.meshgrid(X/X.max(), Y/Y.max())
        inputs = {'x_in': X, 'y_in': Y}
        inputs.update({'tile_{}_in'.format(i): th.Tensor(x[0,i,:,:]) for i in range(self.n_actions)})
        tile_probs = [self.cppn[i](**inputs) for i in range(self.n_actions)]
        multi_hot = th.stack(tile_probs, axis=0)
        multi_hot = multi_hot.unsqueeze(0)
        return multi_hot, True


class Individual(qdpy.phenotype.Individual):
    "An individual for mutating with operators. Assuming we're using vanilla MAP-Elites here."
    def __init__(self, model_cls, n_in_chans, n_actions):
        super(Individual, self).__init__()
        self.model = model_cls(n_in_chans, n_actions)
        self.fitness = Fitness([0])
        self.fitness.delValues()

    def mutate(self):
        self.model.mutate()

    def mate(self, ind_1):
        assert len(self.fitness.values) == 1 == len(ind_1.fitness.values)
        self.model.mate(ind_1.model, fit_0=self.fitness.values[0], fit_1=ind_1.fitness.values[0])

    def __eq__(self, ind_1):
        if not hasattr(ind_1, "model"): return False
        return self.model == ind_1.model


# FIXME: this guy don't work
class GeneratorNNDenseSqueeze(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, observation_shape, n_flat_actions):
        super().__init__()
        n_hid_1 = 16
        # Hack af. Pad the input to make it have root 2? idk, bad
        sq_i = 2
        assert observation_shape[-1] == observation_shape[-2]

        #       while sq_i < observation_shape[-1]:
        #           sq_i = sq_i**2
        #       pad_0 = sq_i - observation_shape[-1]
        self.l1 = Conv2d(n_in_chans, n_hid_1, 3, 1, 0, bias=True)
        self.l2 = Conv2d(n_hid_1, n_hid_1, 3, 2, 0, bias=True)
        self.flatten = th.nn.Flatten()
        n_flat = self.flatten(
            self.l2(self.l1(th.zeros(size=observation_shape)))
        ).shape[-1]
        # n_flat = n_hid_1
        self.d1 = Linear(n_flat, n_flat_actions)
        #       self.d2 = Linear(16, n_flat_actions)
        self.layers = [self.l1, self.l2, self.d1]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = self.l1(x)
            x = th.nn.functional.relu(x)
            x = self.l2(x)
            x = th.nn.functional.relu(x)
            #           for i in range(int(np.log2(x.shape[2])) + 1):
            #               x = self.l2(x)
            #               x = th.nn.functional.relu(x)
            x = self.flatten(x)
            x = self.d1(x)
            x = th.sigmoid(x)
            #           x = self.d2(x)
            #           x = th.sigmoid(x)

        return x, False


class GeneratorNNDense(ResettableNN):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, n_in_chans, n_actions, observation_shape, n_flat_actions):
        super().__init__()
        n_hid_1 = 16
        n_hid_2 = 32
        self.conv1 = Conv2d(n_in_chans, n_hid_1, kernel_size=3, stride=2)
        self.conv2 = Conv2d(n_hid_1, n_hid_2, kernel_size=3, stride=2)
        self.conv3 = Conv2d(n_hid_2, n_hid_2, kernel_size=3, stride=2)
        self.flatten = th.nn.Flatten()
        n_flat = self.flatten(
            self.conv3(self.conv2(self.conv1(th.zeros(size=observation_shape))))
        ).shape[-1]
        #       self.fc1 = Linear(n_flat, n_flat_actions)
        self.fc1 = Linear(n_flat, n_hid_2)
        self.fc2 = Linear(n_hid_2, n_flat_actions)
        self.layers = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2]
        self.apply(init_weights)

    def forward(self, x):
        with th.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = F.softmax(self.fc2(x), dim=1)

        return x, False


class PlayerNN(ResettableNN):
    def __init__(self, n_tile_types, n_actions=4):
        super().__init__()
        self.n_tile_types = n_tile_types
        assert "zelda" in PROBLEM
        self.l1 = Conv2d(n_tile_types, 16, 3, 1, 0, bias=True)
        self.l2 = Conv2d(16, 16, 3, 2, 1, bias=True)
        self.l3 = Conv2d(16, n_actions, 3, 1, 1, bias=True)
        self.layers = [self.l1, self.l2, self.l3]
        self.apply(init_play_weights)
        self.flatten = th.nn.Flatten()
        self.net_reward = 0
        self.n_episodes = 0

    def forward(self, x):
        x = th.Tensor(get_one_hot_map(x, self.n_tile_types))
        x = x.unsqueeze(0)
        with th.no_grad():
            x = th.relu(self.l1(x))

            for i in range(int(np.log2(x.shape[2])) + 1):
                #           for i in range(1):
                x = th.relu(self.l2(x))
            x = th.relu(self.l3(x))

            #           x = x.argmax(1)
            #           x = x[0]
            x = x.flatten()
            x = th.softmax(x, axis=0)
            # x = [x.argmax().item()]
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
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv2d:
        th.nn.init.orthogonal_(m.weight)


def init_play_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform(m.weight, gain=0)
        m.bias.data.fill_(0.00)

    if type(m) == th.nn.Conv2d:
        #       th.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        th.nn.init.constant_(m.weight, 0)


def set_nograd(nn):
    for param in nn.parameters():
        param.requires_grad = False


def get_init_weights(nn):
    """
    Use to get flat vector of weights from PyTorch model
    """
    init_params = []

    if isinstance(nn, CPPN):
        for node in nn.cppn:
            if isinstance(node, Leaf):
                continue
            init_params.append(node.weights)
            init_params.append(node.bias)
    else:
        for lyr in nn.layers:
            init_params.append(lyr.weight.view(-1).numpy())
            init_params.append(lyr.bias.view(-1).numpy())
    init_params = np.hstack(init_params)
    print("number of initial NN parameters: {}".format(init_params.shape))

    return init_params


def set_weights(nn, weights):
    if ALGO == "ME":
        # then out nn is contained in the individual
        individual = weights  # I'm sorry mama
        return individual.model
    with th.no_grad():
        n_el = 0

        if isinstance(nn, CPPN):
            for node in nn.cppn:
                l_weights = weights[n_el : n_el + len(node.weights)]
                n_el += len(node.weights)
                node.weights = l_weights
                b_weight = weights[n_el: n_el + 1]
                n_el += 1
                node.bias = b_weight
        else:
            for layer in nn.layers:
                l_weights = weights[n_el : n_el + layer.weight.numel()]
                n_el += layer.weight.numel()
                l_weights = l_weights.reshape(layer.weight.shape)
                layer.weight = th.nn.Parameter(th.Tensor(l_weights))
                layer.weight.requires_grad = False
                b_weights = weights[n_el : n_el + layer.bias.numel()]
                n_el += layer.bias.numel()
                b_weights = b_weights.reshape(layer.bias.shape)
                layer.bias = th.nn.Parameter(th.Tensor(b_weights))
                layer.bias.requires_grad = False

    return nn


def get_one_hot_map(int_map, n_tile_types):
    obs = (np.arange(n_tile_types) == int_map[..., None]).astype(int)
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
    # FIXME: make this robust to different action spaces
    n_classes = len(env._prob._prob)
    max_val = -(1 / n_classes) * np.log(1 / n_classes) * n_classes
    total = len(int_map.flatten())
    entropy = 0.0

    for tile in range(n_classes):
        p = (tile == int_map.flatten()).astype(int).sum() / total

        if p != 0:
            entropy -= p * np.log(p)

    return entropy / max_val


def get_counts(int_map, env):
    """
    Function to calculate the tile counts for all possible tiles
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a python list with tile counts for each tile normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width * env._prob._height  # for example 14*14=196

    return [
        np.sum(int_map.flatten() == tile) / max_val
        for tile in range(len(env._prob._prob))
    ]


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
    max_val = env._prob._width * env._prob._height / 2  # for example 14*14/2=98
    m = 0

    if int(int_map.shape[0]) % 2 == 0:
        m = np.sum(
            (
                int_map[: int(int_map.shape[0] / 2)]
                == np.flip(int_map[int(int_map.shape[0] / 2) :], 0)
            ).astype(int)
        )
        m = m / max_val
    else:
        m = np.sum(
            (
                int_map[: int(int_map.shape[0] / 2)]
                == np.flip(int_map[int(int_map.shape[0] / 2) + 1 :], 0)
            ).astype(int)
        )
        m = m / max_val

    return m


def get_ver_sym(int_map, env):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    max_val = env._prob._width * env._prob._height / 2  # for example 14*14/2=98
    m = 0

    if int(int_map.shape[1]) % 2 == 0:
        m = np.sum(
            (
                int_map[:, : int(int_map.shape[1] / 2)]
                == np.flip(int_map[:, int(int_map.shape[1] / 2) :], 1)
            ).astype(int)
        )
        m = m / max_val
    else:
        m = np.sum(
            (
                int_map[:, : int(int_map.shape[1] / 2)]
                == np.flip(int_map[:, int(int_map.shape[1] / 2) + 1 :], 1)
            ).astype(int)
        )
        m = m / max_val

    return m


# SYMMETRY


def get_sym(int_map, env):
    """
    Function to get the vertical symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    result = (get_ver_sym(int_map, env) + get_hor_sym(int_map, env)) / 2.0

    return result


# CO-OCCURRANCE


def get_co(int_map, env):
    max_val = env._prob._width * env._prob._height * 4
    result = (
        np.sum((np.roll(int_map, 1, axis=0) == int_map).astype(int))
        + np.sum((np.roll(int_map, -1, axis=0) == int_map).astype(int))
        + np.sum((np.roll(int_map, 1, axis=1) == int_map).astype(int))
        + np.sum((np.roll(int_map, -1, axis=1) == int_map).astype(int))
    )

    return result / max_val


def get_regions(stats):
    return stats["regions"]


def get_path_length(stats):
    return stats["path-length"]


# TODO: call this once to return the releveant get_bc function, then call this after each eval, so that we don't have to repeatedly compare strings


def get_bc(bc_name, int_map, stats, env):
    if bc_name in stats.keys():
        return stats[bc_name]
    elif bc_name == "co-occurance":
        return get_co(int_map, env)
    elif bc_name == "symmetry":
        return get_sym(int_map, env)
    elif bc_name == "symmetry-vertical":
        return get_ver_sym(int_map, env)
    elif bc_name == "symmetry-horizontal":
        return get_hor_sym(int_map, env)
    elif bc_name == "emptiness":
        return get_emptiness(int_map, env)
    elif bc_name == "entropy":
        return get_entropy(int_map, env)
    elif bc_name == "NONE":
        return 0
    else:
        print("The BC {} is not recognized.".format(bc_name))
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
    if ALGO == "ME":
        # Do this inside optimizer ..?
        return
    # TensorBoard Logging.
    df = archive.as_pandas(include_solutions=False)
    elapsed_time = time.time() - start_time
    writer.add_scalar("{} ArchiveSize".format(name), len(df), itr)
    writer.add_scalar("{} score/mean".format(name), df["objective"].mean(), itr)
    writer.add_scalar("{} score/max".format(name), df["objective"].max(), itr)
    writer.add_scalar("{} score/min".format(name), df["objective"].min(), itr)

    # Change: log mean, max, and min for all stats

    if level_json:
        stats = ["batch_reward", "targets"]

        if N_INIT_STATES > 1:
            stats += ["variance", "diversity"]
        # level_json = {'level': final_levels.tolist(),'batch_reward':[batch_reward] * len(final_levels.tolist()), 'variance': [variance_penalty] * len(final_levels.tolist()), 'diversity':[diversity_bonus] * len(final_levels.tolist()),'targets':trg.tolist(), **bc_dict}

        for stat in stats:
            writer.add_scalar(
                "Training {}/min".format(stat), np.min(level_json[stat]), itr
            )
            writer.add_scalar(
                "Training {}/mean".format(stat), np.mean(level_json[stat]), itr
            )
            writer.add_scalar(
                "Training {}/max".format(stat), np.max(level_json[stat]), itr
            )

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
        action = player(p_obs["map"])

        if isinstance(action, th.Tensor):
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
        # net_p_rew += p_rew
        net_p_rew = p_rew

        if p_done:
            break

    #   player.assign_reward(net_p_rew)
    action_freqs = np.bincount(action_hist, minlength=len(env.player_actions))
    action_entropy = scipy.stats.entropy(action_freqs)
    local_action_entropy = np.mean(
        [
            scipy.stats.entropy(
                np.bincount(action_hist[i : i + 10], minlength=len(env.player_actions))
            )
            for i in np.arange(0, len(action_hist) - 10, 6)
        ]
    )
    local_action_entropy = np.nan_to_num(local_action_entropy)

    return net_p_rew, [action_entropy, local_action_entropy]


@ray.remote
def multi_evo(
    env,
    model,
    model_w,
    n_tile_types,
    init_states,
    bc_names,
    static_targets,
    seed,
    player_1,
    player_2,
    proc_id=None,
    init_states_archive=None,
    index=None,
):
    if init_states is None:
        init_states = get_init_states(init_states_archive, tuple(index))

    if proc_id is not None:
        print("simulating id: {}".format(proc_id))
    model = set_weights(model, model_w)
    result = simulate(
        env=env,
        model=model,
        n_tile_types=n_tile_types,
        init_states=init_states,
        bc_names=bc_names,
        static_targets=static_targets,
        seed=seed,
        player_1=player_1,
        player_2=player_2,
    )

    return result


@ray.remote
def multi_play_evo(
    env,
    gen_model,
    player_1_w,
    n_tile_types,
    init_states,
    play_bc_names,
    static_targets,
    seed,
    player_1,
    player_2,
    playable_levels,
    proc_id=None,
):

    if proc_id is not None:
        print("simulating id: {}".format(proc_id))
    player_1 = set_weights(player_1, player_1_w)
    obj, bcs = player_simulate(
        env=env,
        n_tile_types=n_tile_types,
        play_bc_names=play_bc_names,
        seed=seed,
        player_1=player_1,
        playable_levels=playable_levels,
    )

    return obj, bcs


def gen_playable_levels(env, gen_model, init_states, n_tile_types):
    """ To get only the playable levels of a given generator, so that we can run player evaluations on them more quickly."""
    final_levels = []

    for int_map in init_states:
        obs = get_one_hot_map(int_map, n_tile_types)

        if RENDER:
            env.render()
        done = False
        n_step = 0
        last_int_map = None

        while not done:
            int_tensor = th.unsqueeze(th.Tensor(obs), 0)
            action, done = gen_model(int_tensor)[0].numpy()
            obs = action
            int_map = done or action.argmax(axis=0)
            env._rep._map = int_map
            done = done or (int_map == last_int_map).all() or n_step >= N_STEPS

            #           if INFER and not EVALUATE:
            #               time.sleep(1 / 30)

            if done:
                gen_model.reset()
                env._rep._old_map = int_map
                env._rep._random_start = False
                _ = env.reset()

                if env.is_playable():
                    final_levels.append(int_map)
            n_step += 1

    return final_levels


def player_simulate(
    env, n_tile_types, play_bc_names, player_1, playable_levels, seed=None
):
    n_evals = 10
    net_reward = 0
    bcs = []

    for int_map in playable_levels * n_evals:
        if INFER:
            #           env.render()
            input("ready player 1")
        p_1_rew, p_bcs = play_level(env, int_map, player_1)
        bcs.append(p_bcs)

        if INFER:
            print("p_1 reward: ", p_1_rew)
        net_reward += p_1_rew

    reward = net_reward / len(playable_levels * n_evals)
    bcs = [np.mean([bcs[j][i] for j in range(len(bcs))]) for i in range(len(bcs[0]))]

    return reward, bcs


def simulate(
    env,
    model,
    n_tile_types,
    init_states,
    bc_names,
    static_targets,
    seed=None,
    player_1=None,
    player_2=None,
    render_levels=False
):
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
    global N_INIT_STATES

    if seed is not None:
        env.seed(seed)

    if PLAY_LEVEL:
        assert player_1 is not None
        assert player_2 is not None

    if CMAES:
        bc_names = ["NONE", "NONE"]
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
    # if SAVE_LEVELS:
    trg = np.empty(shape=(n_init_states))
    final_levels = np.empty(shape=init_states.shape, dtype=np.uint8)
    batch_reward = 0
    batch_time_penalty = 0
    batch_targets_penalty = 0
    batch_play_bonus = 0

    if render_levels:
        level_frames = []

    for (n_episode, init_state) in enumerate(init_states):
        # NOTE: Sneaky hack. We don't need initial stats. Never even reset. Heh. Be careful!!
        # Set the representation to begin in the upper left corner
        env._rep._map = init_state.copy()
        env._prob.path_coords = []
        env._prob.path_length = None
        # Only applies to narrow and turtle. Better than using reset, but ugly, and not optimal
        # TODO: wrap the env instead
        env._rep._x = env._rep._y = 0
        # env._rep._x = np.random.randint(env._prob._width)
        # env._rep._y = np.random.randint(env._prob._height)
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
            if render_levels:
                level_frames.append(env.render(mode="rgb_array"))
            #           in_tensor = th.unsqueeze(
            #               th.unsqueeze(th.tensor(np.float32(obs['map'])), 0), 0)
            in_tensor = th.unsqueeze(th.Tensor(obs), 0)
            action, done = model(in_tensor)
            action = action[0].numpy()
            # There is probably a better way to do this, so we are not passing unnecessary kwargs, depending on representation
            action, skip = preprocess_action(
                action,
                int_map=env._rep._map,
                x=env._rep._x,
                y=env._rep._y,
                n_dirs=N_DIRS,
                n_tiles=n_tile_types,
            )
            change, x, y = env._rep.update(action)
            int_map = env._rep._map
            obs = get_one_hot_map(env._rep.get_observation()["map"], n_tile_types)
            preprocess_observation(obs, x=env._rep._x, y=env._rep._y)
            #           int_map = action.argmax(axis=0)
            #           obs = get_one_hot_map(int_map, n_tile_types)
            #           env._rep._map = int_map
            done = done or not (change or skip) or n_step >= N_STEPS
            # done = n_step >= N_STEPS

            #           if INFER and not EVALUATE:
            #               time.sleep(1 / 30)

            if done:
                model.reset()
                if render_levels:
                    # get final level state
                    level_frames.append(env.render(mode="rgb_array"))
                # we'll need this to compute Hamming diversity
                final_levels[n_episode] = int_map
                stats = env._prob.get_stats(
                    get_string_map(int_map, env._prob.get_tile_types()),
                    # lenient_paths = True,
                )

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
                        # take the smallest distance from current value to any point in range
                        # NOTE: we're assuming this metric is integer-valued
                        targets_penalty += abs(
                            np.arange(static_targets[k][0], static_targets[k][1]) - stats[k]
                        ).min()
                    else:
                        targets_penalty += abs(static_targets[k] - stats[k])
                #                   targets_penalty = np.sum([abs(static_targets[k] - stats[k]) if not isinstance(static_targets[k], tuple) else abs(np.arange(*static_targets[k]) - stats[k]).min() for k in static_targets])
                batch_targets_penalty -= targets_penalty
                # if SAVE_LEVELS:
                trg[n_episode] = -targets_penalty

                if PLAY_LEVEL:
                    if INFER:
                        env.render()
                        input("ready player 1")
                    p_1_rew, p_bcs = play_level(env, int_map, player_1)

                    if INFER:
                        print("p_1 reward: ", p_1_rew)
                        input("ready player 2")
                    p_2_rew, p_bcs = play_level(env, int_map, player_2)

                    if INFER:
                        print("p_2 reward: ", p_2_rew)

                    max_regret = env._prob.max_reward - env._prob.min_reward
                    # add this in case we get worst possible regret (don't want to punish a playable map)
                    batch_play_bonus += max_regret + p_1_rew - p_2_rew

            if RENDER:
                if INFER:
                    stats = env._prob.get_stats(
                        get_string_map(int_map, env._prob.get_tile_types()),
                        # lenient_paths=True,
                    )
                env.render()


            if done and INFER:  # and not (EVALUATE and THREADS):
                if not EVALUATE:
                    #                   time.sleep(5 / 30)
                    print(
                        "stats: {}\n\ntime_penalty: {}\n targets_penalty: {}".format(
                            stats, time_penalty, targets_penalty
                        )
                    )
            last_int_map = int_map
            n_step += 1
    final_bcs = [bcs[i].mean() for i in range(bcs.shape[0])]
    batch_targets_penalty = TARGETS_PENALTY_WEIGHT * batch_targets_penalty / max(N_INIT_STATES, 1)
    # batch_targets_penalty = batch_targets_penalty / N_INIT_STATES
    batch_reward += batch_targets_penalty

    if PLAY_LEVEL:
        batch_reward += batch_play_bonus / max(N_INIT_STATES, 1)
        time_penalty, targets_penalty, variance_penalty, diversity_bonus = (
            None,
            None,
            None,
            None,
        )
    else:
        #       batch_time_penalty = batch_time_penalty / N_INIT_STATES

        N_INIT_STATES = n_init_states

        if N_INIT_STATES > 1 and (batch_targets_penalty == 0 or not CASCADE_REWARD):
            # Calculate stats that depend on having generated multiple levels. If using gated reward, only calculate these additional components of reward if level is
            # perfectly valid.
            # Variance penalty is the negative average (per-BC) standard deviation from the mean BC vector.
            variance_penalty = (
                -np.sum([bcs[i].std() for i in range(bcs.shape[0])]) / bcs.shape[0]
            )
            # Diversity bonus. We want minimal variance along BCS *and* diversity in terms of the map.
            # Sum pairwise hamming distances between all generated maps.
            diversity_bonus = np.sum(
                [
                    np.sum(final_levels[j] != final_levels[k]) if j != k else 0
                    for k in range(N_INIT_STATES)
                    for j in range(N_INIT_STATES)
                ]
            ) / (N_INIT_STATES * N_INIT_STATES - 1)
            # ad hoc scaling :/
            diversity_bonus = 10 * diversity_bonus / (width * height)
            batch_reward = batch_reward + max(0, variance_penalty + diversity_bonus)
        else:
            variance_penalty = None
            diversity_bonus = None

    if SAVE_LEVELS:
        bc_dict = {}

        for i in range(len(bc_names)):
            bc_name = bc_names[i]
            bc_dict[bc_name] = bcs[i, :].tolist()
        level_json = {
            "level": final_levels.tolist(),
            "batch_reward": [batch_reward] * len(final_levels.tolist()),
            "variance": [variance_penalty] * len(final_levels.tolist()),
            "diversity": [diversity_bonus] * len(final_levels.tolist()),
            "targets": trg.tolist(),
            **bc_dict,
        }
    else:
        level_json = {
            "level": final_levels.tolist(),
            "batch_reward": [batch_reward] * len(final_levels.tolist()),
            "variance": [variance_penalty] * len(final_levels.tolist()),
            "diversity": [diversity_bonus] * len(final_levels.tolist()),
            "targets": trg.tolist(),
        }

    if render_levels:
        return level_frames

    if not INFER:
        return level_json, batch_reward, final_bcs
    else:
        return (
            level_json,
            batch_reward,
            final_bcs,
            (
                batch_time_penalty,
                batch_targets_penalty,
                variance_penalty,
                diversity_bonus,
            ),
        )


class EvoPCGRL:
    def __init__(self):
        self.init_env()
        assert self.env.observation_space["map"].low[0, 0] == 0
        # get number of tile types from environment's observation space
        # here we assume that all (x, y) locations in the observation space have the same upper/lower bound
        self.n_tile_types = self.env.observation_space["map"].high[0, 0] + 1
        self.width = self.env._prob._width
        self.height = self.env._prob._height

        # FIXME why not?
        # self.width = self.env._prob._width

        # TODO: make reward a command line argument?
        # TODO: multi-objective compatibility?
        self.bc_names = BCS

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        self.bc_bounds = self.env._prob.cond_bounds
        self.bc_bounds.update(
            {
                "co-occurance": (0.0, 1.0),
                "symmetry": (0.0, 1.0),
                "symmetry-vertical": (0.0, 1.0),
                "symmetry-horizontal": (0.0, 1.0),
                "emptiness": (0.0, 1.0),
                "entropy": (0.0, 1.0),
            }
        )

        self.static_targets = self.env._prob.static_trgs

        if REEVALUATE_ELITES or (RANDOM_INIT_LEVELS and args.n_init_states != 0):
            init_level_archive_args = (N_INIT_STATES, self.height, self.width)
        else:
            init_level_archive_args = ()

        if ALGO == "ME":
            if RANDOM_INIT_LEVELS and args.n_init_states != 0:
                gen_archive_cls = MEInitStatesArchive
            else:
                gen_archive_cls = MEGrid

        elif REEVALUATE_ELITES:
            # If we are constantly providing new random seeds to generators, we may want to regularly re-evaluate
            # elites
            gen_archive_cls = FlexArchive
        elif RANDOM_INIT_LEVELS and not args.n_init_states == 0:
            # If we have random seeds each generation but are not re-evaluating elites, then we want to hang onto these
            # random seeds.
            gen_archive_cls = InitStatesArchive
        #           gen_archive_cls = GridArchive
        else:
            gen_archive_cls = GridArchive
            init_level_archive_args = ()

        if PLAY_LEVEL:
            self.play_bc_names = ["action_entropy", "local_action_entropy"]
            self.play_bc_bounds = {
                "action_entropy": (0, 4),
                "local_action_entropy": (0, 4),
            }
            self.gen_archive = gen_archive_cls(
                [100 for _ in self.bc_names],
                # [1],
                # [(-1, 1)],
                [self.bc_bounds[bc_name] for bc_name in self.bc_names],
            )
            self.play_archive = FlexArchive(
                # minimum of: 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                # [min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                [100 for _ in self.play_bc_names],
                # min/max for each BC
                [self.play_bc_bounds[bc_name] for bc_name in self.play_bc_names],
            )
        else:
            if CMAES:
                # Restrict the archive to 1 cell so that we are effectively doing CMAES. BCs should be ignored.
                self.gen_archive = gen_archive_cls(
                    [1, 1], [(0, 1), (0, 1)], *init_level_archive_args
                )
            else:
                self.gen_archive = gen_archive_cls(
                    # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                    # [min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                    [100 for _ in self.bc_names],
                    # min/max for each BC
                    [self.bc_bounds[bc_name] for bc_name in self.bc_names],
                    *init_level_archive_args,
                )

        reps_to_out_chans = {
            "cellular": self.n_tile_types,
            "wide": self.n_tile_types,
            "narrow": self.n_tile_types + 1,
            "turtle": self.n_tile_types + N_DIRS,
        }

        reps_to_in_chans = {
            "cellular": self.n_tile_types,
            "wide": self.n_tile_types,
            "narrow": self.n_tile_types + 1,
            "turtle": self.n_tile_types + 1,
        }

        n_out_chans = reps_to_out_chans[REPRESENTATION]
        n_in_chans = reps_to_in_chans[REPRESENTATION]

        if MODEL == "CNN":
            # Adding n_tile_types as a dimension here. Why would this not be in the env's observation space though? Should be one-hot by default?
            observation_shape = (
                1,
                self.n_tile_types,
                *self.env.observation_space["map"].shape,
            )

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
                raise NotImplementedError(
                    "I don't know how to handle this action space: {}".format(
                        type(self.env.action_space)
                    )
                )
            self.gen_model = GeneratorNNDense(
                n_in_chans=self.n_tile_types,
                n_actions=n_out_chans,
                observation_shape=observation_shape,
                n_flat_actions=n_flat_actions,
            )
        # TODO: remove this, just call model "NCA"
        elif MODEL == "NCA":
            self.gen_model = globals()["GeneratorNN"](
                n_in_chans=self.n_tile_types, n_actions=n_out_chans
            )
        else:
            self.gen_model = globals()[MODEL](
                n_in_chans=self.n_tile_types, n_actions=n_out_chans
            )
        set_nograd(self.gen_model)
        initial_w = get_init_weights(self.gen_model)
        assert len(initial_w.shape) == 1
        self.n_generator_weights = initial_w.shape[0]
        self.n_player_weights = 0
        # TODO: different initial weights per emitter as in pyribs lunar lander relanded example?

        if MODEL == "NCA":
            init_step_size = 1
        elif MODEL == "CNN":
            init_step_size = 1
        else:
            init_step_size = 1

        if CMAES:
            # The optimizing emitter will prioritize fitness over exploration of behavior space
            emitter_type = OptimizingEmitter
        else:
            emitter_type = ImprovementEmitter

        batch_size = 30
        n_emitters = 5
        if ALGO == "ME":
            pass

        elif args.mega:
            gen_emitters = [
                GradientImprovementEmitter(
                    self.gen_archive,
                    initial_w.flatten(),
                    # TODO: play with initial step size?
                    sigma_g=10.0,
                    stepsize=0.002,  # Initial step size.
                    gradient_optimizer="adam",
                    selection_rule="mu",
                    batch_size=batch_size,
                )
                for _ in range(n_emitters)  # Create 5 separate emitters.
            ]
        else:
            gen_emitters = [
                #           ImprovementEmitter(
                emitter_type(
                    self.gen_archive,
                    initial_w.flatten(),
                    # TODO: play with initial step size?
                    init_step_size,  # Initial step size.
                    batch_size=batch_size,
                )
                for _ in range(n_emitters)  # Create 5 separate emitters.
            ]

        if PLAY_LEVEL:
            # Concatenate designer and player weights
            self.play_model = PlayerNN(
                self.n_tile_types, n_actions=len(self.env.player_actions)
            )
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
                    batch_size=batch_size,
                )
                for _ in range(n_emitters)  # Create 5 separate emitters.
            ]
            self.play_optimizer = Optimizer(self.play_archive, play_emitters)
        if ALGO == "ME":
            ind_cls_args = {
                    'model_cls': globals()[MODEL],
                   'n_in_chans': self.n_tile_types,
                   'n_actions': self.n_tile_types,
            }

            self.gen_optimizer = MEOptimizer(self.gen_archive,
                                             ind_cls=Individual,
                                             batch_size=n_emitters*batch_size,
                                             ind_cls_args=ind_cls_args,
                                             )
        else:
            self.gen_optimizer = Optimizer(self.gen_archive, gen_emitters)

        # These are the initial maps which will act as seeds to our NCA models

        if args.n_init_states == 0:
            # special square patch
            self.init_states = np.zeros(shape=(1, self.height, self.width))
            self.init_states[0, 5:-5, 5:-5] = 1
        else:
            #           self.init_states = np.random.randint(
            #               0, self.n_tile_types, (N_INIT_STATES, self.width, self.height)
            #           )
            self.init_states = gen_random_levels(N_INIT_STATES, self.env)

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
        with open(os.path.join(SAVE_PATH, "settings.json"), "w", encoding="utf-8") as f:
            json.dump(arg_dict, f, ensure_ascii=False, indent=4)

    def evolve(self):

        net_p_itr = 0

        for itr in tqdm(range(self.n_itr, self.total_itrs + 1)):
            # Request models from the optimizer.

            if args.mega:
                gen_sols = self.gen_optimizer.ask(grad_estimate=True)
            else:
                # if algo is ME, these are "individual" objects
                gen_sols = self.gen_optimizer.ask()

            # Evaluate the models and record the objectives and BCs.
            objs, bcs = [], []
            stats = ["batch_reward", "variance", "diversity", "targets"]
            stat_json = {
                "batch_reward": [],
                "variance": [],
                "diversity": [],
                "targets": [],
            }

            if RANDOM_INIT_LEVELS and args.n_init_states != 0:
                init_states = gen_random_levels(N_INIT_STATES, self.env)
            else:
                init_states = self.init_states

            if THREADS:
                n_sols = len(gen_sols)
                if N_PROC is not None:
                    n_proc = N_PROC
                else:
                    n_proc = n_sols
                n_launches = np.ceil(n_sols / n_proc)
                results = []
                for n_launch in range(int(n_launches)):
                    futures = [
                        multi_evo.remote(
                            self.env,
                            self.gen_model,
                            model_w,
                            self.n_tile_types,
                            init_states,
                            self.bc_names,
                            self.static_targets,
                            seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                        )
                        for model_w in gen_sols
                    ]
                    results += ray.get(futures)
                    del futures
                    auto_garbage_collect()

                for result in results:
                    level_json, m_obj, m_bcs = result

                    if SAVE_LEVELS:
                        df = pd.DataFrame(level_json)
                        df = df[df["targets"] == 0]

                        if len(df) > 0:
                            df.to_csv(
                                os.path.join(SAVE_PATH, "levels.csv"),
                                mode="a",
                                header=False,
                                index=False,
                            )
                    objs.append(m_obj)
                    bcs.append([*m_bcs])
                    [stat_json[stat].extend(level_json[stat]) for stat in stats]
                del results
                auto_garbage_collect()
            else:
                for model_w in gen_sols:
                    gen_model = set_weights(self.gen_model, model_w)
                    level_json, m_obj, m_bcs = simulate(
                        env=self.env,
                        model=gen_model,
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
                        df = df[df["targets"] == 0]

                        if len(df) > 0:
                            df.to_csv(
                                os.path.join(SAVE_PATH, "levels.csv"),
                                mode="a",
                                header=False,
                                index=False,
                            )
                    objs.append(m_obj)
                    bcs.append(m_bcs)
                    [stat_json[stat].extend(level_json[stat]) for stat in stats]

            if RANDOM_INIT_LEVELS:
                # Tell the archive what the initial states are, so that we can record them in case an individual is
                # added.
                self.gen_archive.set_init_states(init_states)
            # Send the results back to the optimizer.
            if args.mega:
                # TODO: Here we need the jacobian
                jacobian = None
                self.gen_optimizer.tell(objs, bcs, jacobian=jacobian)
            else:
                self.gen_optimizer.tell(objs, bcs)
#               for emitter in self.gen_optimizer.emitters:
#

            # Re-evaluate elite generators. If doing CMAES, re-evaluate every iteration. Otherwise, try to let the archive grow.

            if REEVALUATE_ELITES and (CMAES or self.n_itr % 1 == 0):
                df = self.gen_archive.as_pandas()
                #               curr_archive_size = len(df)
                high_performing = df.sample(frac=1)
                elite_models = np.array(high_performing.loc[:, "solution_0":])
                elite_bcs = np.array(high_performing.loc[:, "behavior_0":"behavior_1"])

                if THREADS:
                    futures = [
                        multi_evo.remote(
                            self.env,
                            self.gen_model,
                            elite_models[i],
                            self.n_tile_types,
                            init_states,
                            self.bc_names,
                            self.static_targets,
                            seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                        )
                        for i in range(min(max(len(elite_models) // 2, 1), 150 // 2))
                    ]
                    results = ray.get(futures)

                    for (el_i, result) in enumerate(results):
                        old_el_bcs = elite_bcs[el_i]
                        level_json, el_obj, el_bcs = result

                        if SAVE_LEVELS:
                            # Save levels to disk
                            df = pd.DataFrame(level_json)
                            df = df[df["targets"] == 0]

                            if len(df) > 0:
                                df.to_csv(
                                    os.path.join(SAVE_PATH, "levels.csv"),
                                    mode="a",
                                    header=False,
                                    index=False,
                                )
                        #                       mean_obj, mean_bcs, obj_hist, bc_hist = self.gen_archive.pop_elite(el_obj, el_bcs, old_el_bcs)
                        results[el_i] = self.gen_archive.pop_elite(
                            el_obj, el_bcs, old_el_bcs
                        )
                        [stat_json[stat].extend(level_json[stat]) for stat in stats]

                    for (el_i, result) in enumerate(results):
                        self.gen_archive.update_elite(*result)
                    del results
                    auto_garbage_collect()

                else:
                    # 150 to match number of new-model evaluations

                    for elite_i in range(min(max(len(elite_models) // 2, 1), 150 // 2)):
                        # print(elite_i)
                        # pprint.pprint(self.gen_archive.obj_hist, width=1)
                        # pprint.pprint(self.gen_archive.bc_hist, width=1)
                        old_el_bcs = elite_bcs[elite_i]
                        gen_model_weights = elite_models[elite_i]
                        gen_model = set_weights(self.gen_model, gen_model_weights)

                        level_json, el_obj, el_bcs = simulate(
                            env=self.env,
                            model=gen_model,
                            n_tile_types=self.n_tile_types,
                            init_states=init_states,
                            bc_names=self.bc_names,
                            static_targets=self.static_targets,
                            seed=seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                        )
                        idx = self.gen_archive.get_index(old_el_bcs)
                        [stat_json[stat].extend(level_json[stat]) for stat in stats]
                        self.gen_archive.update_elite(
                            *self.gen_archive.pop_elite(el_obj, el_bcs, old_el_bcs)
                        )

            #               last_archive_size = len(self.gen_archive.as_pandas(include_solutions=False))

            log_archive(self.gen_archive, "Generator", itr, self.start_time, stat_json)

            # FIXME: implement these
            #           self.play_bc_names = ['action_entropy', 'action_entropy_local']

            if PLAY_LEVEL:
                # elite_model_w = self.gen_archive.get_random_elite()[0]
                df = self.gen_archive.as_pandas()
                high_performing = df.sort_values("objective", ascending=False)
                models = np.array(high_performing.loc[:, "solution_0":])
                np.random.shuffle(models)
                playable_levels = []

                for m_i in range(len(models)):
                    elite_model_w = models[m_i]
                    gen_model = set_weights(self.gen_model, elite_model_w)
                    playable_levels += gen_playable_levels(
                        self.env, self.gen_model, self.init_states, self.n_tile_types
                    )

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
                            futures = [
                                multi_play_evo.remote(
                                    self.env,
                                    gen_model,
                                    player_w,
                                    self.n_tile_types,
                                    init_states,
                                    self.play_bc_names,
                                    self.static_targets,
                                    seed,
                                    player_1=self.player_1,
                                    player_2=self.player_2,
                                    playable_levels=playable_levels,
                                )
                                for player_w in play_sols
                            ]
                            results = ray.get(futures)

                            for result in results:
                                m_obj, m_bcs = result
                                objs.append(m_obj)
                                bcs.append([*m_bcs])
                            del results
                            auto_garbage_collect()
                        else:
                            play_i = 0

                            for play_w in play_sols:
                                play_i += 1
                                play_model = set_weights(self.play_model, play_w)
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

                        # TODO: parallelize me
                        df = self.play_archive.as_pandas()
                        high_performing = df.sort_values("objective", ascending=False)
                        elite_models = np.array(high_performing.loc[:, "solution_0":])

                        for elite_i in range(10):
                            play_model_weights = elite_models[elite_i]
                            init_nn = set_weights(self.play_model, play_model_weights)

                            obj, bcs = player_simulate(
                                self.env,
                                self.n_tile_types,
                                self.play_bc_names,
                                init_nn,
                                playable_levels=playable_levels,
                            )

                            self.play_archive.update_elite(obj, bcs)

                        #    m_objs.append(obj)
                        # bc_a = get_bcs(init_nn)
                        # obj = np.mean(m_objs)
                        # objs.append(obj)
                        # bcs.append([bc_a])
                        log_archive(self.play_archive, "Player", p_itr, play_start_time)

                        if net_p_itr > 0 and net_p_itr % SAVE_INTERVAL == 0:
                            # Save checkpoint during player evolution loop
                            self.save()

                        df = self.play_archive.as_pandas()
                        high_performing = df.sort_values("objective", ascending=False)
                        elite_scores = np.array(high_performing.loc[:, "objective"])

                        if np.array(elite_scores).max() >= self.env._prob.max_reward:
                            break

                    # TODO: assuming an archive of one here! Make it more general, like above for generators
                    play_model = set_weights(
                        self.play_model, self.play_archive.get_random_elite()[0]
                    )

            if itr % SAVE_INTERVAL == 0 or itr == 1:
                # Save checkpoint during generator evolution loop
                self.save()

#           if itr % VIS_INTERVAL == 0 or itr == 1:
#               ckp_dir = os.path.join(SAVE_PATH, "checkpoint_{}".format(itr))

#               if not os.path.isdir(ckp_dir):
#                   os.mkdir(ckp_dir)

#               if not CMAES:
#                   # Otherwise the heatmap would just be a single cell
#                   self.visualize(itr=itr)
#               archive_objs = np.array(
#                   self.gen_archive.as_pandas(include_solutions=False).loc[
#                       :, "objective"
#                   ]
#               )
#               save_train_stats(archive_objs, itr=itr)

            self.n_itr += 1

    def save(self):
        global ENV
        ENV = self.env
        self.env = None
        evo_path = os.path.join(SAVE_PATH, "evolver.pkl")

        os.system(
            'mv "{}" "{}"'.format(evo_path, os.path.join(SAVE_PATH, "last_evolver.pkl"))
        )
        pickle.dump(
            self, open(os.path.join(SAVE_PATH, "evolver.pkl"), "wb"), protocol=4
        )
        self.env = ENV

    def init_env(self):
        """Initialize the PCGRL level-generation RL environment and extract any useful info from it."""

        env_name = "{}-{}-v0".format(PROBLEM, REPRESENTATION)
        self.env = gym.make(env_name)
        self.env.adjust_param(render=RENDER)

        if CMAES:
            # Give a little wiggle room from targets, to allow for some diversity

            if "binary" in PROBLEM:
                path_trg = self.env._prob.static_trgs["path-length"]
                self.env._prob.static_trgs.update(
                    {"path-length": (path_trg - 20, path_trg)}
                )
            elif "zelda" in PROBLEM:
                path_trg = self.env._prob.static_trgs["path-length"]
                self.env._prob.static_trgs.update(
                    {"path-length": (path_trg - 40, path_trg)}
                )
            elif "sokoban" in PROBLEM:
                sol_trg = self.env._prob.static_trgs["sol-length"]
                self.env._prob.static_trgs.update(
                    {"sol-length": (sol_trg - 10, sol_trg)}
                )
            elif "smb" in PROBLEM:
                pass
            else:
                raise NotImplemented

        global N_DIRS

        if hasattr(self.env._rep, "_dirs"):
            N_DIRS = len(self.env._rep._dirs)
        else:
            N_DIRS = 0

        global N_STEPS

        #       if N_STEPS is None:
        #       if REPRESENTATION != "cellular":
        max_ca_steps = args.n_steps
        max_changes = self.env._prob._height * self.env._prob._width
        reps_to_steps = {
            "cellular": max_ca_steps,
            "wide": max_changes,
            #           "narrow": max_changes,
            "narrow": max_changes,
            #           "turtle": max_changes * 2,
            "turtle": 2 * max_changes,
            # So that it can move around to each tile I guess
        }
        N_STEPS = reps_to_steps[REPRESENTATION]

    def visualize(self, itr=None):
        archive = self.gen_archive
        # # Visualize Result
        #       grid_archive_heatmap(archive, vmin=self.reward_bounds[self.reward_names[0]][0], vmax=self.reward_bounds[self.reward_names[0]][1])
        #       if PROBLEM == 'binary':
        #           vmin = -20
        #           vmax = 20
        #       elif PROBLEM == 'zelda':
        #           vmin = -20
        #           vmax = 20
        #       grid_archive_heatmap(archive, vmin=vmin, vmax=vmax)
        if ALGO == "ME":
            obj_min, obj_max = archive.fitness_extrema[0]
            qdpy.plots.plotGridSubplots(archive.quality_array[..., 0], os.path.join(SAVE_PATH, 'fitness.pdf'),
                                        plt.get_cmap("inferno_r"), archive.features_domain,
                                        archive.fitness_domain[0], nbTicks=None)
        else:
            plt.figure(figsize=(8, 6))
            df_obj = archive.as_pandas()["objective"]
            obj_min = df_obj.min()
            obj_max = df_obj.max()
            vmin = np.floor(obj_min)
            vmax = np.ceil(obj_max)
            grid_archive_heatmap(archive, vmin=vmin, vmax=vmax)
            if not CMAES:
                plt.xlabel(self.bc_names[0])
                plt.ylabel(self.bc_names[1])
            if itr is not None:
                save_path = os.path.join(SAVE_PATH, "checkpoint_{}".format(itr))
            else:
                save_path = SAVE_PATH
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "fitness.png"))
            #       plt.gca().invert_yaxis()  # Makes more sense if larger BC_1's are on top.


            if SHOW_VIS:
                plt.show()
            plt.close()

        # Print table of results
#       df = archive.as_pandas()
        # high_performing = df[df["objective"] > 200].sort_values("objective", ascending=False)

    #       print(df)

    def infer(self, concat_gifs=True):
        assert INFER
        self.init_env()
        archive = self.gen_archive
        if args.algo == "ME":
            nonempty_idxs = np.stack(np.where(
                np.isnan(archive.quality_array) == False), axis=1)
            # Sort according to 2nd BC
            idxs = nonempty_idxs.tolist()
            idxs.sort(key=lambda x: x[1])
            idxs_T = tuple(np.array(idxs).T)
            objs = archive.quality_array[idxs_T]
            TT()  # TODO: get and sort BCs by objectives
        else:
            df = archive.as_pandas()
            #       high_performing = df[df["behavior_1"] > 50].sort_values("behavior_1", ascending=False)
            

    #       if "binary" in PROBLEM:
    #           #           high_performing = df.sort_values("behavior_1", ascending=False)
    #           high_performing = df.sort_values("objective", ascending=False)

    #       if "zelda" in PROBLEM:
    #           # path lenth
    #           #           high_performing = df.sort_values("behavior_1", ascending=False)
    #           # nearest enemies
    #           #           high_performing = df.sort_values("behavior_0", ascending=False)
    #           high_performing = df.sort_values("objective", ascending=False)
    #       else:
    #           high_performing = df.sort_values("objective", ascending=False)

            # Assume 2nd BC is a measure of complexity
            rows = df.sort_values("behavior_1", ascending=False)
            models = np.array(rows.loc[:, "solution_0":])
            bcs_0 = np.array(rows.loc[:, "behavior_0"])
            bcs_1 = np.array(rows.loc[:, "behavior_1"])
            objs = np.array(rows.loc[:, "objective"])
            # FIXME: don't need these
            idxs = np.array(rows.loc[:, "index_0":"index_1"])
        global N_INIT_STATES
        global N_EVAL_STATES
        global RENDER
        global RANDOM_INIT_LEVELS

        if RENDER_LEVELS:
            RENDER = False
            #           N_INIT_STATES = 1

            if "smb" in PROBLEM:
                d = 4
                figw, figh = 32, 4
            else:
                d = 6  # number of rows and columns
                figw, figh = self.env._prob._width, self.env._prob._height

            if CMAES:
                n_rows = 2
                n_cols = 5
                n_figs = n_rows * d
                fig, axs = plt.subplots(
                    ncols=d,
                    nrows=n_rows,
                    figsize=(figw * n_cols / d, figh * n_rows / d),
                )
                df_g = df.sort_values(by=["objective"], ascending=False)
                grid_models = np.array(df_g.loc[:, "solution_0":])
                level_frames = []

                for (i, model) in enumerate(grid_models):
                    for j in range(n_figs):
                        n_row = j // d
                        n_col = j % d
                        axs[n_row, n_col].set_axis_off()
                        # TODO: select for diversity?
                        # parallelization would be kind of pointelss here
                        init_nn = set_weights(self.gen_model, model)
                        # run simulation, but only on a single level-seed
#                       init_state = gen_random_levels(1, self.env)
                        #                       init_state = np.random.randint(
                        #                           0, self.n_tile_types, size=(1, *self.init_states.shape[1:])
                        #                       )
                        #                       _, _, _, (
                        #                           time_penalty,
                        #                           targets_penalty,
                        #                           variance_penalty,
                        #                           diversity_bonus,
                        #                       ) = simulate(

                        TT()  # don't have a way of rendering CMAES yet??
                        level_frames_i = simulate(
                            self.env,
                            init_nn,
                            self.n_tile_types,
                            self.init_states[0:1],
                            self.bc_names,
                            self.static_targets,
                            seed=None,
                            render_levels=True,
                        )
                        if not concat_gifs:
                            save_level_frames(level_frames_i, i)
                        else:
                            level_frames += level_frames_i
                        # Get image
#                       img = self.env.render(mode="rgb_array")
                        img = level_frames[-1]
                        axs[n_row, n_col].imshow(img, aspect=1)
                if concat_gifs:
                    save_level_frames(level_frames, 'concat')

            else:
                fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figw, figh))
                df_g = df.sort_values(by=["behavior_0", "behavior_1"], ascending=False)

                df_g["row"] = np.floor(
                    np.linspace(0, d, len(df_g), endpoint=False)
                ).astype(int)
                level_frames = []

                for row_num in range(d):
                    row = df_g[df_g["row"] == row_num]
                    row = row.sort_values(by=["behavior_1"], ascending=True)
                    row["col"] = np.arange(0, len(row), dtype=int)
                    idx = np.floor(np.linspace(0, len(row) - 1, d)).astype(int)
                    row = row[row["col"].isin(idx)]
                    row = row.drop(["row", "col"], axis=1)
                    grid_models = np.array(row.loc[:, "solution_0":])

                    for col_num in range(len(row)):
                        model = grid_models[col_num]
#                       axs[row_num, col_num].set_axis_off()
                        axs[-col_num-1, -row_num-1].set_axis_off()

                        # initialize weights
                        gen_model = set_weights(self.gen_model, model)

                        # run simulation, but only on the first level-seed
                        #                       _, _, _, (
                        #                           time_penalty,
                        #                           targets_penalty,
                        #                           variance_penalty,
                        #                           diversity_bonus,
                        #                       ) = simulate(
                        level_frames_i = simulate(
                            self.env,
                            gen_model,
                            self.n_tile_types,
                            self.init_states[0:1],
                            self.bc_names,
                            self.static_targets,
                            seed=None,
                            render_levels=True,
                        )
                        if not concat_gifs:
                            save_level_frames(level_frames_i, '{}_{}'.format(row_num, col_num))
                        else:
                            level_frames += level_frames_i
                        # Get image
                        #                       img = self.env.render(mode="rgb_array")
                        img = level_frames[-1]
#                       axs[row_num, col_num].imshow(img, aspect="auto")
                        axs[-col_num-1, -row_num-1].imshow(img, aspect="auto")
                if concat_gifs:
                    save_level_frames(level_frames, 'concat')
            fig.subplots_adjust(hspace=0.01, wspace=0.01)
            plt.tight_layout()
            fig.savefig(
                os.path.join(SAVE_PATH, "levelGrid_{}-bin.png".format(d)), dpi=300
            )
            plt.close()

        if PLAY_LEVEL:
            player_simulate(
                self.env,
                self.n_tile_types,
                self.play_bc_names,
                self.play_model,
                playable_levels=self.playable_levels,
                seed=None,
            )
        i = 0

        if EVALUATE:
            # First, visualize and aggregate the scores of the elites as they currently stand in the grid

            if not VISUALIZE:
                # visualize if we haven't already
                self.visualize()
            # aggregate scores of individuals currently in the grid
            save_train_stats(objs, archive, self.env, self.bc_names)

            # The level spaces which we will attempt to map to
            problem_eval_bc_names = {
                "binary": [("regions", "path-length")],
                "zelda": [
                    ("nearest-enemy", "path-length"),
                    ("symmetry", "path-length"),
                    ("emptiness", "path-length"),
                ],
                "sokoban": [("crate", "sol-length")],
                "smb": [("emptiness", "jumps")],
            }

            for k in problem_eval_bc_names.keys():
                problem_eval_bc_names[k] += [
                    # ("NONE"),
                    ("emptiness", "symmetry")
                ]

            for (k, v) in problem_eval_bc_names.items():
                if k in PROBLEM:
                    eval_bc_names = v

                    break
            # toss our elites into an archive with different BCs. For fun!

            if not CMAES:
                eval_archives = [
                    GridArchive(
                        # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                        # [min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                        [100 for _ in eval_bcs],
                        # min/max for each BC
                        [self.bc_bounds[bc_name] for bc_name in eval_bcs],
                    )
                    for eval_bcs in eval_bc_names
                ]
                [
                    eval_archive.initialize(solution_dim=len(models[0]))
                    for eval_archive in eval_archives
                ]

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
            fitness_scores = np.full((y_dim, x_dim), np.nan)
            playability_scores = np.full((y_dim, x_dim), np.nan)
            diversity_scores = np.full((y_dim, x_dim), np.nan)
            reliability_scores = np.full((y_dim, x_dim), np.nan)
            eval_fitness_scores = []
            eval_playability_scores = []
            eval_diversity_scores = []
            eval_reliability_scores = []

            if not CMAES:
                for j in range(len(eval_archives)):
                    eval_fitness_scores.append(np.full((y_dim, x_dim), np.nan))
                    eval_playability_scores.append(np.full((y_dim, x_dim), np.nan))
                    eval_reliability_scores.append(np.full((y_dim, x_dim), np.nan))
                    eval_diversity_scores.append(np.full((y_dim, x_dim), np.nan))

            def record_scores(
                id_0,
                id_1,
                batch_reward,
                targets_penalty,
                diversity_bonus,
                variance_penalty,
                fitness_scores,
                playability_scores,
                diversity_scores,
                reliability_scores,
            ):
                fitness_scores[id_0, id_1] = batch_reward
                playability_scores[id_0, id_1] = targets_penalty

                if diversity_bonus is not None:
                    diversity_scores[id_0, id_1] = diversity_bonus

                if variance_penalty is not None:
                    reliability_scores[id_0, id_1] = variance_penalty

            def save_levels(level_json, overwrite=False, headers=False):
                df = pd.DataFrame.from_dict(level_json
                                  )
                #               df = df[df['targets'] == 0]

                if overwrite:
                    write_mode = "w"
                else:
                    write_mode = "a"

                if len(df) > 0:
                    csv_name = "eval_levels"

                    if not RANDOM_INIT_LEVELS:
                        csv_name += "_fixLvls"
                    csv_name += ".csv"
                    if headers:
                        header = df.columns
                    else:
                        header = None
                    df.to_csv(
                        os.path.join(SAVE_PATH, csv_name),
                        mode=write_mode,
                        header=header,
                        index=False,
                    )

            init_states_archive = None

            if RANDOM_INIT_LEVELS:
                # Effectively doing inference on a (presumed) held-out set of levels

                if CMAES:
                    N_EVAL_STATES = N_INIT_STATES = 100
                else:
                    N_EVAL_STATES = N_INIT_STATES = 100  # e.g. 10

                init_states = gen_random_levels(N_INIT_STATES, self.env)
            #               init_states = np.random.randint(
            #                   0,
            #                   self.n_tile_types,
            #                   size=(N_EVAL_STATES, *self.init_states.shape[1:]),
            #               )
            elif args.fix_level_seeds or args.n_init_states == 0:
                # If level seeds were fixed throughout training, use those
                init_states = self.init_states
                N_EVAL_STATES = N_INIT_STATES = init_states.shape[0]
            else:
                init_states_archive = self.gen_archive.init_states_archive
                init_states = None
                # Otherwise, use the init level seeds that were entered into the archive with each elite

            n_train_bcs = len(self.bc_names)

            if THREADS:
                futures = [
                    multi_evo.remote(
                        self.env,
                        self.gen_model,
                        model_w,
                        self.n_tile_types,
                        init_states,
                        self.bc_names
                        + [bc for bc_names in eval_bc_names for bc in bc_names],
                        self.static_targets,
                        seed,
                        player_1=self.player_1,
                        player_2=self.player_2,
                        proc_id=i,
                        init_states_archive=init_states_archive,
                        index=tuple(idxs[i]),
                    )
                    for (i, model_w) in enumerate(models)
                ]
                results = ray.get(futures)
                i = 0

                for result in results:

                    level_json, batch_reward, final_bcs, (
                        time_penalty,
                        batch_targets_penalty,
                        variance_penalty,
                        diversity_bonus,
                    ) = result
                    # id_0 = idxs_0[i]
                    # id_1 = idxs_1[i]
                    id_0, id_1 = archive.get_index(np.array(final_bcs[:n_train_bcs]))

                    if SAVE_LEVELS:
                        save_levels(level_json, overwrite=i == 0, headers=i==0)
                    # Record directly from evolved archive since we are guaranteed to have only one elite per cell
                    record_scores(
                        id_0,
                        id_1,
                        batch_reward,
                        batch_targets_penalty,
                        diversity_bonus,
                        variance_penalty,
                        fitness_scores,
                        playability_scores,
                        diversity_scores,
                        reliability_scores,
                    )

                    if not CMAES:
                        for j, eval_archive in enumerate(eval_archives):
                            # Record componentes of the fitness for each cell in each evaluation archive
                            # NOTE: assume 2 BCs per eval archive
                            eval_bcs = np.array(
                                final_bcs[n_train_bcs + 2 * j : n_train_bcs + 2 * j + 2]
                            )
                            id_0, id_1 = eval_archive.get_index(eval_bcs)
                            # Add dummy solution weights for now
                            status, _ = eval_archive.add(
                                np.zeros(eval_archive.solution_dim),
                                batch_reward,
                                eval_bcs,
                            )

                            if status != AddStatus.NOT_ADDED:
                                # For eval archive, only record new best individuals in each filled cell
                                record_scores(
                                    id_0,
                                    id_1,
                                    batch_reward,
                                    batch_targets_penalty,
                                    diversity_bonus,
                                    variance_penalty,
                                    eval_fitness_scores[j],
                                    eval_playability_scores[j],
                                    eval_diversity_scores[j],
                                    eval_reliability_scores[j],
                                )
                    i += 1
                auto_garbage_collect()

            else:
                # NOTE: Note maintaining this single-threaded code at the moment, can refactor and bring it up to date later

                while i < len(models):
                    # iterate through all models and record stats, on either training seeds or new ones (to test generalization)
                    model = models[i]
                    id_0, id_1 = idxs[i]

                    if init_states is None:
                        init_states_archive = archive.init_states_archive
                    else:
                        init_states_archive = None

                    if init_states is None:
                        init_states = get_init_states(
                            init_states_archive, np.array(idxs[i])
                        )

                    # TODO: Parallelize me
                    gen_model = set_weights(self.gen_model, model)
                    level_json, batch_reward, final_bcs, (
                        time_penalty,
                        targets_penalty,
                        variance_penalty,
                        diversity_bonus,
                    ) = simulate(
                        self.env,
                        gen_model,
                        self.n_tile_types,
                        init_states,
                        self.bc_names,
                        self.static_targets,
                        seed=None,
                        player_1=self.player_1,
                        player_2=self.player_2,
                    )

                    if SAVE_LEVELS:
                        save_levels(level_json)
                    record_scores(
                        id_0,
                        id_1,
                        batch_reward,
                        targets_penalty,
                        diversity_bonus,
                        variance_penalty,
                        fitness_scores,
                        playability_scores,
                        diversity_scores,
                        reliability_scores,
                    )

            def plot_score_heatmap(scores, score_name, bc_names, cmap_str="magma"):
                scores = scores.T
                ax = plt.gca()
                ax.set_xlim(lower_bounds[0], upper_bounds[0])
                ax.set_ylim(lower_bounds[1], upper_bounds[1])
                ax.set_xlabel(bc_names[0])
                ax.set_ylabel(bc_names[1])
                vmin = np.nanmin(scores)
                vmax = np.nanmax(scores)
                t = ax.pcolormesh(
                    x_bounds,
                    y_bounds,
                    scores,
                    cmap=matplotlib.cm.get_cmap(cmap_str),
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.figure.colorbar(t, ax=ax, pad=0.1)

                if SHOW_VIS:
                    plt.show()
                f_name = score_name + "_" + "-".join(bc_names)

                if not RANDOM_INIT_LEVELS:
                    f_name = f_name + "_fixLvls"
                f_name += ".png"
                plt.savefig(os.path.join(SAVE_PATH, f_name))
                plt.tight_layout()
                plt.close()

            assert len(models) == len(archive._occupied_indices)
            qd_score = get_qd_score(archive, self.env, self.bc_names)
            stats = {
                "generations completed": self.n_itr,
                "% train archive full": len(models) / archive.bins,
                "% eval archives full": {},
                "QD score": qd_score,
            }

            if not CMAES:
                plot_score_heatmap(playability_scores, "playability", self.bc_names)
                plot_score_heatmap(diversity_scores, "diversity", self.bc_names)
                plot_score_heatmap(reliability_scores, "reliability", self.bc_names)
                plot_score_heatmap(fitness_scores, "fitness_eval", self.bc_names)

                for j, eval_archive in enumerate(eval_archives):
                    bc_names = eval_bc_names[j]

                    if bc_names != ("NONE"):
                        plot_score_heatmap(
                            eval_playability_scores[j], "playability", bc_names
                        )
                        plot_score_heatmap(
                            eval_diversity_scores[j], "diversity", bc_names
                        )
                        plot_score_heatmap(
                            eval_reliability_scores[j], "reliability", bc_names
                        )
                        plot_score_heatmap(
                            eval_fitness_scores[j], "fitness_eval", bc_names
                        )

                    if bc_names == tuple(self.bc_names):
                        # FIXME: there's a bug somewhere here, include this redundant data to try and pinpoint it
                        pct_archive_full = (
                            len(eval_archive._occupied_indices) / eval_archive.bins
                        )

                        if not RANDOM_INIT_LEVELS:
                            # then this will be the same as the
                            #                           if not len(eval_archive._occupied_indices) / eval_archive.bins == stats["% train archive full"]:
                            #                           continue
                            pass
                        else:
                            stats["% elites maintained"] = (
                                pct_archive_full / stats["% train archive full"]
                            )
                        stats["% fresh train archive full"] = pct_archive_full
                    n_occupied = len(eval_archive.as_pandas(include_solutions=False))
                    assert n_occupied == len(eval_archive._occupied_indices)
                    stats["% eval archives full"].update(
                        {
                            "-".join(bc_names): len(eval_archive._occupied_indices)
                            / eval_archive.bins
                        }
                    )

            stats.update(
                {
                    "playability": get_stats(playability_scores),
                    "diversity": get_stats(diversity_scores),
                    "reliability": get_stats(reliability_scores),
                }
            )
            f_name = "stats"

            if not RANDOM_INIT_LEVELS:
                f_name = f_name + "fixLvls"
            f_name += ".json"
            with open(os.path.join(SAVE_PATH, f_name), "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=4)

            return

        while i < len(models):
            #           model = self.archive.get_random_elite()[0]
            #           model = models[np.random.randint(len(models))]

            model = models[i]
            gen_model = set_weights(self.gen_model, model)

            #           RANDOM_INIT_LEVELS = not opts.fix_level_seeds

            if RANDOM_INIT_LEVELS and args.n_init_states != 0:
                init_states = gen_random_levels(N_INIT_STATES, self.env)
            elif not args.fix_level_seeds and args.n_init_states != 0:
                init_states_archive = archive.init_states_archive
                init_states = get_init_states(init_states_archive, tuple(idxs[i]))
            else:
                init_states = self.init_states
            _, _, _, (
                time_penalty,
                targets_penalty,
                variance_penalty,
                diversity_bonus,
            ) = simulate(
                self.env,
                gen_model,
                self.n_tile_types,
                init_states,
                self.bc_names,
                self.static_targets,
                seed=None,
                player_1=self.player_1,
                player_2=self.player_2,
            )
            #           input("Mean behavior characteristics:\n\t{}: {}\n\t{}: {}\nMean reward:\n\tTotal: {}\n\ttime: {}\n\ttargets: {}\n\tvariance: {}\n\tdiversity: {}\nPress any key for next generator...".format(
            #               self.bc_names[0], bcs_0[i], self.bc_names[1], bcs_1[i], objs[i], time_penalty, targets_penalty, variance_penalty, diversity_bonus))
            i += 1


#           if i == len(models):
#               i=0


def gen_random_levels(n_init_states, env):
    init_states = np.random.randint(
        0, len(env._prob.get_tile_types()), (N_INIT_STATES, env._prob._height, env._prob._width)
    )
    return init_states

#   init_states = np.zeros(shape=(n_init_states, env._prob._height, env._prob._width))
#   init_state_maps = []

#   for i in range(N_INIT_STATES):
#       env._rep.reset(
#           env._prob._width,
#           env._prob._height,
#           get_int_prob(env._prob._prob, env._prob.get_tile_types()),
#       )
#       #                   init_state_maps.append(np.expand_dims(get_one_hot_map(self.env._rep._map, self.n_tile_types), axis=0))
#       init_state_maps.append(np.expand_dims(env._rep._map, axis=0))

#   init_states[:] = np.vstack(init_state_maps)
#   # init_states = np.zeros(
#   #    0, self.n_tile_types, size=self.init_states.shape
#   # )

#   return init_states


if __name__ == "__main__":
    """
    Set Parameters
    """
    seed = 420
    CA_ACTION = True
    args, arg_dict = get_args()
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
    global REEVALUATE_ELITES
    global preprocess_action
    global N_PROC
    global ALGO
    N_PROC = arg_dict["n_cpu"]
    MODEL = arg_dict["model"]
    ALGO = arg_dict["algo"]
    if ALGO == "ME":
        # TODO: implement wrapper around other models generically
        assert MODEL in ["CPPN", "GenCPPN", "CPPNCA"]
    else:
        assert ALGO == "CMAME"
    REPRESENTATION = arg_dict["representation"]
    CASCADE_REWARD = arg_dict["cascade_reward"]
    REEVALUATE_ELITES = not arg_dict["fix_elites"] and arg_dict["n_init_states"] != 0
    RANDOM_INIT_LEVELS = (
        not arg_dict["fix_level_seeds"]
        and arg_dict["n_init_states"] != 0
        or REEVALUATE_ELITES
    )

    if REEVALUATE_ELITES:
        # Otherwise there is no point in re-evaluating them
        assert RANDOM_INIT_LEVELS
    CMAES = arg_dict["behavior_characteristics"] == ["NONE"]
    EVALUATE = arg_dict["evaluate"]
    PLAY_LEVEL = arg_dict["play_level"]
    BCS = arg_dict["behavior_characteristics"]
    N_GENERATIONS = arg_dict["n_generations"]
    N_INIT_STATES = arg_dict["n_init_states"]
    N_STEPS = arg_dict["n_steps"]

    SHOW_VIS = arg_dict["show_vis"]
    PROBLEM = arg_dict["problem"]
    CUDA = False
    VISUALIZE = arg_dict["visualize"]
    INFER = arg_dict["infer"] or EVALUATE
    N_INFER_STEPS = N_STEPS
    #   N_INFER_STEPS = 100
    RENDER_LEVELS = arg_dict["render_levels"]
    THREADS = arg_dict["multi_thread"]  # or EVALUATE
    SAVE_INTERVAL = arg_dict["save_interval"]
    VIS_INTERVAL = 50
    if "CPPN" in MODEL:
        if MODEL != "CPPNCA" and "Gen" not in MODEL:
            assert N_INIT_STATES == 0 and not RANDOM_INIT_LEVELS and not REEVALUATE_ELITES
        if MODEL != "CPPNCA":
            assert N_STEPS == 1

    SAVE_LEVELS = arg_dict["save_levels"] or EVALUATE

    #   exp_name = 'EvoPCGRL_{}-{}_{}_{}-batch_{}-step_{}'.format(PROBLEM, REPRESENTATION, BCS, N_INIT_STATES, N_STEPS, arg_dict['exp_name'])
    #   exp_name = "EvoPCGRL_{}-{}_{}_{}_{}-batch".format(
    #       PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES
    #   )
    exp_name = "EvoPCGRL_"
    if ALGO == "ME":
        exp_name += "ME_"
    exp_name += "{}-{}_{}_{}_{}-batch_{}-pass".format(
        PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES, N_STEPS
    )

    if CASCADE_REWARD:
        exp_name += "_cascRew"

    if not RANDOM_INIT_LEVELS:
        exp_name += "_fixLvls"

    if not REEVALUATE_ELITES:
        exp_name += "_fixElites"

    if args.mega:
        exp_name += "_MEGA"
    exp_name += "_" + arg_dict["exp_name"]
    SAVE_PATH = os.path.join("evo_runs", exp_name)
    if MODEL not in preprocess_action_funcs:
        if "CPPN" in MODEL:
            preprocess_action = preprocess_action_funcs['CPPN'][REPRESENTATION]
        else:
            preprocess_action = preprocess_action_funcs['NCA'][REPRESENTATION]
    else:
        preprocess_action = preprocess_action_funcs[MODEL][REPRESENTATION]
    if MODEL not in preprocess_observation_funcs:
        preprocess_observation = preprocess_observation_funcs['NCA'][REPRESENTATION]
    else:
        preprocess_observation = preprocess_observation_funcs[MODEL][REPRESENTATION]

    def init_tensorboard():
        assert not INFER
        # Create TensorBoard Log Directory if does not exist
        LOG_NAME = "./runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + exp_name
        writer = SummaryWriter(LOG_NAME)

        return writer


    if THREADS:
        ray.init()

    try:
        try:
            evolver = pickle.load(open(os.path.join(SAVE_PATH, "evolver.pkl"), "rb"))
        except:
            evolver = pickle.load(
                open(os.path.join(SAVE_PATH, "last_evolver.pkl"), "rb")
            )
        print("Loaded save file at {}".format(SAVE_PATH))

        if VISUALIZE:
            evolver.visualize()

        if INFER:
            global RENDER
            RENDER = True
            N_STEPS = N_INFER_STEPS

            #           if not RANDOM_INIT_LEVELS:
            # evaluate on initial level seeds that each generator has seen before
            RANDOM_INIT_LEVELS = False
            evolver.infer()
            save_grid(csv_name="eval_levels_fixLvls")
            # evaluate on random initial level seeds
            RANDOM_INIT_LEVELS = True
            evolver.infer()
            save_grid(csv_name="eval_levels")
        #           save_grid(csv_name="levels")

        if not (INFER or VISUALIZE):
            writer = init_tensorboard()
            # then we train
            RENDER = arg_dict["render"]
            evolver.init_env()
            evolver.total_itrs = arg_dict["n_generations"]
            evolver.evolve()
    except FileNotFoundError as e:
        if not INFER:
            RENDER = arg_dict["render"]
            print(
                "Failed loading from an existing save-file. Evolving from scratch. The error was: {}".format(
                    e
                )
            )
            writer = init_tensorboard()
            evolver = EvoPCGRL()
            evolver.evolve()
        else:
            print(
                "Loading from an existing save-file failed. Cannot run inference. The error was: {}".format(
                    e
                )
            )
