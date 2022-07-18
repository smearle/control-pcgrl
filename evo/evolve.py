import gc
import json
import os
import pickle
import time
from functools import reduce
from timeit import default_timer as timer
from pdb import set_trace as TT

from operator import mul
from typing import Tuple

import gym
import matplotlib
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import psutil
import ray
import scipy
from skimage import measure
import torch as th
from tqdm import tqdm
from qdpy.phenotype import Fitness, Features
from ribs.archives import GridArchive
from ribs.emitters import (
    # GradientImprovementEmitter,
    ImprovementEmitter,
    OptimizingEmitter,
)
from ribs.archives._add_status import AddStatus
from ribs.optimizers import Optimizer
from ribs.visualize import grid_archive_heatmap
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import deap
import deap.tools
import deap.algorithms
from qdpy import tools
from deap.base import Toolbox
import copy


from args import get_args, get_exp_dir, get_exp_name
from archives import CMAInitStatesGrid, get_qd_score, MEGrid, MEInitStatesArchive, FlexArchive
from models import Individual, GeneratorNNDense, PlayerNN, set_nograd, get_init_weights, \
    set_weights, Decoder, NCA, NCA3D, GenCPPN2, GenSin2CPPN2, Sin2CPPN, CPPN, DirectEncoding
from utils import get_one_hot_map
from gym_pcgrl.conditional_wrappers import ConditionalWrapper
from gym_pcgrl.envs.helper import get_string_map
from gym_pcgrl.envs.helper_3D import get_string_map as get_string_map_3d
from qdpy import plots as qdpy_plots

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
see setup.sh

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

matplotlib.use('Agg')


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


def save_train_stats(objs, archive, args, itr=None):
    train_time_stats = {
            "QD score": get_qd_score(archive, args),
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
    fontsize = 32
    if "zelda" in PROBLEM:
        d = 3
        fontsize = int(fontsize * d / 4)
    elif "smb" in PROBLEM:
        d = 4
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
    env = ConditionalWrapper(env)
    map_width = env.unwrapped._prob._width
    map_height = env.unwrapped._prob._height
    if ENV3D:
        map_length = env.unwrapped._prob._length

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
    while len(df) < d**2 and targets_thresh > og_df['targets'].min():
        last_len = len(df)
        # Raise the threshold so it includes at least one more individual
        targets_thresh = og_df[og_df['targets'] < targets_thresh]['targets'].max()
        df = og_df[og_df['targets'] >= targets_thresh]
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
            if CONTINUOUS:
                level = np.zeros((3, map_width, map_height), dtype=int)
            if ENV3D:
                level = np.zeros((map_length, map_width, map_height), dtype=int)
            else:
                level = np.zeros((map_height, map_width), dtype=int)

            for i, l_rows in enumerate(grid_models[col_num].split("], [")):
                for j, l_col in enumerate(l_rows.split(",")):
                    level[i, j] = int(
                        l_col.replace("[", "").replace("]", "").replace(" ", "")
                    )

            # Set map
            env.unwrapped._rep.unwrapped._x = env.unwrapped._rep.unwrapped._y = 0
            env.unwrapped._rep.unwrapped._map = level

            # TODO: this won't work for minecraft! Find a workaround?
            img = env.render(mode="rgb_array")

#           axs[row_num, col_num].imshow(img, aspect="auto")
            axs[-col_num-1, -row_num-1].imshow(img, aspect="auto")

    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    levels_png_path = os.path.join(SAVE_PATH, "{}_grid.png".format(csv_name))
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


def cut_border_action_3d(action, **kwargs):
    return action[:, 1:-1, 1:-1, 1:-1], False


preprocess_action_funcs = {
    "NCA": {
        "cellular": id_action,
        "wide": wide_action,
        "narrow": narrow_action,
        "turtle": turtle_action,
    },
    "NCA3D": {
        "cellular3D": id_action,
        "cellular3Dholey": cut_border_action_3d,
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
    # Might be some inconsistencies in ordering of x, y?
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
    "NCA3D": {
        "cellular3D": id_observation,
        "cellular3Dholey": id_observation,
    },
    "CNN": {
        "cellular": id_observation,
        "wide": id_observation,
        "narrow": local_observation,
        "turtle": local_observation,
    },
}


@njit
def get_init_states(init_states_archive, door_coords_archive, index):
    return init_states_archive[index], door_coords_archive[index]



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
        self.logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + \
            ["meanFitness", "maxFitness", "elapsed"]
        self.i = 0


    def tell(self, objective_values, behavior_values):
        """Tell MAP-Elites about the performance (and diversity measures) of new offspring / candidate individuals, 
        after evaluation on the task."""
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

        assert len(self.grid._best_fitness.values) == 1, "Multi-objective evolution is not supported."

        # FIXME: something is wrong here, this is the min, not max.
        # maxFitness = self.grid._best_fitness[0]

        fits = [ind.fitness.values[0] for ind in self.grid]
        maxFitness = np.max(fits)
        meanFitness = np.mean(fits)
        self.logbook.record(iteration=self.i, containerSize=self.grid.size_str(), evals=len(self.inds), 
                            nbUpdated=nb_updated, elapsed=timer()-self.start_time, meanFitness=meanFitness, maxFitness=maxFitness,
                            **record)
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

# class GeneratorNN(ResettableNN):


#class ReluCPPN(ResettableNN):


# Sin2 is siren-type net (i.e. sinusoidal, fixed-topology CPPN), with proper activation as per paper

# CPPN2 takes latent seeds not onehot levels



"""
Behavior Characteristics Functions
"""

def get_blur(float_map, env):
    return measure.blur_effect(float_map)


def get_entropy(int_map, env):
    """
    Function to calculate entropy of levels represented by integers
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns the entropy of the level normalized roughly to a range of 0.0 to 1.0
    """
    if CONTINUOUS:
        a = 0
        b = 15
        return (measure.shannon_entropy(int_map) - a) / (b - a)
    # FIXME: make this robust to different action spaces
    n_classes = len(env.unwrapped._prob._prob)
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
    if not ENV3D:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height  # for example 14*14=196
    else:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height * env.unwrapped._prob.length

    return [
        np.sum(int_map.flatten() == tile) / max_val
        for tile in range(len(env.unwrapped._prob._prob))
    ]


def get_brightness(float_map, env):
    assert np.min(float_map) >= 0.0 and np.max(float_map) <= 1.0
    return np.sum(float_map) / reduce(mul, float_map.shape)

rand_sols = {}


def get_rand_sol(float_map, env, idx=0):
    # TODO: discrete version
    if idx not in rand_sols:
        rand_sols[idx] = np.random.uniform(0, 1, size=float_map.shape)
    return np.sum(np.abs(float_map - rand_sols[idx])) / reduce(mul, float_map.shape)


def get_emptiness(int_map, env):
    """
    Function to calculate how empty the level is
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns an emptiness value normalized to a range of 0.0 to 1.0
    """
    # TODO: double check that the "0th" tile-type actually corresponds to empty tiles
    if not ENV3D:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height  # for example 14*14=196
    else:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height * env.unwrapped._prob._length

    return np.sum(int_map.flatten() == 0) / max_val

#from pymks import PrimitiveTransformer, plot_microstructures, two_point_stats, TwoPointCorrelation


def get_hor_sym(int_map, env):
    """
    Function to get the horizontal symmetry of a level
    int_map (numpy array of ints): representation of level
    env (gym-pcgrl environment instance): used to get the action space dims
    returns a symmetry float value normalized to a range of 0.0 to 1.0
    """
    if not ENV3D:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height / 2  # for example 14*14/2=98
    else:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._length / 2 * env.unwrapped._prob._height  
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
    if not ENV3D:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._height / 2  # for example 14*14/2=98
    else:
        max_val = env.unwrapped._prob._width * env.unwrapped._prob._length / 2 * env.unwrapped._prob._height  
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
    max_val = env.unwrapped._prob._width * env.unwrapped._prob._height * 4
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


# TODO: call this once to return the relevant get_bc function, then call this after each eval, so that we don't have to repeatedly compare strings


def get_bc(bc_name, int_map, stats, env, idx):
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
    elif bc_name == "brightness":
        return get_brightness(int_map, env)  # FIXME: name incorrect, this a float map
    elif bc_name == "entropy":
        return get_entropy(int_map, env)
    elif bc_name == 'blur':
        return get_blur(int_map, env)
    elif bc_name == 'rand_sol':
        return get_rand_sol(int_map, env, idx=idx)
    elif bc_name == "NONE":
        return 0
    # elif bc_name == "two_spatial":
    #     return get_two_spatial(int_map, env)
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


def log_archive(archive, name, itr, start_time, args, level_json=None):
    # TensorBoard Logging.
    if args.algo == "CMAME":
        df = archive.as_pandas(include_solutions=False)
        archive_size = len(df)
        objs = df["objective"]
    else:
        archive_size = len(archive)
        objs = archive.quality_array

    elapsed_time = time.time() - start_time
    writer.add_scalar("{} ArchiveSize".format(name), archive_size, itr)
    writer.add_scalar("{} score/mean".format(name), np.nanmean(objs), itr)
    writer.add_scalar("{} score/max".format(name), np.nanmax(objs), itr)
    writer.add_scalar("{} score/min".format(name), np.nanmin(objs), itr)
    writer.add_scalar(f"{name} QD score", get_qd_score(archive, args), itr)

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

    # Logging to console.
    if ALGO == "ME":
        # This is handled by the qdpy/deap Logbook.
        return

    if itr % 1 == 0:
        print(f"> {itr} itrs completed after {elapsed_time:.2f} s")
        print(f"  - {name} Archive Size: {len(df)}")
        print(f"  - {name} Max Score: {df['objective'].max()}")
        print(f"  - {name} Mean Score: {df['objective'].mean()}")
        print(f"  - {name} Min Score: {df['objective'].min()}")


N_PLAYER_STEPS = 100


def play_level(env, level, player):
    env.unwrapped._rep.unwrapped._old_map = level
    env.unwrapped._rep.unwrapped._random_start = False
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
            player_coords = env.unwrapped._prob.player.coords
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
    target_weights,
    seed,
    player_1,
    player_2,
    proc_id=None,
    init_states_archive=None,
    door_coords_archive=None,
    index=None,
    door_coords=None,
):
    if init_states is None:
        init_states, door_coords = get_init_states(init_states_archive, door_coords_archive, tuple(index))

    # if proc_id is not None:
    #     print("simulating id: {}".format(proc_id))
    model = set_weights(model, model_w, algo=ALGO)
    result = simulate(
        env=env,
        model=model,
        n_tile_types=n_tile_types,
        init_states=init_states,
        bc_names=bc_names,
        static_targets=static_targets,
        target_weights=target_weights,
        seed=seed,
        player_1=player_1,
        player_2=player_2,
        door_coords=door_coords,
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
#           obs = action
            int_map = done or action.argmax(axis=0)
            env.unwrapped._rep.unwrapped._map = int_map
            done = done or (int_map == last_int_map).all() or n_step >= N_STEPS

            #           if INFER and not EVALUATE:
            #               time.sleep(1 / 30)

            if done:
                gen_model.reset()
                env.unwrapped._rep.unwrapped._old_map = int_map
                env.unwrapped._rep.unwrapped._random_start = False
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

def plot_score_heatmap(scores, score_name, bc_names, cmap_str="magma", bcs_in_filename=True,
                       lower_bounds=None, upper_bounds=None,
                       x_bounds=None, y_bounds=None):
    scores = scores.T
    ax = plt.gca()
    ax.set_xlim(lower_bounds[0], upper_bounds[0])
    ax.set_ylim(lower_bounds[1], upper_bounds[1])
    label_fontdict = {
        'fontsize': 16,
    }
    ax.set_xlabel(bc_names[0], fontdict=label_fontdict)
    ax.set_ylabel(bc_names[1], fontdict=label_fontdict)
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
    if bcs_in_filename:
        f_name = score_name + "_" + "-".join(bc_names)
    else:
        f_name = score_name

    if not RANDOM_INIT_LEVELS:
        f_name = f_name + "_fixLvls"
    f_name += ".png"
    plt.title(score_name, fontdict={'fontsize': 24})
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH, f_name))
    plt.close()

def simulate(
        env,
        model,
        n_tile_types,
        init_states,
        bc_names,
        static_targets,
        target_weights,
        seed=None,
        player_1=None,
        player_2=None,
        render_levels=False,
        door_coords=None,
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

    #   env.unwrapped._rep._random_start = False
    #   if n_episode == 0 and False:
    #       env.unwrapped._rep._old_map = init_state
    #       obs = env.reset()
    #       int_map = obs['map']
    n_init_states = init_states.shape[0]
    width = init_states.shape[1]
    height = init_states.shape[2]
    bcs = np.empty(shape=(len(bc_names), n_init_states))
    # if SAVE_LEVELS:
    trg = np.empty(shape=(n_init_states))

    # init_states has shape (n_episodes, n_chan, height, width)
    if not ENV3D:
        final_levels = np.empty(shape=(init_states.shape[0], env.unwrapped._prob._height, env.unwrapped._prob._width), dtype=np.uint8)
    else:
        final_levels = np.empty(shape=(init_states.shape[0], env.unwrapped._prob._height, env.unwrapped._prob._width, env.unwrapped._prob._length), dtype=np.uint8)
    batch_reward = 0
    batch_time_penalty = 0
    batch_targets_penalty = 0
    batch_play_bonus = 0

    if render_levels:
        level_frames = []

    for (n_episode, init_state) in enumerate(init_states):
        # TODO: wrap the env instead
        env.unwrapped._rep.unwrapped._x = env.unwrapped._rep.unwrapped._y = 0
        env.reset() # Initialize the bordered map
        # Decoder and CPPN models will observe continuous latent seeds. #TODO: implement for CPPNs
        if ("Decoder" in MODEL) or ("CPPN" in MODEL):
            obs = init_state
        else:
            # NOTE: Sneaky hack. We don't need initial stats. Never even reset. Heh. Be careful!!
            # Set the representation to begin in the upper left corner
            if IS_HOLEY:
                if ENV3D:
                    entrance_coords, exit_coords = door_coords[n_episode]
                    env.unwrapped._prob._hole_queue = [(entrance_coords, exit_coords)]
                    # env.unwrapped._prob.entrance_coords, env.unwrapped._prob.exit_coords = entrance_coords, exit_coords
                    # env.unwrapped._rep.set_holes(entrance_coords, exit_coords)
                else:
                    raise NotImplementedError
            env.unwrapped._rep.unwrapped._map = init_state.copy()
            env.unwrapped._rep._update_bordered_map()
            env.unwrapped._prob.path_coords = []
            env.unwrapped._prob.path_length = None
            # Only applies to narrow and turtle. Better than using reset, but ugly, and not optimal
            # env.unwrapped._rep._x = np.random.randint(env.unwrapped._prob._width)
            # env.unwrapped._rep._y = np.random.randint(env.unwrapped._prob._height)
            if IS_HOLEY:
                int_map = env.unwrapped._get_rep_map()
            else:
                int_map = init_state
            obs = get_one_hot_map(int_map, n_tile_types)

        if RENDER or RENDER_LEVELS:
            if INFER:
                if ENV3D:
                    stats = env.unwrapped._prob.get_stats(
                        get_string_map_3d(int_map, env.unwrapped._prob.get_tile_types()),
                        # lenient_paths=True,
                    )
                    print(stats)
                else:
                    stats = env.unwrapped._prob.get_stats(
                        get_string_map(int_map, env.unwrapped._prob.get_tile_types(), continuous=CONTINUOUS),
                        # lenient_paths=True,
                    )

        if RENDER:
            env.render()

            if INFER:
                pass
                # time.sleep(10/30)
                #               input()
        done = False

        n_step = 0

        # Simulate an episode of level generation.
        while not done:
            if env.unwrapped._rep.unwrapped._map is not None:
                if render_levels:
                    level_frames.append(env.render(mode="image"))
            #           in_tensor = th.unsqueeze(
            #               th.unsqueeze(th.tensor(np.float32(obs['map'])), 0), 0)
            in_tensor = th.unsqueeze(th.Tensor(obs), 0)
            action, done = model(in_tensor)
            action = action[0].numpy()
            # There is probably a better way to do this, so we are not passing unnecessary kwargs, depending on representation
            if not IS_HOLEY:
                int_map = env.unwrapped._rep.unwrapped._map  # TODO: can use `_get_rep_map()` as below, right?
            else:
                int_map = env.unwrapped._get_rep_map()
            action, skip = preprocess_action(
                action,
                int_map=int_map,
                x=env.unwrapped._rep.unwrapped._x,
                y=env.unwrapped._rep.unwrapped._y,
                n_dirs=N_DIRS,
                n_tiles=n_tile_types,
            )
            if not ENV3D:
                change, [x, y] = env.unwrapped._rep.update(action, continuous=CONTINUOUS)
            else:
                change, [x, y, z] = env.unwrapped._rep.update(action, continuous=CONTINUOUS)
            if not IS_HOLEY:
                int_map = env.unwrapped._rep.unwrapped._map
            else:
                int_map = env.unwrapped._get_rep_map()
            obs = get_one_hot_map(env.unwrapped._rep.get_observation()["map"], n_tile_types)
            preprocess_observation(obs, x=env.unwrapped._rep.unwrapped._x, y=env.unwrapped._rep.unwrapped._y)
            #           int_map = action.argmax(axis=0)
            #           obs = get_one_hot_map(int_map, n_tile_types)
            #           env.unwrapped._rep._map = int_map
            done = done or not (change or skip) or n_step >= N_STEPS - 1
            # done = n_step >= N_STEPS

            #           if INFER and not EVALUATE:
            #               time.sleep(1 / 30)

            if done:
                # we'll need this to compute Hamming diversity
                if ENV3D and IS_HOLEY:
                    final_levels[n_episode] = int_map[1:-1, 1:-1, 1:-1]
                else:
                    final_levels[n_episode] = int_map
                if not ENV3D:
                    stats = env.unwrapped._prob.get_stats(
                        get_string_map(int_map, env.unwrapped._prob.get_tile_types(), continuous=CONTINUOUS),
                        # lenient_paths = True,
                    )
                else:
                    stats = env.unwrapped._prob.get_stats(
                        get_string_map_3d(int_map, env.unwrapped._prob.get_tile_types()),
                        # lenient_paths = True,
                    )
                if render_levels:
                    # get final level state
                    level_frames.append(env.render(mode="image"))
                model.reset()

                # get BCs
                # Resume here. Use new BC function.

                for i in range(len(bc_names)):
                    bc_name = bc_names[i]
                    bcs[i, n_episode] = get_bc(bc_name, int_map, stats, env, idx=i)

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
                        trg_penalty_k = abs(
                            np.arange(static_targets[k][0], static_targets[k][1]) - stats[k]
                        ).min()
                    else:
                        trg_penalty_k = abs(static_targets[k] - stats[k])
                    trg_penalty_k *= target_weights[k]
                    targets_penalty += trg_penalty_k
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

                    max_regret = env.unwrapped._prob.max_reward - env.unwrapped._prob.min_reward
                    # add this in case we get worst possible regret (don't want to punish a playable map)
                    batch_play_bonus += max_regret + p_1_rew - p_2_rew

                    #TODO Add discriminator here

            if RENDER or RENDER_LEVELS:
                if INFER:
                    if ENV3D:
                        stats = env.unwrapped._prob.get_stats(
                            get_string_map_3d(int_map, env.unwrapped._prob.get_tile_types()),
                            # lenient_paths=True,
                        )
                        print(stats)
                    else:
                        stats = env.unwrapped._prob.get_stats(
                            get_string_map(int_map, env.unwrapped._prob.get_tile_types(), continuous=CONTINUOUS),
                            # lenient_paths=True,
                        )
            if RENDER:
                env.render()


            if done and INFER:  # and not (EVALUATE and THREADS):
                if not EVALUATE:
                    #                   time.sleep(5 / 30)
                    print(
                        "stats: {}\n\ntime_penalty: {}\n targets_penalty: {}".format(
                            stats, time_penalty, targets_penalty
                        )
                    )
                if RENDER:
                    pass
                    time.sleep(0.2)
            last_int_map = int_map
            n_step += 1
    final_bcs = [bcs[i].mean() for i in range(bcs.shape[0])]
    batch_targets_penalty = args.targets_penalty_weight * batch_targets_penalty / max(N_INIT_STATES, 1)
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
            # FIXME: Removing this for ad-hoc comparison for now (re: loderunner)
#           batch_reward = batch_reward + max(0, variance_penalty + diversity_bonus)
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
    def __init__(self, args):
        self.init_env()
        args.max_loss = self.env.get_max_loss(ctrl_metrics=args.behavior_characteristics) * args.targets_penalty_weight
        self.args = args
        if not ENV3D:
            assert self.env.observation_space["map"].low[0, 0] == 0
            # get number of tile types from environment's observation space
            # here we assume that all (x, y) locations in the observation space have the same upper/lower bound
            self.n_tile_types = self.env.observation_space["map"].high[0, 0] + 1
        else:
            assert self.env.observation_space["map"].low[0,0,0] == 0
            self.n_tile_types = self.env.observation_space["map"].high[0, 0, 0] + 1
            self.length = self.env.unwrapped._prob._length
        self.width = self.env.unwrapped._prob._width
        self.height = self.env.unwrapped._prob._height

        # FIXME why not?
        # self.width = self.env.unwrapped._prob._width

        # TODO: make reward a command line argument?
        # TODO: multi-objective compatibility?
        self.bc_names = BCS

        # calculate the bounds of our behavioral characteristics
        # NOTE: We assume a square map for some of these (not ideal).
        # regions and path-length are applicable to all PCGRL problems
        self.bc_bounds = self.env.unwrapped._prob.cond_bounds
        self.bc_bounds.update(
            {
                "co-occurance": (0.0, 1.0),
                "symmetry": (0.0, 1.0),
                "symmetry-vertical": (0.0, 1.0),
                "symmetry-horizontal": (0.0, 1.0),
                "emptiness": (0.0, 1.0),
                "entropy": (0.0, 1.0),
                "brightness": (0.0, 1.0),
                "blur": (0.0, 1.0),
                "rand_sol": (0.0, 1.0),
                "two_spatial": (0.0, 1.0),
            }
        )

        self.static_targets = self.env.unwrapped._prob.static_trgs

        init_level_archive_args = {'n_init_states': N_INIT_STATES}
        if REEVALUATE_ELITES or (RANDOM_INIT_LEVELS and args.n_init_states != 0) and (not ENV3D):
            init_level_archive_args.update({
                'map_dims': (self.height, self.width),
                'n_init_states': N_INIT_STATES,
            })
        elif REEVALUATE_ELITES or (RANDOM_INIT_LEVELS and args.n_init_states != 0) and ENV3D:
            init_level_archive_args.update({
                'map_dims': (self.height, self.width, self.length),
                'n_init_states': N_INIT_STATES,
            })
        if "Decoder" in MODEL or "CPPN" in MODEL:
            init_level_archive_args.update({"map_dims": (N_LATENTS, self.height // 4, self.width // 4)})
        self.init_level_archive_args = init_level_archive_args

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
            gen_archive_cls = CMAInitStatesGrid
        #           gen_archive_cls = GridArchive
        else:
            gen_archive_cls = GridArchive
            init_level_archive_args = {}
        self.gen_archive_cls = gen_archive_cls

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
                    [1, 1], [(0, 1), (0, 1)], **init_level_archive_args
                )
            else:

                for bc_name in self.bc_names:
                    if bc_name not in self.bc_bounds:
                        raise Exception(f"Behavior characteristic / measure `{bc_name}` not found in self.bc_bounds."
                        "You probably need to specify the lower/upper bounds of this measure in prob.cond_bounds.")

                self.gen_archive = gen_archive_cls(
                    # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                    # [min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                    [100 for _ in self.bc_names],
#                   [1 for _ in self.bc_names],
                    # min/max for each BC
                    [self.bc_bounds[bc_name] for bc_name in self.bc_names],
                    **init_level_archive_args,
                )

        # TODO: different initial weights per emitter as in pyribs lunar lander relanded example?
        self._init_model()

        init_step_size = args.step_size
#       if MODEL == "NCA":
#           init_step_size = args.step_size
#       elif MODEL == "CNN":
#           init_step_size = args.step_size
#       else:
#           init_step_size = args.step_size

        if CMAES:
            # The optimizing emitter will prioritize fitness over exploration of behavior space
            emitter_type = OptimizingEmitter
        else:
            emitter_type = ImprovementEmitter

        if ALGO == "ME":
            batch_size = 150
            self.n_generator_weights = None

        # elif args.mega:
        #     gen_emitters = [
        #         GradientImprovementEmitter(
        #             self.gen_archive,
        #             initial_w.flatten(),
        #             # TODO: play with initial step size?
        #             sigma_g=10.0,
        #             stepsize=0.002,  # Initial step size.
        #             gradient_optimizer="adam",
        #             selection_rule="mu",
        #             batch_size=batch_size,
        #         )
        #         for _ in range(n_emitters)  # Create 5 separate emitters.
        #     ]

        # Otherwise, we're using CMAME. 
        else:
            n_emitters = 5
            batch_size = 30
            # Get the initial (continuous) weights so that we can feed them to CMAME for covariance matrix 
            # adaptation.
            initial_w = get_init_weights(self.gen_model)
            assert len(initial_w.shape) == 1
            self.n_generator_weights = initial_w.shape[0]
            self.n_player_weights = 0
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
                    # NOTE: Big step size, no good otherwise
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
                   'step_size': args.step_size,
            }
            if MODEL == "DirectEncoding":
                ind_cls_args.update({'map_width': self.env.unwrapped._prob._width,
                    'map_dims': self.env.get_map_dims()[:-1]})

            self.gen_optimizer = MEOptimizer(grid=self.gen_archive,
                                             ind_cls=Individual,
                                             batch_size=batch_size,
                                             ind_cls_args=ind_cls_args,
                                             )
        else:
            self.gen_optimizer = Optimizer(self.gen_archive, gen_emitters)

        # These are the initial maps which will act as seeds to our NCA models

        self.init_states = gen_latent_seeds(N_INIT_STATES, self.env)
        if IS_HOLEY:
            self.door_coords = gen_door_coords(N_INIT_STATES, self.env)
        else:
            self.door_coords = None

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

    def _init_model(self):
        global N_DIRS

        if hasattr(self.env.unwrapped._rep.unwrapped, "_dirs"):
        # if hasattr(self.env.unwrapped._rep, "_dirs"):
            N_DIRS = len(self.env.unwrapped._rep.unwrapped._dirs)
        else:
            N_DIRS = 0
        reps_to_out_chans = {
            "cellular": self.n_tile_types,
            "cellular3D": self.n_tile_types,
            "cellular3Dholey": self.n_tile_types,
            "wide": self.n_tile_types,
            "narrow": self.n_tile_types + 1,
            "turtle": self.n_tile_types + N_DIRS,
        }

        reps_to_in_chans = {
            "cellular": self.n_tile_types,
            "cellular3D": self.n_tile_types,
            "cellular3Dholey": self.n_tile_types,
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
#       elif MODEL == "NCA":
#           self.gen_model = globals()["GeneratorNN"](
#               n_in_chans=self.n_tile_types, n_actions=n_out_chans
#           )
#       else:
        n_observed_tiles = 0 if "Decoder" in MODEL or "CPPN" in MODEL else self.n_tile_types
        self.gen_model = globals()[MODEL](
            n_in_chans=n_observed_tiles + N_LATENTS, n_actions=n_out_chans, map_width=self.env.unwrapped._prob._width,
            map_dims=self.env.get_map_dims()[:-1],
            render=RENDER, n_aux_chan=args.n_aux_chan)
        # TODO: toggle CUDA/GPU use with command line argument.
        if CUDA:
            self.gen_model.cuda()
        set_nograd(self.gen_model)

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
            # targets = "validity", variance = "reliability"
            stats = ["batch_reward", "variance", "diversity", "targets"]
            stat_json = {
                "batch_reward": [],
                "variance": [],
                "diversity": [],
                "targets": [],
            }

            if RANDOM_INIT_LEVELS and args.n_init_states != 0:
                init_states = gen_latent_seeds(N_INIT_STATES, self.env)
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
                            self.env.unwrapped._reward_weights,
                            seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                            door_coords=self.door_coords,
                        )
                        for model_w in gen_sols[n_launch * n_proc: (n_launch+1) * n_proc]
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
                    gen_model = set_weights(self.gen_model, model_w, algo=ALGO)
                    level_json, m_obj, m_bcs = simulate(
                        env=self.env,
                        model=gen_model,
                        n_tile_types=self.n_tile_types,
                        init_states=init_states,
                        bc_names=self.bc_names,
                        static_targets=self.static_targets,
                        target_weights=self.env.unwrapped._reward_weights,
                        seed=seed,
                        player_1=self.player_1,
                        player_2=self.player_2,
                        door_coords=self.door_coords,
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
                self.gen_archive.set_init_states(init_states, door_coords=self.door_coords)
            # Send the results back to the optimizer.
            if args.mega:
                # TODO: Here we need the jacobian
                jacobian = None
                self.gen_optimizer.tell(objs, bcs, jacobian=jacobian)
            else:
                self.gen_optimizer.tell(objs, bcs)
#               for emitter in self.gen_optimizer.emitters:
#

            # Re-evaluate elite generators. If doing CMAES,re-evaluate every iteration. Otherwise, try to let the archive grow.

            if REEVALUATE_ELITES and (CMAES or self.n_itr % 1 == 0):
                df = self.gen_archive.as_pandas()
                #               curr_archive_size = len(df)
                high_performing = df.sample(frac=1)
                elite_models = np.array(high_performing.loc[:, "solution_0":])
                if 'behavior_1' in high_performing.columns:
                    elite_bcs = np.array(high_performing.loc[:, "behavior_0":"behavior_1"])
                else:
                    elite_bcs = np.array(high_performing.loc[:, "behavior_0"])
                #if there is not behavior_1
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
                            self.env.unwrapped._reward_weights,
                            seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                            door_coords=self.door_coords,
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
                        if not isinstance(old_el_bcs,np.ndarray):
                            old_el_bcs = np.array([old_el_bcs])
                        #TODO fix here
                        gen_model_weights = elite_models[elite_i]
                        gen_model = set_weights(self.gen_model, gen_model_weights, algo=ALGO)

                        level_json, el_obj, el_bcs = simulate(
                            env=self.env,
                            model=gen_model,
                            n_tile_types=self.n_tile_types,
                            init_states=init_states,
                            bc_names=self.bc_names,
                            static_targets=self.static_targets,
                            target_weights=self.env.unwrapped._reward_weights,
                            seed=seed,
                            player_1=self.player_1,
                            player_2=self.player_2,
                            door_coords=self.door_coords,
                        )
                        idx = self.gen_archive.get_index(old_el_bcs)
                        [stat_json[stat].extend(level_json[stat]) for stat in stats]
                        self.gen_archive.update_elite(
                            *self.gen_archive.pop_elite(el_obj, el_bcs, old_el_bcs)
                        )

            #               last_archive_size = len(self.gen_archive.as_pandas(include_solutions=False))

            log_archive(self.gen_archive, "Generator", itr, self.start_time, args=self.args, level_json=stat_json)

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
                    gen_model = set_weights(self.gen_model, elite_model_w, algo=ALGO)
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
                                play_model = set_weights(self.play_model, play_w, algo=ALGO)
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
                            init_nn = set_weights(self.play_model, play_model_weights, algo=ALGO)

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
                        log_archive(self.play_archive, "Player", p_itr, play_start_time, args=args)

                        if net_p_itr > 0 and net_p_itr % SAVE_INTERVAL == 0:
                            # Save checkpoint during player evo loop
                            self.save()

                        df = self.play_archive.as_pandas()
                        high_performing = df.sort_values("objective", ascending=False)
                        elite_scores = np.array(high_performing.loc[:, "objective"])

                        if np.array(elite_scores).max() >= self.env.unwrapped._prob.max_reward:
                            break

                    # TODO: assuming an archive of one here! Make it more general, like above for generators
                    play_model = set_weights(
                        self.play_model, self.play_archive.get_random_elite()[0], algo=ALGO
                    )

            if itr % SAVE_INTERVAL == 0 or itr == 1:
                # Save checkpoint during generator evo loop
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
        self.env = ConditionalWrapper(self.env)
        self.env.adjust_param(render=RENDER, change_percentage=None, model=None, max_board_scans=1, static_prob=0.0)
        self.env.unwrapped._get_stats_on_step = False

#       if CMAES:
#           # Give a little wiggle room from targets, to allow for some diversity (or not)
#           if "binary" in PROBLEM:
#               path_trg = self.env._prob.static_trgs["path-length"]
#               self.env._prob.static_trgs.update(
#                   {"path-length": (path_trg - 20, path_trg)}
#               )
#           elif "zelda" in PROBLEM:
#               path_trg = self.env._prob.static_trgs["path-length"]
#               self.env._prob.static_trgs.update(
#                   {"path-length": (path_trg - 40, path_trg)}
#               )
#           elif "sokoban" in PROBLEM:
#               sol_trg = self.env._prob.static_trgs["sol-length"]
#               self.env._prob.static_trgs.update(
#                   {"sol-length": (sol_trg - 10, sol_trg)}
#               )
#           elif "smb" in PROBLEM:
#               pass
#           elif "microstructure" in PROBLEM:
#               pass
#           else:
#               raise NotImplementedError
        global N_STEPS
        global CONTINUOUS
        CONTINUOUS = PROBLEM == 'face_ctrl'

        #       if N_STEPS is None:
        #       if REPRESENTATION != "cellular":
        max_ca_steps = args.n_steps
        
        max_changes = self.env.unwrapped._prob._height * self.env.unwrapped._prob._width
        if ENV3D:
            max_changes *= self.env.unwrapped._prob._length

        reps_to_steps = {
            "cellular": max_ca_steps,
            "cellular3D": max_ca_steps,
            "cellular3Dholey": max_ca_steps,

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
            qdpy_plots.plotGridSubplots(archive.quality_array[..., 0], os.path.join(SAVE_PATH, 'fitness.pdf'),
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
            label_fontdict = {
                'fontsize': 16,
            }
            if not CMAES:
                plt.xlabel(self.bc_names[0], fontdict=label_fontdict)
                plt.ylabel(self.bc_names[1], fontdict=label_fontdict)
            if itr is not None:
                save_path = os.path.join(SAVE_PATH, "checkpoint_{}".format(itr))
            else:
                save_path = SAVE_PATH
            plt.title('fitness', fontdict={'fontsize': 24})
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
        self.door_coords = None if not hasattr(self, 'door_coords') else self.door_coords  # HACK for backward compatibility
        args = self.args
        self.init_env()
        archive = self.gen_archive
        if args.algo == "ME":
            nonempty_idxs = np.stack(np.where(
                np.isnan(archive.quality_array) == False), axis=1)
            # Assume 2nd BC is a measure of complexity
            # Sort according to 2nd BC
            idxs = nonempty_idxs.tolist()
            idxs.sort(key=lambda x: x[1])
            idxs_T = tuple(np.array(idxs).T)
            objs = archive.quality_array[idxs_T]
            # Get list of individuals in same order. First get list of features belonging to individuals in bin,
            # then get individual by bin-coordinate
            bcs = [archive.features[tuple(idx[:-1])][idx[-1]].values for idx in idxs]
            models = [archive.solutions[tuple(idx[:-1])][idx[-1]] for idx in idxs]
            # Get rid of bin coordinate for our purposes
            # TODO: for more flexibility, instead adapt the below to get this bin coordinate
            idxs = [idx[:-1] for idx in idxs]
        else:
            df = archive.as_pandas()
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
            elif "zelda" in PROBLEM:
                d = 3
                figw, figh = self.env.unwrapped._prob._width, self.env.unwrapped._prob._height
            else:
                d = 6  # number of rows and columns
                figw, figh = self.env.unwrapped._prob._width, self.env.unwrapped._prob._height

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
                        init_nn = set_weights(self.gen_model, model, algo=ALGO)
                        # run simulation, but only on a single level-seed
                        # init_state = (1, self.env)
                        #                       init_state = np.random.randint(
                        #                           0, self.n_tile_types, size=(1, *self.init_states.shape[1:])
                        #                       )
                        #                       _, _, _, (
                        #                           time_penalty,
                        #                           targets_penalty,
                        #                           variance_penalty,
                        #                           diversity_bonus,
                        #                       ) = simulate(

                        raise NotImplementedError
                        raise Exception
                        # don't have a way of rendering CMAES yet??
                        level_frames_i = simulate(
                            self.env,
                            init_nn,
                            self.n_tile_types,
                            self.init_states[0:1],
                            self.bc_names,
                            self.static_targets,
                            target_weights=self.env.unwrapped._reward_weights,
                            seed=None,
                            render_levels=True,
                        )
                        if not concat_gifs:
                            save_level_frames(level_frames_i, i)
                        else:
                            level_frames += level_frames_i
                        # Get image
                        # img = self.env.render(mode="rgb_array")
                        img = level_frames[-1]
                        axs[n_row, n_col].imshow(img, aspect=1)
                if concat_gifs:
                    save_level_frames(level_frames, 'concat')

            else:
                fig, axs = plt.subplots(ncols=d, nrows=d, figsize=(figw, figh))
                if ALGO == "ME":
                    pass
                else:
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
                        gen_model = set_weights(self.gen_model, model, algo=ALGO)

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
                            target_weights=self.env.unwrapped._reward_weights,
                            seed=None,
                            render_levels=True,
                            door_coords=self.door_coords,
                        )
                        if not concat_gifs:
                            save_level_frames(level_frames_i, '{}_{}'.format(row_num, col_num))
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
            save_train_stats(objs=objs, archive=archive, args=args)

            # Basically deprecated, not really fuckin' with this.
            # Toss our elites into an archive with different BCs. For fun!
            # The level spaces which we will attempt to map to
            problem_eval_bc_names = {
                "binary": [
#                   ("regions", "path-length")
                      ],
                "zelda": [
#                   ("nearest-enemy", "path-length"),
#                   ("symmetry", "path-length"),
#                   ("emptiness", "path-length"),
                ],
                "sokoban": [
#                   ("crate", "sol-length")
                ],
                "smb": [
#                   ("emptiness", "jumps")
                ],
                "loderunner": [
#                   ("emptiness", "path-length"),
#                   ("symmetry", "path-length"),
                ],
                "face": [
                    ("brightness", "entropy"),
                ],
                "microstructure": []
            }

#           for k in problem_eval_bc_names.keys():
#               problem_eval_bc_names[k] += [
#                   # ("NONE"),
#                   ("emptiness", "symmetry")
#               ]

            eval_bc_names = []
            for (k, v) in problem_eval_bc_names.items():
                if k in PROBLEM:
                    eval_bc_names = v
                    break

            eval_bc_names = list(set([tuple(self.bc_names)] + eval_bc_names))

            if not CMAES:
                if ALGO == "ME":
                    eval_archives = [
                        MEGrid(
                            [N_BINS for _ in eval_bcs],
                            [self.bc_bounds[bc_name] for bc_name in eval_bcs],
                        )
                        for eval_bcs in eval_bc_names
                    ]
                else:
                    eval_archives = [
                        GridArchive(
                            # minimum of 100 for each behavioral characteristic, or as many different values as the BC can take on, if it is less
                            # [min(100, int(np.ceil(self.bc_bounds[bc_name][1] - self.bc_bounds[bc_name][0]))) for bc_name in self.bc_names],
                            [N_BINS for _ in eval_bcs],
                            # min/max for each BC
                            [self.bc_bounds[bc_name] for bc_name in eval_bcs],
                        )
                        for eval_bcs in eval_bc_names
                    ]
                    [
                        eval_archive.initialize(solution_dim=len(models[0]))
                        for eval_archive in eval_archives
                    ]
            else:
                eval_archive = gen_archive_cls(
                    [1, 1], [(0, 1), (0, 1)], **self.init_level_archive_args
                )

            RENDER = False
            # Iterate through our archive of trained elites, evaluating them and storing stats about them.
            # Borrowing logic from grid_archive_heatmap from pyribs.

            # Retrieve data from archive
            if ALGO == 'ME':
                lower_bounds = [archive.features_domain[i][0] for i in range(len(archive.features_domain))]
                upper_bounds = [archive.features_domain[i][1] for i in range(len(archive.features_domain))]
                x_dim, y_dim = archive.shape
            else:
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
            door_coords_archive = None

            if RANDOM_INIT_LEVELS:
                # Effectively doing inference on a (presumed) held-out set of levels

                if CMAES:
                    N_EVAL_STATES = N_INIT_STATES = 100
                else:
                    N_EVAL_STATES = N_INIT_STATES = 20  #= 100  # e.g. 10

                init_states = gen_latent_seeds(N_INIT_STATES, self.env)
            #               init_states = np.random.randint(
            #                   0,
            #                   self.n_tile_types,
            #                   size=(N_EVAL_STATES, *self.init_states.shape[1:]),
            #               )
            elif args.fix_level_seeds or args.n_init_states == 0:
                # If level seeds were fixed throughout training, use those
                init_states = self.init_states
                door_coords = self.door_coords
                N_EVAL_STATES = N_INIT_STATES = init_states.shape[0]
            else:
                init_states_archive = self.gen_archive.init_states_archive
                door_coords_archive = self.gen_archive.door_coords_archive
                init_states = None
                door_coords = None
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
                        [bc for bc_names in eval_bc_names for bc in bc_names],
                        self.static_targets,
                        self.env.unwrapped._reward_weights,
                        seed,
                        player_1=self.player_1,
                        player_2=self.player_2,
                        proc_id=i,
                        init_states_archive=init_states_archive,
                        door_coords_archive=door_coords_archive,
                        index=tuple(idxs[i]),
                        door_coords=self.door_coords,
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
                    grid_bcs = final_bcs[:n_train_bcs]
                    # TODO: remove this (it's for backward compatibility) since we've implemented get_index for qdpy
                    #   grid
                    if ALGO == "ME":
                        # Clip features to within the feature domain (shouldn't be outside of this domain in theory 
                        # though).
                        grid_bcs = [np.clip(bc, *archive.features_domain[i]) for i, bc in enumerate(grid_bcs)]
                        id_0, id_1 = archive.index_grid(tuple(grid_bcs))
                    else:
                        id_0, id_1 = archive.get_index(np.array(grid_bcs))

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
#                               final_bcs[n_train_bcs + 2 * j : n_train_bcs + 2 * j + 2]
                                final_bcs[2 * j: 2 * (j + 1)]
                            )
                            if ALGO == "ME":
                                eval_bcs = [np.clip(bc, *archive.features_domain[i]) for i, bc in enumerate(eval_bcs)]
                                id_0, id_1 = archive.index_grid(tuple(eval_bcs))
                                # Dummy individual
                                individual = Individual(type(self.gen_model), self.n_tile_types, self.n_tile_types,
                                    map_dims=self.env.unwrapped.get_map_dims()[:-1])
                                individual.fitness = Fitness([batch_reward])
                                individual.features = Features(final_bcs)
                                idx = eval_archive.add(individual)
                                ind_added = idx is not None
                            else:
                                id_0, id_1 = eval_archive.get_index(eval_bcs)
                                # Add dummy solution weights for now
                                status, _ = eval_archive.add(
                                    np.zeros(eval_archive.solution_dim),
                                    batch_reward,
                                    eval_bcs,
                                )
                                ind_added = status != AddStatus.NOT_ADDED

                            if ind_added:
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
                # NOTE: Not maintaining this single-threaded code at the moment, can refactor and bring it up to date later

                while i < len(models):
                    # iterate through all models and record stats, on either training seeds or new ones (to test evaluation)
                    model = models[i]
                    id_0, id_1 = idxs[i]

                    if init_states is None:
                        init_states_archive = archive.init_states_archive
                    else:
                        init_states_archive = None

                    if init_states is None:
                        init_states, door_coords = get_init_states(
                            init_states_archive, door_coords_archive, tuple(idxs[i])
                        )

                    gen_model = set_weights(self.gen_model, model, algo=ALGO)
                    level_json, batch_reward, final_bcs, (
                        time_penalty,
                        targets_penalty,
                        variance_penalty,
                        diversity_bonus,
                    ) = simulate(
                        env=self.env,
                        model=gen_model,
                        n_tile_types=self.n_tile_types,
                        init_states=init_states,
                        bc_names=self.bc_names,
                        static_targets=self.static_targets,
                        target_weights=self.env.unwrapped._reward_weights,
                        seed=None,
                        player_1=self.player_1,
                        player_2=self.player_2,
                        door_coords=door_coords,
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

            if ALGO == "ME":
                n_filled_bins = eval_archive.filled_bins
                assert len(models) == archive.filled_bins
                n_total_bins = archive.size
            else:
                n_filled_bins = len(eval_archive._occupied_indices)
                assert len(models) == len(archive._occupied_indices)
                n_total_bins = archive.bins
            qd_score = get_qd_score(archive, self.args)
            eval_qd_score = get_qd_score(eval_archive, self.args)
            stats = {
                "generations completed": self.n_itr,
                "% train archive full": len(models) / n_total_bins,
                "archive size": n_filled_bins,
                "QD score": qd_score,
                "eval QD score": eval_qd_score,
                "% eval archives full": {},
                "eval archive sizes": {},
                "eval QD scores": {},
            }

            if not CMAES:
                plot_args = {
                    'lower_bounds': lower_bounds,
                    'upper_bounds': upper_bounds,
                    'x_bounds': x_bounds,
                    'y_bounds': y_bounds,
                }
                plot_score_heatmap(playability_scores, "playability", self.bc_names, **plot_args,
                                   bcs_in_filename=False)
                plot_score_heatmap(diversity_scores / 10, "diversity", self.bc_names, **plot_args, bcs_in_filename=False)
                plot_score_heatmap(reliability_scores, "reliability", self.bc_names, **plot_args, bcs_in_filename=False)
                plot_score_heatmap(fitness_scores, "fitness_eval", self.bc_names, **plot_args, bcs_in_filename=False)

                for j, eval_archive in enumerate(eval_archives):
                    bc_names = eval_bc_names[j]

                    if bc_names != ("NONE") and bc_names != tuple(self.bc_names):
                        plot_score_heatmap(
                            eval_playability_scores[j], "playability", bc_names, **plot_args,
                        )
                        plot_score_heatmap(
                            eval_diversity_scores[j] / 10, "diversity", bc_names, **plot_args,
                        )
                        plot_score_heatmap(
                            eval_reliability_scores[j], "reliability", bc_names, **plot_args,
                        )
                        plot_score_heatmap(
                            eval_fitness_scores[j], "fitness_eval", bc_names, **plot_args,
                        )

                    if bc_names == tuple(self.bc_names):
                        # in case a bug appears here, where performance differs from training to inference,
                        # include this redundant data to try and pinpoint it. Note that this is only redundant in
                        # stats_fixLvls, though, because otherwise, we are doing evaluation in the same BC space.
                        pct_archive_full = (
                            n_filled_bins / n_total_bins
                        )

                        if not RANDOM_INIT_LEVELS:
                            # then this will be the same as the
                            #                           if not len(eval_archive._occupied_indices) / eval_archive.bins == stats["% train archive full"]:
                            #                           continue
                            pass
                        else:
                            pass
                        stats["% elites maintained"] = (
                            pct_archive_full / stats["% train archive full"]
                        )
                        stats["% QD score maintained"] = stats["eval QD score"] / stats["QD score"]

                        stats["% fresh train archive full"] = pct_archive_full
                        stats["% fresh train archive full"] = pct_archive_full
                    n_occupied = n_filled_bins
#                   assert n_occupied == len(eval_archive._occupied_indices)
                    bcs_key = "-".join(bc_names)
                    stats["% eval archives full"].update(
                        {
                            bcs_key: n_occupied / n_total_bins,
                    })
                    stats["eval archive sizes"].update({
                        bcs_key: n_occupied,
                    })
                    stats["eval QD scores"].update({
                        bcs_key: get_qd_score(eval_archive, self.args),
                    })

            stats.update(
                {
                    "playability": get_stats(playability_scores),
                    "diversity": get_stats(diversity_scores / 10),
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

        # This is the inference code, which will play back models for our own enjoyment/observation.
        while i < len(models):
            #           model = self.archive.get_random_elite()[0]
            #           model = models[np.random.randint(len(models))]

            model = models[i]
            gen_model = set_weights(self.gen_model, model, algo=ALGO)

            #           RANDOM_INIT_LEVELS = not opts.fix_level_seeds

            if RANDOM_INIT_LEVELS and args.n_init_states != 0:
                init_states = gen_latent_seeds(N_INIT_STATES, self.env)
            elif not args.fix_level_seeds and args.n_init_states != 0:
                init_states_archive = archive.init_states_archive
                door_coords_archive = archive.door_coords_archive
                init_states, door_coords = get_init_states(init_states_archive, door_coords_archive, tuple(idxs[i]))
            else:
                init_states = self.init_states
                door_coords = self.door_coords
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
                target_weights=self.env.unwrapped._reward_weights,
                seed=None,
                player_1=self.player_1,
                player_2=self.player_2,
                door_coords=door_coords,
            )
            #           input("Mean behavior characteristics:\n\t{}: {}\n\t{}: {}\nMean reward:\n\tTotal: {}\n\ttime: {}\n\ttargets: {}\n\tvariance: {}\n\tdiversity: {}\nPress any key for next generator...".format(
            #               self.bc_names[0], bcs_0[i], self.bc_names[1], bcs_1[i], objs[i], time_penalty, targets_penalty, variance_penalty, diversity_bonus))
            i += 1


#           if i == len(models):
#               i=0


def gen_door_coords(n_init_states, env):
    all_holes = env.unwrapped._prob.gen_all_holes()
    door_idxs = np.random.choice(len(all_holes), max(1, n_init_states), replace=False)
    return np.array(all_holes)[door_idxs]


def gen_latent_seeds(n_init_states, env):
    width = env.unwrapped._prob._width
    height = env.unwrapped._prob._height
    if n_init_states == 0:
        # special square patch with all 1s in a box in the middle
        sw = width // 3
        sh = height // 3
        if not ENV3D:
            if CONTINUOUS:
                init_states = np.zeros(shape=(1, 3, height, width))
                init_states[0, :, height//2-sh//2:height//2+sh//2, width//2-sw//2: width//2+sw//2] = 1
            else:
                init_states = np.zeros(shape=(1, height, width))
                init_states[0, height//2-sh//2:height//2+sh//2, width//2-sw//2: width//2+sw//2] = 1
        else:
            length = env.unwrapped._prob._length
            init_states = np.zeros(shape=(1, height, width, length))
            init_states[0, height//2-sh//2:height//2+sh//2, width//2-sw//2: width//2+sw//2,
                                length//2-sw//2: length//2+sw//2] = 1
        return init_states
    if ENV3D:
        im_dims = (env.unwrapped._prob._height, env.unwrapped._prob._width, env.unwrapped._prob._length)
    else:
        im_dims = (env.unwrapped._prob._height, env.unwrapped._prob._width)
    if env.unwrapped._prob.is_continuous():  # AD HOC continous representation
        init_states = np.random.uniform(0, 1, size=(N_INIT_STATES, 3, *im_dims))
    elif "CPPN" in MODEL or "Decoder" in MODEL:
        init_states = np.random.normal(0, 1, (N_INIT_STATES, N_LATENTS))
        if "CPPN" in MODEL:
            init_states = np.tile(init_states[:, :, None, None], (1, 1, *im_dims))
        if "Decoder" in MODEL:
            assert env.unwrapped._prob._width % 4 == env.unwrapped._prob._height % 4 == 0
            init_states = np.tile(init_states[:, :, None, None], (1, 1, *tuple(np.array(im_dims) // 4)))
    else:
        init_states = np.random.randint(
            0, len(env.unwrapped._prob.get_tile_types()), (N_INIT_STATES, *im_dims)
        )
    return init_states

#   init_states = np.zeros(shape=(n_init_states, env.unwrapped._prob._height, env.unwrapped._prob._width))
#   init_state_maps = []

#   for i in range(N_INIT_STATES):
#       env.unwrapped._rep.reset(
#           env.unwrapped._prob._width,
#           env.unwrapped._prob._height,
#           get_int_prob(env.unwrapped._prob._prob, env.unwrapped._prob.get_tile_types()),
#       )
#       #                   init_state_maps.append(np.expand_dims(get_one_hot_map(self.env.unwrapped._rep._map, self.n_tile_types), axis=0))
#       init_state_maps.append(np.expand_dims(env.unwrapped._rep._map, axis=0))

#   init_states[:] = np.vstack(init_state_maps)
#   # init_states = np.zeros(
#   #    0, self.n_tile_types, size=self.init_states.shape
#   # )

#   return init_states


if __name__ == "__main__":
    """
    Set Parameters
    """
    N_BINS = 100
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
    global seed

    CONCAT_GIFS = False
    if arg_dict["exp_name"] == '5':
        seed = 420
    else:
        try:
            seed = int(arg_dict["exp_name"])
        except Exception:
            print("Assigning random seed")
            seed = np.random.randint(10000)
    print("Random number seed is: {}".format(seed))
    N_PROC = arg_dict["n_cpu"]
    MODEL = arg_dict["model"]
    ALGO = arg_dict["algo"]
    if ALGO == "ME":
        # TODO: implement wrapper around other models generically
        pass
        # assert MODEL in ["CPPN", "GenCPPN", "CPPNCA", "DirectBinaryEncoding"]
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
    CMAES = arg_dict["behavior_characteristics"] == ["NONE", "NONE"]
    EVALUATE = arg_dict["evaluate"]
    PLAY_LEVEL = arg_dict["play_level"]
    BCS = arg_dict["behavior_characteristics"]
    N_GENERATIONS = arg_dict["n_generations"]

    # Number of generation episodes (i.e. number of latent seeds or initial states in the case of NCA)
    N_INIT_STATES = arg_dict["n_init_states"]

    # How many latents for Decoder and CPPN architectures
    # TODO: Try making this nonzero for NCA?
    N_LATENTS = 0 if "NCA" in MODEL else 2
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
    args.targets_penalty_weight = 10
    VIS_INTERVAL = 50
    ENV3D = "3D" in PROBLEM
    IS_HOLEY = "holey" in PROBLEM

    if "CPPN" in MODEL:
        if MODEL != "CPPNCA" and "Gen" not in MODEL:
            assert N_INIT_STATES == 0 and not RANDOM_INIT_LEVELS and not REEVALUATE_ELITES
        if MODEL != "CPPNCA":
            assert N_STEPS == 1
    if ("Decoder" in MODEL) or ("CPPN" in MODEL):
        assert N_STEPS == 1

    SAVE_LEVELS = arg_dict["save_levels"] or EVALUATE

    # TODO: This is redundant. Re-use `utils.get_exp_name()`
    #   exp_name = 'EvoPCGRL_{}-{}_{}_{}-batch_{}-step_{}'.format(PROBLEM, REPRESENTATION, BCS, N_INIT_STATES, N_STEPS, arg_dict['exp_name'])
    #   exp_name = "EvoPCGRL_{}-{}_{}_{}_{}-batch".format(
    #       PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES
    #   )
    # exp_name = 'EvoPCGRL_'
    # if ALGO == "ME":
    #     exp_name += "ME_"
    # exp_name += "{}-{}_{}_{}_{}-batch_{}-pass".format(
    #     PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES, N_STEPS
    # )

    # # TODO: remove this! Ad hoc, for backward compatibility.
    # if ALGO == "CMAME" and arg_dict["step_size"] != 1 or ALGO == "ME" and arg_dict["step_size"] != 0.01:
    #     exp_name += f"_{arg_dict['step_size']}-stepSize"

    # if CASCADE_REWARD:
    #     exp_name += "_cascRew"

    # if not RANDOM_INIT_LEVELS:
    #     exp_name += "_fixLvls"

    # if not REEVALUATE_ELITES:
    #     exp_name += "_fixElites"

    # if args.mega:
    #     exp_name += "_MEGA"
    # exp_name += "_" + arg_dict["exp_name"]
    exp_name = get_exp_name(args, arg_dict)
    SAVE_PATH = get_exp_dir(exp_name)
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
        # LOG_NAME = "./runs/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + exp_name
        LOG_NAME = SAVE_PATH
        writer = SummaryWriter(LOG_NAME)

        return writer


    if THREADS:
        ray.init()

    evolver: EvoPCGRL = None

    try:
        try:
            evolver = pickle.load(open(os.path.join(SAVE_PATH, "evolver.pkl"), "rb"))
        except:
            evolver = pickle.load(
                open(os.path.join(SAVE_PATH, "last_evolver.pkl"), "rb")
            )
        print("Loaded save file at {}".format(SAVE_PATH))

        if INFER:
            RENDER = True
            N_STEPS = N_INFER_STEPS
            RANDOM_INIT_LEVELS = False
        else:
            RENDER = False

        evolver.init_env()
        evolver._init_model()

        if VISUALIZE:
            if not ENV3D:
                evolver.visualize()

        if INFER:
            # evaluate on initial level seeds that each generator has seen before
            evolver.infer(concat_gifs=CONCAT_GIFS)
            save_grid(csv_name="eval_levels_fixLvls")

            if not isinstance(evolver.gen_model, DirectEncoding):
                # evaluate on random initial level seeds
                RANDOM_INIT_LEVELS = True
                evolver.infer(concat_gifs=CONCAT_GIFS)
                save_grid(csv_name="eval_levels")

        if not (INFER or EVALUATE or VISUALIZE):
            writer = init_tensorboard()
            # then we train
            RENDER = arg_dict["render"]
            evolver.total_itrs = arg_dict["n_generations"]
            evolver.evolve()
    except FileNotFoundError as e:
        if not (INFER or EVALUATE or RENDER_LEVELS):
            RENDER = arg_dict["render"]
            print(
                "Failed loading from an existing save-file. Evolving from scratch. The error was: {}".format(
                    e
                )
            )
            writer = init_tensorboard()
            evolver = EvoPCGRL(args)
            evolver.evolve()
        else:
            print(
                "Loading from an existing save-file failed. Cannot run inference. The error was: {}".format(
                    e
                )
            )
