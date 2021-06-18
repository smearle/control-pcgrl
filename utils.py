"""
Helper functions for train, infer, and eval modules.
"""
from pdb import set_trace as TT
import glob
import os
import re

import numpy as np
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from gym_pcgrl import wrappers

MAP_WIDTHS = {"binary": 16, "zelda": 16, "sokoban": 5}

PROB_CONTROLS = {
    "binary_ctrl": [
#       ["regions"],
#       ["path-length"],
        ["regions", "path-length"],
        # ['emptiness', 'path-length'],
        # ["symmetry", "path-length"]
    ],
    "zelda_ctrl": [
#       ["nearest-enemy"],
#       ["path-length"],
        ["nearest-enemy", "path-length"],
        # ["emptiness", "path-length"],
        # ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
        # ["crate"],
#       ["sol-length"],
        ["crate", "sol-length"],
        # ["emptiness", "sol-length"],
        # ["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
        ['enemies', 'jumps'],
        # ["emptiness", "jumps"],
        # ["symmetry", "jumps"],
    ],
    "RCT": [
        # ['income'],
    ],
}


def get_map_width(game):
    for (k, v) in MAP_WIDTHS.items():
        if k in game:
            return v
    return 16

def get_crop_size(game):
    if "binary" in game:
        return 32
    elif "zelda" in game:
        return 32
    elif "sokoban" in game:
        return 10
    else:
        return None


class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """

    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        global_render = kwargs.get("render", False)
        render_rank = kwargs.get("render_rank", 0)
        self.render_me = False
        if global_render and self.rank == render_rank:
            self.render_me = True

        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_me:
            self.render()
        ret = Monitor.step(self, action)

        return ret


def get_action(obs, env, model, action_type=True):
    action = None

    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(
            a=list(range(len(action_prob))), size=1, p=action_prob
        )
    else:
        action = np.array([env.action_space.sample()])

    return action


def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    """
    Return a function that will initialize the environment when called.
    """
    max_step = kwargs.get("max_step", None)
    render = kwargs.get("render", False)

    def _thunk():
        if representation == "wide":
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get("cropped_size", 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)

        if max_step is not None:
            env = wrappers.MaxStep(env, max_step)

        if log_dir is not None and kwargs.get("add_bootstrap", False):
            env = wrappers.EliteBootStrapping(
                env, os.path.join(log_dir, "bootstrap{}/".format(rank))
            )
        # RenderMonitor must come last

        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)

        return env

    return _thunk


def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    """
    Prepare a vectorized environment using a list of 'make_env' functions.
    """
    n_cpu = kwargs.pop("n_cpu", 1)

    if n_cpu > 1:
        env_lst = []

        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])

    return env


def get_env_name(game, representation):
    if "RCT" in game or "Micropolis" in game:
        env_name = "{}-v0".format(game)
    else:
        env_name = "{}-{}-v0".format(game, representation)

    return env_name


def get_exp_name(game, representation, **kwargs):
    exp_name = "{}_{}".format(game, representation)
    change_percentage = kwargs.get("change_percentage")

    if kwargs.get("conditional"):
        exp_name += "_conditional"
        exp_name += "_" + "-".join(["ctrl"] + kwargs.get("cond_metrics"))
        if change_percentage != 1.0:
            exp_name += "_chng-{}".format(change_percentage)
    else:
        exp_name += "_vanilla"
        exp_name += "_chng-{}".format(change_percentage)

    if kwargs.get("midep_trgs"):
        exp_name += "_midEpTrgs"

    if kwargs.get("ca_action"):
        exp_name += "_CAaction"

    if kwargs.get("alp_gmm"):
        exp_name += "_ALPGMM"

    return exp_name


#def max_exp_idx(exp_name):
#    log_dir = os.path.join("./rl_runs", exp_name)
#    log_files = glob.glob("{}*".format(log_dir))
#
#    if len(log_files) == 0:
#        n = 0
#    else:
#        log_ns = [int(re.search("_(\d+)", f).group(1)) for f in log_files]
#        n = max(log_ns)
#
#    return int(n)


def load_model(log_dir, n_tools=None, load_best=False):
    if load_best:
        name = "best"
    else:
        name = "latest"
    model_path = os.path.join(log_dir, "{}_model.pkl".format(name))

    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, "{}_model.zip".format(name))

    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if ".pkl" in f or ".zip" in f]

        if len(files) > 0:
            # selects the last file listed by os.listdir
            # What the fuck is up with the random thing
            model_path = os.path.join(log_dir, np.random.choice(files))
        else:
            print("No models are saved at {}".format(model_path))
            return None
#           raise Exception("No models are saved at {}".format(model_path))
    print("Loading model at {}".format(model_path))

    if n_tools:
        policy_kwargs = {"n_tools": n_tools}
    else:
        policy_kwargs = {}
    model = PPO2.load(model_path)

    return model
