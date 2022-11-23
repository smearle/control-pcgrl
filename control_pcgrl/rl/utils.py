"""
Helper functions for train, infer, and eval modules.
"""
from pdb import set_trace as TT
import glob
import os
import ray
import re

import numpy as np
# from stable_baselines import PPO2
# from stable_baselines.bench import Monitor
#from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from control_pcgrl.configs.config import ControlPCGRLConfig
from control_pcgrl import wrappers

# NOTE: minecraft has to precede zelda since minecraft zelda maze has both phrases in its name.
MAP_WIDTHS = [("binary", 16), ("minecraft_3D_rain", 7), ("minecraft_3D", 15), ("zelda", 16), ("sokoban", 5)]

PROB_CONTROLS = {
    "binary_ctrl": [
        ["regions"],
        ["path-length"],
        ["regions", "path-length"],
        # ['emptiness', 'path-length'],
        # ["symmetry", "path-length"]
    ],
    "zelda_ctrl": [
        ["nearest-enemy"],
        ["path-length"],
        ["nearest-enemy", "path-length"],
        # ["emptiness", "path-length"],
        # ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
        # ["crate"],
        ["sol-length"],
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

@ray.remote
class IdxCounter:
    ''' When using rllib trainer to train and simulate on evolved maps, this global object will be
    responsible for providing unique indices to parallel environments.'''

    def __init__(self):
        self.count = 0
        self.keys = None

    def get(self, hsh):
        world_key_queue = self.hashes_to_keys[hsh]

        if not world_key_queue:
            raise Exception("No world keys provided.")

        return world_key_queue

    def set(self, i):
        # For inference
        self.count = i

    def set_keys(self, keys):
        self.count = 0
        self.keys = keys

    def set_hashes(self, hashes):
        """
        Note that we may assign multiple worlds to a single environment, or a single world to multiple environments.

        We will only assign a single world to multiple environments if duplicate keys were provided to `set_idxs()`.

        Args:
            hashes: A list of hashes, one per environment object.
        """
        hashes_to_keys = {h: [] for h in hashes}

        for i, wk in enumerate(self.keys):
            h = hashes[i % len(hashes)]
            hashes_to_keys[h].append(wk)
        
        self.hashes_to_keys = hashes_to_keys

    def scratch(self):
        return self.hashes_to_keys

def get_map_width(game):
    for k, v in MAP_WIDTHS:
        if k in game:
            return v
    raise Exception("Unknown game")

# def get_crop_size(game):
#     if "binary" in game:
#         return 32
#     elif "minecraft_3D" in game:
#         return 14
#     elif "zelda" in game:
#         return 32
#     elif "sokoban" in game:
#         return 10
#     else:
#         raise Exception("Unknown game")


#class RenderMonitor(Monitor):
#    """
#    Wrapper for the environment to save data in .csv files.
#    """
#
#    def __init__(self, env, rank, log_dir, **kwargs):
#        self.log_dir = log_dir
#        self.rank = rank
#        global_render = kwargs.get("render", False)
#        render_rank = kwargs.get("render_rank", 0)
#        self.render_me = False
#        if global_render and self.rank == render_rank:
#            self.render_me = True
#
#        if log_dir is not None:
#            log_dir = os.path.join(log_dir, str(rank))
#        Monitor.__init__(self, env, log_dir)
#
#    def step(self, action):
#        if self.render_me:
#            self.render()
#        ret = Monitor.step(self, action)
#
#        return ret


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


#def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
#    """
#    Return a function that will initialize the environment when called.
#    """
#    max_step = kwargs.get("max_step", None)
#    render = kwargs.get("render", False)
#
#    def _thunk():
#        if representation == "wide":
#            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
#        else:
#            crop_size = kwargs.get("crop_size", 28)
#            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
#
#        if max_step is not None:
#            env = wrappers.MaxStep(env, max_step)
#
#        if log_dir is not None and kwargs.get("add_bootstrap", False):
#            env = wrappers.EliteBootStrapping(
#                env, os.path.join(log_dir, "bootstrap{}/".format(rank))
#            )
#        # RenderMonitor must come last
#
#        if render or log_dir is not None and len(log_dir) > 0:
#            env = RenderMonitor(env, rank, log_dir, **kwargs)
#
#        return env
#
#    return _thunk


#def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
#    """
#    Prepare a vectorized environment using a list of 'make_env' functions.
#    """
#    n_cpu = kwargs.pop("n_cpu", 1)
#
#    if n_cpu > 1:
#        env_lst = []
#
#        for i in range(n_cpu):
#            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
#        env = SubprocVecEnv(env_lst)
#    else:
#        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
#
#    return env


def get_env_name(game, representation):
    if "RCT" in game or "Micropolis" in game:
        env_name = "{}-v0".format(game)
    else:
        env_name = "{}-{}-v0".format(game, representation)

    return env_name


def get_exp_name(cfg: ControlPCGRLConfig):

    exp_name = os.path.join(
        cfg.problem.name, 
        "weights_" + "-".join(f"{k}-{v}" for k, v in cfg.problem.weights.items()),
        cfg.representation,
    )

    exp_name += '/'

    if cfg.model.name is not None:
        exp_name += cfg.model.name

    if cfg.controls is not None:
        exp_name += "_" + "-".join(["ctrl"] + cfg.controls)
    
    if cfg.change_percentage is not None:
        exp_name += "_chng-{}".format(cfg.change_percentage)

    if cfg.max_board_scans != 1:
        exp_name += "_{}-scans".format(cfg.max_board_scans)

    if hasattr(cfg, "midep_trgs") and cfg.midep_trgs:
        exp_name += "_midEpTrgs"

    if hasattr(cfg, "alp_gmm") and cfg.alp_gmm:
        exp_name += "_ALPGMM"    

    if cfg.multiagent.n_agents != 0:
        exp_name += f"_{cfg.multiagent['n_agents']}-player"

    # TODO: Can have subdirectories for given settings of a given model type.
    if cfg.model.name is not None:
        exp_name += f"_{cfg.model.conv_filters}-convSz" if cfg.model.conv_filters != 64 else ""
        exp_name += f"_{cfg.model.fc_size}-fcSz" if cfg.model.fc_size != 64 and cfg.model.name != 'NCA' else ""

    if cfg.n_aux_tiles > 0:
        exp_name += f"_{cfg.n_aux_tiles}-aux"

    if cfg.static_prob is not None:
        exp_name += f"_{cfg.static_prob}-static"
    
    if cfg.learning_rate:
        exp_name += f"_lr-{cfg.learning_rate:.1e}"
    
    if cfg.observation_size is not None:
        exp_name += f"_obs-{cfg.observation_size}"

    # Can't control `n_frame`, but if we did, wouldn't want to have this in experiment name in case we watned to extent
    # training later.
    # if cfg.n_frame is not None:
    #     exp_name += f"_nframe-{cfg.n_frame}"

    return exp_name


# def load_model(log_dir, n_tools=None, load_best=False):
#     if load_best:
#         name = "best"
#     else:
#         name = "latest"
#     model_path = os.path.join(log_dir, "{}_model.pkl".format(name))
# 
#     if not os.path.exists(model_path):
#         model_path = os.path.join(log_dir, "{}_model.zip".format(name))
# 
#     if not os.path.exists(model_path):
#         files = [f for f in os.listdir(log_dir) if ".pkl" in f or ".zip" in f]
# 
#         if len(files) > 0:
#             # selects the last file listed by os.listdir
#             # What the fuck is up with the random thing
#             model_path = os.path.join(log_dir, np.random.choice(files))
#         else:
#             print("No models are saved at {}".format(model_path))
#             return None
# #           raise Exception("No models are saved at {}".format(model_path))
#     print("Loading model at {}".format(model_path))
# 
#     if n_tools:
#         policy_kwargs = {"n_tools": n_tools}
#     else:
#         policy_kwargs = {}
#     model = PPO2.load(model_path, reset_num_timesteps=False)
# 
#     return model


def max_exp_idx(exp_name):
    log_dir = os.path.join("../runs", exp_name)

    # Collect log directories corresponding to this experiment.
    log_files = glob.glob('{}*'.format(log_dir))

    if len(log_files) == 0:
        n = 1
    else:
        # Get the IDs of past log directories, assign the next one to this experiment (should only apply when reloading!)
        log_ns = [re.search('_(\d+)(_log)?$', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)

