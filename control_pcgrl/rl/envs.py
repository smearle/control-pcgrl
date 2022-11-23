from argparse import Namespace
from collections import namedtuple
import os
from pdb import set_trace as TT
from typing import Dict
from control_pcgrl import wrappers

from gym import spaces
from control_pcgrl.envs.pcgrl_env import PcgrlEnv
import numpy as np

from control_pcgrl import control_wrappers
from control_pcgrl.envs.probs import PROBLEMS
from control_pcgrl.envs.probs.holey_prob import HoleyProblem
from control_pcgrl.envs.probs.problem import Problem3D
from control_pcgrl.envs.reps import REPRESENTATIONS
from control_pcgrl.envs.reps.ca_rep import CARepresentation
from control_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from control_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
#from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
# from utils import RenderMonitor, get_map_width

def make_env(cfg):
    """
    Initialize and wrap the environment.

    Args:
        cfg_dict: dictionary of configuration parameters
    """
    cfg_dict = cfg # tehe
    cfg = Namespace(**cfg)

    # Turn dictionary into an object with attributes instead of keys.
    # cfg = namedtuple("env_cfg", cfg_dict.keys())(*cfg_dict.values())

    env: PcgrlEnv = None
    rep_cls = REPRESENTATIONS[cfg.representation], 
    if cfg.representation in ['wide']:
    # if issubclass(rep_cls, WideRepresentation):
        if '3D' in cfg.problem.name:
        # if issubclass(rep_cls, Representation3DABC):
            # elif cfg.representation in ['wide3D', 'wide3Dholey']:
                # raise NotImplementedError("3D wide representation not implemented")
            env = wrappers.ActionMap3DImagePCGRLWrapper(cfg.env_name, **cfg_dict)
        else:
            # HACK
            if 'holey' in cfg.problem.name:
            # if issubclass(rep_cls, HoleyRepresentationABC):
                env = wrappers.ActionMapImagePCGRLWrapper(cfg.env_name, bordered_observation=True, **cfg_dict)
            else:
                env = wrappers.ActionMapImagePCGRLWrapper(cfg.env_name, **cfg_dict)

    elif rep_cls == CARepresentation:
        # env = wrappers.CAWrapper(env_name, **kwargs)
        env = wrappers.CAactionWrapper(cfg.env_name, **cfg_dict)

    elif np.any(issubclass(rep_cls, c) for c in [NarrowRepresentation, TurtleRepresentation]):
    # elif cfg.representation in ['narrow', 'turtle', 'narrowholey', 'turtleholey']:
        # if '3D' in cfg.problem.name:
        # if issubclass(rep_cls, Representation3DABC):
            # env = wrappers.Cropped3DImagePCGRLWrapper(cfg.env_name, **cfg_dict)
        # else:  
        env = wrappers.CroppedImagePCGRLWrapper(cfg.env_name, **cfg_dict)
    # elif cfg.representation in ['narrow3D', 'turtle3D', 'narrow3Dholey', 'turtle3Dholey']:

    else:
        raise Exception('Unknown representation: {}'.format(rep_cls))
    env.configure(**cfg_dict)
    # if cfg.max_step is not None:
        # env = wrappers.MaxStep(env, cfg.max_step)
#   if log_dir is not None and cfg.get('add_bootstrap', False):
#       env = wrappers.EliteBootStrapping(env,
#                                           os.path.join(log_dir, "bootstrap{}/".format(rank)))
    env = control_wrappers.ControlWrapper(env, ctrl_metrics=cfg.controls, **cfg_dict)
    if not cfg.evaluate:
        if cfg.controls is not None:
            if cfg.controls.alp_gmm:
                env = control_wrappers.ALPGMMTeacher(env, **cfg_dict)
            else:
                env = control_wrappers.UniformNoiseyTargets(env, **cfg_dict)
    # it not conditional, the ParamRew wrapper should just be fixed at default static targets
#   if render or log_dir is not None and len(log_dir) > 0:
#       # RenderMonitor must come last
#       env = RenderMonitor(env, rank, log_dir, **kwargs)

    if cfg.multiagent.n_agents != 0:
        env = wrappers.MultiAgentWrapper(env, **cfg_dict)

    return env
