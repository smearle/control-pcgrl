from argparse import Namespace
from collections import namedtuple
import os
from pdb import set_trace as TT
import json
from typing import Dict

import ray
from control_pcgrl import wrappers

from gymnasium import spaces
from control_pcgrl.envs.pcgrl_env import PcgrlEnv
import numpy as np

from control_pcgrl import control_wrappers
from control_pcgrl.configs.config import Config
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

def make_env(cfg: Config):
    """
    Initialize and wrap the environment.

    Args:
        cfg_dict: dictionary of configuration parameters
    """
    env: PcgrlEnv = None
    try:
        rep_cls = REPRESENTATIONS[cfg.representation]     # when we first go pass this function, cfg is <class 'omegaconf.dictconfig.DictConfig'>
    except:
        # when we pass here the second time, cfg is <class 'ray.rllib.env.env_context.EnvContext'>, it's a dictionary
        cfg = Namespace(**cfg)        # it is so ugly
        rep_cls = REPRESENTATIONS[cfg.representation]  
    if cfg.representation in ['wide']:
    # if issubclass(rep_cls, WideRepresentation):
        if '3D' in cfg.task.problem:
        # if issubclass(rep_cls, Representation3DABC):
            # elif cfg.representation in ['wide3D', 'wide3Dholey']:
                # raise NotImplementedError("3D wide representation not implemented")
            env = wrappers.ActionMap3DImagePCGRLWrapper(cfg.env_name, cfg)
        else:
            # HACK
            if 'holey' in cfg.task.problem:
            # if issubclass(rep_cls, HoleyRepresentationABC):
                env = wrappers.ActionMapImagePCGRLWrapper(cfg.env_name, bordered_observation=True, cfg=cfg)
            else:
                env = wrappers.ActionMapImagePCGRLWrapper(cfg.env_name, cfg=cfg)

    elif rep_cls == CARepresentation:
        # env = wrappers.CAWrapper(env_name, **kwargs)
        env = wrappers.CAactionWrapper(cfg.env_name, cfg)

    elif np.any(issubclass(rep_cls, c) for c in [NarrowRepresentation, TurtleRepresentation]):
        env = wrappers.CroppedImagePCGRLWrapper(game=cfg.env_name, cfg=cfg)

    else:
        raise Exception('Unknown representation: {}'.format(rep_cls))
    env = control_wrappers.ControlWrapper(env, ctrl_metrics=cfg.controls, cfg=cfg)
    if not cfg.evaluate:
        if cfg.controls is not None:
            if cfg.controls.alp_gmm:
                env = control_wrappers.ALPGMMTeacher(env, cfg)
            else:
                env = control_wrappers.UniformNoiseyTargets(env, cfg)

    if cfg.multiagent.n_agents != 0:
        env = wrappers.MultiAgentWrapper(env, cfg)
        env = ray.rllib.env.wrappers.multi_agent_env_compatibility.MultiAgentEnvCompatibility(env)

    return env
