"""
Helper functions for train, infer, and eval modules.
"""
import os
import re
import glob
import numpy as np
from gym_pcgrl import wrappers
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
import gym

import grpc
import minecraft_pb2_grpc
from minecraft_pb2 import *
import time

CHANNEL = grpc.insecure_channel('localhost:5001')
CLIENT = minecraft_pb2_grpc.MinecraftServiceStub(CHANNEL)

b_map = [DIRT, AIR, QUARTZ_BLOCK]
block_map = dict(zip(range(len(b_map)), b_map))
inv_block_map = dict(zip(b_map, range(len(b_map))))
N_BLOCK_TYPE = 3

def clear(n, e):
    '''
    Clear a background of size (n e) in position (0 0 0) for rendering in Minecraft
    n stands for length in NORTH direction
    e stands for length in EAST direction
    '''
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=0, y=4, z=0),
            max=Point(x=n, y=7, z=e)
        ),
        type=AIR
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-3, y=4, z=-3),
            max=Point(x=n+3, y=4, z=e+3)
        ),
        type=QUARTZ_BLOCK
    ))

def get_tile(tile):
    '''
    return the types blocks of each tiles in Minecraft
    '''
    # TODO add

def spawn_maze(map, base=5):
    '''
    Spawn maze iterately in Minecraft
    '''
    blocks = []
    for j in range(len(map)):
        for i in range(len(map[j])):
            item = get_tile(map[j][i])
            blocks.append(Block(position=Point(x=i, y=base,z=j),
                               type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    #time.sleep(0.2)

class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        return Monitor.step(self, action)

class RenderMinecraftWrapper(gym.Wrapper):
    """
    Wrapper class for rendering in Evocraft environment for Minecraft game.
    """
    def __init__(self, env, **kwargs):
        mode = kwargs.get('render_mode', 'human')
        self.env
        self.pcgrl_env = gym.make(game)
        self.pcgrl_env.adjust_param(**kwargs)

    def render(self, mode='human'):
        if mode == 'evocraft':
            # TODO add func calls

            return
        else:
            return self.env.render(mode=mode)

def get_action(obs, env, model, action_type=True):
    action = None
    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(a=list(range(len(action_prob))), size=1, p=action_prob)
    else:
        action = np.array([env.action_space.sample()])
    return action

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    render_mode = kwargs.get('render_mode', 'human')
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 28)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        if render_mode == 'evocraft':
            env = RenderMinecraftWrapper(env, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)

def load_model(log_dir):
    model_path = os.path.join(log_dir, 'latest_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'latest_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if '.pkl' in f or '.zip' in f]
        if len(files) > 0:
            # selects the last file listed by os.listdir
            model_path = os.path.join(log_dir, np.random.choice(files))
        else:
            raise Exception('No models are saved')
    model = PPO2.load(model_path)
    return model
