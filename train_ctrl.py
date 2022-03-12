
import json
import os
from pathlib import Path
from pdb import set_trace as TT
import shutil
import gym

import numpy as np

import gym_pcgrl
from model import CustomFeedForwardModel  # noqa : F401
from rl_args import parse_args
from envs import make_env
#from stable_baselines3.common.policies import ActorCriticCnnPolicy
#from model import CustomPolicyBigMap, CApolicy, WidePolicy
#from model import (CustomPolicyBigMap, CustomPolicySmallMap,
#                   FullyConvPolicyBigMap, FullyConvPolicySmallMap, CAPolicy)
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
#from stable_baselines3 import PPO
# from stable_baselines import PPO2
#from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines.results_plotter import load_results, ts2xy
# import tensorflow as tf
from utils import (get_crop_size, get_env_name, get_exp_name, get_map_width,
#                  max_exp_idx
                   )

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0


def main(game, representation, n_frames, n_cpu, render, logging, **kwargs):
    if (game not in ["binary_ctrl", "sokoban_ctrl", "zelda_ctrl", "smb_ctrl", "MicropolisEnv", "RCT"]) and ("minecraft" not in game):
        raise Exception(
            "Not a controllable environment. Maybe add '_ctrl' to the end of the name? E.g. 'sokoban_ctrl'")
    kwargs['n_cpu'] = n_cpu
    env_name = get_env_name(game, representation)
    print('env name: ', env_name)
    exp_name = get_exp_name(game, representation, **kwargs)


    load = kwargs.get('load', True)
    ca_action = kwargs.get('ca_action')
    crop_size = kwargs.get('cropped_size')
    exp_id = kwargs.get('experiment_id')
    global log_dir
    exp_name_id = f'{exp_name}_{exp_id}'
    log_dir = f'rl_runs/{exp_name_id}_log'

    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
        'representation': representation,
        'env_name': env_name,
    }

    if not load:

        if not opts.overwrite:
            if os.path.isdir(log_dir):
                raise Exception("Log directory exists. Delete it first (or use command line argument `--load`).")

            # Create the log directory if training from scratch.
            os.mkdir(log_dir)

        # Save the experiment settings for future reference.
        with open(os.path.join(log_dir, 'settings.json'),
                'w',
                encoding='utf-8') as f:
            json.dump(kwargs, f, ensure_ascii=False, indent=4)

    # Don't use any rllib remote workers if using 1 process (local worker only).
    num_workers = 0 if n_cpu == 1 else n_cpu

    # Using this simple feedforward model for now by default
    ModelCatalog.register_custom_model("feedforward", CustomFeedForwardModel)

#   # Define a few default rllib models for different square crop sizes. These are lists of conv layers, where each conv
#   # layer is a list of [n_filters, kernel_size, stride]. Padding is automatic.
#   if crop_size == 32:
#       conv_filters = [
#               [32, [3, 3], 2],
#               [64, [3, 3], 2],
#               [64, [3, 3], 1],
#               [128, [6, 6], 1],  # Effectively a dense layer. stride of 3 only serves to prevent padding (?)
#           ]
#   else:
#       raise NotImplementedError(f"Have not defined conv_filters for a crop width of {crop_size}.") 

    # The rllib trainer config (see the docs here: https://docs.ray.io/en/latest/rllib/rllib-training.html)
    trainer_config = {
        'framework': 'torch',
        'num_workers': num_workers,
        'num_gpus': 1,
        'env_config': kwargs,
        'num_envs_per_worker': 10,
        'render_env': opts.render,
        'model': {
            'custom_model': 'feedforward',
        }
    }

    register_env('pcgrl', make_env)

    trainer = ppo.PPOTrainer(env='pcgrl', config=trainer_config)

    checkpoint_path_file = os.path.join(log_dir, 'checkpoint_path.txt')

    # Super ad-hoc re-loading. Note that we reset the number of training steps to be executed. Need to clearn this up if
    # we were to use it in actual publication-worthy experiments. Good for debugging though, maybe.
    if load:
        with open(checkpoint_path_file, 'r') as f:
            checkpoint_path = f.read()

        trainer.load_checkpoint(checkpoint_path=checkpoint_path)

    n_params = 0
    param_dict = trainer.get_weights()['default_policy']
    for v in param_dict.values():
        n_params += np.prod(v.shape)
    print(f'default_policy has {n_params} parameters.')
    print('model overview: \n', trainer.get_policy('default_policy').model)

    # Do inference, i.e., observe agent behavior for endless episodes.
    if opts.infer:
        env = make_env(kwargs)
        while True:
            obs = env.reset()
            # TODO: get max steps or set this manually
            for i in range(1000):
                action = trainer.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                env.render()

        # Quit the program before agent starts training.
        sys.exit()

    # TODO: makes this controllable, i.e., by dividing the number of frames by the train_batch_size
    n_updates = 10000

    # The training loop.
    for i in range(n_updates):
        result = trainer.train()
        print(pretty_print(result))
        trainer.train()

        # Intermittently save model checkpoints.
        if i % 10 == 0:
            checkpoint = trainer.save(checkpoint_dir=log_dir)

            # Remove the old checkpoint file if it exists.
            if os.path.isfile(checkpoint_path_file):
                with open(checkpoint_path_file, 'r') as f:
                    old_checkpoint_path = f.read()

                shutil.rmtree(Path(old_checkpoint_path).parent)
            
            # Record the path of the new checkpoint.
            with open(checkpoint_path_file, 'w') as f:
                f.write(checkpoint)

            print("checkpoint saved at", checkpoint)


################################## MAIN ########################################

opts = parse_args()

# User settings
conditional = len(opts.conditionals) > 0
problem = opts.problem
representation = opts.representation
n_frames = opts.n_frames
render = opts.render
logging = True
n_cpu = opts.n_cpu
load = opts.load
# load = True
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
alp_gmm = opts.alp_gmm
#################


max_step = opts.max_step
# TODO: play with change percentage for controllable agents?
change_percentage = opts.change_percentage
global COND_METRICS

if conditional:
    COND_METRICS = opts.conditionals
#   change_percentage = 1.0
else:
    COND_METRICS = []
#   change_percentage = opts.change_percentage
map_width = get_map_width(problem)
cropped_size = opts.crop_size
cropped_size = map_width * 2 if cropped_size == -1 else cropped_size

kwargs = {
    'map_width': get_map_width(problem),
    'change_percentage': change_percentage,
    'conditional': conditional,
    'cond_metrics': COND_METRICS,
    'load': load,
    'max_step': max_step,
    'midep_trgs': midep_trgs,
    'ca_action': ca_action,
    'cropped_size': cropped_size,
    'alp_gmm': alp_gmm,
    'experiment_id': opts.experiment_id,
}

if __name__ == '__main__':
    main(problem, representation, int(n_frames), n_cpu, render, logging, **kwargs)