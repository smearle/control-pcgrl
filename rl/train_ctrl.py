
from functools import partial
import json
import os
from pathlib import Path
from pdb import set_trace as TT
import shutil
import sys
import gym

import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import DEFAULT_LOGGERS, pretty_print
from ray.rllib.agents.ppo import PPOTrainer as RlLibPPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import gym_pcgrl
from models import CustomFeedForwardModel, CustomFeedForwardModel3D, WideModel3D, WideModel3DSkip, NCA # noqa : F401
from args import parse_args
from envs import make_env
#from stable_baselines3.common.policies import ActorCriticCnnPolicy
#from model import CustomPolicyBigMap, CApolicy, WidePolicy
#from model import (CustomPolicyBigMap, CustomPolicySmallMap,
#                   FullyConvPolicyBigMap, FullyConvPolicySmallMap, CAPolicy)
#from stable_baselines3.common.results_plotter import load_results, ts2xy
# from stable_baselines.results_plotter import load_results, ts2xy
# import tensorflow as tf
from utils import (get_env_name, get_exp_name, get_map_width,
#                  max_exp_idx
                   )
from callbacks import StatsCallbacks

n_steps = 0
PROJ_DIR = Path(__file__).parent.parent
best_mean_reward, n_steps = -np.inf, 0


class PPOTrainer(RlLibPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_path_file = kwargs['config']['checkpoint_path_file']
        # self.checkpoint_path_file = checkpoint_path_file

    @classmethod
    def get_default_config(cls):
        def_cfg = RlLibPPOTrainer.get_default_config()
        def_cfg.update({'checkpoint_path_file': None})
        return def_cfg

    def save(self, *args, **kwargs):
        ckp_path = super().save(*args, **kwargs)
        with open(self.checkpoint_path_file, 'w') as f:
            f.write(ckp_path)
        return ckp_path


def main(cfg):
    if (cfg.problem not in ["binary_ctrl", "binary_ctrl_holey", "sokoban_ctrl", "zelda_ctrl", "smb_ctrl", "MicropolisEnv", "RCT"]) and \
        ("minecraft" not in cfg.problem):
        raise Exception(
            "Not a controllable environment. Maybe add '_ctrl' to the end of the name? E.g. 'sokoban_ctrl'")

    is_3D_env = False
    if "3D" in cfg.problem:
        assert "3D" in cfg.representation
        is_3D_env = True

    cfg.env_name = get_env_name(cfg.problem, cfg.representation)
    print('env name: ', cfg.env_name)
    exp_name = get_exp_name(cfg)
    exp_name_id = f'{exp_name}_{cfg.experiment_id}'
    log_dir = os.path.join(PROJ_DIR, f'rl_runs/{exp_name_id}_log')

    if not cfg.load:

        if not cfg.overwrite:
            # New experiment
            if os.path.isdir(log_dir):
                raise Exception(f"Log directory rl_runs/{exp_name_id} exists. Please delete it first (or use command "
                "line argument `--load` to load experiment, or `--overwrite` to overwrite it).")

            # Create the log directory if training from scratch.
            os.mkdir(log_dir)

        else:
            # Overwrite the log directory.
            shutil.rmtree(log_dir)
            os.mkdir(log_dir)

        # Save the experiment settings for future reference.
        with open(os.path.join(log_dir, 'settings.json'), 'w', encoding='utf-8') as f:
            json.dump(cfg.__dict__, f, ensure_ascii=False, indent=4)

    if not is_3D_env:
        # Using this simple feedforward model for now by default
        model_cls = globals()[cfg.model] if cfg.model else CustomFeedForwardModel
    else:
        if cfg.representation == "wide3D":
            model_cls = globals()[cfg.model] if cfg.model else WideModel3D
        else:
            model_cls = globals()[cfg.model] if cfg.model else CustomFeedForwardModel3D

    ModelCatalog.register_custom_model("custom_model", model_cls)

    # If n_cpu is 0 or 1, we only use the local rllib worker. Specifying n_cpu > 1 results in use of remote workers.
    num_workers = 0 if cfg.n_cpu == 1 else cfg.n_cpu
    stats_callbacks = partial(StatsCallbacks, cfg=cfg)

    # ray won't stfu about this anyway lol
    dummy_env = make_env(vars(cfg))
    ray.rllib.utils.check_env(dummy_env)

    checkpoint_path_file = os.path.join(log_dir, 'checkpoint_path.txt')

    # The rllib trainer config (see the docs here: https://docs.ray.io/en/latest/rllib/rllib-training.html)
    trainer_config = {
        'env': 'pcgrl',
        'framework': 'torch',
        'num_workers': num_workers,
        'num_gpus': cfg.n_gpu,
        'env_config': vars(cfg),  # Maybe env should get its own config? (A subset of the original?)
        # 'env_config': {
            # 'change_percentage': cfg.change_percentage,
        # },
        'num_envs_per_worker': 20 if not cfg.infer else 1,
        'render_env': cfg.render,
        'lr': cfg.lr,
        'gamma': cfg.gamma,
        'model': {
            'custom_model': 'custom_model',
            'custom_model_config': {
                **cfg.model_cfg,
            }
        },
        "evaluation_config": {
            "explore": True,
        },
        "logger_config": {
                        "wandb": {
                "project": "PCGRL",
                "name": exp_name_id,
                "id": exp_name_id,
                # "api_key_file": "~/.wandb_api_key"
            },
            "type": "ray.tune.logger.TBXLogger",
            # Optional: Custom logdir (do not define this here
            # for using ~/ray_results/...).
            "logdir": log_dir,
        },
#       "exploration_config": {
#           "type": "Curiosity",
#       }
#       "log_level": "INFO",
#       "train_batch_size": 32,
#       "sgd_minibatch_size": 32,
        'callbacks': stats_callbacks,

        # `ray.tune` seems to need these spaces specified here.
        # 'observation_space': dummy_env.observation_space,
        # 'action_space': dummy_env.action_space,

        # 'create_env_on_driver': True,
        'checkpoint_path_file': checkpoint_path_file
    }

    register_env('pcgrl', make_env)

    # Log the trainer config, excluding overly verbose entries (i.e. Box observation space printouts).
    trainer_config_loggable = trainer_config.copy()
    # trainer_config_loggable.pop('observation_space')
    # trainer_config_loggable.pop('action_space')
    # trainer_config_loggable.pop('multiagent')
    print(f'Loading trainer with config:')
    print(pretty_print(trainer_config_loggable))

    # NOTE: `ray.tune` now handles re-loading. Remove the below code when we're sure all the functionality has migrated
    #   over successfully.
    # Super ad-hoc re-loading. Note that we reset the number of training steps to be executed. Need to clearn this up if
    # we were to use it in actual publication-worthy experiments. Good for debugging though, maybe.
    # if cfg.load:
    #     trainer = ppo.PPOTrainer(env='pcgrl', config=trainer_config)
    #     with open(checkpoint_path_file, 'r') as f:
    #         checkpoint_path = f.read()

    #     # TODO: are we failing to save/load certain optimizer states? For example, the kl coefficient seems to shoot
    #     #  back up when reloading (see tensorboard logs).
    #     trainer.load_checkpoint(checkpoint_path=checkpoint_path)
    #     print(f"Loaded checkpoint from {checkpoint_path}.")

    #     n_params = 0
    #     param_dict = trainer.get_weights()['default_policy']

    #     for v in param_dict.values():
    #         n_params += np.prod(v.shape)
    #     print(f'default_policy has {n_params} parameters.')
    #     print('model overview: \n', trainer.get_policy('default_policy').model)

    # Do inference, i.e., observe agent behavior for many episodes.
    if cfg.infer:
        trainer = PPOTrainer(env='pcgrl', config=trainer_config)
        with open(checkpoint_path_file, 'r') as f:
            checkpoint_path = f.read()
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}.")

        env = make_env(vars(cfg))
        for i in range(10000):
            obs = env.reset()
            done = False
            while not done:
                action = trainer.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                # print(env.unwrapped._rep_stats["path-length"])
                print(env.unwrapped._rep_stats)
                env.render()

        # Quit the program before agent starts training.
        sys.exit()

    log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']
    best_mean_reward = -np.inf

    # TODO: makes this controllable, i.e., by dividing the number of frames by the train_batch_size
    n_updates = 10000

    # TODO: We've given the main loop over to `ray.tune`. Make sure we keep this functionality aroundn! In particular:
    #   the printout is ugly and verbose. And it would be nice to log `fps`. (Probably just add to PPOTrainer subclass
    #   and override the `train_step` or `print_result` methods?)
    # The training loop.
    # for i in range(n_updates):
    def train_fn(config={}):
        result = trainer.train()
        log_result = {k: v for k, v in result.items() if k in log_keys}
        log_result['info: learner:'] = result['info']['learner']

        # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
        log_result['fps'] = result['timesteps_this_iter'] / result['time_this_iter_s']

        print('-----------------------------------------')
        print(pretty_print(log_result))

        # NOTE: `ray.tune` does this better now.
        # # Intermittently save model checkpoints.
        # if i % 10 == 0:
        #     checkpoint = trainer.save(checkpoint_dir=log_dir)

        #     # Remove the old checkpoint file if it exists.
        #     if os.path.isfile(checkpoint_path_file):
        #         with open(checkpoint_path_file, 'r') as f:
        #             old_checkpoint = f.read()

        #         # FIXME: sometimes this does not exist (when overwriting?)
        #         shutil.rmtree(Path(old_checkpoint).parent)
            
        #     # Record the path of the new checkpoint.
        #     with open(checkpoint_path_file, 'w') as f:
        #         f.write(checkpoint)

        #     print("checkpoint saved at", checkpoint)
    tune.register_trainable("CustomPPO", PPOTrainer)
    
    ray.init()

    analysis = tune.run(
        "CustomPPO",
        resume=cfg.load,
        config={
            **trainer_config,
            # "wandb": {
                # "project": "PCGRL",
            # },
        },
        checkpoint_at_end=True,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        local_dir=log_dir,
        verbose=3,
        # loggers=DEFAULT_LOGGERS + (WandbLogger, ),
        loggers=[WandbLogger]
    )

################################## MAIN ########################################

cfg = parse_args()

cfg.evaluate = False  # TODO: implement evaluation
cfg.ca_actions = False  # Not using NCA-type actions.
cfg.logging = True  # Always log


# NOTE: change percentage currently has no effect! Be warned!! (We fix number of steps for now.)

cfg.map_width = get_map_width(cfg.problem)
crop_size = cfg.crop_size
if "holey" in cfg.problem:
    crop_size = cfg.map_width * 2 + 1 if crop_size == -1 else crop_size
else:
    crop_size = cfg.map_width * 2 if crop_size == -1 else crop_size
cfg.crop_size = crop_size

if __name__ == '__main__':
    main(cfg)
