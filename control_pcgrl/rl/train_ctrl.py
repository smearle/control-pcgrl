
import copy
import json
import os
import pickle
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from pdb import set_trace as TT
from typing import Dict

import gym
import hydra
import matplotlib
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
import ray
import torch as th
import wandb
from matplotlib import pyplot as plt
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.algorithms import ppo
# from ray.rllib.algorithms.a3c import A2CTrainer
# from ray.rllib.algorithms.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import check_env
from ray.tune import CLIReporter
from ray.tune.integration.wandb import (WandbLoggerCallback,
                                        WandbTrainableMixin, wandb_mixin)
from ray.tune.logger import DEFAULT_LOGGERS, pretty_print
from ray.tune.registry import register_env

from control_pcgrl.rl.args import parse_args
from control_pcgrl.rl.callbacks import StatsCallbacks
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.evaluate import evaluate
from control_pcgrl.rl.models import (NCA, ConvDeconv2d,  # noqa : F401
                       CustomFeedForwardModel, CustomFeedForwardModel3D,
                       Decoder, DenseNCA, SeqNCA, SeqNCA3D, WideModel3D,
                       WideModel3DSkip)
from control_pcgrl.rl.utils import IdxCounter, get_env_name, get_exp_name, get_map_width
from control_pcgrl.rl.rllib_utils import PPOTrainer
from control_pcgrl.configs.config import ControlPCGRLConfig
import control_pcgrl
from control_pcgrl.envs.probs import PROBLEMS
from control_pcgrl.envs.probs.minecraft.minecraft_3D_holey_maze_prob import \
    Minecraft3DholeymazeProblem
from control_pcgrl.task_assignment import set_map_fn

# Annoying, but needed since we can't go through `globals()` from inside hydra SLURM job. Is there a better way?
MODELS = {"NCA": NCA, "DenseNCA": DenseNCA, "SeqNCA": SeqNCA, "SeqNCA3D": SeqNCA3D}

matplotlib.use('Agg')

n_steps = 0
PROJ_DIR = Path(__file__).parent.parent.parent
best_mean_reward, n_steps = -np.inf, 0


# TODO: Render this bloody scatter plot of control targets/vals!
# class CustomWandbLogger(WandbLogger):
#     def on_result(self, result: Dict):
#         res = super().on_result(result)
#         if 'custom_plots' in result:
#             for k, v in result['custom_plots'].items():
#                 wandb.log({k: v}, step=result['training_iteration'])



@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: ControlPCGRLConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("Current working directory:", os.getcwd())

    if cfg.crop_shape is None:
        # This guy gotta observe the holes.
        if "holey" in cfg.problem.name:
            crop_shape = cfg.map_shape * 2 + 1
        else:
            crop_shape = cfg.map_shape * 2
        cfg.crop_shape = crop_shape

    # FIXME: Check for a 3D problem parent class.
    is_3D_env = False
    if "3D" in cfg.problem.name:
        is_3D_env = True

    cfg.env_name = get_env_name(cfg.problem.name, cfg.representation)
    print('env name: ', cfg.env_name)
    exp_name = get_exp_name(cfg)
    exp_name_id = f'{exp_name}_{cfg.exp_id}'
    cfg.log_dir = log_dir = os.path.join(PROJ_DIR, f'rl_runs/{exp_name_id}_log')

    if not cfg.load:

        # New experiment, or automatically load
        if not cfg.overwrite:
            if os.path.isdir(log_dir):
                print(f"Log directory {log_dir} already exists. Will attempt to load.")

            else:
                os.makedirs(log_dir)
                print(f"Created new log directory {log_dir}")

        # Try to overwrite the saved directory.
    if cfg.overwrite:
        if not os.path.exists(log_dir):
            print(f"Log directory rl_runs/{exp_name_id} does not exist. Will create new directory.")
        else:
            # Overwrite the log directory.
            print(f"Overwriting log directory rl_runs/{exp_name_id}")
            shutil.rmtree(log_dir, ignore_errors=True)
        os.makedirs(log_dir, exist_ok=True)

    # Save the experiment settings for future reference.
    # with open(os.path.join(log_dir, 'settings.json'), 'w', encoding='utf-8') as f:
    #     json.dump(cfg.__dict__, f, ensure_ascii=False, indent=4)

    if not is_3D_env:
        if cfg.model.name is None:
            if cfg.representation == "wide":
                model_cls = ConvDeconv2d
            else:
                model_cls = CustomFeedForwardModel
        else:
            model_cls = MODELS[cfg.model.name]
    else:
        if cfg.representation == "wide3D":
            model_cls = MODELS[cfg.model.name] if cfg.model else WideModel3D
        else:
            model_cls = MODELS[cfg.model.name] if cfg.model else CustomFeedForwardModel3D

    ModelCatalog.register_custom_model("custom_model", model_cls)

    # If n_cpu is 0 or 1, we only use the local rllib worker. Specifying n_cpu > 1 results in use of remote workers.
    num_workers = num_workers = 0 if cfg.hardware.n_cpu == 1 else cfg.hardware.n_cpu
    stats_callbacks = partial(StatsCallbacks, cfg=cfg)

    dummy_cfg = copy.copy(cfg)
    # dummy_cfg["render"] = False
    dummy_cfg.evaluation_env = False
    dummy_env = make_env(dummy_cfg)
    check_env(dummy_env)

    if issubclass(type(dummy_env), MultiAgentEnv):
        agent_obs_space = dummy_env.observation_space["agent_0"]
        agent_act_space = dummy_env.action_space["agent_0"]
    else:
        agent_obs_space = dummy_env.observation_space
        agent_act_space = dummy_env.action_space

    ### DEBUG ###
    if cfg.debug:
        for _ in range(100):
            obs = dummy_env.reset()
            for i in range(500):
                # if i > 3:
                act = dummy_env.action_space.sample()
                # act = 0 
                # else:
                    # act = 0
                obs, rew, done, info = dummy_env.step(act)
                # print(obs.transpose(2, 0, 1)[:, 10:-10, 10:-10])
                if cfg.render:
                    dummy_env.render()
        print('DEBUG: Congratulations! You can now use the environment.')
        sys.exit()

    checkpoint_path_file = os.path.join(log_dir, 'checkpoint_path.txt')
    # FIXME: nope
    num_envs_per_worker = cfg.hardware.num_envs_per_worker if not cfg.infer else 1
    logger_type = {"type": "ray.tune.logger.TBXLogger"} if not (cfg.infer or cfg.evaluate) else {}
    eval_num_workers = eval_num_workers = num_workers if cfg.evaluate else 0
    model_cfg = {**cfg.model}

    # rllib will pass its own `name`
    model_cfg.pop('name')

    if cfg.multiagent.n_agents != 0:
        multiagent_config = {
            "policies": {
                f"default_policy": PolicySpec(
                    policy_class=None,
                    observation_space=agent_obs_space,
                    action_space=agent_act_space,
                    config=None,)
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
            "count_steps_by": "agent_steps",
        }
        multiagent_config = {"multiagent": multiagent_config}
    else:
        multiagent_config = {}

    # The rllib trainer config (see the docs here: https://docs.ray.io/en/latest/rllib/rllib-training.html)
    trainer_config = {
        'env': 'pcgrl',
        **multiagent_config,
        'framework': 'torch',
        'num_workers': num_workers if not (cfg.evaluate or cfg.infer) else 0,
        'num_gpus': cfg.hardware.n_gpu,
        'env_config': {
            **cfg,  # Maybe env should get its own config? (A subset of the original?)
            "evaluation_env": False,
        },
        # 'env_config': {
            # 'change_percentage': cfg.change_percentage,
        # },
        'num_envs_per_worker': num_envs_per_worker,
        'render_env': cfg.render,
        'lr': cfg.learning_rate,
        'gamma': cfg.gamma,
        'model': {
            'custom_model': 'custom_model',
            'custom_model_config': {
                "dummy_env_obs_space": copy.copy(agent_obs_space),
            **model_cfg,
            },
        },
        "evaluation_interval" : 1 if cfg.evaluate else 1,
        "evaluation_duration": max(1, num_workers),
        "evaluation_duration_unit": "episodes",
        "evaluation_num_workers": eval_num_workers,
        "env_task_fn": set_map_fn,
        "evaluation_config": {
            "env_config": {
                **cfg,
                "evaluation_env": True,
                "num_eval_envs": num_envs_per_worker * eval_num_workers,
            },
            "explore": True,
        },
        "logger_config": {
                # "wandb": {
                    # "project": "PCGRL",
                    # "name": exp_name_id,
                    # "id": exp_name_id,
                    # "api_key_file": "~/.wandb_api_key"
            # },
            **logger_type,
            # Optional: Custom logdir (do not define this here
            # for using ~/ray_results/...).
            "logdir": log_dir,
        },
#       "exploration_config": {
#           "type": "Curiosity",
#       }
#       "log_level": "INFO",
        # "train_batch_size": 50,
        # "sgd_minibatch_size": 50,
        'callbacks': stats_callbacks,

        # To take random actions while changing all tiles at once seems to invite too much chaos.
        'explore': True,

        # `ray.tune` seems to need these spaces specified here.
        # 'observation_space': dummy_env.observation_space,
        # 'action_space': dummy_env.action_space,

        # 'create_env_on_driver': True,
        'checkpoint_path_file': checkpoint_path_file,
        # 'record_env': log_dir,
        # 'stfu': True,
        'disable_env_checking': True,
    }

    register_env('pcgrl', make_env)

    # Log the trainer config, excluding overly verbose entries (i.e. Box observation space printouts).
    trainer_config_loggable = trainer_config.copy()
    # trainer_config_loggable.pop('observation_space')
    # trainer_config_loggable.pop('action_space')
    # trainer_config_loggable.pop('multiagent')
    print(f'Loading trainer with config:')
    print(pretty_print(trainer_config_loggable))

    # Do inference, i.e., observe agent behavior for many episodes.
    if cfg.infer or cfg.evaluate:
        trainer_config.update({
            # FIXME: The name of this config arg seems to have changed in rllib?
            # 'record_env': log_dir if cfg.record_env else None,
        })
        trainer = PPOTrainer(env='pcgrl', config=trainer_config)

        if cfg.load:
            with open(checkpoint_path_file, 'r') as f:
                checkpoint_path = f.read()
        
            # HACK (should probably be logging relative paths in the first place?)
            checkpoint_path = checkpoint_path.split('control-pcgrl/')[-1]
        
            # HACK wtf (if u edit the checkpoint path some funkiness lol)
            if not os.path.exists(checkpoint_path):
                assert os.path.exists(checkpoint_path[:-1]), f"Checkpoint path does not exist: {checkpoint_path}."
                checkpoint_path = checkpoint_path[:-1]

            # trainer.load_checkpoint(checkpoint_path=checkpoint_path)
            trainer.restore(checkpoint_path=checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}.")

        # TODO: Move this into its own script and re-factor the redundant bits!(?)
        if cfg.evaluate:
            evaluate(trainer, dummy_env, cfg)
            sys.exit()

        elif cfg.infer:
            while True:
                trainer.evaluate()

        # Quit the program before agent starts training.
        sys.exit()

    tune.register_trainable("CustomPPO", PPOTrainer)

    # Limit the number of rows.
    reporter = CLIReporter(
        metric_columns={
            # "training_iteration": "itr",
            "timesteps_total": "timesteps",
            "custom_metrics/path-length_mean": "path-length",
            "custom_metrics/connected-path-length_mean": "cnct-path-length",
            "custom_metrics/regions_mean": "regions",
            "episode_reward_mean": "reward",
            "fps": "fps",
        },
        max_progress_rows=10,
        )
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    # reporter.add_metric_column("custom_metrics/path-length_mean")
    # reporter.add_metric_column("episode_reward_mean")
    
    ray.init()
    # loggers_dict = {'loggers': [WandbLoggerCallback]} if cfg.wandb else {}
    # loggers_dict = {'loggers': [CustomWandbLogger]} if cfg.wandb else {}
    callbacks_dict = {'callbacks': [WandbLoggerCallback(
        project="PCGRL_AIIDE_0",
        name=exp_name_id,
        id=exp_name_id,
    )]} if cfg.wandb else {}

    try:
        # TODO: ray overwrites the current config with the re-loaded one. How to avoid this?
        analysis = tune.run(
            "CustomPPO",
            resume="AUTO" if (cfg.load and not cfg.overwrite) else False,
            config={
                **trainer_config,
            },
            # checkpoint_score_attr="episode_reward_mean",
            # TODO: makes timestep total input by user.(n_frame)
            stop={"timesteps_total": 1e10},
            checkpoint_at_end=True,
            checkpoint_freq=10,
            keep_checkpoints_num=2,
            local_dir=log_dir,
            verbose=1,
            # loggers=DEFAULT_LOGGERS + (WandbLogger, ),
            # **loggers_dict,
            **callbacks_dict,
            progress_reporter=reporter,
        )
    except KeyboardInterrupt:
        ray.shutdown()

################################## MAIN ########################################

if __name__ == '__main__':
    main()
    # cfg = parse_args()
    # main(cfg)
