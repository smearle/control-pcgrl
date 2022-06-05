
import copy
from functools import partial
import json
import os
from pathlib import Path
from pdb import set_trace as TT
import shutil
import sys
import time
import gym
from matplotlib import pyplot as plt
from typing import Dict

import numpy as np
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.logger import DEFAULT_LOGGERS, pretty_print
from ray.rllib.agents.ppo import PPOTrainer as RlLibPPOTrainer
# from ray.rllib.agents.a3c import A2CTrainer
# from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import check_env
from ray.tune import CLIReporter
from ray.tune.integration.wandb import wandb_mixin, WandbTrainableMixin
from ray.tune.registry import register_env

import gym_pcgrl
import wandb
from models import CustomFeedForwardModel, CustomFeedForwardModel3D, WideModel3D, WideModel3DSkip, Decoder, DenseNCA, \
    NCA, SeqNCA, SeqNCA3D # noqa : F401
from args import parse_args
from envs import make_env
from utils import get_env_name, get_exp_name, get_map_width
from callbacks import StatsCallbacks

n_steps = 0
PROJ_DIR = Path(__file__).parent.parent
best_mean_reward, n_steps = -np.inf, 0
log_keys = ['episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 'episode_len_mean']


# TODO: Render this bloody scatter plot of control targets/vals!
# class CustomWandbLogger(WandbLogger):
#     def on_result(self, result: Dict):
#         res = super().on_result(result)
#         if 'custom_plots' in result:
#             for k, v in result['custom_plots'].items():
#                 wandb.log({k: v}, step=result['training_iteration'])


class PPOTrainer(RlLibPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # wandb.init(**self.config['wandb'])
        self.checkpoint_path_file = kwargs['config']['checkpoint_path_file']
        self.ctrl_metrics = self.config['env_config']['conditionals']
        cbs = self.workers.foreach_env(lambda env: env.unwrapped.cond_bounds)
        cbs = [cb for worker_cbs in cbs for cb in worker_cbs if cb is not None]
        cond_bounds = cbs[0]
        self.metric_ranges = {k: v[1] - v[0] for k, v in cond_bounds.items()}
        # self.checkpoint_path_file = checkpoint_path_file

    def setup(self, config):
        ret = super().setup(config)
        n_params = 0
        param_dict = self.get_weights()['default_policy']

        for v in param_dict.values():
            n_params += np.prod(v.shape)
        print(f'default_policy has {n_params} parameters.')
        print('model overview: \n', self.get_policy('default_policy').model)
        return ret

    @classmethod
    def get_default_config(cls):
        # def_cfg = super().get_default_config()
        def_cfg = RlLibPPOTrainer.get_default_config()
        def_cfg.update({
            'checkpoint_path_file': None,
            'wandb': {
                'project': 'PCGRL',
                'name': 'default_name',
                'id': 'default_id',
            },
        })
        return def_cfg

    def save(self, *args, **kwargs):
        ckp_path = super().save(*args, **kwargs)
        with open(self.checkpoint_path_file, 'w') as f:
            f.write(ckp_path)
        return ckp_path

    # @wandb_mixin
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        log_result = {k: v for k, v in result.items() if k in log_keys}
        log_result['info: learner:'] = result['info']['learner']

        # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
        result['fps'] = result['timesteps_this_iter'] / result['time_this_iter_s']

        # TODO: Send a heatmap to tb/wandb representing success reaching various control targets?
        if len(result['custom_metrics']) > 0:
            n_bins = 20
            result['custom_plots'] = {}
            for metric in self.ctrl_metrics:

                # Scatter plots via wandb
                # trgs = result['hist_stats'][f'{metric}-trg']
                # vals = result['hist_stats'][f'{metric}-val']
                # data = [[x, y] for (x, y) in zip(trgs, vals)]
                # table = wandb.Table(data=data, columns=['trg', 'val'])
                # scatter = wandb.plot.scatter(table, "trg", "val", title=f"{metric}-trg-val")
                # result['custom_plots']["scatter_{}".format(metric)] = scatter
                # scatter.save(f"{metric}-trg-val.png")
                # wandb.log({f'{metric}-scc': scatter}, step=self.iteration)

                # Spoofed histograms
                # FIXME: weird interpolation behavior here???
                bin_size = self.metric_ranges[metric] / n_bins  # 30 is the default number of tensorboard histogram bins (HACK)
                trg_dict = {}

                for i, trg in enumerate(result['hist_stats'][f'{metric}-trg']):
                    val = result['hist_stats'][f'{metric}-val'][i]
                    scc = 1 - abs(val - trg) / self.metric_ranges[metric]
                    trg_bin = trg // bin_size
                    if trg not in trg_dict:
                        trg_dict[trg_bin] = [scc]
                    else:
                        trg_dict[trg_bin] += [scc]
                # Get average success rate in meeting each target.
                trg_dict = {k: np.mean(v) for k, v in trg_dict.items()}
                # Repeat each target based on how successful we were in reaching it. (Appears at least once if sampled)
                spoof_data = [[trg * bin_size] * (1 + int(20 * scc)) for trg, scc in trg_dict.items()]
                spoof_data = [e for ee in spoof_data for e in ee]  # flatten the list
                result['hist_stats'][f'{metric}-scc'] = spoof_data

                # Make a heatmap.
                # ax, fig = plt.subplots(figsize=(10, 10))
                # data = np.zeros(n_bins)
                # for trg, scc in trg_dict.items():
                    # data[trg] = scc
                # wandb.log({f'{metric}-scc': wandb.Histogram(data, n_bins=n_bins)})

                # plt.imshow(data, cmap='hot')
                # plt.savefig(f'{metric}.png')

            

        # for k, v in result['hist_stats'].items():
            # if '-trg' in k or '-val' in k:
                # result['custom_metrics'][k] = [v]

        # print('-----------------------------------------')
        # print(pretty_print(log_result))
        return result


def main(cfg):
    # if (cfg.problem not in ["binary_ctrl", "binary_ctrl_holey", "sokoban_ctrl", "zelda_ctrl", "smb_ctrl", "MicropolisEnv", "RCT"]) and \
        # ("minecraft" not in cfg.problem):
        # raise Exception(
            # "Not a controllable environment. Maybe add '_ctrl' to the end of the name? E.g. 'sokoban_ctrl'")

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

    dummy_cfg = copy.copy(vars(cfg))
    dummy_cfg['render'] = False
    dummy_env = make_env(dummy_cfg)
    # check_env(dummy_env)

    # DEBUG ###
    for _ in range(10):
        obs = dummy_env.reset()
        for i in range(300):
            if i > 3:
                act = dummy_env.action_space.sample()
            else:
                act = 0
            dummy_env.step(act)
            dummy_env.render()
    sys.exit()

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
                # 'orig_obs_space': copy.copy(dummy_env.observation_space),
            }
        },
        "evaluation_config": {
            "explore": True,
        },
        "logger_config": {
                # "wandb": {
                    # "project": "PCGRL",
                    # "name": exp_name_id,
                    # "id": exp_name_id,
                    # "api_key_file": "~/.wandb_api_key"
            # },
            "type": "ray.tune.logger.TBXLogger",
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

    # NOTE: `ray.tune` now handles re-loading. Remove the below code when we're sure all the functionality has migrated
    #   over successfully.
    # Super ad-hoc re-loading. Note that we reset the number of training steps to be executed. Need to clearn this up if
    # we were to use it in actual publication-worthy experiments. Good for debugging though, maybe.
    # if cfg.load:
    #     trainer = Trainer(env='pcgrl', config=trainer_config)
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
        trainer_config.update({
            'record_env': log_dir,
            'explore': True,
        })
        trainer = PPOTrainer(env='pcgrl', config=trainer_config)
        with open(checkpoint_path_file, 'r') as f:
            checkpoint_path = f.read()
        
        # HACK (should probably be logging relative paths in the first place?)
        checkpoint_path = checkpoint_path.split('control-pcgrl/')[1]
        
        if not os.path.exists(checkpoint_path):
            assert os.path.exists(checkpoint_path[:-1]), f"Checkpoint path does not exist: {checkpoint_path}."
            checkpoint_path = checkpoint_path[:-1]
        trainer.load_checkpoint(checkpoint_path=checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}.")

        for _ in range(100):
            trainer.evaluate()

        # env = make_env(vars(cfg))
        # for i in range(10000):
        #     obs = env.reset()
        #     done = False
        #     time.sleep(0.5)
        #     while not done:
        #         action = trainer.compute_single_action(obs, explore=False)
        #         obs, reward, done, info = env.step(action)
        #         # print(env.unwrapped._rep_stats["path-length"])
        #         print(env.unwrapped._rep_stats)
        #         env.render()

        # Quit the program before agent starts training.
        sys.exit()

    best_mean_reward = -np.inf

    # TODO: makes this controllable, i.e., by dividing the number of frames by the train_batch_size
    n_updates = 10000

    # TODO: We've given the main loop over to `ray.tune`. Make sure we keep this functionality aroundn! In particular:
    #   the printout is ugly and verbose. And it would be nice to log `fps`. (Probably just add to PPOTrainer subclass
    #   and override the `train_step` or `print_result` methods?)
    # The training loop.
    # for i in range(n_updates):
    # def train_fn(config={}):
    #     result = trainer.train()
    #     log_result = {k: v for k, v in result.items() if k in log_keys}
    #     log_result['info: learner:'] = result['info']['learner']

    #     # FIXME: sometimes timesteps_this_iter is 0. Maybe a ray version problem? Weird.
    #     log_result['fps'] = result['timesteps_this_iter'] / result['time_this_iter_s']

    #     print('-----------------------------------------')
    #     print(pretty_print(log_result))

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

    class TrialProgressReporter(CLIReporter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.num_timesteps = []

        def should_report(self, trials, done=False):
            """Reports only on trial termination events."""
            old_num_timesteps = self.num_timesteps
            self.num_timesteps = [t.last_result['timesteps_total'] if 'timesteps_total' in t.last_result else 0 for t in trials]
            # self.num_terminated = len([t for t in trials if t.status == Trial.TERMINATED])
            done = np.any(self.num_timesteps > old_num_timesteps)
            return done

    # Limit the number of rows.
    reporter = TrialProgressReporter(
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

    # TODO: ray overwrites the current config with the re-loaded one. How to avoid this?
    analysis = tune.run(
        "CustomPPO",
        resume=cfg.load,
        config={
            **trainer_config,
        },
        checkpoint_score_attr="episode_reward_mean",
        
        checkpoint_at_end=True,
        checkpoint_freq=10,
        keep_checkpoints_num=3,
        local_dir=log_dir,
        verbose=1,
        # loggers=DEFAULT_LOGGERS + (WandbLogger, ),
        # **loggers_dict,
        **callbacks_dict,
        progress_reporter=reporter,
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
