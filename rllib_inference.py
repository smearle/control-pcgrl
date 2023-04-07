'''
Mostly deprecated, just use some of the fitness functions inside
'''
import argparse
import copy
import json
import os
import uuid

from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import gymnasium as gym
from gymnasium.spaces import Tuple

from ray import air, tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.algorithms.qmix import QMix
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from control_pcgrl.configs.config import Config
from control_pcgrl import wrappers
from control_pcgrl.rl import models
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.rllib_utils import ControllableTrainerFactory as trainer_factory


def load_config(experiment_path):
    with open(Path(experiment_path, 'params.json'), 'r') as f:
        config = json.load(f)
    # override multiagent policy mapping function

    if 'multiagent' in config:
        if 'default_policy' in config:
            config['multiagent']['policy_mapping_fn'] = lambda agent_id: 'default_policy'
        else:
            config['multiagent']['policy_mapping_fn'] = lambda agent_id: agent_id
        config['env_config']['multiagent'] = json.loads(config['env_config']['multiagent'].replace("\'", "\""))

    config['evaluation_env'] = True
    config['explore'] = False # turn off expl.loadoration for evaluation
    config['env_config']['obs_window'] = json(config['env_config']['obs_window'])
    config['env_config']['problem'] = json.loads(config['env_config']['problem'].replace("\'", "\""))
    
    env_name = config['env_config']['env_name']
    return config


def setup_multiagent_config(config, model_cfg):
    dummy_env = make_env(config)
    obs_space = dummy_env.observation_space['agent_0']
    act_space = dummy_env.action_space['agent_0']
    multiagent_config = {}
    if config['multiagent']['policies'] == "centralized":
        multiagent_config['policies'] = {
            'default_policy': PolicySpec(
                policy_class=None,
                observation_space=obs_space,
                action_space=act_space,
                config={
                    'custom_model': 'custom_model',
                    'custom_model_config': {
                        "dummy_env_obs_space": copy.copy(obs_space),
                        **json.loads(model_cfg.replace('\'', '\"')),
                    }
                }
            )
        }
        multiagent_config['policy_mapping_fn'] = lambda agent_id: 'default_policy'
    elif config['multiagent']['policies'] == "decentralized":
        multiagent_config['policies'] = {
            f'agent_{i}': PolicySpec(
                policy_class=None,
                observation_space=obs_space,
                action_space=act_space,
                config={
                    'custom_model': 'custom_model',
                    'custom_model_config': {
                        "dummy_env_obs_space": copy.copy(obs_space),
                        **json.loads(model_cfg.replace('\'', '\"')),
                    }
                }
            ) for i in range(config['multiagent']['n_agents'])
        }
        multiagent_config['policy_mapping_fn'] = lambda agent_id: agent_id
    return multiagent_config

def checkpoints_iter(experiment_path):
    experiment_path = Path(experiment_path)
    return filter(lambda f: 'checkpoint' in f.name, experiment_path.iterdir())


def get_best_checkpoint(experiment_path, config):
    # load progress.csv
    progress = pd.read_csv(Path(experiment_path, 'progress.csv'))

    max_episode_reward = float('-inf')
    max_checkpoint = None
    max_checkpoint_name = None
    for checkpoint in checkpoints_iter(experiment_path):
        # get number after underscore in checkpoint
        trainer = restore_best_ckpt(log_dir)
        iteration = trainer._iteration
        # look up iteration in progress dataframe
        trainer_performance = progress.loc[progress['training_iteration'] == iteration]
        trainer_reward = trainer_performance['episode_reward_mean'].values[0]
        # sometimes the first checkpoint has a nan reward
        if np.isnan(trainer_reward) or trainer_reward > max_episode_reward:
            max_episode_reward = float('-inf') if np.isnan(trainer_reward) else trainer_reward
            max_checkpoint = trainer
            max_checkpoint_name = checkpoint
    print(f'Loaded from checkpoint: {max_checkpoint_name}')
    return max_checkpoint


def restore_best_ckpt(log_dir):
    tuner = tune.Tuner.restore(log_dir)
    best_result = tuner.get_results().get_best_result()
    ckpt = best_result.best_checkpoints[0][0]
    return ckpt


def init_trainer(config):
    config.pop('checkpoint_path_file') # huh?
    if config['env_config']['algorithm'] == 'QMIX':
        trainer = QMix(config=config)
    else:
        trainer = PPOTrainer(config=config)
    return trainer


def register_model(config):
    MODELS = {"NCA": models.NCA, "DenseNCA": models.DenseNCA, "SeqNCA": models.SeqNCA, "SeqNCA3D": models.SeqNCA3D}
    model_conf_str = config['env_config']['model'].replace('\'', '\"')
    model_name_default = model_conf_str.find('None')
    if model_name_default > 0:
        model_conf_str = model_conf_str[:model_name_default-1] + f' \"None\"' + model_conf_str[model_name_default+4:]
    model_config = json.loads(model_conf_str)
    if model_config.get('name') == "None":
        if config['env_config']['representation'] == 'wide':
            model_cls = models.ConvDeconv2D
        else:
            model_cls = models.CustomFeedForwardModel
    else:
        model_cls = MODELS[model_config['name']]
    ModelCatalog.register_custom_model('custom_model', model_cls)


def rollout(env_config, trainer, policy_mapping_fn=None, seed=None):
    env = make_env(env_config)
    env.seed(seed)
    env.reset()
    env.seed(seed)
    #env.unwrapped._max_iterations *= 2
    obs = env.reset()
    done = False
    acts, obss, rews, infos, frames = [], [], [], [], []
    while not done:
        if policy_mapping_fn is not None:
            actions = get_multi_agent_actions(trainer, obs, policy_mapping_fn)
            acts.append({agent: int(act) for agent, act in actions.items()})
        elif env_config['algorithm'] == 'QMIX':
            actions = get_qmix_actions(trainer, obs)
            acts.append({agent: int(act) for agent, act in actions.items()})
        else:
            actions = get_single_agent_actions(trainer, obs)
            acts.append({'agent_0': int(actions)})
            
        # build action histogram
        obs, rew, done, info = env.step(actions)
        #import pdb; pdb.set_trace()
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        rews.append(rew)
        infos.append(int(env.unwrapped._rep_stats['path-length']))
        #infos.append(info)
        if isinstance(done, dict):
            done = done['__all__']
    
    return {
        'actions': acts,
        'rewards': rews,
        'infos': infos,
        'frames': frames,
        'success': env.unwrapped._prob.get_episode_over(env.unwrapped._rep_stats, None),
        'heatmaps': env.unwrapped._rep.heatmaps
    }


def save_trial_metrics(metrics, logdir):
    # save initial frame, final frame, and gif of frames
    imageio.imsave(Path(logdir, 'initial_map.png'), metrics['frames'][0])
    imageio.imsave(Path(logdir, 'final_map.png'), metrics['frames'][-1])
    imageio.mimsave(Path(logdir, 'frames.gif'), metrics['frames'])
    # save rewards in json file
    with open(Path(logdir, 'rewards.json'), 'w+') as f:
        f.write(json.dumps(metrics['rewards']))
    # graph rewards over time
    # save infos in json file
    with open(Path(logdir, 'infos.json'), 'w+') as f:
        f.write(json.dumps(metrics['infos']))
    # plot path length over time
    # save actions in json file
    with open(Path(logdir, 'actions.json'), 'w+') as f:
        f.write(json.dumps(list(metrics['actions'])))

    # check success
    with open(Path(logdir, 'success.json'), 'w+') as f:
        f.write(json.dumps({'success': bool(metrics['success'])}))

    for i, heatmap in enumerate(metrics['heatmaps']):
        fig, ax = plt.subplots()
        im = ax.imshow(heatmap)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('changes', rotation=-90, va="bottom")
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        fig.savefig(Path(logdir, f'{i}_heatmap.png'), dpi=400)
        plt.close(fig) # close figure to prevent memory issues


def get_qmix_actions(trainer, observations):
    actions = trainer.compute_single_action(tuple(observations.values()))
    return {agent: action for agent, action in zip(observations.keys(), actions)}


def get_single_agent_actions(trainer, observations):
    return trainer.compute_single_action(observations)

def get_multi_agent_actions(trainer, observations, policy_mapping_fn):
    return {
        agent_id: trainer.compute_single_action(agent_obs, policy_id=policy_mapping_fn(agent_id))
        for agent_id, agent_obs in observations.items()
    }

def make_grouped_env(config):
    
    try:
        n_agents = config['multiagent']['n_agents']
    except:
        n_agents = json.loads(config['multiagent'].replace('\'', '\"'))['n_agents']
    dummy_env = make_env(config)
    groups = {'group_1': list(dummy_env.observation_space.keys())}
    obs_space = Tuple(dummy_env.observation_space.values())
    act_space = Tuple(dummy_env.action_space.values())
    #import pdb; pdb.set_trace()
    register_env(
        'grouped_pcgrl',
        lambda config: wrappers.GroupedEnvironmentWrapper(make_env(config).with_agent_groups(
            groups, obs_space=obs_space, act_space=act_space))
        
    )

# run evals with the checkpoint
def evaluate(experiment_path):
    # load and setup config
    config = load_config(experiment_path)
    if 'multiagent' in config:
        config['multiagent'] = setup_multiagent_config(config['env_config'], config['env_config']['model'])
    # delete keys not recognized by rllib
    del config['checkpoint_path_file']
    del config['evaluation_env']
    del config['callbacks']
    del config['num_workers']
    del config['num_envs_per_worker']
    #del config['multiagent']
    #import pdb; pdb.set_trace()
    if config['env_config']['algorithm'] == 'PPO':
        register_env('pcgrl', make_env)
    else:
        make_grouped_env(config['env_config'])
        #register_env('grouped_pcgrl', make_grouped_env)
    config['num_gpus'] = 0
    register_model(config)
    # load trainer from checkpoint
    trainer = get_best_checkpoint(experiment_path, config)
    # rollout the model for n trials
    logdir = Path(experiment_path, f'eval_best_{uuid.uuid4()}')
    logdir.mkdir()

    try:
        policy_mapping_fn = config['multiagent']['policy_mapping_fn']
        breakpoint()
    except KeyError:
        policy_mapping_fn = None

    paths = []
    max_changes = 0
    for trial in tqdm(range(40)):
        results = rollout(config['env_config'], trainer, policy_mapping_fn, seed=trial*100)
        #results = rollout(config['env_config'], trainer, config['multiagent']['policy_mapping_fn'], seed=trial*100)
        trial_log_dir = Path(logdir, f'{trial}')
        trial_log_dir.mkdir()
        paths.append(results['infos'][-1])
        #changes.append(results['infos'][-1]['changes'] / results['infos'][-1]['iterations'])
        save_trial_metrics(results, trial_log_dir)

    print(f'Wrote logs to: {logdir}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--experiment_path',
            '-e',
            dest='experiment_path',
            type=str,
            required=True
            )

    #parser.add_argument('checkpoint_loader') # just load the best for now
    args = parser.parse_args()
    evaluate(Path(args.experiment_path))
