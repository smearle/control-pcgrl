from copy import copy
import argparse
from pathlib import Path
import uuid
from tqdm import tqdm
from pathlib import Path
import json
import imageio
import pandas as pd
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog
from control_pcgrl.rl import models
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.rllib_utils import PPOTrainer

def load_config(experiment_path):
    with open(Path(experiment_path, 'params.json'), 'r') as f:
        config = json.load(f)
    # override multiagent policy mapping function

    if 'default_policy' in config:
        config['multiagent']['policy_mapping_fn'] = lambda agent_id: 'default_policy'
    else:
        config['multiagent']['policy_mapping_fn'] = lambda agent_id: agent_id

    config['evaluation_env'] = True
    config['explore'] = False # turn off exploration for evaluation
    config['env_config']['multiagent'] = json.loads(config['env_config']['multiagent'].replace("\'", "\""))
    config['env_config']['crop_shape'] = json.loads(config['env_config']['crop_shape'])
    config['env_config']['problem'] = json.loads(config['env_config']['problem'].replace("\'", "\""))
    
    env_name = config['env_config']['env_name']
    return config

def setup_multiagent_config(config, model_cfg):
    dummy_env = make_env(config)
    #import pdb; pdb.set_trace()
    obs_space = dummy_env.observation_space['agent_0']
    act_space = dummy_env.action_space['agent_0']
    multiagent_config = {}
    if config['multiagent']['policies'] == "centralized":
        multiagent_config['policies'] = {
            'default_policy': PolicySpec(
                policy_class=None,
                observation_space=obs_space,
                action_space=act_space,
                config=None 
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
                        **model_cfg,
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
        trainer = restore_trainer(Path(checkpoint), config)
        iteration = trainer._iteration
        # look up iteration in progress dataframe
        trainer_performance = progress.loc[progress['training_iteration'] == iteration]
        trainer_reward = trainer_performance['episode_reward_mean'].values[0]
        if trainer_reward > max_episode_reward:
            max_episode_reward = trainer_reward
            max_checkpoint = trainer
            max_checkpoint_name = checkpoint
    print(f'Loaded from checkpoint: {max_checkpoint_name}')
    return max_checkpoint

def restore_trainer(checkpoint_path, config):
    #trainer = qmix.QMixTrainer(config=config)
    trainer = ppo.PPOTrainer(config=config)
    print(checkpoint_path)
    trainer.restore(str(checkpoint_path))
    return trainer

def register_model(config):
    MODELS = {"NCA": models.NCA, "DenseNCA": models.DenseNCA, "SeqNCA": models.SeqNCA, "SeqNCA3D": models.SeqNCA3D}
    if config['model'].get('name', None) is None:
        if config['env_config']['representation'] == 'wide':
            model_cls = models.ConvDeconv2D
        else:
            model_cls = models.CustomFeedForwardModel
    else:
        model_cls = MODELS[config['model']['name']]
    ModelCatalog.register_custom_model('custom_model', model_cls)

def rollout(env_config, trainer, policy_mapping_fn, seed=None):
    env = make_env(env_config)
    env.seed(seed)
    env.reset()
    env.seed(seed)
    obs = env.reset()
    done = False
    acts, obss, rews, infos, frames = [], [], [], [], []
    while not done:
        actions = get_agent_actions(trainer, obs, policy_mapping_fn)
        obs, rew, done, info = env.step(actions)
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        acts.append({agent: int(act) for agent, act in actions.items()})
        rews.append(rew)
        done = done['__all__']
    
    return {
        'actions': acts,
        'rewards': rews,
        'infos': infos,
        'frames': frames,
        'success': env.unwrapped._prob.get_episode_over(env.unwrapped._rep_stats, None)
    }

def save_metrics(metrics, logdir):
    # save initial frame, final frame, and gif of frames
    imageio.imsave(Path(logdir, 'initial_map.png'), metrics['frames'][0])
    imageio.imsave(Path(logdir, 'final_map.png'), metrics['frames'][-1])
    imageio.mimsave(Path(logdir, 'frames.gif'), metrics['frames'])
    # save rewards in json file
    with open(Path(logdir, 'rewards.json'), 'w+') as f:
        json.dumps(metrics['rewards'])
    # graph rewards over time
    # save infos in json file
    with open(Path(logdir, 'infos.json'), 'w+') as f:
        json.dumps(metrics['infos'])
    # plot path length over time
    # save actions in json file
    with open(Path(logdir, 'actions.json'), 'w+') as f:
        json.dumps(list(metrics['actions']))

    # check success
    with open(Path(logdir, 'success.json'), 'w+') as f:
        json.dumps({'success': bool(metrics['success'])})

def get_agent_actions(trainer, observations, policy_mapping_fn):
    return {
        agent_id: trainer.compute_single_action(agent_obs, policy_id=policy_mapping_fn(agent_id))
        for agent_id, agent_obs in observations.items()
    }

# run evals with the checkpoint
def evaluate(experiment_path):
    # load and setup config
    config = load_config(experiment_path)
    config['multiagent'] = setup_multiagent_config(config['env_config'], config['model'])
    # delete keys not recognized by rllib
    del config['checkpoint_path_file']
    del config['evaluation_env']
    del config['callbacks']
    register_env('pcgrl', make_env)
    config['num_gpus'] = 0
    register_model(config)
    # load trainer from checkpoint
    trainer = get_best_checkpoint(experiment_path, config)
    # rollout the model for n trials
    logdir = Path(experiment_path, f'eval_best_{uuid.uuid4()}')
    logdir.mkdir()
    for trial in tqdm(range(40)):
        results = rollout(config['env_config'], trainer, config['multiagent']['policy_mapping_fn'], seed=trial*100)
        trial_log_dir = Path(logdir, f'{trial}')
        trial_log_dir.mkdir()
        save_metrics(results, trial_log_dir)
        


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
