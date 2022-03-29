#pip install tensorflow==1.15
#Install stable-baselines as described in the documentation

import model_sb2
from model_sb2 import FullyConvPolicy, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from utils import get_exp_name, max_exp_idx, load_model, make_vec_envs
#from stable_baselines import PPO2
from policy import PPO2
from stable_baselines.results_plotter import load_results, ts2xy

import tensorflow as tf
import numpy as np
import os
from pdb import set_trace as TT

n_steps = 0
log_dir = './'
best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if n_steps % 10 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        # if len(x) > 100:
        #pdb.set_trace()
        mean_reward = np.mean(y[-100:])
        ti = 0 if len(x) == 0 else x[-1]
        print(f'{ti} timesteps')
        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

        # New best model, we save the agent here
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
        else:
            print("Saving latest model")
            _locals['self'].save(os.path.join(log_dir, 'latest_model.pkl'))
#   else:
#       print('{} monitor entries'.format(len(x)))
#       pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, **kwargs):
    kwargs['n_cpu'] = n_cpu
    env_name = '{}-{}-v0'.format(game, representation)
    exp_name = get_exp_name(game, representation, experiment, **kwargs)
    resume = kwargs.get('resume', False)
    if representation == 'wide':
        policy = FullyConvPolicy
        if game == "sokoban":
            policy = FullyConvPolicySmallMap
    else:
        policy = CustomPolicyBigMap
        if game == "sokoban":
            policy = CustomPolicySmallMap
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10

    # What's the ID of the latest log directory corresponding to this experiment?
    n = max_exp_idx(exp_name)

    # If reloading, load the latest model
    if resume:
        old_log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')
        model = load_model(old_log_dir)
        n = n + 1

    # Name of the new log directory to be created and logged to
    global log_dir
    log_dir = 'runs/{}_{}_{}'.format(exp_name, n, 'log')

    # If the new log directory exists, we have a problem
    if os.path.isdir(log_dir):
        raise Exception(f'Log directory {log_dir} already exists.')

    # Create the new log directory
    os.makedirs(log_dir)
    # log_dir = 'runs/{}_{}'.format(exp_name, 'log')
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
    }
    used_dir = log_dir
    if not logging:
        used_dir = None
    env = make_vec_envs(env_name, representation, log_dir, **kwargs)
    if not resume or model is None:
        model = PPO2(policy, env, verbose=1, tensorboard_log="./runs")
    else:
        model.set_env(env)

    n_params = 0
    for param in model.params:
        n_params += np.prod(param.shape)
    print(f'Model has {n_params} parameters.')

    if not logging:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name)
    else:
        model.learn(total_timesteps=int(steps), tb_log_name=exp_name, callback=callback)

################################## MAIN ########################################
# game = 'minecraft_3D_maze'
game = 'minecraft_3D_zelda'
representation = ['narrow3D']
experiment = 'logTest'
steps = 1e8
render = False
logging = True
n_cpu = 20
kwargs = {
    'resume': True
}

if __name__ == '__main__':
    for repre in representation:
        main(game, repre, experiment, steps, n_cpu, render, logging, **kwargs)
