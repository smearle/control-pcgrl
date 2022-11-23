"""
Run a trained agent and get generated maps
"""
import model_sb2
from stable_baselines import PPO2

import time
from utils import make_vec_envs


def infer(game, representation, model_path, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    env_name = '{}-{}-v0'.format(game, representation)
    if game == "binary":
        model_sb2.FullyConvPolicy = model_sb2.FullyConvPolicyBigMap
        kwargs['crop_size'] = 28
    elif game == "zelda":
        model_sb2.FullyConvPolicy = model_sb2.FullyConvPolicyBigMap
        kwargs['crop_size'] = 22
    elif game == "sokoban":
        model_sb2.FullyConvPolicy = model_sb2.FullyConvPolicySmallMap
        kwargs['crop_size'] = 10
    # elif game == "minecraft_2Dmaze":
    #     model.FullyConvPolicy = model.FullyConvPolicyBigMap
    #     kwargs['crop_size'] = 28

    kwargs['render'] = True

    agent = PPO2.load(model_path)
    env = make_vec_envs(env_name, representation, None, 1, **kwargs)
    obs = env.reset()

    dones = False
    for i in range(kwargs.get('trials', 1)):
        while not dones:
            action, _ = agent.predict(obs)
            obs, _, dones, info = env.step(action)
            if kwargs.get('verbose', False):
                print(info[0])
            if dones:
                break
        # time.sleep(2)


################################## MAIN ########################################
# game = 'minecraft_3D_maze'
game = 'minecraft_3D_zelda'
representation = 'narrow3D'
exp_id = "0_1"
model_path = 'runs/{}_{}_{}_log/best_model.pkl'.format(game, representation, exp_id)

# game = 'binary'
# representation = 'narrow'
# exp_id = 1
# model_path = 'models/{}/{}/model_1.pkl'.format(game, representation, exp_id)
kwargs = {
    'change_percentage': 1,
    'trials': 100,
    'verbose': True
}

if __name__ == '__main__':
    infer(game, representation, model_path, **kwargs)
