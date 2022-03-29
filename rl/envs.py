import os
from pdb import set_trace as TT

from gym import spaces

from gym_pcgrl import wrappers, conditional_wrappers
#from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
# from utils import RenderMonitor, get_map_width

# def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
def make_env(cfg):
    """
    Return a function that will initialize the environment when called.
    """
    log_dir = cfg.get('log_dir')
    env_name = cfg.get('env_name')
    representation = cfg.get('representation')
    max_step = cfg.get('max_step', None)
    render = cfg.get('render', False)
    conditional = cfg.get('conditional', False)
    evaluate = cfg.get('evaluate', False)
    ALP_GMM = cfg.get('alp_gmm', False)
    map_width = cfg.get('map_width')
    # map_width = get_map_width(env_name)
    # kwargs['map_width'] = map_width
    n_cpu = cfg.pop('n_cpu')

    if representation == 'wide':
        env = wrappers.ActionMapImagePCGRLWrapper(env_name, **cfg)
    elif representation == 'wide3D':
        raise NotImplementedError("3D wide representation not implemented")
        # env = wrappers.ActionMapImage3DPCGRLWrapper(env_name, **cfg)
    elif representation == 'cellular':
        # env = wrappers.CAWrapper(env_name, **kwargs)
        env = wrappers.CAactionWrapper(env_name, **cfg)
    elif representation in ['narrow', 'turtle']:
        crop_size = cfg.get('cropped_size')
        env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **cfg)
    elif representation in ['narrow3D', 'turtle3D']:
        crop_size = cfg.get('cropped_size')
        env = wrappers.Cropped3DImagePCGRLWrapper(env_name, crop_size, **cfg)
    else:
        raise Exception('Unknown representation: {}'.format(representation))
    env.configure(**cfg)
    if max_step is not None:
        env = wrappers.MaxStep(env, max_step)
#   if log_dir is not None and cfg.get('add_bootstrap', False):
#       env = wrappers.EliteBootStrapping(env,
#                                           os.path.join(log_dir, "bootstrap{}/".format(rank)))
    env = conditional_wrappers.ConditionalWrapper(env, ctrl_metrics=cfg.pop('cond_metrics', []), **cfg)
    if not evaluate:
        if not ALP_GMM:
            env = conditional_wrappers.UniformNoiseyTargets(env, **cfg)
        else:
            env = conditional_wrappers.ALPGMMTeacher(env, **cfg)
    # it not conditional, the ParamRew wrapper should just be fixed at default static targets
#   if render or log_dir is not None and len(log_dir) > 0:
#       # RenderMonitor must come last
#       env = RenderMonitor(env, rank, log_dir, **kwargs)

    return env



