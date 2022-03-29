from gym_pcgrl import wrappers, conditional_wrappers
#from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from utils import RenderMonitor, get_map_width
from gym import spaces
from pdb import set_trace as TT

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    conditional = kwargs.get('conditional', False)
    evaluate = kwargs.get('evaluate', False)
    ALP_GMM = kwargs.get('alp_gmm', False)
    map_width = kwargs.get('map_width')

#   evo_compare = kwargs.get('evo_compare', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)

        elif representation == 'cellular':
           # env = wrappers.CAWrapper(env_name, **kwargs)
            env = wrappers.CAactionWrapper(env_name, **kwargs)
        elif representation in ['narrow', 'turtle']:
            crop_size = kwargs.get('cropped_size')
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        elif representation in ['narrow3D', 'turtle3D']:
            crop_size = kwargs.get('cropped_size')
            env = wrappers.Cropped3DImagePCGRLWrapper(env_name, crop_size, **kwargs)
        else:
            raise Exception('Unknown representation: {}'.format(representation))
#       if evo_compare:
#           # FIXME: THIS DOES NOT WORK

#           # Give a little wiggle room from targets, to allow for some diversity
#           if "binary" in env_name:
#               path_trg = env.unwrapped._prob.static_trgs['path-length']
#               env.unwrapped._prob.static_trgs.update({'path-length': (path_trg - 20, path_trg)})
#           elif "zelda" in env_name:
#               path_trg = env.unwrapped._prob.static_trgs['path-length']
#               env.unwrapped._prob.static_trgs.update({'path-length': (path_trg - 40, path_trg)})
#           elif "sokoban" in env_name:
#               sol_trg = env.unwrapped._prob.static_trgs['sol-length']
#               env.unwrapped._prob.static_trgs.update({'sol-length': (sol_trg - 10, sol_trg)})
#           elif "smb" in env_name:
#               pass
#           else:
#               raise NotImplementedError
        env.configure(**kwargs)
        if max_step is not None:
            env = wrappers.MaxStep(env, max_step)
        if log_dir is not None and kwargs.get('add_bootstrap', False):
            env = wrappers.EliteBootStrapping(env,
                                              os.path.join(log_dir, "bootstrap{}/".format(rank)))
        env = conditional_wrappers.ConditionalWrapper(env, ctrl_metrics=kwargs.pop('cond_metrics', []), **kwargs)
        if not evaluate:
            if not ALP_GMM:
                env = conditional_wrappers.UniformNoiseyTargets(env, **kwargs)
            else:
                env = conditional_wrappers.ALPGMMTeacher(env, **kwargs)
        # it not conditional, the ParamRew wrapper should just be fixed at default static targets
        if render or log_dir is not None and len(log_dir) > 0:
            # RenderMonitor must come last
            env = RenderMonitor(env, rank, log_dir, **kwargs)

        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    map_width = get_map_width(env_name)
    cropped_size = kwargs.get('cropped_size')
    cropped_size = map_width * 2 if cropped_size == -1 else cropped_size
    kwargs['cropped_size'] = cropped_size
    kwargs['map_width'] = map_width
    n_cpu = kwargs.pop('n_cpu')
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    # A hack :~)  Use a dummy env to get the action space
    dummy_env = make_env(env_name, representation, -1, None, **kwargs)()
    action_space = dummy_env.action_space
    if isinstance(action_space, spaces.Discrete):
        n_tools = action_space.n // (map_width ** 2)
    elif isinstance(action_space, spaces.MultiDiscrete):
        n_tools = action_space.nvec[2]
    elif isinstance(action_space, spaces.Box):
        n_tools = action_space.shape[0] // map_width ** 2
    else:
        raise Exception
    dummy_env.env.close()
    del(dummy_env)

    return env, action_space, n_tools