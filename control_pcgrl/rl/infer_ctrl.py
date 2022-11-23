"""
Run a trained agent for qualitative analysis.
"""
from pdb import set_trace as TT

import cv2
import numpy as np
from control_pcgrl.envs.helper import get_string_map

from args import parse_args
# from envs import make_vec_envs
from utils import get_crop_size, get_env_name, get_exp_name

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


def infer(game, representation, infer_kwargs, **kwargs):
    """
     - max_trials: The number of trials per evaluation.
     - infer_kwargs: Args to pass to the environment.
    """
    infer_kwargs = {**infer_kwargs, "inference": True, "render": True, "compute_stats": True}
    max_trials = kwargs.get("max_trials", -1)
#   n = kwargs.get("n", None)
    exp_id = infer_kwargs.get('exp_id')
    map_width = infer_kwargs.get("map_width")
    env_name = get_env_name(game, representation)
    exp_name = get_exp_name(game, representation, **infer_kwargs)

#   if n is None:
#       if EXPERIMENT_ID is None:
#           n = max_exp_idx(exp_name)
#       else:
#           n = EXPERIMENT_ID

#   if n == 0:
#       raise Exception(
#           "Did not find ranked saved model of experiment: {}".format(exp_name)
#       )
    crop_size = infer_kwargs.get("crop_size")

    if crop_size == -1:
        infer_kwargs["crop_size"] = get_crop_size(game)
#   log_dir = "{}/{}_{}_log".format(EXPERIMENT_DIR, exp_name, n)
    log_dir = "{}/{}_{}_log".format(EXPERIMENT_DIR, exp_name, exp_id)
    # no log dir, 1 parallel environment
    n_cpu = infer_kwargs.get("n_cpu")
    env, dummy_action_space, n_tools = make_vec_envs(
        env_name, representation, None, **infer_kwargs
    )
    print("loading model at {}".format(log_dir))
    model = load_model(
        log_dir, load_best=infer_kwargs.get("load_best"), n_tools=n_tools
    )
    if model is None:
        raise Exception("No model loaded")
    #   model.set_env(env)
    env.action_space = dummy_action_space
    obs = env.reset()
    # Record final values of each trial
    #   if 'binary' in env_name:
    #       path_lengths = []
    #       changes = []
    #       regions = []
    #       infer_info = {
    #           'path_lengths': [],
    #           'changes': [],
    #           'regions': [],
    #           }
    n_trials = 0
    n_step = 0

    while n_trials != max_trials:
        # action = get_action(obs, env, model)
        action, _ = model.predict(obs)
        obs, rewards, dones, info = env.step(action)

        #       print('reward: {}'.format(rewards))
        #       reward = rewards[0]
        #       n_regions = info[0]['regions']
        #       readouts = []
        #       if 'binary' in env_name:
        #           curr_path_length = info[0]['path-length']
        #           readouts.append('path length: {}'.format(curr_path_length) )
        #           path_lengths.append(curr_path_length)
        #           changes.append(info[0]['changes'])
        #           regions.append(info[0]['regions'])

        #       readouts += ['regions: {}'.format(n_regions), 'reward: {}'.format(reward)]
        #       stringexec = ""
        #       m=0
        #       y0, dy = 50, 40
        #       img = np.zeros((256,512,3), np.uint8)
        #       scale_percent = 60 # percent of original size
        #       width = int(img.shape[1] * scale_percent / 100)
        #       height = int(img.shape[0] * scale_percent / 100)
        #       dim = (width, height)
        #       # resize image
        #       for i, line in enumerate(readouts):
        #           y = y0 + i*dy
        #           cv2.putText(img, line, (50, y), font, fontScale, fontColor, lineType)
        #          #stringexec ="cv2.putText(img, TextList[" + str(TextList.index(i))+"], (100, 100+"+str(m)+"), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 100, 100), 1, cv2.LINE_AA)\n"
        #          #m += 100
        #       #cv2.putText(
        #       #    img,readout,
        #       #    topLeftCornerOfText,
        #       #    font,
        #       #    fontScale,
        #       #    fontColor,
        #       #    lineType)
        #       #Display the image
        #       resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #       cv2.imshow("img",resized)
        #       cv2.waitKey(1)
        #      #for p, v in model.get_parameters().items():
        #      #    print(p, v.shape)
        n_step += 1

        if dones:
            env.reset()
            #          #show_state(env, path_lengths, changes, regions, n_step)
            #           if 'binary' in env_name:
            #               infer_info['path_lengths'] = path_lengths[-1]
            #               infer_info['changes'] = changes[-1]
            #               infer_info['regions'] = regions[-1]
            n_step = 0
            n_trials += 1

    #       print(env.envs[0].metrics)
    #       print(n_step)


opts = parse_args()

# For locating trained model
global EXPERIMENT_ID
global EXPERIMENT_DIR
# EXPERIMENT_DIR = 'hpc_runs/runs'

if not opts.HPC:
    EXPERIMENT_DIR = "../rl_runs"
else:
    EXPERIMENT_DIR = "hpc_runs"
EXPERIMENT_ID = opts.exp_id
problem = opts.problem
representation = opts.representation
conditional = len(opts.controls) > 0
midep_trgs = opts.midep_trgs
ca_action = opts.ca_action
alp_gmm = opts.alp_gmm
kwargs = {
    'map_width': opts.map_width,
    # 'change_percentage': 1,
    # 'target_path': 105,
    # 'n': 4, # rank of saved experiment (by default, n is max possible)
}

#if problem == "sokobangoal":
#    map_width = 5
#else:
#    map_width = 16

max_step = opts.max_step
# if max_step is None:
#    max_step = 1000
change_percentage = opts.change_percentage

if conditional:
    cond_metrics = opts.controls

    if ca_action:
        max_step = 50
#   change_percentage = 1.0

else:
    cond_metrics = None

# For inference
infer_kwargs = {
    "change_percentage": change_percentage,
    # 'target_path': 200,
    "conditional": conditional,
    "cond_metrics": cond_metrics,
    "max_step": max_step,
    "render": True,
    "n_cpu": 1,
    "load_best": opts.load_best,
    "midep_trgs": midep_trgs,
    "infer": True,
    "ca_action": ca_action,
    "map_width": opts.map_width,
    "crop_size": opts.crop_size,
    "alp_gmm": alp_gmm,
    "exp_id": opts.exp_id,
}

if __name__ == "__main__":
    infer(problem, representation, infer_kwargs, **kwargs)
#   evaluate(test_params, game, representation, experiment, infer_kwargs, **kwargs)
#   analyze()
