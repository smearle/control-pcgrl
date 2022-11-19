"""Command-line arguments for the reinforcement learning algorithm launched by running `python train_ctrl.py`."""
import argparse
import json
import sys
from pdb import set_trace as TT

prob_cond_metrics = {
    "binary_ctrl": ["regions", "path-length"],
    "zelda_ctrl": ["nearest-enemy", "path-length"],
    "sokoban_ctrl": ["crate", "sol-length"],
    "MicropolisEnv": ["res_pop"],
    "RCT": ["income"],
}

all_metrics = {
    "binarygoal": ["regions", "path-length"],
    "zeldagoal": [
        "player",
        "key",
        "door",
        "enemies",
        "regions",
        "nearest-enemy",
        "path-length",
    ],
    "sokobangoal": ["player", "crate", "sol-length"],
}


def parse_args():
    args = get_args()

    return parse_pcgrl_args(args)


def parse_pcgrl_args(args):
    opts = args.parse_args()
    opts.conditional = True
    opts.model_cfg = {}
    opts_dict = vars(opts)

    if opts.load_args is not None:
        with open(f'configs/rl/auto/settings_{opts.load_args}.json') as f:
            new_arg_dict = json.load(f)
            opts_dict.update(new_arg_dict)

        if opts.max_step == -1:
            opts.max_step = None
    if opts.controls == []:
        opts.conditional = False
    elif opts.controls == ["DEFAULT"]:
        opts.controls = prob_cond_metrics[opts.problem]
    elif opts.controls == ["ALL"]:
        opts.controls = all_metrics[opts.problem]

    return opts


def get_args():
    args = argparse.ArgumentParser(description="Conditional PCGRL")
    args.add_argument(
        "-p",
        "--problem",
        help="which problem (i.e. game) to generate levels for (binary, sokoban, zelda, mario, ..."
        "roller coaster tycoon, simcity???)",
        default="binary_ctrl",
    )
    args.add_argument(
        "-r",
        "--representation",
        help="Which representation to use (narrow, turtle, wide, ... cellular-automaton???)",
        default="narrow",
    )
    args.add_argument(
        "-mo",
        "--model",
        type=str,
        help="Which model to use.",
        default=None,
    )
    args.add_argument(
        "-mw",
        "--map_width",
        help="Width of the game level.",
        type=int,
        default=None,
    )
    args.add_argument(
        "-ca",
        "--ca_action",
        help="Cellular automaton-type action. The entire next game state is sampled from the model output.",
        action="store_true",
    )
    args.add_argument(
        "-c",
        "--controls",
        nargs="+",
        help="Which game level metrics to use as controls for the generator",
        default=None,
    )
#   opts.add_argument(
#       "--resume",
#       help="Are we resuming from a saved training run?",
#       action="store_true",
#   )
    args.add_argument(
        "--exp_id",
        help="An experiment ID for tracking different runs of experiments with identical hyperparameters.",
        type=int, 
        default=0,
    )
    args.add_argument(
        "--midep_trgs",
        help="Do we sample new (random) targets mid-episode, or nah?",
        action="store_true",
    )
    args.add_argument(
        "--n_cpu",
        help="How many environments to run in parallel.",
        type=int,
        default=50,
    )
    args.add_argument(
        "--n_gpu",
        help="How many GPUs to use.",
        type=int,
        default=1,
    )
    args.add_argument("--render",
        help="Render an environment?", 
        action="store_true"
    )
    args.add_argument(
        "--load_best",
        help="Whether to load the best saved model of a given run rather than the latest.",
        action="store_true",
    )
    args.add_argument(
        "--crop_size",
        help="Hackishly control crop size of agent observation.",
        type=int,
        default=-1,
    )
    args.add_argument(
        "--change_percentage",
        help="How much of the content can an agent can edit before episode termination. Between 0 and 1.",
        type=float,
        default=0.2,
    )
    args.add_argument(
        "--max_step",
        help="How many steps in an episode, maximum.",
        type=int,
        default=-1,
    )
    args.add_argument(
        "--alp_gmm",
        help="Fancy ish teacher algorithm for controllable targets.",
        action="store_true",
    )
#   opts.add_argument(
#       "--evo_compare",
#       help="Compare with work in evo-pcgrl using pyribs to train NNs as generators.",
#       action="store_true",
#   )
    
    args.add_argument(
        "--n_frames",
        help="The net total number of gameplay frames to be experienced by the agent during training.",
        type=int,
        default=5e8,
    )
    args.add_argument(
        "-la",
        "--load_args",
        help='Rather than having the above opts supplied by the command-line, load them from a settings.json file. (Of '
        'course, the value of this arg in the json will have no effect.)',
        type=str,
        default=None,
    )
    args.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite previous experiment with same name."
    )
    args.add_argument(
        "--cuda",
        help="Whether to use CUDA (GPU) or not.",
        action="store_true",
    )
    args.add_argument(
        '--wandb',
        help='Whether to use wandb for logging.',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args.add_argument(
        '--record_env',
        help='Whether to record the environment during inference.',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args.add_argument(
        '--max_board_scans',
        help='Number of max iterations in terms of maximum number of times the board can be scanned by the agent.',
        type=int,
        default=1,
    )
    args.add_argument(
        '--n_aux_tiles',
        help='Number of auxiliary tiles to use (for the agent to leave itself messages for later).',
        type=int,
        default=0,
    )
    args.add_argument(
        '--lr',
        help='Learning rate for the agent.',
        type=float,
        default=5e-6,
    )
    args.add_argument(
        '--load',
        help='Whether to load a model from a checkpoint.',
        action='store_true',
    )
    args.add_argument(
        '--evaluate',
        help='Whether to evaluate a model.',
        action='store_true',
    )
    args.add_argument(
        '--infer',
        help='Whether to do inference with a model.',
        action='store_true',
    )
    args.add_argument(
        "-ga",
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor of the MDP."
    )

    return args
