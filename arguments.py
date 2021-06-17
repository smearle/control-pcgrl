import argparse
import json
from pdb import set_trace as T

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

    arg_dict = vars(opts)

    if opts.load_args is not None:
        with open('configs/rl/settings_{}.json'.format(opts.load_args)) as f:
            new_arg_dict = json.load(f)
            arg_dict.update(new_arg_dict)

        if opts.max_step == -1:
            opts.max_step = None

    if opts.conditionals == ["NONE"]:
        opts.conditionals = []
        opts.conditional = False
    elif opts.conditionals == ["DEFAULT"]:
        opts.conditionals = prob_cond_metrics[opts.problem]
    elif opts.conditionals == ["ALL"]:
        opts.conditionals = all_metrics[opts.problem]

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
        "--conditionals",
        nargs="+",
        help="Which game level metrics to use as conditionals for the generator",
        default=["NONE"],
    )
#   args.add_argument(
#       "--resume",
#       help="Are we resuming from a saved training run?",
#       action="store_true",
#   )
    args.add_argument(
        "--experiment_id",
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
    args.add_argument("--render", help="Render an environment?", action="store_true")
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
#   args.add_argument(
#       "--evo_compare",
#       help="Compare with work in evo-pcgrl using pyribs to train NNs as generators.",
#       action="store_true",
#   )

    # Not for training:
    args.add_argument(
        "--HPC",
        help='Load from "hpc_runs" (rather than "runs") directory.',
        action="store_true",
    )
    args.add_argument(
        "--n_frames",
        help="The net total number of gameplay frames to be experienced by the agent during training.",
        type=int,
        default=5e8,
    )
    args.add_argument(
        '-la',
        '--load_args',
        help='Rather than having the above args supplied by the command-line, load them from a settings.json file. (Of '
        'course, the value of this arg in the json will have no effect.)',
        type=int,
        default=None,
    )

    return args
