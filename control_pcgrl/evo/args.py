"""Command-line arguments for the evolutionary algorithm launched by running `python evolve.py`."""
import pathlib
from pdb import set_trace as TT
import os
import sys
import argparse 
import json


def get_args(load_args=None):
    if load_args is not None:
        sys.argv = sys.argv[:1]
    opts = argparse.ArgumentParser(
        description="Evolving Neural Cellular Automata for PCGRL"
    )
    opts.add_argument(
        "-p",
        "--problem",
        help='Which game to generate levels for (PCGRL "problem").',
        default="binary_ctrl",
    )
    opts.add_argument(
        "-e",
        "--exp_name",
        help="Name of the experiment, for save files.",
        default="test_0",
    )
    opts.add_argument(
        "-ng",
        "--n_generations",
        type=int,
        help="Number of generations for which to run evo.",
        default=10000,
    )
    opts.add_argument(
        "-si",
        "--save_interval",
        type=int,
        help="Number of generations after which to save.",
        default=10,
    )
    opts.add_argument(
        "-nis",
        "--n_init_states",
        help="The number of initial states on which to evaluate our models. 0 for a single fixed map with a square of wall in the centre.",
        type=int,
        default=10,
    )
    opts.add_argument(
        "-ns",
        "--n_steps",
        help="Maximum number of steps in each generation episode. Only applies to NCA model and cellular"
        "representation at the moment.",
        type=int,
        default=10,
    )
    opts.add_argument(
        "-bcs",
        "--behavior_characteristics",
        nargs="+",
        help="A list of strings corresponding to the behavior characteristics that will act as the dimensions for our grid of elites during evo.",
        default=["NONE", "NONE"],
    )
    opts.add_argument(
        "-r", "--render", help="Render the environment.", action="store_true"
    )
    opts.add_argument(
        "-i",
        "--infer",
        help="Run inference with evolved archive of individuals.",
        action="store_true",
    )
    opts.add_argument(
        "-v",
        "--visualize",
        help="Visualize heatmap of the archive of individuals.",
        action="store_true",
    )
    opts.add_argument(
        "--show_vis",
        help="Render visualizations in matplotlib rather than saving them to png.",
        action="store_true",
    )
    opts.add_argument(
        "-g", "--render_levels", help="Save grid of levels to png.", action="store_true"
    )
    opts.add_argument(
        "-m",
        "--multi_thread",
        help="Use multi-thread evo process.",
        type=bool,
        default=False,
    )
    opts.add_argument(
        "--play_level",
        help="Use a playing agent to evaluate generated levels.",
        action="store_true",
    )
    opts.add_argument(
        "-ev",
        "--evaluate",
        help="Run evaluation on the archive of elites.",
        action="store_true",
    )
    opts.add_argument(
        "-s", "--save_levels", help="Save all levels to a csv.", action="store_true"
    )
    opts.add_argument(
        "--fix_level_seeds",
        help="Use a fixed set of random levels throughout evo, rather than providing the generators with new random initial levels during evaluation.",
        action="store_true",
    )
    opts.add_argument(
        "-cr",
        "--cascade_reward",
        help="Incorporate diversity/variance bonus/penalty into fitness only if targets are met perfectly (rather than always incorporating them).",
        action="store_true",
    )
    opts.add_argument(
        "-rep",
        "--representation",
        help="The interface between generator-agent and environment. cellular: agent acts as cellular automaton, observing and"
        " supplying entire next stats. wide: agent observes entire stats, and changes any one tile. narrow: agent "
        "observes state and target tile, and selects built at target tile. turtle: agent selects build at current "
        "tile or navigates to adjacent tile.",
        default="cellular",
    )
    opts.add_argument(
        "-la",
        "--load_args",
        help="Rather than having the above opts supplied by the command-line, load them from a settings.json file. (Of "
        "course, the value of this arg in the json will have no effect.)",
        type=int,
        default=None,
    )
    opts.add_argument(
        "--model",
        help="Which neural network architecture to use for the generator. NCA: just conv layers. CNN: Some conv layers, then a dense layer.",
        default="NCA",
    )
    opts.add_argument(
        "--fix_elites",
        help="(Do not) re-evaluate the elites on new random seeds to ensure their generality.",
        action="store_true",
    )
    opts.add_argument(
        "--save_gif",
        help="Save screenshots (and gif?) of level during agent generation process.",
        action="store_true",
    )
    opts.add_argument(
        "--n_cpu",
        type=int,
        default=None,
    )
    opts.add_argument(
        "--algo",
        help="Which evolutionary algorithm to run. (CMAME, ME)",
        default="CMAME",
    )
    opts.add_argument(
        "-ss",
        "--step_size",
        help="Standard deviation of the Gaussian noise added to mutated models.",
        type=float,
        default=1.00,
    )
    opts.add_argument(
        '--n_aux_chan',
        help='Number of auxiliary channels for NCA-type hidden activations.',
        type=int,
        default=0,
    )
    opts.add_argument("--mega", help="Use CMA-MEGA.", action="store_true")

    args = opts.parse_args()
    arg_dict = vars(args)

    if load_args is not None:
        arg_dict.update(load_args)  
    if args.load_args is not None:
        with open("configs/evo/auto/settings_{}.json".format(args.load_args)) as f:
            new_arg_dict = json.load(f)
            arg_dict.update(new_arg_dict)

    return args, arg_dict


# TODO: Clean this up. Unnecessary globals etc. Have evolve.py use it as well.
def get_exp_name(args=None, arg_dict={}):
    global INFER
    global EVO_DIR
    global CUDA
    global RENDER
    global PROBLEM
    global SHOW_VIS
    global VISUALIZE
    global N_STEPS
    global N_GENERATIONS
    global N_INIT_STATES
    global N_INFER_STEPS
    global BCS
    global RENDER_LEVELS
    global THREADS
    global PLAY_LEVEL
    global CMAES
    global EVALUATE
    global SAVE_LEVELS
    global RANDOM_INIT_LEVELS
    global CASCADE_REWARD
    global REPRESENTATION
    global MODEL
    global REEVALUATE_ELITES
    global preprocess_action
    MODEL = arg_dict["model"]
    REPRESENTATION = arg_dict["representation"]
    CASCADE_REWARD = arg_dict["cascade_reward"]
    REEVALUATE_ELITES = not arg_dict["fix_elites"] and arg_dict["n_init_states"] != 0
    RANDOM_INIT_LEVELS = (
        not arg_dict["fix_level_seeds"]
        and arg_dict["n_init_states"] != 0
        or REEVALUATE_ELITES
    )

    if REEVALUATE_ELITES:
        # Otherwise there is no point in re-evaluating them
        assert RANDOM_INIT_LEVELS
    CMAES = arg_dict["behavior_characteristics"] == ["NONE", "NONE"]
    EVALUATE = arg_dict["evaluate"]
    PLAY_LEVEL = arg_dict["play_level"]
    BCS = arg_dict["behavior_characteristics"]
    N_GENERATIONS = arg_dict["n_generations"]
    N_INIT_STATES = arg_dict["n_init_states"]
    N_STEPS = arg_dict["n_steps"]

    SHOW_VIS = arg_dict["show_vis"]
    PROBLEM = arg_dict["problem"]
    CUDA = False
    VISUALIZE = arg_dict["visualize"]
    INFER = arg_dict["infer"] or EVALUATE
    N_INFER_STEPS = N_STEPS
    #   N_INFER_STEPS = 100
    RENDER_LEVELS = arg_dict["render_levels"]
    THREADS = arg_dict["multi_thread"] or EVALUATE
    #   SAVE_INTERVAL = 100
    SAVE_INTERVAL = 100
    VIS_INTERVAL = 50

    SAVE_LEVELS = arg_dict["save_levels"] or EVALUATE

    #   exp_name = 'EvoPCGRL_{}-{}_{}_{}-batch_{}-step_{}'.format(PROBLEM, REPRESENTATION, BCS, N_INIT_STATES, N_STEPS, arg_dict['exp_name'])
    #   exp_name = "EvoPCGRL_{}-{}_{}_{}_{}-batch".format(
    #       PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES
    #   )
    exp_name = ""
    if arg_dict['algo'] == "ME":
        exp_name += "ME_"
    exp_name += "{}-{}_{}_{}_{}-batch_{}-pass".format(
        PROBLEM, REPRESENTATION, MODEL, BCS, N_INIT_STATES, N_STEPS
    )

    # TODO: remove this! Ad hoc, for backward compatibility.
    if arg_dict["algo"] == "CMAME" and arg_dict["step_size"] != 1 or arg_dict["algo"] == "ME" and arg_dict["step_size"] != 0.01:
        exp_name += f"_{arg_dict['step_size']}-stepSize"

    if CASCADE_REWARD:
        exp_name += "_cascRew"

    if not RANDOM_INIT_LEVELS:
        exp_name += "_fixLvls"

    if not REEVALUATE_ELITES:
        exp_name += "_fixElites"

    if arg_dict['n_aux_chan'] > 0:
        exp_name += f"_{arg_dict['n_aux_chan']}-aux"

    # if arg_dict['mega']:
        # exp_name += "_MEGA"
    exp_name += "_" + arg_dict["exp_name"]
    return exp_name

def get_exp_dir(exp_name):
    evo_runs_dir = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), 'evo_runs')
    SAVE_PATH = os.path.join(evo_runs_dir, exp_name)

    return SAVE_PATH
