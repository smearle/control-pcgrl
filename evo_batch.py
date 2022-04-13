"""
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from
dead processes.  """
import argparse
import copy
import json
import os
import re
from pdb import set_trace as TT
from typing import List

from evo_cross_eval import compile_results
from render_gifs import render_gifs


##### HYPERPARAMETERS #####

GENERATIVE_ONLY_CROSS_EVAL = True
exp_ids = [
        0,
#       1,
#       2,
#       3,
#       4,
#       5,
#       6,
#       7,
#       8,
#       9,
#       10,
]
problems = [
        "microstructure_ctrl"
#       "face_ctrl",
#       "loderunner_ctrl",
#       "binary_ctrl",
#       "zelda_ctrl",
#       "sokoban_ctrl",
#       "smb_ctrl"
#       "loderunner_ctrl",
#       "face_ctrl",
]
representations = [
        "cellular",  # change entire board at each step
#       "wide",  # agent "picks" one tile to change
#       "narrow",  # scan over board in sequence, feed current tile to agent as observation
#       "turtle"  # agent "moves" between adjacent tiles, give positional observation as in narrow, and agent has extra action channels corresponding to movement
]
models = [
#   "NCA",
    "DirectBinaryEncoding",
    # "GenSinCPPN",
    # "GenCPPN",
#   "Decoder",
    # "DeepDecoder",
    # "GenCPPN2",
    # "GenSinCPPN2",
#   "GenSin2CPPN2",
#   "AuxNCA",  # NCA w/ additional/auxiliary "invisible" tile-channels to use as external memory
#   "AttentionNCA",
#   "CPPN",  # Vanilla CPPN. No latents. Only runs with n_init_states = 0
#   "Sin2CPPN",

    #   "CPPNCA",  # NCA followed by a traditional CPPN, not a fixed-size/continuous genome
#   "DoneAuxNCA",  # AuxNCA but with one aux. channel to represent done-ness (agent decides when it's finished)
#   "CoordNCA",  # NCA with additional channels corresponding to x and y coordinates

#   "MixCPPN",
#   "MixNCA",

#   "GenReluCPPN",
#   "GenMixCPPN",

#   "FeedForwardCPPN",
#   "SinCPPN",
    # "CNN"  # Doesn't learn atm
]
# Reevaluate elites on new random seeds after inserting into the archive?
fix_elites = [
        True, 
       ]
# Fix a set of random levels with which to seed the generator, or use new ones each generation?
fix_seeds = [
        True,
#       False
        ]
# How many random initial maps on which to evaluate each agent? (0 corresponds to a single layout with a square of wall
# in the center)
n_init_states_lst = [
#   0,
  1,
    # 10,
#   20,
]
# How many steps in an episode of level editing?
n_steps_lst = [
    1,
#   10,
    # 50,
#   100,
]
global_bcs: List[List] = [
      ["NONE", "NONE"], 
#       ["emptiness", "symmetry"],
]
local_bcs = {
    "binary_ctrl": [
#       ["regions", "path-length"],
#       ["emptiness", "path-length"],
        ["emptiness", "path-length"],
    ],
    "zelda_ctrl": [
#       ["nearest-enemy", "path-length"],
#       ["emptiness", "path-length"],
        ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
#       ["crate", "sol-length"],
        ["emptiness", "sol-length"],
#       ["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
       ["jumps", "sol-length"],
       ["emptiness", "sol-length"],
       ["symmetry", "sol-length"]
       ],
    "loderunner_ctrl": [
        ["emptiness", "path-length"],
#       ["symmetry", "path-length"],
#       ["emptiness", "path-length"],
        ["symmetry", "path-length"],
#       ["win", "path-length"],
#       ["gold", "emptiness"],
    ],
    "face_ctrl": [
#       ["face_1", "brightness"],
#       ['brightness', 'blur'],
        ['brightness', 'entropy'],
#       ['rand_sol', 'rand_sol']
    ],
    "microstructure_ctrl": [
        # ["emptiness", "path-length"],
        # ["path-length", "tortuosity"],
    ]
}

###########################




def launch_batch(exp_name, collect_params=False):
    if collect_params:
        settings_list = []
        assert not EVALUATE

    if LOCAL:
        print("Testing locally.")
    else:
        print("Launching batch of experiments on SLURM.")
    with open("configs/evo/default_settings.json", "r") as f:
        default_config = json.load(f)
    print("Loaded default config:\n{}".format(default_config))

    if LOCAL:
        default_config["n_generations"] = 50000
    i = 0

    for exp_id in exp_ids:
        for prob in problems:
            prob_bcs = global_bcs + local_bcs[prob]

            for rep in representations:
                for model in models:

                    if model == "CNN" and rep == "cellular":
                        # This would necessitate an explosive number of model params so we'll not run it

                        continue

                    for bc_pair in prob_bcs:

                        for fix_el in fix_elites:
                            for fix_seed in fix_seeds:

                                # No reason to re-evaluate other than random seeds so this would cause an error

                                if fix_seed and not fix_el:
                                    continue

                                for n_steps in n_steps_lst:
                                    if rep != "cellular":
                                        if n_steps != n_steps_lst[0]:
                                            continue

                                    if "NCA" in model and n_steps <= 5:
                                        continue

                                    for n_init_states in n_init_states_lst:
                                        # The hand-made seed cannot be randomized
                                        if n_init_states == 0 and not (fix_seed and fix_el):
                                            continue

                                        # The hand-made seed is not valid for Decoders (or CPPNs, handled below)
                                        if n_init_states == 0 and "Decoder" in model:
                                            continue

                                        # For the sake of cross-evaluating over model variable alone, do not look at
                                        # experiments treating models with generative capabilities as indirect encodings
                                        if args.cross_eval and GENERATIVE_ONLY_CROSS_EVAL:
                                            if n_init_states == 0 and not (model == "CPPN" or model == "Sin2CPPN" or model == "SinCPPN"):
                                                continue

                                        if model in ["CPPN", "GenCPPN", "GenCPPN2", "CPPNCA", "DirectBinaryEncoding"]:
                                            algo = "ME"
                                        else:
                                            algo = "CMAME"

                                        if 'CPPN' in model:
                                            if 'Gen' not in model and model != "CPPNCA":
                                                # We could have more initial states, randomized initial states, and re-evaluated elites with generator-CPPNs
                                                if n_init_states != 0 or not fix_seed or not fix_el:
                                                    continue

                                            if model != "CPPNCA" and n_steps != 1:
                                                continue

                                        # The decoder generates levels in a single pass (from a smaller latent)
                                        if 'Decoder' in model and n_steps != 1:
                                            continue

                                        # Edit the sbatch file to load the correct config file

                                        if EVALUATE:
                                            script_name = "evo_eval.sh"
                                        else:
                                            script_name = "evo_train.sh"
                                        with open(script_name, "r") as f:
                                            content = f.read()
                                            new_content = re.sub(
                                                "python evolve.py -la \d+",
                                                "python evolve.py -la {}".format(i),
                                                content,
                                            )
                                        with open(script_name, "w") as f:
                                            f.write(new_content)
                                        # Write the config file with the desired settings
                                        exp_config = copy.deepcopy(default_config)
                                        exp_config.update({
                                                "problem": prob,
                                                "representation": rep,
                                                "behavior_characteristics": bc_pair,
                                                "algo": algo,
                                                "model": model,
                                                "fix_elites": fix_el,
                                                "fix_level_seeds": fix_seed,
#                                               "exp_name": exp_name,
                                                "exp_name": str(exp_id),
                                                "save_levels": False,
                                                "n_steps": n_steps,
                                                "n_init_states": n_init_states,
                                                "n_generations": 50000,
                                                "multi_thread": not args.single_thread,
                                            }
                                        )
                                        if args.render:
                                            exp_config.update(
                                                {
                                                    "infer": True,
                                                    "render": True,
                                                    "visualize": True,
                                                }
                                            )

                                        elif EVALUATE:
                                            # No real point a mapping that takes only one-step (unless we're debugging latent seeds, in which case just use more steps)
                                            render_levels = RENDER_LEVELS and n_steps > 1
                                            # ... Also, because this isn't compatible with qdpy at the moment
                                            render_levels = RENDER_LEVELS and algo != "ME"
                                            exp_config.update(
                                                {
                                                    "infer": True,
                                                    "evaluate": True,
                                                    "render_levels": render_levels,
                                                    "save_levels": True,
                                                    "visualize": True,
                                                }
                                            )
                                        print(
                                            "Saving experiment config:\n{}".format(
                                                exp_config
                                            )
                                         )
                                        with open(
                                            "configs/evo/settings_{}.json".format(i), "w"
                                        ) as f:
                                            json.dump(
                                                exp_config, f, ensure_ascii=False, indent=4
                                            )
                                        # Launch the experiment. It should load the saved settings

                                        if collect_params:
                                            settings_list.append(exp_config)
                                        elif LOCAL:
                                            os.system("python evolve.py -la {}".format(i))
                                            # Turned off for mid-training evals
#                                           os.system("ray stop")
                                        else:
                                            os.system("sbatch {}".format(script_name))
                                        i += 1

    if collect_params:
        return settings_list


if __name__ == "__main__":
    opts = argparse.ArgumentParser(
        description="Launch a batch of experiments/evaluations for evo-pcgrl"
    )

    opts.add_argument(
        "-ex",
        "--experiment_name",
        help="A name to be shared by the batch of experiments.",
        default="",
    )
    opts.add_argument(
        "-ev",
        "--evaluate",
        help="Evaluate a batch of evolution experiments.",
        action="store_true",
    )
    opts.add_argument(
        "-l",
        "--local",
        help="Test the batch script, i.e. run it on a local machine and evolve for minimal number of generations.",
        action="store_true",
    )
    opts.add_argument(
        "-r",
        "--render",
        help="Render and observe",
        action="store_true",
    )
    opts.add_argument(
        "-ce",
        "--cross_eval",
        help="Compile stats from previous evaluations into a table",
        action="store_true",
    )
    opts.add_argument(
        "-tex",
        "--tex",
        help="If compiling cross-eval results, produce latex table (otherwise html).",
        action="store_true",
    )
    opts.add_argument(
        "--gif",
        help="Make gifs from previously-rendered level-generation episodes.",
        action="store_true",
    )
    opts.add_argument(
        "--render_levels",
        help="Save images from level-generation (to be subsequently used to render gifs with --gif).",
        action="store_true",
    )
    opts.add_argument(
        "-st",
        "--single_thread",
        help="Run experiment sequentially, instead of using ray to parallelise evaluation.",
        action="store_true",
    )
    args = opts.parse_args()
    EXP_NAME = args.experiment_name
    EVALUATE = args.evaluate
    LOCAL = args.local
    RENDER_LEVELS = args.render_levels

    if args.cross_eval or args.gif:
        settings_list = launch_batch(EXP_NAME, collect_params=True)
    if args.cross_eval:
        compile_results(settings_list, tex=args.tex)
        if not args.tex:
            print("Produced html at evo_runs/cross_eval_{}.html".format(args.experiment_name))
        else:
            os.chdir('eval_experiment')
            os.system(f'pdflatex tables.tex')
    elif args.gif:
        render_gifs(settings_list)
    else:
        launch_batch(EXP_NAME)
