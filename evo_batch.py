"""
Launch a batch of experiments on a SLURM cluster.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from
dead processes.
"""
import argparse
import copy
import json
import os
import re
from pdb import set_trace as TT
from typing import List

from cross_eval import compile_results

problems = [
        "binary_ctrl", 
        "zelda_ctrl", 
        "sokoban_ctrl", 
        "smb_ctrl"
        ]
representations = [
        "cellular", 
       #"wide", 
       #"narrow", 
       #"turtle"
        ]
global_bcs: List[List] = [
       #["NONE"], 
       #["emptiness", "symmetry"],
        ]
local_bcs = {
    "binary_ctrl": [
       #["regions", "path-length"],
       #["emptiness", "path-length"],
        ["symmetry", "path-length"],
    ],
    "zelda_ctrl": [
       #["nearest-enemy", "path-length"],
        ["emptiness", "path-length"],
       #["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
       #["crate", "sol-length"],
        ["emptiness", "sol-length"],
       #["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
       #["enemies", "jumps"], 
       #["emptiness", "jumps"], 
       ["symmetry", "jumps"]
       ],
}
models = [
    "NCA",
    # "CNN"  # Doesn't learn atm
]
# Reevaluate elites on new random seeds after inserting into the archive?
fix_elites = [
        True, 
       #False
       ]
# Fix a set of random levels with which to seed the generator, or use new ones each generation?
fix_seeds = [
        True, 
        False
        ]
# How many random initial maps on which to evaluate each agent? (0 corresponds to a single layout with a square of wall
# in the center)
n_init_states_lst = [
    0,
    10,
    20,
]
# How many steps in an episode of level editing?
n_steps_lst = [
    10,
    50,
    100,
]


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
        default_config["n_generations"] = 50
    i = 0

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

                                for n_init_states in n_init_states_lst:
                                    if n_init_states == 0 and not (fix_seed and fix_el):
                                        # The hand-made seed cannot be randomized

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
                                    exp_config.update(
                                        {
                                            "problem": prob,
                                            "representation": rep,
                                            "behavior_characteristics": bc_pair,
                                            "model": model,
                                            "fix_elites": fix_el,
                                            "fix_level_seeds": fix_seed,
                                            "exp_name": exp_name,
                                            "save_levels": True,
                                            "n_steps": n_steps,
                                            "n_init_states": n_init_states,
                                            "n_generations": 10000,
                                        }
                                    )

                                    if EVALUATE:
                                        exp_config.update(
                                            {
                                                "infer": True,
                                                "evaluate": True,
                                                "render_levels": True,
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
                                        os.system("ray stop")
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
        default="1",
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
        "-ce",
        "--cross_eval",
        help="Compile stats from previous evaluations into a table",
        action="store_true",
    )
    args = opts.parse_args()
    EXP_NAME = args.experiment_name
    EVALUATE = args.evaluate
    LOCAL = args.local

    if args.cross_eval:
        settings_list = launch_batch(EXP_NAME, collect_params=True)
        compile_results(settings_list)
    else:
        launch_batch(EXP_NAME)
