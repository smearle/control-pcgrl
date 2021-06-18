"""
Launch a batch of experiments on a SLURM cluster.

dead processes.
"""
from pdb import set_trace as TT
import argparse
import copy
import json
import os
import re
from typing import Dict, List

import numpy as np
from rl_cross_eval import compile_results

problems: List[str] = [
    "binary_ctrl",
    "zelda_ctrl",
    "sokoban_ctrl",
    # 'smb_ctrl',
]
representations: List[str] = [
    # 'cellular',
    # "wide",
    "narrow",
    # 'turtle',
]
# TODO: incorporate formal (rather than only functional) metrics as controls
global_controls: List[List] = [
    ["NONE"],
    # ['emptiness', 'symmetry'],
]
local_controls: Dict[str, List] = {
    "binary_ctrl": [
        ["regions"],
        ["path-length"],
        ["regions", "path-length"],
        # ['emptiness', 'path-length'],
        # ["symmetry", "path-length"]
    ],
    "zelda_ctrl": [
        ["nearest-enemy"],
        ["path-length"],
        ["nearest-enemy", "path-length"],
        # ["emptiness", "path-length"],
        # ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
        # ["crate"],
        ["sol-length"],
        ["crate", "sol-length"],
        # ["emptiness", "sol-length"],
        # ["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
        # ['enemies', 'jumps'],
        # ["emptiness", "jumps"],
        # ["symmetry", "jumps"],
    ],
    "RCT": [
        # ['income'],
    ],
}
#change_percentages = np.arange(2, 11, 4) / 10
change_percentages = [
    0.2,
    0.6,
    1.0,
]
alp_gmms = [
    True,
    False
]


def launch_batch(exp_name, collect_params=False):
    if collect_params:
        settings_list = []
        assert not EVALUATE
#   if args.render_levels:
#       print('Rendering levels')
#       n_bins = 4
#       n_maps = 2
    if LOCAL:
        print("Testing locally.")
        n_maps = 2
        n_bins = 13
    else:
        print("Launching batch of experiments on SLURM.")
        n_maps = 50
        n_bins = 25
    with open("configs/rl/default_settings.json", "r") as f:
        default_config = json.load(f)
    print("Loaded default config:\n{}".format(default_config))

    if LOCAL:
        # if running locally, just run a quick test
        default_config["n_frames"] = 100000
    i = 0

    for prob in problems:
        prob_controls = global_controls + local_controls[prob]

        for rep in representations:
            for controls in prob_controls:
                for change_percentage in change_percentages:

#                   if controls != ["NONE"] and change_percentage != 1:

#                       continue

                    for alp_gmm in alp_gmms:

                        if alp_gmm and controls == ["NONE"]:
                            continue

#                       if (not alp_gmm) and len(controls) < 2 and controls != ["NONE"]:
#                           # For now we're only looking at uniform-random target-sampling with both control metrics
#                           continue

                        if EVALUATE:
                            py_script_name = "evaluate_ctrl.py"
                            sbatch_name = "rl_eval.sh"
                        else:
                            py_script_name = "train_ctrl.py"
                            sbatch_name = "rl_train.sh"
                        # Edit the sbatch file to load the correct config file
                        with open(sbatch_name, "r") as f:
                            content = f.read()
                            new_content = re.sub(
                                "python .* -la \d+",
                                "python {} -la {}".format(py_script_name, i),
                                content,
                            )
                        with open(sbatch_name, "w") as f:
                            f.write(new_content)
                        # Write the config file with the desired settings
                        exp_config = copy.deepcopy(default_config)
                        exp_config.update(
                            {
                                "n_cpu": 48,
                                "problem": prob,
                                "representation": rep,
                                "conditionals": controls,
                                "change_percentage": change_percentage,
                                "alp_gmm": alp_gmm,
                                "experiment_id": exp_name,
                            }
                        )

                        if EVALUATE:
                            exp_config.update(
                                {
                                    "resume": True,
                                    "n_maps": n_maps,
                                    "render": False,
#                                   "render_levels": args.render_levels,
                                    "n_bins": (n_bins,),
                                }
                            )
                        print("Saving experiment config:\n{}".format(exp_config))
                        with open("configs/rl/settings_{}.json".format(i), "w") as f:
                            json.dump(exp_config, f, ensure_ascii=False, indent=4)
                        # Launch the experiment. It should load the saved settings

                        if collect_params:
                            settings_list.append(exp_config)
                        elif LOCAL:
                            os.system("python {} -la {}".format(py_script_name, i))
                        else:
                            os.system("sbatch {}".format(sbatch_name))
                        i += 1
    if collect_params:
        return settings_list


if __name__ == "__main__":
    opts = argparse.ArgumentParser(
        description="Launch a batch of experiments/evaluations for (controllable) pcgrl"
    )

#   opts.add_argument(
#       "-rl",
#       "--render_levels",
#       help="",
#       action="store_true",
#   )

    opts.add_argument(
        "-ex",
        "--experiment_name",
        help="A name to be shared by the batch of experiments.",
        default="2",
    )
    opts.add_argument(
        "-ev",
        "--evaluate",
        help="Evaluate a batch of PCGRL experiments.",
        action="store_true",
    )
    opts.add_argument(
        "-l",
        "--local",
        help="Run the batch script locally (i.e. to test it) and train for minimal number of frames.",
        action="store_true",
    )
    opts.add_argument(
        "-v",
        "--vis_only",
        help="Just load data from previous evaluation and visualize it.",
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
