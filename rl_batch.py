"""
Launch a batch of experiments on a SLURM cluster.

dead processes.
"""
import argparse
import copy
import json
import os
import re
from typing import Dict, List

import numpy as np

problems: List[str] = [
    "binary_ctrl",
    'zelda_ctrl',
    'sokoban_ctrl',
    #'smb_ctrl',
]
representations: List[str] = [
    # 'cellular',
    # "wide",
    "narrow",
    # 'turtle',
]
# TODO: incorporate formal (rather than functional) metrics as controls
global_controls: List[List] = [
    ["NONE"],
    # ['emptiness', 'symmetry'],
]
local_controls: Dict[str, List] = {
    "binary_ctrl": [
        # ['regions', 'path-length'],
        # ['emptiness', 'path-length'],
        # ["symmetry", "path-length"]
    ],
    "zelda_ctrl": [
        # ['nearest-enemy', 'path-length'],
        # ["emptiness", "path-length"],
        # ["symmetry", "path-length"],
    ],
    "sokoban_ctrl": [
        # ['crate', 'sol-length'],
        # ["emptiness", "sol-length"],
        # ["symmetry", "sol-length"],
    ],
    "smb_ctrl": [
        # ['enemies', 'jumps'],
        # ["emptiness", "jumps"],
        # ["symmetry", "jumps"],
    ],
}
change_percentages = np.arange(2, 11, 2) / 10


def launch_batch(exp_name):
    if LOCAL:
        print("Testing locally.")
    else:
        print("Launching batch of experiments on SLURM.")
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

                    if controls != ["NONE"] and change_percentage != 1:
                        # TODO: support controllable runs with variable change percentage
                        continue

                    # Edit the sbatch file to load the correct config file
                    with open("rl_train.sh", "r") as f:
                        content = f.read()
                        new_content = re.sub(
                            "python train_controllable.py -la \d+",
                            "python train_controllable.py -la {}".format(i),
                            content,
                        )
                    with open("rl_train.sh", "w") as f:
                        f.write(new_content)
                    # Write the config file with the desired settings
                    exp_config = copy.deepcopy(default_config)
                    exp_config.update(
                        {
                            "problem": prob,
                            "representation": rep,
                            "cond_metrics": controls,
                            "change_percentage": change_percentage,
                            "experiment_id": exp_name,
                        }
                    )

                    if EVALUATE:
                        exp_config.update({})
                    print("Saving experiment config:\n{}".format(exp_config))
                    with open("configs/rl/settings_{}.json".format(i), "w") as f:
                        json.dump(exp_config, f, ensure_ascii=False, indent=4)
                    # Launch the experiment. It should load the saved settings

                    if LOCAL:
                        os.system("python train_controllable.py -la {}".format(i))
                    else:
                        os.system("sbatch rl_train.sh")
                    i += 1


if __name__ == "__main__":
    opts = argparse.ArgumentParser(
        description="Launch a batch of experiments/evaluations for (controllable) pcgrl"
    )

    opts.add_argument(
        "-ex",
        "--experiment_name",
        help="A name to be shared by the batch of experiments.",
        default="0",
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
    args = opts.parse_args()
    EXP_NAME = args.experiment_name
    EVALUATE = args.evaluate
    LOCAL = args.local

    launch_batch(EXP_NAME)
