"""
Launch a batch of experiments on a SLURM cluster.

dead processes.
"""
import argparse
from collections import namedtuple
import copy
import json
import os
from pdb import set_trace as TT
import re
from typing import Dict, List
import yaml

import numpy as np

from rl.cross_eval import compile_results


with open("configs/rl/batch.yaml", "r") as f:
    batch_config = yaml.safe_load(f)
batch_config = namedtuple('batch_config', batch_config.keys())(**batch_config)

def launch_batch(exp_name, collect_params=False):
    if collect_params:
        settings_list = []
        assert not EVALUATE
#   if opts.render_levels:
#       print('Rendering levels')
#       n_bins = 4
#       n_maps = 2
    if LOCAL:
        print("Testing locally.")
        n_maps = 2
        n_bins = 10
#       n_bins = 4
    else:
        print("Launching batch of experiments on SLURM.")
        n_maps = 50
        n_bins = 10
    with open("configs/rl/auto/default_settings.json", "r") as f:
        default_config = json.load(f)
    print("Loaded default config:\n{}".format(default_config))

#   if LOCAL:
#       # if running locally, just run a quick test
#       default_config["n_frames"] = 100000
    i = 0

    for prob in batch_config.problems:
        prob_controls = batch_config.global_controls + batch_config.local_controls[prob]

        for rep, model in batch_config.representations_models:
            for controls in prob_controls:

#                   if controls != ["NONE"] and change_percentage != 1:

#                       continue

                for alp_gmm in batch_config.alp_gmms:
                    for change_percentage in batch_config.change_percentages:

                        if sum(['3D' in name for name in [prob, rep]]) == 1:
                            print('Dimensions (2D or 3D) of problem and representation do not match. Skipping '
                                  'experiment.')
                            continue

                        if alp_gmm and controls == ["NONE", "NONE"]:
                            continue

#                       if (not alp_gmm) and len(controls) < 2 and controls != ["NONE"]:
#                           # For now we're only looking at uniform-random target-sampling with both control metrics
#                           continue

                        # TODO: integrate evaluate with rllib
                        if EVALUATE:
                            py_script_name = "rl/evaluate_ctrl.py"
                            sbatch_name = "rl/eval.sh"
#                       elif opts.infer:
#                           py_script_name = "infer_ctrl_sb2.py"
                        else:
                            py_script_name = "rl/train_ctrl.py"
                            sbatch_name = "rl/train.sh"
                        # Edit the sbatch file to load the correct config file
                        if not opts.render:
                            with open(sbatch_name, "r") as f:
                                content = f.read()

                                # Replace the ``python scriptname --cl_args`` line.
                                content = re.sub(
                                    "python .* --load_args \d+",
                                    "python {} --load_args {}".format(py_script_name, i),
                                    content,
                                )

                                # Replace the job name.
                                content = re.sub(
                                    "rl_runs/pcgrl_\d+", 
                                    f"rl_runs/pcgrl_{i}", 
                                    content
                                )
                            with open(sbatch_name, "w") as f:
                                f.write(content)
                        # Write the config file with the desired settings
                        exp_config = copy.deepcopy(default_config)

                        # Supply the command-line arguments in args.py
                        exp_config.update(
                            {
                                "n_cpu": opts.n_cpu,
                                "problem": prob,
                                "representation": rep,
                                "model": model,
                                "conditionals": controls,
                                "change_percentage": change_percentage,
                                "alp_gmm": alp_gmm,
                                "experiment_id": exp_name,
                                "render": opts.render,
                                "load": opts.load or opts.infer,
                                "infer": opts.infer,
                                "overwrite": opts.overwrite,
                            }
                        )

                        if EVALUATE:
                            exp_config.update(
                                {
                                    "load": True,
                                    "n_maps": n_maps,
                                    "render": False,
#                                   "render_levels": opts.render_levels,
                                    "n_bins": (n_bins,),
                                    "vis_only": opts.vis_only,
                                }
                            )
                        print("Saving experiment config:\n{}".format(exp_config))
                        with open("configs/rl/auto/settings_{}.json".format(i), "w") as f:
                            json.dump(exp_config, f, ensure_ascii=False, indent=4)
                        # Launch the experiment. It should load the saved settings

                        if collect_params:
                            settings_list.append(exp_config)
                        elif LOCAL:
                            os.system("python {} --load_args {}".format(py_script_name, i))
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
    opts.add_argument(
        "-np",
        "--no_plot",
        help="Do no plot output from monitor files during cross-evaluation.",
        action="store_true",
    )
    opts.add_argument(
        "--render",
        action='store_true',
        help="Visualize agent taking actions in environment by calling environments render function."
    )
    opts.add_argument(
        "-in",
        "--infer",
        action="store_true",
        help="Run inference with a trained model.",
    )
    opts.add_argument(
        "--n_cpu",
        type=int,
        default=12,
    )
    opts.add_argument(
        "--load",
        action="store_true",
        help="Load previous checkpoint of model to resume training or do inference or evaluation.",
    )
    opts.add_argument(
        "-ovr",
        "--overwrite",
        action="store_true",
        help="Overwrite previous experiment with same name."
    )

    opts = opts.parse_args()
    EXP_NAME = opts.experiment_name
    EVALUATE = opts.evaluate
    LOCAL = opts.local
    if opts.cross_eval:
        settings_list = launch_batch(EXP_NAME, collect_params=True)
        compile_results(settings_list, no_plot=opts.no_plot)
    else:
        launch_batch(EXP_NAME)
