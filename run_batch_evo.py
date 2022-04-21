"""
Launch a batch of experiments on a SLURM cluster (or sequentially, if running with --local). Uncomment hyperparameters 
below, and this script will launch all valid combinations of uncommented hyperparameters.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from
dead processes.  """
import argparse
from collections import namedtuple
import copy
import json
import yaml
import os
import re
from pdb import set_trace as TT
from typing import List

from evo.cross_eval import compile_results
from evo.render_gifs import render_gifs


GECCO_CROSS_EVAL = True

with open("configs/evo/batch.yaml", "r") as f:
    batch_config = yaml.safe_load(f)
batch_config = namedtuple('batch_config', batch_config.keys())(**batch_config)

def launch_batch(exp_name, collect_params=False):
    if collect_params:
        settings_list = []
        assert not EVALUATE

    if LOCAL:
        print("Testing locally.")
    else:
        print("Launching batch of experiments on SLURM.")
    with open("configs/evo/auto/default_settings.json", "r") as f:
        default_config = json.load(f)
    # print("Loaded default config:\n{}".format(default_config))

    if LOCAL:
        default_config["n_generations"] = 50000
    i = 0

    for exp_id in batch_config.exp_ids:
        for prob in batch_config.problems:
            prob_bcs = batch_config.global_bcs + batch_config.local_bcs[prob]

            for rep in batch_config.representations:
                for algo in batch_config.algos:
                    for model in batch_config.models:

                        if model == "CNN" and rep == "cellular":
                            print("Skipping experiments with CNN model and cellular representation, as this would necessitate "
                                "an explosion of model parameters.")

                            continue

                        for bc_pair in prob_bcs:

                            for fix_el in batch_config.fix_elites:
                                for fix_seed in batch_config.fix_seeds:

                                    # No reason to re-evaluate other than random seeds so this would cause an error

                                    if fix_seed and not fix_el:
                                        print("Skipping experiment with fix_seed=True and fix_elites=False. There is no "
                                            "point re-evaluating generators (which are deterministic) on the same seeds.")
                                        continue

                                    for n_steps in batch_config.n_steps_lst:
                                        if rep != "cellular":
                                            if n_steps != batch_config.n_steps_lst[0]:
                                                continue

                                        if "NCA" in model and n_steps <= 5:
                                            print("Skipping experiments with NCA model and n_steps <= 5.")
                                            continue

                                        for n_init_states in batch_config.n_init_states_lst:
                                            if n_init_states == 0 and not (fix_seed and fix_el):
                                                print("Skipping experiments with n_init_states=0 and fix_seed=False. The "
                                                    "hand-made seed cannot be randomized.")
                                                continue

                                            # The hand-made seed is not valid for Decoders (or CPPNs, handled below)
                                            if n_init_states == 0 and "Decoder" in model:
                                                continue

                                            if model in ["CPPN", "GenCPPN", "GenCPPN2", "CPPNCA", "DirectBinaryEncoding"]:
                                                if algo != "ME":
                                                    print("Skipping experiments with model {model} and algo {algo}. (requires "
                                                    "MAP-Elites.)")
                                                    continue
                                            else:
                                                pass
                                                # algo = "CMAME"

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

                                            for step_size in batch_config.step_sizes:

                                                # For rendering tables for the GECCO paper, we exclude a bunch of experiments.
                                                if args.cross_eval and GECCO_CROSS_EVAL:
                                                    if n_init_states == 0 and not (model == "CPPN" or model == "Sin2CPPN" or model == "SinCPPN"):
                                                        continue
                                                    if algo == "ME" and step_size != 0.01 or algo == "CMAME" and step_size != 1.0:
                                                        continue
                                                    if algo == "ME" and model not in ["CPPN", "GenCPPN2"]:
                                                        continue

                                                # Edit the sbatch file to load the correct config file
                                                if EVALUATE:
                                                    script_name = "evo/eval.sh"
                                                else:
                                                    script_name = "evo/train.sh"
                                                with open(script_name, "r") as f:
                                                    content = f.read()

                                                    # Replace the ``python scriptname --cl_args`` line.
                                                    new_content = re.sub(
                                                        "python evo/evolve.py -la \d+",
                                                        "python evo/evolve.py -la {}".format(i),
                                                        content,
                                                    )

                                                    # Replace the job name.
                                                    new_content = re.sub(
                                                        "evo_runs/evopcg_\d+", 
                                                        "evo_runs/evopcg_{}".format(i), 
                                                        new_content
                                                    )
                                                with open(script_name, "w") as f:
                                                    f.write(new_content)
                                                # Write the config file with the desired settings
                                                exp_config = copy.deepcopy(default_config)
                                                exp_config.update({
                                                        "algo": algo,
                                                        "problem": prob,
                                                        "representation": rep,
                                                        "behavior_characteristics": bc_pair,
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
                                                        "save_interval": 10 if args.local else 100,
                                                        "step_size": step_size,
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
                                                    "configs/evo/auto/settings_{}.json".format(i), "w"
                                                ) as f:
                                                    json.dump(
                                                        exp_config, f, ensure_ascii=False, indent=4
                                                    )
                                                # Launch the experiment. It should load the saved settings

                                                if collect_params:
                                                    settings_list.append(exp_config)
                                                elif LOCAL:
                                                    os.system("python evo/evolve.py -la {}".format(i))
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
        help="Evaluate a batch of evo experiments.",
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
#   opts.add_argument(
#       "-ss",
#       "--step_size",
#       help="Standard deviation of the Gaussian noise added to mutated models.",
#       type=float,
#       default=0.01,
#   )
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
