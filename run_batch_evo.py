"""
Launch a batch of experiments on a SLURM cluster (or sequentially, if running with --local). Uncomment hyperparameters 
below, and this script will launch all valid combinations of uncommented hyperparameters.

WARNING: This will kill all ray processes running on the current node after each experiment, to avoid memory issues from
dead processes.  """
import argparse
from collections import namedtuple
import copy
import itertools
import json
import yaml
import os
import re
from pdb import set_trace as TT
from typing import List

from control_pcgrl.evo.args import get_exp_name
from control_pcgrl.evo.render_gifs import render_gifs


GECCO_CROSS_EVAL = False


def launch_batch(args, exp_name, collect_params=False):
    with open(f"configs/evo/{args.batch}.yaml", "r") as f:
        batch_config = yaml.safe_load(f)
    batch_config = namedtuple('batch_config', batch_config.keys())(**batch_config)

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

    # TODO: refactor with itertools
    settings_prod = itertools.product(batch_config.exp_ids, batch_config.problems, batch_config.representations, 
                                      batch_config.algos, batch_config.models, batch_config.fix_elites, 
                                      batch_config.fix_seeds, batch_config.n_steps_lst, batch_config.n_init_states_lst, 
                                      batch_config.step_sizes, batch_config.n_aux_chans)
    settings_prod = list(settings_prod)
    if settings_prod == 0:
        raise Exception("No valid settings to run.")
    for exp_id, prob, rep, algo, model, fix_el, fix_seed, n_steps, n_init_states, step_size, n_aux_chan \
        in settings_prod:

        prob_bcs = batch_config.global_bcs + batch_config.local_bcs[prob]

        for bc_pair in prob_bcs:

            if model == "CNN" and "cellular" in rep:
                print("Skipping experiments with CNN model and cellular representation, as this would necessitate "
                    "an explosion of model parameters.")

                continue

            # No reason to re-evaluate other than random seeds so this would cause an error

            if fix_seed and not fix_el:
                print("Skipping experiment with fix_seed=True and fix_elites=False. There is no "
                    "point re-evaluating generators (which are deterministic) on the same seeds.")
                continue

            if "cellular" not in rep:
                if n_steps != batch_config.n_steps_lst[0]:
                    print("Number of steps does not apply to non-cellular representations.")
                    continue

            if n_aux_chan > 0 and "NCA" not in model:
                print("Skipping experiment with n_aux_chan > 0 and model != NCA.")
                continue

            if "NCA" in model and n_steps <= 5:
                print("Skipping experiments with NCA model and n_steps <= 5.")
                continue

            if n_init_states == 0 and not (fix_seed and fix_el):
                print("Skipping experiments with n_init_states=0 and fix_seed=False. The "
                    "hand-made seed cannot be randomized.")
                continue

            if n_init_states == 0 and "Decoder" in model:
                print("The hand-made seed is not valid for Decoders")  # (or CPPNs, handled below)
                continue

            if model in ["CPPN", "GenCPPN", "GenCPPN2", "CPPNCA", "DirectEncoding"]:
                if algo != "ME":
                    print(f"Skipping experiments with model {model} and algo {algo}. (requires "
                    "MAP-Elites.)")
                    continue
            else:
                pass
                # algo = "CMAME"

            if model == "DirectEncoding":
                if n_init_states != 1:
                    print("DirectEncoding only works with n_init_states=1. (Seed is ignored.)")
                    continue
                if not (fix_seed and fix_el):
                    print("DirectEncoding only works with fix_seed=True and fix_elites=True. (Seed is ignored.)")
                    continue

            if 'CPPN' in model:
                if 'Gen' not in model and model != "CPPNCA":
                    # We could have more initial states, randomized initial states, and re-evaluated elites with generator-CPPNs
                    if n_init_states != 0 or not fix_seed or not fix_el:
                        continue

                if model != "CPPNCA" and n_steps != 1:
                    continue

            # The decoder generates levels in a single pass (from a smaller latent)
            if ('Decoder' in model or 'DirectEncoding' in model) and n_steps != 1:
                print('Too many steps for 1-pass generator.')
                continue

            # For rendering tables for the GECCO paper, we exclude a bunch of experiments.
            if args.cross_eval and GECCO_CROSS_EVAL:
                if n_init_states == 0 and not (model == "CPPN" or model == "Sin2CPPN" or model == "SinCPPN"):
                    continue
                if algo == "ME" and step_size != 0.01 or algo == "CMAME" and step_size != 1.0:
                    continue
                if algo == "ME" and model not in ["CPPN", "GenCPPN2"]:
                    continue


            # Write the config file with the desired settings
            exp_config = copy.deepcopy(default_config)
            exp_config.update({
                    "algo": algo,
                    "behavior_characteristics": bc_pair,
                    "exp_name": str(exp_id),
                    "fix_elites": fix_el,
                    "fix_level_seeds": fix_seed,
#                                               "exp_name": exp_name,
                    "model": model,
                    "multi_thread": not args.single_thread,
                    "n_aux_chan": n_aux_chan,
                    "n_generations": args.n_generations,
                    "n_init_states": n_init_states,
                    "n_steps": n_steps,
                    "problem": prob,
                    "representation": rep,
                    "save_interval": 10 if args.local else 100,
                    "save_levels": False,
                    "step_size": step_size,
                    "render": args.render,
                    "n_cpu": args.n_cpu,
                }
            )
            if args.render:
                exp_config.update(
                    {
                        "infer": args.infer,
                        "visualize": False,
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

            full_exp_name = get_exp_name(arg_dict=exp_config)
            full_exp_name = full_exp_name.replace(' ', '').replace("'", "")

            # Edit the sbatch file to load the correct config file
            if EVALUATE:
                script_name = "control_pcgrl/evo/eval.sh"
            else:
                script_name = "control_pcgrl/evo/train.sh"
            with open(script_name, "r") as f:
                content = f.read()

                # Replace the ``python scriptname --cl_args`` line.
                new_content = re.sub(
                    "python3 control_pcgrl/evo/evolve.py -la \d+",
                    f"python3 control_pcgrl/evo/evolve.py -la {i}",
                    content,
                )

                # Replace the job name.
                new_content = re.sub(
                    "evo_runs/.*\.out", 
                    f"evo_runs/{full_exp_name}.out", 
                    new_content
                )
            with open(script_name, "w") as f:
                f.write(new_content)
            # Launch the experiment. It should load the saved settings

            if collect_params:
                settings_list.append(exp_config)
            elif LOCAL:
                os.system("python3 control_pcgrl/evo/evolve.py -la {}".format(i))
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
        "-b",
        "--batch",
        type=str,
        default="batch",
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
    opts.add_argument(
        "--n_cpu",
        type=int,
        default=None,
    )

    opts.add_argument(
        "--n_generations",
        type=int,
        default=50000,
        help="For QD optimization - how many iterations will it take to come up with the final archive of solutions (the last generation)."
    )

    opts.add_argument('-i', '--infer', action='store_true')
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
        from evo.cross_eval import compile_results
        settings_list = launch_batch(args, EXP_NAME, collect_params=True)
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
        launch_batch(args, EXP_NAME)
