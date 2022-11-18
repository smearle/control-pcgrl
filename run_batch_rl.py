"""
Launch a batch of experiments on a SLURM cluster.

dead processes.
"""
import argparse
from argparse import Namespace
import copy
import itertools
import json
import os
from pdb import set_trace as TT
import re
import submitit
import yaml

from control_pcgrl.rl.cross_eval import compile_results
from control_pcgrl.rl.utils import get_exp_name
from control_pcgrl.rl import train_ctrl


with open("configs/rl/batch.yaml", "r") as f:
    batch_config = yaml.safe_load(f)

# HACK: deal with nested hyperparameters.
local_controls, global_controls = batch_config.pop("local_controls"), batch_config.pop("global_controls")

# Take product of lists of all hyperparameters in `batch_config`.
keys, vals = zip(*batch_config.items())
exp_hypers = itertools.product(*vals)

# Turn lists in `exp_hypers` into dictionaries.
exp_hypers = [dict(zip(keys, exp_hyper)) for exp_hyper in exp_hypers]

# Turn `batch_config` into a namespace.
batch_config = Namespace(**batch_config)



def launch_batch(collect_params=False):
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

#   if LOCAL:
#       # if running locally, just run a quick test
#       default_config["n_frames"] = 100000
    i = 0


    jobs = []
    for exp_cfg in exp_hypers:
        # exp_config inherits all arguments from opts
        exp_cfg.update(vars(opts))

        # Supply the command-line arguments in args.py
        exp_cfg.update(
            {
                "evaluate": EVALUATE,
                "representation": exp_cfg['representation_model'][0],
                "model": exp_cfg['representation_model'][1],
            }
        )

        # TODO: Revive this functionality and put it somewhere
        #             if EVALUATE:
        #                 exp_config.update(
        #                     {
        #                         # "load": True,
        #                         "n_maps": n_maps,
        #                         # "render": False,
        #                                   "render_levels": opts.render_levels,
        #                         "n_bins": (n_bins,),
        #                         "vis_only": opts.vis_only,
        #                     }
                    # )


        # FIXME: This is a hack. How to iterate through nested hyperparameter loops in a better way?
        # TODO: Hydra would solve this. Empty this file to make room for hydra.
        prob_controls = global_controls + local_controls[exp_cfg['problem']]
        for controls in prob_controls:
            exp_prob_cfg = copy.deepcopy(exp_cfg)
            exp_prob_cfg.update({
                "controls": controls
            })
            exp_prob_cfg = Namespace(**exp_prob_cfg)
#                   if controls != ["NONE"] and change_percentage != 1:

#                       continue
            # if sum(['3D' in name for name in [prob, rep]]) == 1:
                # print(f'Dimensions (2D or 3D) of Problem: {prob} and Representation: {rep} '
                        # 'do not match. Skipping experiment.')
                # continue

            # if sum(['holey' in name for name in [prob, rep]]) == 1:
            #     print(f'Holeyness of Problem: {prob} and Representation: {rep} '
            #             'do not match. Skipping experiment.')
            #     continue

            if exp_prob_cfg.alp_gmm and controls is None:
                continue

#                       if (not alp_gmm) and len(controls) < 2 and controls != ["NONE"]:
#                           # For now we're only looking at uniform-random target-sampling with both control metrics
#                           continue

            # TODO: integrate evaluate with rllib
#             if EVALUATE:
#                 py_script_name = "rl/evaluate_ctrl.py"
#                 sbatch_name = "rl/eval.sh"
# #                       elif opts.infer:
# #                           py_script_name = "infer_ctrl_sb2.py"
            # else:
            py_script_name = "control_pcgrl/control_pcgrl/rl/train_ctrl.py"
            sbatch_name = "control_pcgrl/control_pcgrl/rl/train.sh"
            
            # Write the config file with the desired settings
            # exp_config = copy.deepcopy(default_config)

            print(f"Saving experiment config:\n{exp_cfg}")
            
            # get the experiment name to name the config file
            # config_name = f"{prob}_{rep}_{exp_name}"
            config_name = get_exp_name(exp_prob_cfg)
            config_name += f"_{exp_prob_cfg.exp_id}"
            # Edit the sbatch file to load the correct config file
            if not opts.render:
                if  not LOCAL:
                    with open(sbatch_name, "r") as f:
                        content = f.read()

                        # Replace the ``python scriptname --cl_args`` line.
                        content = re.sub(
                            "python .* --load_args .*",
                            f"python {py_script_name} --load_args {config_name}",
                            content,
                        )

                        # Replace the job name.
                        content = re.sub(
                            "rl_runs/pcgrl_.*",
                            f"rl_runs/pcgrl_{config_name}_%j.out",
                            content
                        )

                        content = re.sub(
                            "--job-name=.*",
                            f"--job-name={config_name}.out",
                            content
                        )
                    with open(sbatch_name, "w") as f:
                        f.write(content)
            
            # Get directory of current file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            with open(os.path.join(dir_path, f"configs/rl/auto/settings_{config_name}.json"), "w") as f:
                json.dump(vars(exp_prob_cfg), f, ensure_ascii=False, indent=4)
            # Launch the experiment. It should load the saved settings

            if not (EVALUATE and not opts.overwrite_eval and \
                os.path.isfile(os.path.join('rl_runs', f'{config_name}_log', 'eval_stats.json'))):
                if collect_params:
                    settings_list.append(exp_cfg)
                elif LOCAL:
                    # full_cmd = f"python {py_script_name} --load_args {config_name}"
                    # Printout for convenience: when debugging on a Mac calling this from a script will break `set_trace()`
                    # so we print the command here to be entered-in manually.
                    # print(f"Running command:\n{full_cmd}")
                    # os.system(full_cmd)
                    train_ctrl.main(exp_prob_cfg)
                else:
                    # TODO: User submitit.
                    os.system(f"sbatch {sbatch_name}")
                    # job = executor.submit(train_ctrl.main, exp_prob_cfg)

                    # jobs.append(job)
            else:
                print('Skipping evaluation (already have stats saved).')
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

    # opts.add_argument(
    #     "-ex",
    #     "--experiment_name",
    #     help="A name to be shared by the batch of experiments.",
    #     default="0",
    # )
    opts.add_argument(
        "-d",
        "--debug",
        help="Debug environment & rendering (render random agent for a bunch of episodes then quit).",
        action="store_true",
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
        help="Do no plot training curves (output from monitor files) during cross-evaluation (What?).",
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
        help="Number of remote workers to use for rllib training.",
    ) 
    opts.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs to use for training.",
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
    # opts.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=float,
    #     default=0.000005,
    #     help="Learning rate for rllib agent, default is 0.0001."
    # )
    opts.add_argument(
        "-ga",
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor of the MDP."
    )
    opts.add_argument(
        '--wandb',
        help='Whether to use wandb for logging.',
        action='store_true',
        # action=argparse.BooleanOptionalAction,
        # default=False,
    )
    opts.add_argument(
        '--record_env',
        help='Whether to record the environment during inference.',
        action='store_true',
        # action=argparse.BooleanOptionalAction,
        # default=False,
    )
    # opts.add_argument(
    #     '--max_board_scans',
    #     help='Number of max iterations in terms of maximum number of times the board can be scanned by the agent.',
    #     type=int,
    #     default=1,
    # )
    opts.add_argument(
        '--overwrite_eval',
        help='Whether to overwrite stats resulting from a previous evaluation.',
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    opts = opts.parse_args()
    # EXP_NAME = opts.experiment_name
    EVALUATE = opts.evaluate
    LOCAL = opts.local
    executor = submitit.AutoExecutor(os.path.join("rl_runs", "submitit"))
    executor.update_parameters(gpus_per_node=1, slurm_mem="30GB", cpus_per_task=max(1, opts.n_cpu), slurm_time="5:00:00",
                                job_name="pcgrl",)
    if opts.cross_eval:
        settings_list = launch_batch(collect_params=True)
        compile_results(settings_list, no_plot=opts.no_plot)
    else:
        with executor.batch():
            launch_batch()
