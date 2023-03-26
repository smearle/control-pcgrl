import glob
import os
import shutil
import hydra
from ray import tune
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from control_pcgrl.configs.config import Config, PoDConfig
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.models import CustomFeedForwardModel
from control_pcgrl.rl.utils import validate_config


@hydra.main(config_path="control_pcgrl/configs", config_name="config")
def main(cfg: PoDConfig):
    validate_config(cfg)

    register_env('pcgrl', make_env)
    model_cls = CustomFeedForwardModel
    ModelCatalog.register_custom_model("custom_model", model_cls)

    bc_config = BCConfig()

    bc_config.model = {
        'custom_model': 'custom_model',
        'custom_model_config': {
        },
    }

    # Print out some default values.
    print(bc_config.beta)  

    # Update the config object.
    bc_config.training(  
        lr=tune.grid_search([0.001, 0.0001]), beta=0.0
    )

    # Get all json files in the directory
    traj_glob = os.path.join(cfg.log_dir, "demo-out", "*.json")

    # Set the config object's data path.
    # Run this from the ray directory root.
    bc_config.offline_data(  
        # input_="./tmp/demo-out/output-2023-0"
        # input_=os.path.join(cfg.log_dir, "demo-out")
        input_=traj_glob,
    )

    # Set the config object's env, used for evaluation.
    bc_config.environment(env='pcgrl')  
    bc_config.env_config = {**cfg}

    bc_config.checkpoint_freq = 1

    bc_config.framework("torch")

    exp_dir = "il_logs/BC"

    if not cfg.overwrite and os.path.exists(exp_dir):
        tuner = tune.Tuner.restore(exp_dir)
    else:
        shutil.rmtree(exp_dir, ignore_errors=True)

        tuner = tune.Tuner(
            "BC",
            param_space=bc_config.to_dict(),
            tune_config = tune.TuneConfig(
                metric="info/learner/default_policy/learner_stats/policy_loss",
                mode="min",
            ),
        )

    if cfg.infer:
        best_result = tuner.get_results().get_best_result()
        breakpoint()

    # Use to_dict() to get the old-style python config dict
    # when running with tune.
    result = tuner.fit()


if __name__ == "__main__":
    main()