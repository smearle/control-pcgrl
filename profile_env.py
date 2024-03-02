import copy
from functools import partial
import json
import os
from pathlib import Path
import shutil
import sys

from tqdm import tqdm
from control_pcgrl.rl.callbacks import StatsCallbacks
from control_pcgrl.rl.envs import make_env
from control_pcgrl.rl.models import ConvDeconv2d, CustomFeedForwardModel, CustomFeedForwardModel3D, WideModel3D
from control_pcgrl.rl.train import MODELS
from control_pcgrl.rl.utils import validate_config
from omegaconf import DictConfig, OmegaConf
import hydra

from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from control_pcgrl.configs.config import ProfileEnvConfig


@hydra.main(version_base=None, config_path='./control_pcgrl/configs', config_name='profile_pcgrl')
def profile(cfg: ProfileEnvConfig):

    PROB_TO_COMPARE = ["binary", "zelda"]
    # ENV_NUM = [1, 10] 
    ENV_NUM = [1, 10, 50, 100, 200, 400, 600]
    # calculate the number of rollout workers and the env of each workers
    N_CPU_CURRENT = 11
    n_rollout_workers = N_CPU_CURRENT - 1 

    problem_n_envs_to_fps = {}
    for total_env in ENV_NUM:
        problem_n_envs_to_fps[total_env] = {}
        for task_name in PROB_TO_COMPARE:
            if total_env == 1: 
                cfg.hardware.n_cpu = 1
                cfg.hardware.n_envs_per_worker = 1
            else:
            # FIXME hardcode some of the setting to for comparison
                env_each_workers = total_env / n_rollout_workers 
                cfg.hardware.n_cpu = N_CPU_CURRENT
                cfg.hardware.n_envs_per_worker = int(env_each_workers)
            cfg.debug= True   # FIXME for now only make dummy random action and see the fps
            cfg.task.problem = cfg.task.name = task_name
            cfg.max_board_scans = 5 
            cfg = validate_config(cfg)
            if cfg is False:
                print("Invalid config!")
                return
            print("OmegaConf.to_yaml(cfg)")
            print(OmegaConf.to_yaml(cfg))
            print("Current working directory:", os.getcwd())

            is_3D_env = False
            if "3D" in cfg.task.problem:
                is_3D_env = True

            log_dir = os.path.join(Path(__file__).parent, "profie_env_result")
            print("########################################")
            print(f"Log directory: {log_dir}")
            print("########################################")

            if cfg.load:
                if not os.path.isdir(log_dir):
                    print(f"Log directory {log_dir} does not exist.")
                    os.makedirs(log_dir)
                else:
                    print(f"Loading from log directory {log_dir}")
            if not cfg.load and not cfg.overwrite:
                if os.path.isdir(log_dir):
                    print(f"Log directory {log_dir} already exists. Will attempt to load.")
            
                else:
                    os.makedirs(log_dir)
                    print(f"Created new log directory {log_dir}")
            if cfg.overwrite:
                if not os.path.exists(log_dir):
                    print(f"Log directory {log_dir} does not exist. Will create new directory.")
                else:
                    # Overwrite the log directory.
                    print(f"Overwriting log directory {log_dir}")
                    shutil.rmtree(log_dir, ignore_errors=True)
                os.makedirs(log_dir, exist_ok=True)


            if not is_3D_env:
                if cfg.model.name is None:
                    if cfg.representation == "wide":
                        model_cls = ConvDeconv2d
                    else:
                        model_cls = CustomFeedForwardModel
                else:
                    model_cls = MODELS[cfg.model.name]
            else:
                if cfg.representation == "wide3D":
                    model_cls = MODELS[cfg.model.name] if cfg.model.name else WideModel3D
                else:
                    model_cls = MODELS[cfg.model.name] if cfg.model.name else CustomFeedForwardModel3D

            ModelCatalog.register_custom_model("custom_model", model_cls)

            # If n_cpu is 0 or 1, we only use the local rllib worker. Specifying n_cpu > 1 results in use of remote workers.
            num_workers = 0 if cfg.hardware.n_cpu == 1 else cfg.hardware.n_cpu
            stats_callbacks = partial(StatsCallbacks, cfg=cfg)

            dummy_cfg = copy.copy(cfg)
            # dummy_cfg["render"] = False
            dummy_cfg.evaluation_env = False
            env = make_env(dummy_cfg)

            if issubclass(type(env), MultiAgentEnv):
                agent_obs_space = env.observation_space["agent_0"]
                agent_act_space = env.action_space["agent_0"]
            else:
                agent_obs_space = env.observation_space
                agent_act_space = env.action_space

            ### DEBUG ###
            if cfg.debug:
                from timeit import default_timer as timer
                n_eps = cfg.N_PROFILE_STEPS
                mean_ep_time = 0
                total_start_time = timer()
                # Randomly step through 100 episodes
                for n_ep in tqdm(range(n_eps)):
                    ep_start_time = timer()
                    obs, info = env.reset()
                    done = False
                    n_step = 0
                    while not done:
                        # if i > 3:
                        act = env.action_space.sample()
                        # act = 0 
                        # else:
                            # act = 0
                        # Print shape of map
                        obs, rew, done, truncated, info = env.step(act)

                        # print(obs.transpose(2, 0, 1)[:, 10:-10, 10:-10])
                        if cfg.render:
                            env.render()
                        if isinstance(done, dict):
                            done = done['__all__']
                        n_step += 1

                    ep_end_time = timer()
                    ep_time = ep_end_time - ep_start_time

                    print(f'Episode {n_ep} finished after {n_step} steps in {ep_time} seconds.')
                    print(f'FPS: {n_step / ep_time}')

                    mean_ep_time += ep_time
                total_end_time = timer()
                mean_ep_time /= n_eps
                total_time = total_end_time - total_start_time
                txt = f'Problem: {task_name}, Total envs: {total_env}\n' + \
                    f'Total time: {total_time} seconds for finishing {n_eps} episodes, ' + \
                    f'Each episode will take {total_time / n_eps} seconds if counting the total runtime \n' + \
                    f'Average episode time: {mean_ep_time} seconds.\n' + \
                    f'Average FPS: {n_step / mean_ep_time}.\n' + \
                    f' ---- \n'
                # print(f'Total time: {total_time} seconds for finishing {n_eps} episodes, ')
                # print(f'Each episode will take {total_time / n_eps} seconds if counting the total runtime')
                # print(f'Average episode time: {mean_ep_time} seconds.')
                # print(f'Average FPS: {n_step / mean_ep_time}.')
                print(txt)
                with open(os.path.join(log_dir, "profile_env_result.txt"), "a") as f:
                    f.write(txt)
                problem_n_envs_to_fps[total_env][task_name] = n_step / mean_ep_time


    with open(os.path.join(log_dir, "profile_env_result.json"), "w") as f:
        json.dump(problem_n_envs_to_fps, f, indent=4)
    
    # make the markdown table out of the result
    with open(os.path.join(log_dir, "profile_env_result_table.md"), "a") as f:
        f.write("\n\n")
        f.write("| Problem | 1 | 10 | 50 | 100 | 200 | 400 | 600 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for prob in PROB_TO_COMPARE:
            f.write(f"| {prob} | ")
            for env in ENV_NUM:
                f.write(f"{problem_n_envs_to_fps[env][prob]:.2f} | ")
            f.write("\n")
    print(f'Congratulations! Check your result in {log_dir}!')
    sys.exit()


if __name__ == "__main__":
    profile()