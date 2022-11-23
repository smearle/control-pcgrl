from pdb import set_trace as TT

import numpy as np
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext

from control_pcgrl.envs.pcgrl_env import PcgrlEnv


def set_map_fn(
    train_results: dict, task_settable_env: PcgrlEnv, env_ctx: EnvContext
) -> TaskType:
    """Function returning a possibly new task to set `task_settable_env` to.

    Args:
        train_results (dict): The train results returned by Trainer.train().
        task_settable_env (TaskSettableEnv): A single TaskSettableEnv object
            used inside any worker and at any vector position. Use `env_ctx`
            to get the worker_index, vector_index, and num_workers.
        env_ctx (EnvContext): The env context object (i.e. env's config dict
            plus properties worker_index, vector_index and num_workers) used
            to setup the `task_settable_env`.

    Returns:
        TaskType: The task to set the env to. This may be the same as the
            current one.
    """
    if not task_settable_env.evaluation_env:
        return None
    if task_settable_env.unwrapped._has_been_assigned_map:
        map_idx = task_settable_env.cur_map_idx + env_ctx['num_eval_envs']
    else:
        # This will assign each eval env a unique map_idx from 0 to n_eval_envs - 1
        map_idx = task_settable_env.cur_map_idx + env_ctx.worker_index * env_ctx['num_envs_per_worker'] + \
            env_ctx.vector_index
        task_settable_env.unwrapped._has_been_assigned_map = True
    map_idx = map_idx % len(task_settable_env.unwrapped._prob.eval_maps)
    print(f"Assigning map {map_idx} to environment {env_ctx.vector_index} of worker {env_ctx.worker_index}.")
    return map_idx
