import json
import os
from pdb import set_trace as TT
import sys

from matplotlib import pyplot as plt
import numpy as np
import pickle
import ray
import seaborn as sns

from utils import IdxCounter


LOAD_STATS = False
CONTROL_DOORS = False
GENERAL_EVAL = True


def evaluate(trainer, env, cfg):
    # Set controls
    if 'holey' in cfg.env_name:
        if CONTROL_DOORS:
            test_doors(trainer, env, cfg)

    if len(cfg.conditionals) == 1:
        test_control(trainer, env, cfg)

    # TODO: If 2 controls, test 2 controls at once. Also test each control independently.

    elif GENERAL_EVAL:
        stats = trainer.evaluate()
        print("Evaluation stats:", stats)
        eval_stats = stats['evaluation']
        hist_stats = eval_stats['hist_stats']
        eval_stats.pop('hist_stats')
        n_eval_eps = list(hist_stats['episode_lengths'])
        eval_stats['n_eval_eps'] = n_eval_eps
        custom_stats = eval_stats['custom_metrics']
        eval_stats.pop('custom_metrics')
        eval_stats.update(custom_stats)
        eval_stats = {k: int(v) if isinstance(v, np.int64) else v for k, v in eval_stats.items()}
        # pop hist stats and custom metrics
        with open(os.path.join(cfg.log_dir, 'eval_stats.json'), 'w') as f:
            json.dump(eval_stats, f)
        # pickle.dump(stats, open(os.path.join(cfg.log_dir, 'eval_stats.pkl'), 'wb'))


def test_doors(trainer, env, cfg):
    if LOAD_STATS:
        ctrl_stats = pickle.load(open(f'{cfg.log_dir}/hole_stats.pkl', 'rb'))
        print(f"Loaded {len(ctrl_stats)} hole stats.")
    else:
        ctrl_stats = {}

    # trainer.evaluate() # HACK get initial episode out of the way, here we assign each env its index
    all_holes = env.unwrapped._prob.gen_all_holes()
    all_holes = [hole for i, hole in enumerate(all_holes) if i % 1 == 0]
    all_holes = [hole for hole in all_holes if (tuple(hole[0][0]), tuple(hole[1][0])) not in ctrl_stats]
    n_envs = max(1, cfg.num_workers) * cfg.num_envs_per_worker
    if len(all_holes) >= n_envs:
        # holes_tpl = [tuple([tuple([coord for coord in hole]) for hole in hole_pair]) for hole_pair in all_holes]
        env_hole_int = len(all_holes) // n_envs
        env_holes = [all_holes[env_hole_int * i:env_hole_int * (i + 1)] for i in range(n_envs)]
        envs = trainer.evaluation_workers.foreach_env(lambda env: env)
        envs = [env for worker_env in envs for env in worker_env]
        idx_counter = IdxCounter.options(name='idx_counter', max_concurrency=1).remote()
        idx_counter.set_keys.remote(all_holes)
        hashes = trainer.evaluation_workers.foreach_env(lambda env: hash(env.unwrapped._prob))
        hashes = [hash for worker_hash in hashes for hash in worker_hash]
        # hashes = [hash(env.unwrapped._prob) for env in envs]
        idx_counter.set_hashes.remote(hashes)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        assert ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        # trainer.workers.foreach_worker(
            # lambda worker: worker.foreach_env(lambda env: env.queue_worlds(worlds=eval_mazes, idx_counter=idx_counter, load_now=True)))

        trainer.evaluation_workers.foreach_env(lambda env: env.unwrapped._prob.queue_holes(idx_counter))

        while len(ctrl_stats) < len(all_holes):
            result = trainer.evaluate()
            hist_stats = result['evaluation']['hist_stats']
            # print(result)
            if 'holes_start' in hist_stats:
                for hole_start, hole_end, path_len in zip(hist_stats['holes_start'], hist_stats['holes_end'], 
                                                            hist_stats['connected-path-length-val']):
                    ctrl_stats[(hole_start, hole_end)] = path_len
                print(f"{len(ctrl_stats)} out of {len(all_holes)} hole stats collected")
                # print(hole_stats)
                pickle.dump(ctrl_stats, open(f'{cfg.log_dir}/hole_stats.pkl', 'wb'))
    # print([e.unwrapped._prob.hole_queue for e in envs])
    width = env.width
    heat = np.zeros((width * 4, width * 4))
    heat.fill(np.nan)
    heat_dict = {(i, j): [] for i in range(width * 4) for j in range(width * 4)}
    for hole_pair in ctrl_stats:
        projs = [None, None]
        (ax, ay, az), (bx, by, bz) = hole_pair
        for i, (z, y, x) in enumerate([(ax, ay, az), (bx, by, bz)]):
            if x == 0 :
                proj = y
            elif y == width +1:
                proj = width + x
            elif x == width + 1:
                proj = 3 * width - y - 1
            elif y == 0:
                proj = 4 * width - x - 1
            else:
                TT()
                raise Exception
            projs[i] = proj
        proj_a, proj_b = projs
        # heat[proj_a, proj_b] = hole_stats[hole_pair]
        heat_dict[(proj_a, proj_b)] = ctrl_stats[hole_pair]

    for k in heat_dict:
        val = np.mean(heat_dict[k])
        heat[k[0], k[1]] = val

    fig, ax = plt.subplots(1, 1)
    # Plot heatmap
    ax_s = sns.heatmap(heat, cmap='viridis', ax=ax, cbar=True, square=True, xticklabels=True, yticklabels=True)
    ax_s.invert_yaxis()
    # im = ax.imshow(heat, cmap='viridis', interpolation='nearest')
    plt.title('Path-length between entrances/exits')
    # Set x axis name
    ax.set_xlabel('Entrance position')
    ax.set_ylabel('Exit position')
    plt.savefig(os.path.join(cfg.log_dir, 'hole_heatmap.png'))

def test_control(trainer, env, cfg):
    ctrl_metrics = env.ctrl_metrics
    ctrl = ctrl_metrics[0]
    ctrl_bounds = env.unwrapped.cond_bounds[ctrl]
    if LOAD_STATS:
        ctrl_stats = pickle.load(open(f'{cfg.log_dir}/ctrl-{ctrl}_stats.pkl', 'rb'))
    else:
        # all_trgs = [i for i in range(int(ctrl_bounds[0]), int(ctrl_bounds[1]))]
        all_trgs = np.arange(ctrl_bounds[0], ctrl_bounds[1], 1)
        all_trgs = [{ctrl: v} for v in all_trgs]
        # holes_tpl = [tuple([tuple([coord for coord in hole]) for hole in hole_pair]) for hole_pair in all_holes]
        n_envs = max(1, cfg.num_workers) * cfg.num_envs_per_worker
        idx_counter = IdxCounter.options(name='idx_counter').remote()
        idx_counter.set_keys.remote(all_trgs)
        hashes = trainer.evaluation_workers.foreach_env(lambda env: hash(env))
        hashes = [hash for worker_hash in hashes for hash in worker_hash]
        # hashes = [hash(env.unwrapped._prob) for env in envs]
        idx_counter.set_hashes.remote(hashes)
        # FIXME: Sometimes hash-to-idx dict is not set by the above call?
        ret = ray.get(idx_counter.scratch.remote())
        # Assign envs to worlds
        # trainer.workers.foreach_worker(
            # lambda worker: worker.foreach_env(lambda env: env.queue_worlds(worlds=eval_mazes, idx_counter=idx_counter, load_now=True)))

        ctrl_stats = {}
        trainer.evaluation_workers.foreach_env(lambda env: env.queue_control_trgs(idx_counter))

        while len(ctrl_stats) < len(all_trgs):
            result = trainer.evaluate()
            hist_stats = result['evaluation']['hist_stats']
            print(result)
            if f'{ctrl}-trg' in hist_stats:
                for ctrl_trg, ctrl_val in zip(hist_stats[f'{ctrl}-trg'], hist_stats[f'{ctrl}-val']):
                    ctrl_stats[ctrl_trg] = ctrl_val
                print(f"{len(ctrl_stats)} out of {len(all_trgs)} ctrl stats collected")
                # print(hole_stats)
                pickle.dump(ctrl_stats, open(f'{cfg.log_dir}/ctrl-{ctrl}_stats.pkl', 'wb'))

    fig, ax = plt.subplots(1, 1)
    xs = list(ctrl_stats.keys())
    ys = [ctrl_stats[x] for x in xs]
    plt.scatter(xs, ys)
    plt.title(f'Controlling for {ctrl}')
    # Set x axis name
    ax.set_xlabel(f'{ctrl} targets')
    ax.set_ylabel(f'{ctrl} values')
    plt.savefig(os.path.join(cfg.log_dir, f'{ctrl}_scatter.png'))

    fig, ax = plt.subplots(1, 1)
    ctrl_range = ctrl_bounds[1] - ctrl_bounds[0]
    ys = [1 - np.abs(x - ctrl_stats[x]) / ctrl_range for x in xs]
    im = ax.imshow(np.array(ys)[...,None].T, aspect="auto", cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    # sns.heatmap(np.array(ys), cmap='viridis', ax=ax, cbar=True, xticklabels=True, yticklabels=True)
    plt.savefig(os.path.join(cfg.log_dir, f'{ctrl}_heatmap.png'))
    plt.close()
    sys.exit()
