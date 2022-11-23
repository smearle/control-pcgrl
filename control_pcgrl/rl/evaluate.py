from http.cookiejar import FileCookieJar
import json
import os
from pdb import set_trace as TT
import sys

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import ray
import seaborn as sns

from control_pcgrl.rl.utils import IdxCounter


LOAD_STATS = True
CONTROL_DOORS = True
CONTROLS = False
GENERAL_EVAL = False


def evaluate(trainer, env, cfg):
    # Set controls
    eval_stats = {}
    eval_stats.update({'timesteps_total': trainer._timesteps_total})

    # Test the generator's ability to adapt to controllable door placement.
    if CONTROL_DOORS:
        if 'holey' in cfg.problem.name:
            door_stats = test_doors(trainer, env, cfg)
            eval_stats.update(door_stats)
        else:
            print('Not a holey environment, so not evaluating door placement.')
        return

    # Test the generator's ability to meet controllable metric targets.
    if CONTROLS:
        if len(cfg.controls) == 1:
            control_stats = test_control(trainer, env, cfg)
            eval_stats.update(control_stats)
        else:
            print('Not a single control, so not evaluating control.')

    # TODO: If 2 controls, test 2 controls at once. Also test each control independently.

    if GENERAL_EVAL:
        general_stats = general_eval(trainer, env, cfg)
        eval_stats.update(general_stats)

    # pop hist stats and custom metrics
    with open(os.path.join(cfg.log_dir, 'eval_stats.json'), 'w') as f:
        json.dump(eval_stats, f, indent=4)
    # pickle.dump(stats, open(os.path.join(cfg.log_dir, 'eval_stats.pkl'), 'wb'))
        

def general_eval(trainer, env, cfg):
    stats = trainer.evaluate()
    print("Evaluation stats:", stats)
    eval_stats = stats['evaluation']
    hist_stats = eval_stats['hist_stats']
    eval_stats.pop('hist_stats')
    n_eval_eps = len(hist_stats['episode_lengths'])
    eval_stats['n_eval_eps'] = n_eval_eps
    custom_stats = eval_stats['custom_metrics']
    eval_stats.pop('custom_metrics')
    eval_stats.update(custom_stats)
    eval_stats = {k: int(v) if isinstance(v, np.int64) else v for k, v in eval_stats.items()}
    return eval_stats


def test_doors(trainer, env, cfg):
    ctrl_stats_fname = f'{cfg.log_dir}/hole_stats.pkl'
    if LOAD_STATS and os.path.isfile(ctrl_stats_fname):
        ctrl_stats = pickle.load(open(ctrl_stats_fname, 'rb'))
        print(f"Loaded {len(ctrl_stats)} hole stats.")
    else:
        ctrl_stats = {}

    # trainer.evaluate() # HACK get initial episode out of the way, here we assign each env its index
    all_holes = env.unwrapped._prob.gen_all_holes()
    all_holes_total = [hole for i, hole in enumerate(all_holes) if i % 10 == 0]
    all_holes = [hole for hole in all_holes_total if (tuple(hole[0][0]), tuple(hole[1][0])) not in ctrl_stats]
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

        while len(ctrl_stats) < len(all_holes_total):
            result = trainer.evaluate()
            hist_stats = result['evaluation']['hist_stats']
            # print(result)
            if 'holes_start' in hist_stats:
                for hole_start, hole_end, path_len in zip(hist_stats['holes_start'], hist_stats['holes_end'], 
                                                            hist_stats['connected-path-length-val']):
                    ctrl_stats[(hole_start, hole_end)] = path_len
                print(f"{len(ctrl_stats)} out of {len(all_holes_total)} hole stats collected")
                # print(hole_stats)
                pickle.dump(ctrl_stats, open(ctrl_stats_fname, 'wb'))
    # print([e.unwrapped._prob.hole_queue for e in envs])
    width, height, length = cfg.map_shape

    HEATMAP = 1
    # if HEATMAP == 0:
    if True:
        # Here, we make a heatmap in which each axis is the top-down circumnference of the maze, unravelled. Take means 
        # over the height axis.
        heat = np.zeros((width * 4, width * 4))
        heat.fill(np.nan)
        fail = np.zeros((width * 4, width * 4))
        fail.fill(np.nan)
        heat_dict = {(i, j): [] for i in range(width * 4) for j in range(width * 4)}
        failed_heat_dict= {(i, j): [] for i in range(width * 4) for j in range(width * 4)}
        # failed_heat_dict = {h: {(i, j): 0 for i in range(length + 2) for j in range(width + 2)} for h in range(height - 1)}
        for hole_pair, hole_stats in ctrl_stats.items():
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
                    raise Exception
                projs[i] = proj
            proj_a, proj_b = projs
            # heat[proj_a, proj_b] = hole_stats[hole_pair]
            if hole_stats > -1:
                heat_dict[(proj_a, proj_b)] += [ctrl_stats[hole_pair]]
                failed_heat_dict[(proj_a, proj_b)] += [0]
            else:
                failed_heat_dict[(proj_a, proj_b)] += [1]
                
        num_pair = np.zeros((width * 4, width * 4))
        num_pair.fill(np.nan)
        for k in heat_dict:
            val = np.mean(heat_dict[k])
            fai = np.mean(failed_heat_dict[k])
            heat[k[0], k[1]] = val
            fail[k[0], k[1]] = fai
            num_pair[k[0], k[1]] = len(failed_heat_dict[k])


        # fig, axs = plt.subplots(1, 2)
        fig, axs = plt.subplots(1, 1)
        # resize the figure so that its contents are not overlapping
        fig.set_size_inches(7, 5)

        # Plot heatmap
        axs = sns.heatmap(heat, cmap='viridis', ax=axs, cbar=True, square=True, xticklabels=True, yticklabels=True)
        # set the interval of the x and y axis
        axs.xaxis.set_major_locator(ticker.MultipleLocator(5))
        axs.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axs.yaxis.set_major_locator(ticker.MultipleLocator(5))
        axs.yaxis.set_major_formatter(ticker.ScalarFormatter())

        axs.invert_yaxis()
        # im = ax.imshow(heat, cmap='viridis', interpolation='nearest')
        axs.set_title('Path-length between entrances/exits')
        # Set x axis name
        axs.set_xlabel('Entrance position')
        axs.set_ylabel('Exit position')
        plt.tight_layout()

        plt.savefig(os.path.join(cfg.log_dir, 'hole_heatmap_0_0.png'))
        plt.close()

        fig, axs = plt.subplots(1, 1)
        fig.set_size_inches(7, 5)
        # Plot failed heatmap using red color map
        axs = sns.heatmap(fail, cmap='Reds', ax=axs, cbar=True, square=True, xticklabels=True, yticklabels=True)
        # set the interval of the x and y axis
        axs.xaxis.set_major_locator(ticker.MultipleLocator(5))
        axs.xaxis.set_major_formatter(ticker.ScalarFormatter())
        axs.yaxis.set_major_locator(ticker.MultipleLocator(5))
        axs.yaxis.set_major_formatter(ticker.ScalarFormatter())

        axs.invert_yaxis()

        axs.set_title('Failed connections between entrances/exits')
        axs.set_xlabel('Entrance position')
        axs.set_ylabel('Exit position')

        # ## remeber to set the subplot number to 3 if you want to plot the number of pairs
        # # Plot the number of hole coordinates pairs
        # axs[2] = sns.heatmap(num_pair, cmap='viridis', ax=axs[2], cbar=True, square=True, xticklabels=True, yticklabels=True)
        # # set the interval of the x and y axis
        # axs[2].xaxis.set_major_locator(ticker.MultipleLocator(5))
        # axs[2].xaxis.set_major_formatter(ticker.ScalarFormatter())
        # axs[2].yaxis.set_major_locator(ticker.MultipleLocator(5))
        # axs[2].yaxis.set_major_formatter(ticker.ScalarFormatter())

        # axs[2].invert_yaxis()

        # axs[2].set_title('Failed connections between entrances/exits')
        # axs[2].set_xlabel('Entrance position')
        # axs[2].set_ylabel('Exit position')
        # ##

        # set suptitle
        # fig.suptitle('Heatmap of path-length between entrances/exits')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.log_dir, 'hole_heatmap_0_1.png'))
        plt.close()

    # elif HEATMAP == 1:
    if True:
        # Create heatmaps in which the x and y axes are the absolute x and y distance between doors, resepectively, and
        # each heatmap corresponds to a different height differnce.
        # Note the max width/length/height differences are quirky due to the map being bordered.
        heat_dict = {h: {(i, j): [] for i in range(length + 2) for j in range(width + 2)} for h in range(height - 1)}
        failed_heat_dict = {h: {(i, j): [] for i in range(length + 2) for j in range(width + 2)} for h in range(height - 1)}
        for hole_pair, hole_stat in ctrl_stats.items():
            if hole_stat > -1:
                (az, ay, ax), (bz, by, bx) = hole_pair
                diff_z = abs(az - bz)
                diff_x = abs(ax - bx)
                diff_y = abs(ay - by)
                heat_dict[diff_z][(diff_x, diff_y)] += [hole_stat]
                failed_heat_dict[diff_z][(diff_x, diff_y)] += [0]
            else:
                failed_heat_dict[diff_z][(diff_x, diff_y)] += [1]

        heats = {h: np.zeros((length + 2, width + 2)) for h in range(height - 1)}
        [heat.fill(np.nan) for heat in heats.values()]
        fails = {h: np.zeros((length + 2, width + 2)) for h in range(height - 1)}
        [fail.fill(np.nan) for fail in fails.values()]
        for h in heat_dict:
            for k in heat_dict[h]:
                val = np.mean(heat_dict[h][k])
                fai = np.mean(failed_heat_dict[h][k])
                heats[h][k[0], k[1]] = val
                fails[h][k[0], k[1]] = fai

        # Height many subplots
        fig, axes = plt.subplots(1, height - 1, sharex=True, sharey=True, figsize=(10, 3))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        # for h, heat in heats.items():
        for i, ax in enumerate(axes.flat):
            heat = heats[i]
            # Declare subplot
            plt.subplot(1, height - 1, i + 1)
            # Plot heatmap
            ax_s = sns.heatmap(heat, cmap='viridis', ax=ax, cbar= i==0 , square=True, xticklabels=i == 0, 
                            yticklabels= i==0, cbar_ax=None if i else cbar_ax)
            ax_s.invert_yaxis()
            # im = ax.imshow(heat, cmap='viridis', interpolation='nearest')
            # Set x axis name
            if i == 0:
                ax.set_xlabel('x difference')
                ax.set_ylabel('z difference')
                ax.set_title(f'Height (y) diff. = {i}')
            else:
                ax.set_title(i)
        # set suptitle
        fig.suptitle('Heatmap of path-length between entrances/exits')

        fig.tight_layout(rect=[0, 0, .9, 1])
        plt.savefig(os.path.join(cfg.log_dir, 'hole_heatmap_1_0.png'))
        plt.close()


        fig, axes = plt.subplots(1, height - 1, sharex=True, sharey=True, figsize=(10, 3))
        cbar_ax = fig.add_axes([.91, .3, .03, .4])
        for i, ax in enumerate(axes.flat):
            fail = fails[i]
            # Declare subplot
            plt.subplot(1, height - 1, i + 1)
            # Plot heatmap
            ax_s = sns.heatmap(fail, cmap='Reds', ax=ax, cbar= i == 0, square=True, xticklabels=i == 0, 
                            yticklabels=i == 0, cbar_ax=None if i else cbar_ax)
            ax_s.invert_yaxis()
            # im = ax.imshow(heat, cmap='viridis', interpolation='nearest')
            # Set x axis name
            if i == 0:
                ax.set_xlabel('x difference')
                ax.set_ylabel('z difference')
                ax.set_title(f'Height (y) diff. = {i}')
            else:
                ax.set_title(i)
        # set suptitle
        fig.suptitle('Heatmap of failed connection between entrances/exits')
        # plt.suptitle('Path-length between entrances/exits')
        fig.tight_layout(rect=[0, 0, .9, 1])
        plt.savefig(os.path.join(cfg.log_dir, 'hole_heatmap_1_1.png'))
        plt.close()


    # TODO: summarize the results of the evaluation with controllable door placement.
    door_stats = {
    }

    return door_stats

def test_control(trainer, env, cfg):
    ctrl_metrics = env.ctrl_metrics
    ctrl = ctrl_metrics[0]
    ctrl_bounds = env.unwrapped.cond_bounds[ctrl]
    if LOAD_STATS:
        ctrl_stats = pickle.load(open(f'{cfg.log_dir}/ctrl-{ctrl}_stats.pkl', 'rb'))
    else:
        # all_trgs = [i for i in range(int(ctrl_bounds[0]), int(ctrl_bounds[1]))]
        all_trg_ints = np.arange(ctrl_bounds[0], ctrl_bounds[1], 1)
        all_trgs = [{ctrl: v} for v in all_trg_ints]
        # Repeat certain targets so we can take the average over noisy behavior (we're assuming that eval explore=True here)
        all_trgs = all_trgs * 5
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

        ctrl_stats = {v: [] for v in all_trg_ints}
        trainer.evaluation_workers.foreach_env(lambda env: env.queue_control_trgs(idx_counter))

        n_eps = 0
        while n_eps < len(all_trgs):
            result = trainer.evaluate()
            hist_stats = result['evaluation']['hist_stats']
            print(result)
            if f'{ctrl}-trg' in hist_stats:
                for ctrl_trg, ctrl_val in zip(hist_stats[f'{ctrl}-trg'], hist_stats[f'{ctrl}-val']):
                    ctrl_stats[ctrl_trg] += [ctrl_val]
                    n_eps += 1
                print(f"{n_eps} out of {len(all_trgs)} ctrl stats collected")
                # print(hole_stats)
                pickle.dump(ctrl_stats, open(f'{cfg.log_dir}/ctrl-{ctrl}_stats.pkl', 'wb'))

    mean_ctrl_stats = {k: np.mean(v) for k, v in ctrl_stats.items()}
    fig, ax = plt.subplots(1, 1)
    xs = list(ctrl_stats.keys())
    ys = [np.mean(ctrl_stats[x]) for x in xs]
    # plt.scatter(xs, ys)
    plt.errorbar(xs, ys, yerr=[np.std(ctrl_stats[x]) for x in ctrl_stats], fmt='o')
    plt.title(f'Controlling for {ctrl}')
    # Set x axis name
    ax.set_xlabel(f'{ctrl} targets')
    ax.set_ylabel(f'{ctrl} values')
    plt.savefig(os.path.join(cfg.log_dir, f'{ctrl}_scatter.png'))

    # fig, ax = plt.subplots(1, 1)
    # ctrl_range = ctrl_bounds[1] - ctrl_bounds[0]
    # ys = [1 - np.abs(x - ctrl_stats[x]) / ctrl_range for x in xs]
    # im = ax.imshow(np.array(ys)[...,None].T, aspect="auto", cmap='viridis')
    # cbar = ax.figure.colorbar(im, ax=ax)
    # # sns.heatmap(np.array(ys), cmap='viridis', ax=ax, cbar=True, xticklabels=True, yticklabels=True)
    # plt.savefig(os.path.join(cfg.log_dir, f'{ctrl}_heatmap.png'))
    # plt.close()
    # sys.exit()
