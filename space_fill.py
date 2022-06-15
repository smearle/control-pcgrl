import math
from pdb import set_trace as TT
from time import sleep

import gym
from gym_pcgrl.envs.helper import get_string_map
import numpy as np
from ray.util.multiprocessing import Pool

import gym_pcgrl


STATIC_BUILDS = True


def construct_path(entrance, exit, width, height):
    (x0, y0), (x1, y1) = sorted([entrance, exit], key=lambda x: x[0])
    # Add one empty tile next to lower-x door, to ensure we can connect with two lines.
    # Which border are we on?
    holes = [(x0, y0), (x1, y1)]
    for i, (x, y) in enumerate(holes):
        if x == 0:
            x = 1
        elif x == width-1:
            x = width-2
        elif y == 0:
            y = 1
        elif y == height-1:
            y = height-2
        holes[i] = (x, y)
    (x0, y0), (x1, y1) = holes
    path0 = [(x, y0) for x in range(x0, x1)]
    y0, y1 = sorted([y0, y1])
    path1 = [(x1, y) for y in range(y0, y1+1)]
    path = path0 + path1
    path = np.array(path) - 1
    return path


def stretch_path(arr, static=None):
    """ arr should never have any `-1` values. """
    k = 5
    p = k // 2 + 1
    arr = np.pad(arr, ((p, p), (p, p)), 'constant', constant_values=-2)
    static = np.pad(static, ((p, p), (p, p)), 'constant', constant_values=0) if static is not None else static
    act = arr.copy()

    trg_0 = np.array([
        [-1, 1, -1, -1, -1],
        [0,  1,  1, -1, -1],
        [0,  1,  1, -1, -1],
        [0,  1,  1, -1, -1],
        [-1, 1, -1, -1, -1],
    ])
    out_0 = np.array([
        [-1, 1, -1, -1, -1],
        [0,  0,  1, -1, -1],
        [1,  0,  1, -1, -1],
        [0,  0,  1, -1, -1],
        [-1, 1, -1, -1, -1],
    ])
    trg_1, out_1 = np.flip(trg_0, axis=1), np.flip(out_0, axis=1)
    trg_2, out_2 = np.transpose(trg_0, (1, 0)), np.transpose(out_0, (1, 0))
    trg_3, out_3 = np.flip(trg_2, axis=0), np.flip(out_2, axis=0)

    trg_4 = np.array([
        [0,  0, -1],
        [0,  1,  1],
        [-1, 1, -1],
    ])
    out_4 = np.array([
        [1,  0, -1],
        [0,  0,  1],
        [-1, 1, -1],
    ])
    
    trg_5, out_5 = np.flip(trg_4, axis=1), np.flip(out_4, axis=1)
    trg_6, out_6 = np.transpose(trg_4, (1, 0)), np.transpose(out_4, (1, 0))
    trg_7, out_7 = np.flip(trg_6, axis=0), np.flip(out_6, axis=0)

    trg_out_pairs = [
        (trg_0, out_0),
        (trg_1, out_1),
        (trg_2, out_2),
        (trg_3, out_3),
        (trg_4, out_4),
        (trg_5, out_5),
        (trg_6, out_6),
        (trg_7, out_7),
    ]
    for trg, out in trg_out_pairs:
        # Identify tiles whose state we will need to change if applying this rule.
        static_chan = out != trg
        # Different rules may have different kernel widths.
        k = out.shape[0]
        assert k == out.shape[1]
        assert out.shape == trg.shape
        for i in range(1, arr.shape[0]-k):
            for j in range(1, arr.shape[1]-k):
                patch = arr[i:i+k, j:j+k]
                if static is not None:
                    static_patch = static[i:i+k, j:j+k]
                    # If static builds are in the way of any tile we would need to edit, we can't apply the rule.
                    print(out.shape, trg.shape)
                    if np.sum(static_patch * static_chan) >= 1:
                        continue
                if np.sum(patch == trg) == np.sum(trg > -1):
                    return write_out(act, out, i, j, k, p)
    return act[p:-p, p:-p]


def write_out(act, out, i, j, k, p):
    curr = act[i:i+k, j:j+k]
    act[i:i+k, j:j+k] = np.where(out != -1, out, curr)
    return act[p:-p, p:-p]


def simulate(env, render=False):
    env.reset()
    action = np.zeros(env.get_map_dims()[:-1], dtype=np.uint8)
    # act = onehot_2chan(action)
    # env.step(act)
    # env.render()
    action.fill(1)
    act = onehot_2chan(action)
    env.step(act)
    # print('stats after fill')
    # print(env._rep_stats)
    # print(env._rep._bordered_map)
    # print(env._rep._map)
    if render:
        env.render()
    path = construct_path(env._prob.entrance_coords, env._prob.exit_coords, 
                            *env._rep._bordered_map.shape)
    action[path[:, 0], path[:, 1]] = 0
    act = onehot_2chan(action)
    # hack_step(env, action)
    env.step(act)
    if render:
        env.render()
    done = False
    last_act = act
    # for _ in range(100):
    i = 0
    while not done:
        action = stretch_path(env._rep._bordered_map, 
            env._rep.static_builds if hasattr(env._rep, 'static_builds') else None)[1:-1, 1:-1]
        act = onehot_2chan(action)
        done = np.all(act == last_act) or i > 100
        last_act = act
        # hack_step(env, action)
        env.step(act)
        if render:
            env.render()
        i += 1
    # sleep(1.5)
    stats = env._prob.get_stats(
        get_string_map(env._rep._bordered_map, env.unwrapped._prob.get_tile_types())
    )
    return stats


def main():
    n_proc = 20
    if STATIC_BUILDS:
        env_str = 'binary_holey-cellularholeystatic-v0'
    else:
        env_str = 'binary_holey-cellularholey-v0'
    env = gym.make(env_str)
    all_holes = env._prob.gen_all_holes()
    env.adjust_param(prob_static=0.1)  # ignored if not static-build environment

    # envs = [env] + [gym.make(env_str) for _ in range(n_proc - 1)]
    # [env.adjust_param(prob_static=0.1) for env in envs]
    # pool = Pool()
    # n_env_holes = math.ceil(len(all_holes) / n_proc)
    # hole_stats = {}
    # for i, env in enumerate(envs):
    #     env._hole_queue = all_holes[i*n_env_holes:(i+1)*n_env_holes]
    # for _ in range(n_env_holes):
    #     results = pool.map(simulate, envs)
    #     for r in results:
    #         print(r)

    env._prob._hole_queue = all_holes
    for _ in range(len(all_holes)):
        stats = simulate(env, render=True)
        print(stats)


def hack_step(env, action):
    # HACK don't fix me
    env._rep._map = action
    env._rep._bordered_map[1:-1, 1:-1] = action


def onehot_2chan(x):
    return np.eye(2)[x].transpose(2, 0, 1)


if __name__ == '__main__':
    main()