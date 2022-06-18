import math
from pdb import set_trace as TT
from time import sleep

import gym
from pyrsistent import s
from gym_pcgrl.envs.helper import get_string_map
import numpy as np
from ray.util.multiprocessing import Pool

import gym_pcgrl


# TODO: Adapt this function to work around static builds where possible.
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


def stretch_path_rules():
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
    return trg_out_pairs


def stretch_diameter_rules():

    # Grow in a straight line.
    trg_0 = np.array([
        [-1, 1, -1],
        [1,  1,  1],
        [1,  0,  1],
    ])
    out_0 = np.array([
        [-1, -1, -1],
        [-1,  0, -1],
        [-1, -1, -1],
    ])
    trg_1, out_1 = np.flip(trg_0, axis=0), np.flip(out_0, axis=0)
    trg_2, out_2 = np.transpose(trg_0, (1, 0)), np.transpose(out_0, (1, 0))
    trg_3, out_3 = np.flip(trg_2, axis=1), np.flip(out_2, axis=1)

    # Grow in a new direction.
    trg_4 = np.array([
        [-1,  0,  1, -1],
        [1,   0,  1,  1],
        [-1,  1,  1, -1],
        [-1, -1, -1, -1],
    ])
    out_4 = np.array([
        [-1, -1, -1, -1],
        [-1, -1,  0, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ])
    trg_5, out_5 = np.flip(trg_4, axis=0), np.flip(out_4, axis=0)
    trg_6, out_6 = np.transpose(trg_4, (1, 0)), np.transpose(out_4, (1, 0))
    trg_7, out_7 = np.flip(trg_6, axis=1), np.flip(out_6, axis=1)

    trg_8, out_8 = np.flip(trg_4, axis=1), np.flip(out_4, axis=1)
    trg_9, out_9 = np.flip(trg_8, axis=0), np.flip(out_8, axis=0)
    trg_10, out_10 = np.transpose(trg_8, (1, 0)), np.transpose(out_8, (1, 0))
    trg_11, out_11 = np.flip(trg_10, axis=1), np.flip(out_10, axis=1)

    # Dig a new hole.
    trg_12 = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    out_12 = np.array([
        [1,  1,  1],
        [1,  0,  1],
        [1,  1,  1],
    ])
    
    trg_out_pairs = [
        (trg_0, out_0),
        (trg_1, out_1),
        (trg_2, out_2),
        (trg_3, out_3),
        (trg_4, out_4),
        (trg_5, out_5),
        (trg_6, out_6),
        (trg_7, out_7),
        (trg_8, out_8),
        (trg_9, out_9),
        (trg_10, out_10),
        (trg_11, out_11),
        (trg_12, out_12),
    ]
    return trg_out_pairs


def apply_rules(arr, static, rules):
    """ arr should never have any `-1` values. """

    k = np.max([rule[0].shape[0] for rule in rules])  # assuming square rules
    p = k // 2 + 1
    arr = np.pad(arr, ((p, p), (p, p)), 'constant', constant_values=-2)  # -2 will not match to any target (?)
    static = np.pad(static, ((p, p), (p, p)), 'constant', constant_values=0) if static is not None else static 
    act = arr.copy()

    for trg, out in rules:
        # Identify tiles whose state we will need to change if applying this rule.
        edit_patch = out != trg
        edit_patch = np.where(out > -1, edit_patch, False)
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
                    if np.sum(static_patch * edit_patch) >= 1:
                        continue
                if np.sum(patch == trg) == np.sum(trg > -1):
                    return write_out(act, out, i, j, k, p)

    return act[p:-p, p:-p]


def write_out(act, out, i, j, k, p):
    curr = act[i:i+k, j:j+k]
    act[i:i+k, j:j+k] = np.where(out != -1, out, curr)
    return act[p:-p, p:-p]


adj_mask = np.array([[
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
]])


def simulate(env, render=False, rules_fn = stretch_path_rules):
    rules = rules_fn()
    env.reset()
    action = np.zeros(env.get_map_dims()[:-1], dtype=np.uint8)
    cmpt_arr = np.zeros(action.shape, np.uint16)
    diam_arr = np.zeros(action.shape, np.uint16)
    p = 1
    cmpt_arr = np.pad(cmpt_arr, p, mode='constant', constant_values=0)
    diam_arr = np.pad(diam_arr, p, mode='constant', constant_values=0)
    cmpt_id = 1
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
    if rules_fn == stretch_path_rules:
        path = construct_path(env.unwrapped._prob.entrance_coords, env.unwrapped._prob.exit_coords, 
                                *env.unwrapped._rep._bordered_map.shape)
        action[path[:, 0], path[:, 1]] = 0
        act = onehot_2chan(action)
        # hack_step(env, action)
        env.step(act)
        if render:
            env.render()
    # elif rules_fn == stretch_diameter_rules:
    done = False
    last_act = act
    # for _ in range(100):
    i = 0
    while not done:
        print(cmpt_arr)
        print(diam_arr)
        action = apply_rules(
                env.unwrapped._rep._bordered_map, 
                env.unwrapped._rep.static_builds if hasattr(env.unwrapped._rep, 'static_builds') else None,
                rules,
            )[1:-1, 1:-1]
        act = onehot_2chan(action)
        done = np.all(act == last_act) or i > 256

        # Any newly traversible tiles are assigned a component id.
        new_empty = np.argwhere((action == 0) & (cmpt_arr[p:-p, p:-p] == 0))
        while new_empty.shape[0] > 0:
            updated = False
            # If adjacent to existing components, take on their id.
            for (x, y) in new_empty:
                cmpt_patch = cmpt_arr[x-1+p:x+2+p, y-1+p:y+2+p]
                diam_patch = diam_arr[x-1+p:x+2+p, y-1+p:y+2+p]
                masked_cmpt_patch = cmpt_patch * adj_mask
                masked_diam_patch = diam_patch * adj_mask
                if np.sum(masked_cmpt_patch > 0):
                    cmpt_arr[x+p, y+p] = np.max(masked_cmpt_patch)  # assuming only one unique component adj. to us
                    diam_arr[x+p, y+p] = np.max(masked_diam_patch) + 1
                    updated = True
            # If not, let one node on the frontier belong to a new component.
            if not updated:
                (x, y) = new_empty[0]
                cmpt_arr[x+p, y+p] = cmpt_id
                diam_arr[x+p, y+p] = 1
                cmpt_id += 1
            new_empty = np.argwhere((action == 0) & (cmpt_arr[p:-p, p:-p] == 0))

        last_act = act
        # hack_step(env, action)
        env.step(act)
        if render:
            env.render()
        i += 1
    # sleep(1.5)
    stats = env.unwrapped._prob.get_stats(
        get_string_map(env.unwrapped._rep._bordered_map, env.unwrapped._prob.get_tile_types())
    )
    return stats


def test_holey_space_fill():
    n_proc = 20
    env_str = 'binary_holey-cellular-v0'
    env = gym.make(env_str)
    env.adjust_param(static_prob=0.0, holey=True)
    all_holes = env.unwrapped._prob.gen_all_holes()

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

    env.unwrapped._prob._hole_queue = all_holes
    for _ in range(len(all_holes)):
        stats = simulate(env, render=True, rules_fn=stretch_path_rules)
        print(stats)


def test_space_fill():
    n_proc = 20
    env_str = 'binary-cellular-v0'
    env = gym.make(env_str)
    env.adjust_param(
        static_prob=0.1
    )

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

    for _ in range(100):
        stats = simulate(env, render=True, rules_fn=stretch_diameter_rules)
        print(stats)


def hack_step(env, action):
    # HACK don't fix me
    env.unwrapped._rep._map = action
    env.unwrapped._rep._bordered_map[1:-1, 1:-1] = action


def onehot_2chan(x):
    return np.eye(2)[x].transpose(2, 0, 1)


if __name__ == '__main__':
    test_holey_space_fill()
    test_space_fill()