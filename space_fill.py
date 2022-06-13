from pdb import set_trace as TT

import gym
import numpy as np

import gym_pcgrl

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


def stretch_path(arr):
    """ arr should never have any `-1` values. """
    k = 5
    p = k // 2 + 1
    arr = np.pad(arr, ((p, p), (p, p)), 'constant', constant_values=-2)
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

    trg_out_pairs = [
        (trg_0, out_0),
        (trg_1, out_1),
        (trg_2, out_2),
        (trg_3, out_3),
    ]
    for trg, out in [(trg_0, out_0), (trg_1, out_1), (trg_2, out_2), (trg_3, out_3)]:
        for i in range(1, arr.shape[0]-k):
            for j in range(1, arr.shape[1]-k):
                patch = arr[i:i+k, j:j+k]
                if np.sum(patch == trg) == 11:
                    return write_out(act, out, i, j, k, p)
    return act[p:-p, p:-p]


def write_out(act, out, i, j, k, p):
    curr = act[i:i+k, j:j+k]
    act[i:i+k, j:j+k] = np.where(out != -1, out, curr)
    return act[p:-p, p:-p]



def main():
    env = gym.make('binary_holey-cellularholey-v0')
    for _ in range(1000):
        env.reset()
        action = np.zeros(env._rep._map.shape, dtype=np.uint8)
        act = onehot_2chan(action)
        env.step(act)
        env.render()
        action.fill(1)
        act = onehot_2chan(action)
        env.step(act)
        env.render()
        path = construct_path(env._prob.entrance_coords, env._prob.exit_coords, 
                                *env._rep._bordered_map.shape)
        action[path[:, 0], path[:, 1]] = 0
        act = onehot_2chan(action)
        env.step(act)
        env.render()
        for _ in range(100):
            action = stretch_path(env._rep._bordered_map)[1:-1, 1:-1]
            act = onehot_2chan(action)
            env.step(act)
            env.render()

def onehot_2chan(x):
    return np.eye(2)[x].transpose(2, 0, 1)

if __name__ == '__main__':
    main()