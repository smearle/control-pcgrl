""" 
This file is for debugging path-finding in Minecraft problems, using hand-crafted levels. 

We need to hackishly set the entire level to be equal to the hand-crafted array, then run the path-finding algorithm by calling get_stats().
"""
from pdb import set_trace as TT

import gym
import numpy as np
import gym_pcgrl
from gym_pcgrl.envs.helper_3D import get_string_map

# shape = (height, width, length)
staircase = np.ones((7, 7, 7))
staircase[0:3, 0, 0] = 0
staircase[0:3, 1, 0] = 0
staircase[1:4, 2, 0] = 0
staircase[2:5, 3, 0] = 0

# test_map_1: 
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 29 = 59
test_map_1 = [
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ],
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1]
    ],
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 1, 0]
    ],
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0]
    ]
]


# test_map_2:
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 25 = 55
test_map_2 = [
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ],
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ],
    [
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ]
]


# test_map_3:
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 27 = 57
# info: identical to test_map_2, except that some unnecessary tiles are removed (to test region number)
test_map_3 = [
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 0]
    ],
    [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1]
    ],
    [
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]           # diff: [0, 0, 0, 1, 0, 0, 0] in test_map_2
    ],
    [
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0]           # diff: [0, 0, 1, 1, 0, 0, 0] in test_map_2
    ]
]


# test_map_4:
# size: 3 * 6 * 6
# longest path length: 2 + 1 + 1 + 1 = 5
# info: small map for testing climbing stairs
test_map_4 = [
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1], 
        [1, 1, 1]
    ],
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
]


# test_map_16:
# size: 3 * 1 * 7
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
# height difference: 0
# info: valid jump
test_map_16 = {
    "name": "test_map_16",
    "map": [
    [
        [1, 0, 1]
    ],
    [
        [1, 0, 1]
    ],
    [
        [1, 0, 1]
    ],
    [
        [1, 0, 1]
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 1]
    ]
],
    "size": (3, 1, 7),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 0,
    "info": "valid jump",
}

# test_map_22:
# size: 5 * 1 * 7
# jump distance: 1
# path length: 4
# region number: 1
# jump: 1
# height difference: 0
# info: valid jump
test_map_22 = {
    "name": "test_map_22",
    "map": [
        [
            [1, 1, 0, 1, 1]
        ],
        [
            [1, 1, 0, 1, 1]
        ],
        [
            [1, 1, 0, 1, 1]
        ],
        [
            [1, 1, 0, 1, 1]
        ],
        [
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0]
        ],
        [
            [0, 0, 0, 0, 0]
        ]
    ],
    "size": (5, 1, 7),
    "jump_distance": 1,
    "path_length": 4,
    "region_number": 1,
    "jump": 1,
    "height_difference": 0,
    "info": "valid jump",
}

# Here we have a path in the shape of a loop, at a high level. The player needs to jump over the same chasm twice to 
# reach the end.
test_map_crossjump = {
    "name": "test_map_22",
    "map": [
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
    ],
    "size": (5, 5, 6),
    "jump_distance": 2,
    "path_length": 13,
    "region_number": 1,
    "jump": 2,
    "height_difference": 0,
    "info": "valid jump",
}

test_maps = [np.array(test_map_crossjump["map"])]

for int_map in test_maps:
    # FIXME: why does this seem to take so long? Path-finding is fast. Just creating the environment itself?
    prob = "minecraft_3D_maze_ctrl"
    repr = "narrow3D"
    env_name = f"{prob}-{repr}-v0"
    env = gym.make(env_name)
    env.adjust_param(render=True, change_percentage=None)
    # env.unwrapped._rep._x = env.unwrapped._rep._y = 0
    env.unwrapped._rep._old_coords = env.unwrapped._rep._new_coords = (0, 0, 0)
    env._rep._map = int_map
    env.render()
    stats = env.unwrapped._prob.get_stats(
        get_string_map(int_map, env.unwrapped._prob.get_tile_types()),
    )
    print(stats)
    env.render()

