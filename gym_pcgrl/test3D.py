"""
Run random agent to test the 3D environment
"""
import numpy as np
import gym
import gym_pcgrl
from pdb import set_trace as TT
# from utils import make_vec_envs
from gym_pcgrl.envs.helper_3D import calc_num_regions, debug_path, get_string_map,\
                                        get_tile_locations, calc_longest_path, run_dijkstra
import matplotlib.pyplot as plt

################################################################################
# test the helper functions
tile_types = ["AIR", "DIRT"]

######## Test the path finding func and region counting func in stairing logic #########
# Note: the path length is calculated by the dijkstra map in helper_3D.py (the max value in the dijkstra map), which 
# starts the counting from 0. As a result, the path length is 1 less than the actual path length / len(path_coords).
# test_map_1: 
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 29 = 59
test_map_1 = {
    "name": "test_map_1",
    "map": [
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
],
    "size": (7, 7, 5),
    "path_length": 58,
    "region_number": 1,
    "info": "Two-layer test map with perpendicular corridors."
}


# test_map_2:
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 27 = 57
test_map_2 = {
    "name": "test_map_2",
    "map": [
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
], 
    "size": (7, 7, 5),
    "path_length": 54, # previously I think it was 56, but it's not because there's a short cut at the beginning of the second layer
    "region_number": 1,
    "info": "Two-layer test map with parallel corridor",
}


# test_map_3:
# size: 7 * 7 * 5
# longest path length: 28 + 2 + 27 = 57
# info: identical to test_map_2, except that some unnecessary tiles are removed (to test region number)
test_map_3 = {
    "name": "test_map_3",
    "map": [
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
],
    "size": (7, 7, 5),
    "path_length": 54,
    "region_number": 1,
    "info": "identical to test_map_2, except that some unnecessary tiles are removed (to test region number)",
}


# test_map_4:
# size: 3 * 6 * 6
# longest path length: 2 + 1 + 1 + 1 = 5
# info: small map for testing climbing stairs
test_map_4 = {
    "name": "test_map_4",
    "map": [
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
],
    "size": (3, 6, 6),
    "path_length": 4,
    "region_number": 1,
    "info": "small map for testing climbing stairs",
}


########### For testing the 3D plotting ###########
# test_map_5:
# size: 3 * 3 * 3
test_map_5 = {
    "name": "test_map_5",
    "map": [
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
],
    "size": (3, 3, 3),
}


############ Test the path finding func in the jumping logic #############
# Note: In Minecraft jumping, the extra head room of the staring position and extra head room of the position 1 before 
# foothold needs to be garanteded
# 
#       |__
# O
# 大_    __
#   |   |
#   |   |                                                          



# test_map_6:
# size: 5 * 1 * 6
# This is the max jump distance in Minecraft (press double w + space to jump)
# path length: 2
# region number: 1
# jump: 1
# jump distance: 3
test_map_6 = {
    "name": "test_map_6",
    "map": [
    [
        [1, 0, 0, 0, 1]    
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
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
    "size": (5, 1, 6),
    "jump_distance": 3,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 0,
    "info": "valid jump",
}


# test_map_7:
# size: 5 * 1 * 6
# This is the max jump distance in Minecraft (press double w + space to jump)
# path length: 2
# region number: 1
# jump: 1
# jump distance: 3
# info: valid jump, the head room of the foothold position is trivial
test_map_7 = {
    "name": "test_map_7",
    "map": [
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 1]     # the head room of the foothold position is trivial
    ]
],
    "size": (5, 1, 6),
    "jump_distance": 3,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 0,
    "info": "valid jump, the head room of the foothold position is trivial",
}

# test_map_8:
# size: 5 * 1 * 6
# This is the max jump distance in Minecraft (press double w + space to jump)
# path length: 1
# region number: 1
# jump: 0
# jump distance: 3
# info: head blocked in starting position in either direction
test_map_8 = {
    "name": "test_map_8",
    "map": [
    [
        [1, 0, 0, 0, 1]    
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [1, 0, 0, 0, 1]     # head blocked in starting position in either direction
    ]
],
    "size": (5, 1, 6),
    "jump_distance": 3,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 0,
    "info": "head blocked in starting position in either direction",
}



# test_map_9:
# size: 5 * 1 * 6
# This is the max jump distance in Minecraft (press double w + space to jump)
# path length: 1
# region number: 1
# jump: 0
# jump distance: 3
# info: head blocked in the position before foothold position
test_map_9 = {
    "name": "test_map_9",
    "map": [
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [1, 0, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 1, 1]     # head blocked in the position before foothold position
    ]
],
    "size": (5, 1, 6),
    "jump_distance": 3,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 0,
    "info": "head blocked in the position before foothold position",
}


# test_map_10:
# size: 4 * 1 * 6
# jump distance: 2
# path length: 2
# region number: 1
# jump: 1
test_map_10 = {
    "name": "test_map_10",
    "map": [
    [
        [1, 0, 0, 1]
    ],
    [
        [1, 0, 0, 1]
    ],
    [
        [1, 0, 0, 1]
    ],
    [
        [1, 0, 0, 1]
    ],
    [
        [0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0]
    ],
    [
        [0, 0, 0, 0]
    ]
],
    "size": (4, 1, 6),
    "jump_distance": 2,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 0,
    "info": "valid jump",
}


# test_map_11:
# size: 3 * 1 * 6
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
test_map_11 = {
    "name": "test_map_11",
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
        [0, 0, 0]
    ]
],
    "size": (3, 1, 6),
    "jump_distance": 1,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 0,
    "info": "valid jump",
}


# test_map_12:
# size: 3 * 1 * 6
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
# height difference: 1
# info: the height difference of starting point and foothold position is 1
test_map_12 = {
    "name": "test_map_12",
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
        [1, 0, 0]       # the height difference of starting point and foothold position is 1
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 0]     
    ]
],
    "size": (3, 1, 6),
    "jump_distance": 1,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 1,
    "info": "the height difference of starting point and foothold position is 1",
}


# test_map_13:
# size: 3 * 1 * 6
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
# height difference: 2
# info: the height difference of starting point and foothold position is 2
test_map_13 = {
    "name": "test_map_13",
    "map": [
    [
        [1, 0, 1]
    ],
    [
        [1, 0, 1]
    ],
    [
        [1, 0, 0]
    ],
    [
        [1, 0, 0]       # the height difference of starting point and foothold position is 2
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 0]
    ]
],
    "size": (3, 1, 6),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 2,
    "info": "the height difference of starting point and foothold position is 1",
}


# test_map_14:
# size: 3 * 1 * 6
# jump distance: 1
# path length: 1
# region number: 1
# jump: 0
# height difference: 0
# info: head blocked in starting position in either direction
test_map_14 = {
    "name": "test_map_14",
    "map":[
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
        [1, 0, 1]
    ]
],
    "size": (3, 1, 6),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 0,
    "info": "head blocked in starting position in either direction",
}

# test_map_15:
# size: 3 * 1 * 6
# jump distance: 1
# path length: 1
# region number: 1
# jump: 0
# height difference: 0
# info: head blocked in foothold position
test_map_15 = {
    "name": "test_map_15",
    "map": 
    [
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
        [0, 1, 1]
    ]
],
    "size": (3, 1, 6),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 0,
    "info": "head blocked in foothold position",
}


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


# test_map_17:
# size: 3 * 1 * 7
# jump distance: 1
# path length: 1
# region number: 1
# jump: 0
# height difference: -1
# info: valid jump but not returnable, so we don't count the jump
test_map_17 = {
    "name": "test_map_17",
    "map":[
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
        [1, 0, 0]
    ],
    [
        [0, 0, 0]
    ],
    [
        [0, 0, 1]
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
    "height_difference": -1,
    "info": "valid jump but not returnable, so we don't count the jump",
}


# test_map_18:
# size: 3 * 1 * 8
# jump distance: 1
# path length: 1
# region number: 1
# jump: 0
# height difference: 1
# info: valid jump but not returnable, so we don't count the jump
test_map_18 = {
    "name": "test_map_18",
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
            [1, 0, 0]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ],
        [
            [1, 0, 0]
        ],
        [
            [1, 0, 0]
        ]
    ],
    "size": (3, 1, 8),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": 1,
    "info": "valid jump but not returnable, so we don't count the jump",
}


# test_map_19:
# size: 3 * 1 * 7
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
# height difference: -1
# info: valid jump
test_map_19 = {
    "name": "test_map_19",
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
            [1, 0, 0]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ]
    ],
    "size": (3, 1, 7),
    "jump_distance": 1,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": -1,
    "info": "valid jump",
}


# test_map_20:
# size: 3 * 1 * 7
# jump distance: 1
# path length: 1
# region number: 1
# jump: 0
# height difference: 1
# info: valid jump
test_map_20 = {
    "name": "test_map_20",
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
            [0, 0, 1]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ]
    ],
    "size": (3, 1, 7),
    "jump_distance": 1,
    "path_length": 0,
    "region_number": 1,
    "jump": 0,
    "height_difference": -1,
    "info": "valid jump, but not returnable",
}


# test_map_21:
# size: 3 * 1 * 7
# jump distance: 1
# path length: 2
# region number: 1
# jump: 1
# height difference: 1
# info: valid jump
test_map_21 = {
    "name": "test_map_21",
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
            [0, 0, 1]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ],
        [
            [0, 0, 0]
        ]
    ],
    "size": (3, 1, 7),
    "jump_distance": 1,
    "path_length": 1,
    "region_number": 1,
    "jump": 1,
    "height_difference": 1,
    "info": "valid jump",
}
# TODO: test map for falling distance > 1 and <= 3

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


"""
get the state of the test maps
"""
def get_test_state(map_list, tile_types):
    all_pass_test = True
    if len(map_list) == 0:
        raise ValueError("test_map_list is empty")
    for test_map_dict in map_list:
        test_map = np.array(test_map_dict["map"])
        test_string_map = get_string_map(test_map, tile_types)
        map_locations = get_tile_locations(test_string_map, tile_types)

        # get the state of the test map
        path_length, path_coords, n_jump = calc_longest_path(test_string_map, map_locations, ["AIR"], get_path=True)
        num_regions = calc_num_regions(test_string_map, map_locations, ["AIR"])
        debug_path_coords = debug_path(path_coords, test_string_map, ["AIR"])

        pass_test = path_length == test_map_dict["path_length"] and \
                    num_regions == test_map_dict["region_number"] and \
                    n_jump == test_map_dict["jump"] and \
                    debug_path_coords

        if not pass_test:
            print("-------------------------")
            print(f"Testing on {test_map_dict['name']}")
            print(
                f"longest path length: {path_length}, it should be {test_map_dict['path_length']}, "
                f"pass the test? {path_length == test_map_dict['path_length']}")
            print(
                f"num_regions: {num_regions}, it should be {test_map_dict['region_number']}, pass the test? {num_regions == test_map_dict['region_number']}")
            print(f"The path is valid? {debug_path_coords}") 
        all_pass_test = all_pass_test and pass_test
    print(f"All tests passed? {all_pass_test}")
    return path_length, path_coords, num_regions


"""
plot the test maps using matplotlib 3D voxel / volumetric plotting
"""
def plot_3d_map(test_map):
    test_map = np.array(test_map)

    # change the map axis for plotting
    test_map = np.moveaxis(test_map, (0, 2), (2, 1))

    # create the boolen map of the maze
    boolen_map = np.array(test_map) == 1

    # create the color map of the maze
    color_map = np.empty(test_map.shape, dtype=object)
    color_map[boolen_map] = "green"

    # plot it out!
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_box_aspect([test_map.shape[0]/test_map.shape[1],
                         1,
                         test_map.shape[2]/test_map.shape[1]])
    # ax.set_box_aspect([1,
    #                    1,
    #                    5/7])
    print('test_map.shape:', test_map.shape)
    ax.voxels(boolen_map, facecolors=color_map, edgecolor='k')
    plt.show()



if __name__=="__main__":
    ################################################################################
    # test the 3D environment

    # env = gym.make('minecraft_3D_zelda-narrow3D-v0')
    # while True:
    #     observation = env.reset()
    #     for step in range(500):
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         print(env._rep_stats)
    #         env.render()

    ################################################################################
    # test the path finding algorithm

    # # path_length_1, path_coords_1, num_regions_1 = get_test_state(test_map_1, tile_types)
    # # path_length_2, path_coords_2, num_regions_2 = get_test_state(test_map_2, tile_types)
    # # path_length_3, path_coords_3, num_regions_3 = get_test_state(test_map_3, tile_types)
    # path_length_4, path_coords_4, num_regions_4 = get_test_state(test_map_4, tile_types)
    
    # dijkstra_map_4, _ = run_dijkstra(1, 0, 0, get_string_map(np.array(test_map_4), tile_types), ["AIR"])
    # print("dijkstra_map_4 is \n", dijkstra_map_4)

    ################################################################################
    # test the 3D plotting using matplotlib 3D voxel / volumetric plotting

    # plot_3d_map(test_map_5)

    ################################################################################
    # test the jumping logic
    # jumping distance: 1
    test_map_list = []
    for i in range(11, 22):
        test_map_list.append(globals()[f"test_map_{i}"])
    path_length, path_coords, num_regions = get_test_state(test_map_list, tile_types)

