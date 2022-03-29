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

################################################################################
# test the helper functions
tile_types = ["AIR", "DIRT"]

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
# longest path length: 28 + 2 + 27 = 57
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

"""
get the state of the test maps
"""
def get_test_state(test_map, tile_types):
    test_map = np.array(test_map)
    test_string_map = get_string_map(test_map, tile_types)
    map_locations = get_tile_locations(test_string_map, tile_types)

    # get the state of the test map
    path_length, path_coords = calc_longest_path(test_string_map, map_locations, ["AIR"], get_path=True)
    num_regions = calc_num_regions(test_string_map, map_locations, ["AIR"])
    debug_path_coords = debug_path(path_coords, test_string_map, ["AIR"])
    print("longest path length:", path_length)
    print("number of regions:", num_regions)
    print(f"The path is: {debug_path_coords}")
    return path_length, path_coords, num_regions


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
  
    # path_length_1, path_coords_1, num_regions_1 = get_test_state(test_map_1, tile_types)
    # path_length_2, path_coords_2, num_regions_2 = get_test_state(test_map_2, tile_types)
    # path_length_3, path_coords_3, num_regions_3 = get_test_state(test_map_3, tile_types)
    path_length_4, path_coords_4, num_regions_4 = get_test_state(test_map_4, tile_types)
    
    dijkstra_map_4, _ = run_dijkstra(1, 0, 0, get_string_map(np.array(test_map_4), tile_types), ["AIR"])
    print("dijkstra_map_4 is \n", dijkstra_map_4)
    """
    dijkstra_map_1
    array([[[ 0,  1,  2,  3,  4,  5,  6],
            [-1, -1, -1, -1, -1, -1,  7],
            [14, 13, 12, 11, 10,  9,  8],
            [15, -1, -1, -1, -1, -1, -1],
            [16, 17, 18, 19, 20, 21, 22],
            [-1, -1, -1, -1, -1, -1, 23],
            [-1, -1, -1, 27, 26, 25, 24]],

           [[-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1]],

           [[-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1]],

           [[-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1]],

           [[-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1]]])

        still some issues with the passable function
"""
    print(path_coords_4)
