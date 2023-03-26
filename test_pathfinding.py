from pdb import set_trace as TT
from timeit import default_timer as timer

import networkx as nx
import numpy as np

from rl.envs import make_env
from rl.utils import get_env_name
from gym_pcgrl.envs.helper import calc_longest_path, get_string_map, get_tile_locations


def get_graph(arr, passable="empty"):
    graph = nx.Graph()
    width, height = arr.shape
    size = width * height
    graph.add_nodes_from(range(size))
    # ret = scipy.sparse.csgraph.floyd_warshall(dist)
    for u in range(size):
        ux, uy = u // width, u % width
        if arr[ux, uy] != passable:
            continue
        neighbs_xy = [(ux - 1, uy), (ux, uy-1), (ux+1, uy), (ux, uy+1)]
        neighbs = [x * width + y for x, y in neighbs_xy]
        for v, (vx, vy) in zip(neighbs, neighbs_xy):
            if not 0 <= v < size or not ((0 <= vx < width) and (0 <= vy < height)) or arr[vx, vy] != passable:
                continue
            graph.add_edge(u, v)
    return graph


if __name__ == "__main__":
    problem = 'binary_ctrl'
    representation = 'narrow'
    env_name = get_env_name(problem, representation)
    cfg_dict = {
        'crop_size': 32,
        'map_width': 16,
        'max_step': 400,
        'problem': problem,
        'conditionals': [],
        'evaluate': False,
        'alp_gmm': False,
        'representation': representation,
        'env_name': env_name,
    }
    env = make_env(cfg_dict)
    env.reset()
    env.render()

    str_map = env.get_string_map(env.unwrapped._rep._map, env.unwrapped._prob.get_tile_types())
    start_time = timer()
    for _ in range(1000):
        str_map = np.array(str_map)
        width = str_map.shape[0]
        graph = get_graph(str_map, passable="empty")
        all_paths = dict(nx.all_pairs_dijkstra_path(graph))
        max_path = []
        for u, paths in all_paths.items():
            for k, path in paths.items():
                if len(path) > len(max_path):
                    max_path = path
        path_coords = [(u // width, u % width) for u in max_path]
        path_length = len(max_path)
    print("Time taken: {}".format(timer() - start_time))

    map_locations = get_tile_locations(str_map, env.unwrapped._prob.get_tile_types())
    start_time = timer()
    for _ in range(1000):
        path_length, path_coords = calc_longest_path(str_map, map_locations, ["empty"], get_path=env.unwrapped._prob.render_path)
    print("Time taken: {}".format(timer() - start_time))