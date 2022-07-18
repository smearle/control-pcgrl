"""
A helper module that can be used by all problems
"""
import numpy as np
from pdb import set_trace as TT

"""
Public function to get a dictionary of all location of all tiles

Parameters:
    map (any[][]): the current map
    tile_values (any[]): an array of all the tile values that are possible

Returns:
    Dict(string,(int,int)[]): positions for every certain tile_value
"""
def get_tile_locations(map, tile_values):
    tiles = {}
    for t in tile_values:
        tiles[t] = []
    for y in range(len(map)):
        for x in range(len(map[y])):
            tiles[map[y][x]].append((x,y))
    return tiles

"""
Get the vertical distance to certain type of tiles

Parameters:
    map (any[][]): the actual map
    x (int): the x position of the start location
    y (int): the y position of the start location
    types (any[]): an array of types of tiles

Returns:
    int: the distance to certain types underneath a certain location
"""
def _calc_dist_floor(map, x, y, types):
    for dy in range(len(map)):
        if y+dy >= len(map):
            break
        if map[y+dy][x] in types:
            return dy-1
    return len(map) - 1

"""
Public function to calculate the distance of a certain tiles to the floor tiles

Parameters:
    map (any[][]): the current map
    from (any[]): an array of all the tile values that the method is calculating the distance to the floor
    floor (any[]): an array of all the tile values that are considered floor

Returns:
    int: a value of how far each tile from the floor where 0 means on top of floor and positive otherwise
"""
def get_floor_dist(map, fromTypes, floorTypes):
    result = 0
    for y in range(len(map)):
        for x in range(len(map[y])):
            if map[y][x] in fromTypes:
                result += _calc_dist_floor(map, x, y, floorTypes)
    return result

"""
Get number of tiles that have certain value arround certain position

Parameters:
    map (any[][]): the current map
    x (int): the x position of the start location
    y (int): the y position of the start location
    types (any[]): an array of types of tiles
    relLocs ((int,int)[]): a tuple array of all the relative positions

Returns:
    int: the number of similar tiles around a certain location
"""
def _calc_group_value(map, x, y, types, relLocs):
    result = 0
    for l in relLocs:
        nx, ny = x+l[0], y+l[1]
        if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
            continue
        if map[ny][nx] in types:
            result += 1
    return result

"""
Get the number of tiles that is a group of certain size

Parameters:
    map (any[][]): the current map
    types (any[]): an array of types of tiles
    relLocs ((int,int)[]): a tuple array of all the relative positions
    min (int): min number of tiles around
    max (int): max number of tiles around

Returns:
    int: the number of tiles that have surrounding between min and max
"""
def get_type_grouping(map, types, relLocs, min, max):
    result = 0
    for y in range(len(map)):
        for x in range(len(map[y])):
            if map[y][x] in types:
                value = _calc_group_value(map, x, y, types, relLocs)
                if value >= min and value <= max:
                    result += 1
    return result

"""
Get the number of changes of tiles in either vertical or horizontal direction

Parameters:
    map (any[][]): the current map
    vertical (boolean): calculate the vertical changes instead of horizontal

Returns:
    int: number of different tiles either in vertical or horizontal direction
"""
def get_changes(map, vertical=False):
    start_y = 0
    start_x = 0
    if vertical:
        start_y = 1
    else:
        start_x = 1
    value = 0
    for y in range(start_y, len(map)):
        for x in range(start_x, len(map[y])):
            same = False
            if vertical:
                same = map[y][x] == map[y-1][x]
            else:
                same = map[y][x] == map[y][x-1]
            if not same:
                value += 1
    return value

"""
Private function to get a list of all tile locations on the map that have any of
the tile_values

Parameters:
    map_locations (Dict(string,(int,int)[])): the histogram of locations of the current map
    tile_values (any[]): an array of all the tile values that the method is searching for

Returns:
    (int,int)[]: a list of (x,y) position on the map that have a certain value
"""
def _get_certain_tiles(map_locations, tile_values):
    tiles=[]
    for v in tile_values:
        tiles.extend(map_locations[v])
    return tiles

"""
Private function that runs flood fill algorithm on the current color map

Parameters:
    x (int): the starting x position of the flood fill algorithm
    y (int): the starting y position of the flood fill algorithm
    color_map (int[][]): the color map that is being colored
    map (any[][]): the current tile map to check
    color_index (int): the color used to color in the color map
    passable_values (any[]): the current values that can be colored over

Returns:
    int: the number of tiles that has been colored
"""
def _flood_fill(x, y, color_map, map, color_index, passable_values):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map[cy][cx] not in passable_values:
            continue
        num_tiles += 1
        color_map[cy][cx] = color_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny))
    return num_tiles

"""
Calculates the number of regions in the current map with passable_values

Parameters:
    map (any[][]): the current map being tested
    map_locations(Dict(string,(int,int)[])): the histogram of locations of the current map
    passable_values (any[]): an array of all the passable tile values

Returns:
    int: number of regions in the map
"""
def calc_num_regions(map, map_locations, passable_values):
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    region_index=0
    color_map = np.full((len(map), len(map[0])), -1)
    for (x,y) in empty_tiles:
        num_tiles = _flood_fill(x, y, color_map, map, region_index + 1, passable_values)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index


"""
Public function that runs dijkstra algorithm and return the map

Parameters:
    x (int): the starting x position for dijkstra algorithm
    y (int): the starting y position for dijkstra algorithm
    map (any[][]): the current map being tested
    passable_values (any[]): an array of all the passable tile values

Returns:
    int[][]: returns the dijkstra map after running the dijkstra algorithm
"""
def run_dijkstra(x, y, map, passable_values):
    dijkstra_map = np.full((len(map), len(map[0])),-1)
    visited_map = np.zeros((len(map), len(map[0])))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map[cy][cx] not in passable_values or (dijkstra_map[cy][cx] >= 0 and dijkstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dijkstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny, cd + 1))
    return dijkstra_map, visited_map

ADJ_FILTER = np.array([[0,1,0],[1,0,1],[0,1,0]])

"""
Calculate the longest path on the map

Parameters:
    map (any[][]): the current map being tested
    map_locations (Dict(string,(int,int)[])): the histogram of locations of the current map
    passable_values (any[]): an array of all passable tiles in the map

Returns:
    int: the longest path in tiles in the current map
"""
def calc_longest_path(map, map_locations, passable_values, get_path=False):
    width, height = len(map), len(map[0])
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    final_visited_map = np.zeros((width, height))
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dijkstra_map, visited_map = run_dijkstra(x, y, map, passable_values)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dijkstra_map, axis=None), dijkstra_map.shape)
        dijkstra_map, _ = run_dijkstra(mx, my, map, passable_values)
        max_value = np.max(dijkstra_map)
        if max_value > final_value:
            final_value = max_value
            if get_path:
                path_map = dijkstra_map
    # Return path for the purpose of rendering (binary problem)
    path = []
    if get_path and final_value > 0:
        path = get_path_coords(path_map)
    return final_value, path

def calc_tortuosity(map,map_locations,passable_values,get_path=False):
    """
    To calculate tortuosity, we follow the logic of calc_longest_path above, but instead of tracking the longest
    shortest path, we track the longest shortest path divided by the euclidean distance between the correspoding 
    start/end points.

    #TODO: this doesn't actually compute all shortest paths. We'll plug in better code here soon.
    """
    width, height = len(map), len(map[0])
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    final_visited_map = np.zeros((width, height))
    max_path_length = 0
    torts = []

    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = run_dijkstra(x, y, map, passable_values)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = run_dijkstra(mx, my, map, passable_values)
        max_path_xy = np.max(dikjstra_map)

        if max_path_xy > max_path_length:
            max_path_length = max_path_xy
            if get_path:
                path_map = dikjstra_map
        
        l2_dist = np.sqrt((x-mx)**2 + (y-my)**2)
        l2_dist = l2_dist if l2_dist > 0 else 1
        tort_xy = max_path_xy / l2_dist
        torts.append(tort_xy)

    mean_tortuosity = np.mean(torts) if torts else 0

    # Return path for the purpose of rendering (binary problem)
    path = None
    if get_path and max_path_length > 0:
        path = get_path_coords(path_map)

    return mean_tortuosity, max_path_length, path


def get_path_coords(path_map, init_coords=None):
    '''
    Recover a shortest path (as list of coords) from a dijkstra map, using either some initial coords, 
    or else from the furthest point.

    When width = heigth = 3

    map
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])

    The dijkstra map starts from 1 to max
    pad_path_map
    array([[0., 0., 0., 0., 0.],
           [0., 1., 1., 1., 0.],
           [0., 1., 5., 2., 0.],
           [0., 10., 1., 3., 0.],
           [0., 0., 0., 0., 0.]])
    
    max_cell
    10.0

    np.where(pad_path_map == max_cell)
    (array([3]), array([1])) 

    np.where(any_other_map == max_cell)
    (array([], dtype=int64), array([], dtype=int64))

    curr = np.array(np.where(pad_path_map == max_cell))
    array([[3],
           [1]])

    a = (1, 2)

    b = np.array([a])
    array([[1, 2]])

    b = np.array([a]).T
    array([[1],
           [2]])

    xi, yi = b[:, 0]
    xi
    1
    yi
    2

    adj_mask
    array([[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=int32)
    xi, yi = 3, 3
    x0, x1, y0, y1 = xi - 1, xi + 2, yi - 1, yi + 2
    2,  5,  2,  5
    adj_mask[x0: x1, y0: y1] = ADJ_FILTER   # ADJ_FILTER = np.array([[0,1,0],[1,0,1],[0,1,0]])
    adj_mask
    array([[0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]], dtype=int32)
    
    '''
    width, height = len(path_map), len(path_map[0])
    pad_path_map = np.zeros(shape=(width + 2, height + 2), dtype=np.int32)
    pad_path_map.fill(0)
    pad_path_map[1:width + 1, 1:height + 1] = path_map + 1
    if not init_coords:
        # Work from the greatest cell value (end of the path) backward
        max_cell = pad_path_map.max()
        curr = np.array(np.where(pad_path_map == max_cell))
    else:
        # add 1 because there are 0s padding around the dijkstra map
        curr = np.array([init_coords], dtype=np.int32).T + 1
        max_cell = pad_path_map[curr[0][0], curr[1][0]]
    
    # current position in pad_path_map
    xi, yi = curr[:, 0]
    path = np.zeros(shape=(max_cell, 2), dtype=np.int32)
    i = 0
    while max_cell > 1:
        # turn xi, yi into real dijstra map entry coordinates
        path[i, :] = [xi - 1, yi - 1]

        # delete the current position's value
        pad_path_map[xi, yi] = -1

        # path length - 1
        max_cell -= 1
        
        x0, x1, y0, y1 = xi - 1, xi + 2, yi - 1, yi + 2
        adj_mask = np.zeros((width + 2, height + 2), dtype=np.int32)
        adj_mask[x0: x1, y0: y1] = ADJ_FILTER

        curr = np.array(np.where(adj_mask * pad_path_map == max_cell))
        xi, yi = curr[:, 0]
        i += 1
    if i > 0:
        path[i, :] = [xi - 1, yi - 1]

    return path
    # This func is too hard to understand. Do we have a better way?

def calc_longest_path_old(map, map_locations, passable_values):
    empty_tiles = _get_certain_tiles(map_locations, passable_values)
    final_visited_map = np.zeros((len(map), len(map[0])))
    final_value = 0
    for (x,y) in empty_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dijkstra_map, visited_map = run_dijkstra(x, y, map, passable_values)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dijkstra_map, axis=None), dijkstra_map.shape)
        dijkstra_map, _ = run_dijkstra(mx, my, map, passable_values)
        max_value = np.max(dijkstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value



"""
Calculate the number of tiles that have certain values in the map

Returns:
    int: get number of tiles in the map that have certain tile values
"""
def calc_certain_tile(map_locations, tile_values):
    return len(_get_certain_tiles(map_locations, tile_values))

"""
Calculate the number of reachable tiles of a certain values from a certain starting value
The starting value has to be one on the map

Parameters:
    map (any[][]): the current map
    start_value (any): the start tile value it has to be only one on the map
    passable_values (any[]): the tile values that can be passed in the map
    reachable_values (any[]): the tile values that the algorithm trying to reach

Returns:
    int: number of tiles that has been reached of the reachable_values
"""
def calc_num_reachable_tile(map, map_locations, start_value, passable_values, reachable_values):
    (sx,sy) = _get_certain_tiles(map_locations, [start_value])[0]
    dijkstra_map, _ = run_dijkstra(sx, sy, map, passable_values)
    tiles = _get_certain_tiles(map_locations, reachable_values)
    total = 0
    for (tx,ty) in tiles:
        if dijkstra_map[ty][tx] >= 0:
            total += 1
    return total

"""
Generate random map based on the input Parameters

Parameters:
    random (numpy.random): random object to help generate the map
    width (int): the generated map width
    height (int): the generated map height
    prob (dict(int,float)): the probability distribution of each tile value

Returns:
    int[][]: the random generated map
"""
def gen_random_map(random, dims, prob):
# def gen_random_map(random, width, height, prob):
    map = random.choice(list(prob.keys()),size=dims[::-1],p=list(prob.values())).astype(np.uint8)
    return map

"""
A method to convert the map to use the tile names instead of tile numbers

Parameters:
    map (numpy.int[][]): a numpy 2D array of the current map
    tiles (string[]): a list of all the tiles in order

Returns:
    string[][]: a 2D map of tile strings instead of numbers
"""
def get_string_map(map, tiles, continuous=False):
    if continuous:
        return map  # AD HOC this is not applicable to continuous domains
    int_to_string = dict((i, s) for i, s in enumerate(tiles))
    result = []
    for y in range(map.shape[0]):
        result.append([])
        for x in range(map.shape[1]):
            result[y].append(int_to_string[int(map[y][x])])
    return result

"""
A method to convert the probability dictionary to use tile numbers instead of tile names

Parameters:
    prob (dict(string,float)): a dictionary of the probabilities for each tile name
    tiles (string[]): a list of all the tiles in order

Returns:
    Dict(int,float): a dictionary of tile numbers to probability values (sum to 1)
"""
def get_int_prob(prob, tiles):
    string_to_int = dict((s, i) for i, s in enumerate(tiles))
    result = {}
    total = 0.0
    for t in tiles:
        result[string_to_int[t]] = prob[t]
        total += prob[t]
    for i in result:
        result[i] /= total
    return result

"""
A method to help calculate the reward value based on the change around optimal region

Parameters:
    new_value (float): the new value to be checked
    old_value (float): the old value to be checked
    low (float): low bound for the optimal region
    high (float): high bound for the optimal region

Returns:
    float: the reward value for the change between new_value and old_value
"""
def get_range_reward(new_value, old_value, low, high):
    if new_value >= low and new_value <= high and old_value >= low and old_value <= high:
        return 0
    if old_value <= high and new_value <= high:
        return min(new_value,low) - min(old_value,low)
    if old_value >= low and new_value >= low:
        return max(old_value,high) - max(new_value,high)
    if new_value > high and old_value < low:
        return high - new_value + old_value - low
    if new_value < low and old_value > high:
        return high - old_value + new_value - low
