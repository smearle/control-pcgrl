# import gym
import math
import os
import numpy as np
from time import sleep
from turtle import position
import grpc
# import minecraft_pb2_grpc
# from minecraft_pb2 import *
import control_pcgrl.envs.probs.minecraft.minecraft_pb2_grpc as minecraft_pb2_grpc
from control_pcgrl.envs.probs.minecraft.minecraft_pb2 import *
import pyscreenshot as ImageGrab
from pdb import set_trace as TT

SCREENSHOT_SIZE = (512, 512)
BBOX_OFFSET = (0, 66)  # For 16" M1 MacBook Pro
BBOX = (BBOX_OFFSET[0], BBOX_OFFSET[1], BBOX_OFFSET[0] + SCREENSHOT_SIZE[0], BBOX_OFFSET[1] + SCREENSHOT_SIZE[1])

SOLID_BLOCK = STAINED_GLASS
# SOLID_BLOCK = STAINED_GLASS
b_map = [AIR, SOLID_BLOCK, CHEST, PUMPKIN, PUMPKIN]
string_map = ["AIR", "DIRT","CHEST", "SKULL", "PUMPKIN"]
PATH_BLOCK = PURPUR_SLAB  # pretty
# PATH_BLOCK = TRAPDOOR  # navigable (debug)
PLAYER_BLOCK = RED_GLAZED_TERRACOTTA

# map string map entries into Minecraft item type
block_map = dict(zip(string_map, b_map))

# map Minecraft item type into string map entries
inv_block_map = dict(zip(b_map, string_map))

N_BLOCK_TYPE = 3
# RENDER_PATH_SEQUENCE = False
CLIENT = None


def render_blocks(blocks, base_pos=5):
    """ Render blocks through the gRPC interface.

    Args:
        blocks (dict): A dictionary mapping block coordinates to block type.
    """
    # FIXME: we're not using this base_pos argument directly??
    block_lst = [Block(position=Point(x=i, y=k+5, z=j), type=block_type, orientation=NORTH) 
                    for (i, j, k), block_type in blocks.items()]
    CLIENT.spawnBlocks(Blocks(blocks=block_lst))


def clear(n, e, boundary_size=3, backgroud_type=QUARTZ_BLOCK):
    '''
    Clear a background of the map whose size is (n e) in position (0 0 0) for rendering in Minecraft

    Parameters:
        n (int): length in x direction (map[][x])
        e (int): length in y direction (map[y][])
        boundary_size (int): the border of the background
        backgroud_type (any): the block type of the background
    '''
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-boundary_size, y=4, z=-boundary_size),
            max=Point(x=n+boundary_size, y=10, z=e+boundary_size)
        ),
        type=AIR
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-boundary_size, y=4, z=-boundary_size),
            max=Point(x=n+boundary_size-2, y=4, z=e+boundary_size-2)
        ),
        type=backgroud_type
    ))


def get_tile(tile):
    '''
    Return the block type of the given tile in Minecraft.
    '''
    if tile:
        return block_map[tile]
    else:
        return AIR


def spawn_2D_maze(map, border_tile, border_size=(1,1), base_pos=5, maze_height=3):
    '''
    Spawn maze iterately in Minecraft

    Parameters:
        map (string[][]): the current game map
        border_tile (string): the tile name of the border
        border_size ((int,int)): an offeset in tiles if the borders are not part of the level
        base_pos (int): the horizontal height of the maze
        maze_height (int): the height of the walls in the maze
    '''
    blocks = []
    # clear(len(map[0])+border_size[0], len(map)+border_size[1])

    # rendering the border
    item = get_tile(border_tile)
    for h in range(maze_height):
        for j in range(-border_size[0], 0):
            for i in range(-border_size[1], len(map[0])+border_size[1]):
                blocks.append(Block(position=Point(x=i, y=base_pos+h, z=j),
                                    type=item))
        for j in range(len(map), len(map)+border_size[0]):
            for i in range(-border_size[1], len(map[0])+border_size[1]):
                blocks.append(Block(position=Point(x=i, y=base_pos+h, z=j),
                                    type=item))
        for j in range(-border_size[1], 0):
            for i in range(0, len(map)):
                blocks.append(Block(position=Point(x=j, y=base_pos+h, z=i),
                                    type=item))
        for j in range(len(map[0]), len(map[0])+border_size[1]):
            for i in range(0, len(map)):
                blocks.append(Block(position=Point(x=j, y=base_pos+h, z=i),
                                    type=item))

    # rendering the map
    for j in range(len(map)):
        for i in range(len(map[j])):
            item = get_tile(map[j][i])
            for h in range(maze_height):
                blocks.append(Block(position=Point(x=i, y=base_pos+h, z=j),
                                    type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    #time.sleep(0.2)


def spawn_2D_path(path=None, base_pos=5, item=LEAVES):
    blocks = []
    if path:
        for pos in path:
            blocks.append(Block(position=Point(x=pos[0], y=base_pos, z=pos[1]),
                                    type=item))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    return

def spawn_base(map, border_size=(1, 1, 1), base_pos=5,\
                    boundary_size=3, backgroud_type=QUARTZ_BLOCK):
    i, k, j = len(map[0][0]), len(map), len(map[0])
    # render the base
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-boundary_size-border_size[0] + 2, 
                      y=base_pos - 2, z=-boundary_size-border_size[1] + 2),
            max=Point(x=i + boundary_size + border_size[0] - 3,
                      y=base_pos - 1, z=j + boundary_size + border_size[1] - 3)
        ),
        type=backgroud_type
    ))

def spawn_3D_border(map, border_tile, border_size=(1, 1, 1), base_pos=5,\
                    boundary_size=3, entrance_coords=None, exit_coords=None, backgroud_type=QUARTZ_BLOCK):
    '''
    Spawn the border of the maze
    The boundary contains five sides of the cube except the base

    Parameters:
        border_tile (string): the tile name of the border
        border_size ((int,int,int)): an offeset in tiles if the borders are not part of the level
        base_pos (int): the vertical height of the bottom of the maze
        boundary_size (int): the border of the background
        backgroud_type (any): the block type of the background
    '''
    item = get_tile(border_tile)
    i, k, j = len(map[0][0]), len(map), len(map[0])

    spawn_base(map, border_size, base_pos, boundary_size, backgroud_type)

    # render the border
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-border_size[0], y=base_pos, z=-border_size[1]),
            max=Point(x=i+border_size[0]-1, y=base_pos + k +
                      border_size[2]-1, z=j+border_size[1]-1)
        ),
        type=item
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=0, y=base_pos, z=0),
            max=Point(x=i-1, y=k+base_pos-1, z=j-1)
        ),
        type=AIR
    ))

    # render the entrance's door on the border 
    # entrance_coords and exit_coords are (z,y,x), the coordinates of Evocraft is (x,z,y)
    if entrance_coords is not None:
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=entrance_coords[0][2], y=entrance_coords[0][0], z=entrance_coords[0][1]),
                max=Point(x=entrance_coords[1][2], y=entrance_coords[1][0], z=entrance_coords[1][1])
            ),
            type=AIR
        ))
    # else:
    #     CLIENT.fillCube(FillCubeRequest(
    #         cube=Cube(
    #             min=Point(x=-border_size[0], y=base_pos, z=0),
    #             max=Point(x=-1,y=base_pos+1, z=0)
    #         ),
    #         type=AIR
    #     ))                                                             # Change to GOLD_BLOCK to highlight the entrance
        # CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=-1, y=base_pos-1, z=0),type=item, orientation=NORTH)]))

    # render the exit on the border
    if exit_coords is not None:
        CLIENT.fillCube(FillCubeRequest(
            cube=Cube(
                min=Point(x=entrance_coords[0][2], y=entrance_coords[0][0], z=entrance_coords[0][1]),
                max=Point(x=entrance_coords[1][2], y=entrance_coords[1][0], z=entrance_coords[1][1])
            ),
            type=AIR
        ))
    # else:
    #     CLIENT.fillCube(FillCubeRequest(
    #         cube=Cube(
    #             min=Point(x=i, y=base_pos+k-2, z=j-1), 
    #             max=Point(x=i+border_size[0]-1, y=base_pos+k-1, z=j-1)
    #         ),
    #         type=AIR
    #     ))                                                              # Change to DIOMAND_BLOCK to highlight the exit
        # CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=i, y=base_pos+k-3, z=j-1),type=item, orientation=NORTH)]))

    return


def spawn_3D_maze(map, base_pos=5):
    '''
    Note that: in Minecraft, the vertical direction is y (map[z][][])
                             the horizontal direction x is (map[][][x])
                             the horizontal direction z is (map[][y][])

    Spawn maze iterately in Minecraft

    Parameters:
        map (string[][][]): the current game map
        base_pos (int): the vertical height of the bottom of the maze
    '''
    blocks = []
    for k in range(len(map)):
        for j in range(len(map[k])):
            for i in range(len(map[k][j])):
                item = get_tile(map[k][j][i])
                # FIXME: why base_pos is str? Because sometimes we are incorrectlyproviding self._border_tile as the 
                #  second arguement from inside the problem.
                blocks.append(Block(position=Point(x=i, y=k+5,  z=j),   
                                    type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    return

def spawn_3D_bordered_map(map, base_pos=5, offset=(0, 0, 0)):
    blocks = []
    for k in range(len(map)):
        for j in range(len(map[k])):
            for i in range(len(map[k][j])):
                item = get_tile(map[k][j][i])
                # FIXME: why base_pos is str? Because sometimes we are incorrectlyproviding self._border_tile as the 
                #  second arguement from inside the problem.
                blocks.append(Block(
                    position=Point(x=i+1+offset[0], y=k+base_pos+1+offset[2],  z=j+1+offset[1]),   # NOTE: the -1 may cause a problem when the border is thicker than 1
                                    type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    return


def get_3D_maze_blocks(map):
    return {(i, k, j): get_tile(map[k][j][i]) 
                for k in range(len(map)) for j in range(len(map[k])) for i in range(len(map[k][j]))}


# NEXT: change these 2 funcs into 1
def spawn_3D_path(path, base_pos=5, item=PATH_BLOCK, offset=(0, 0, 0), ordered_path=None, 
                    render_path_sequence = False, **kwargs):
    '''
    Spawn path once or iteratively in Minecraft

    Parameters:
        path (list[(int, int, int)]): the path to be spawned
        base_pos (int): the vertical height of the bottom of the maze
        item (int): the type of the block to be spawned
        offset (tuple(int, int, int)): the offset of the path
        sequence (bool): if True, spawn the path iteratively
    '''
    if len(path) == 0:
            return

    if render_path_sequence is False or ordered_path is None:
        blocks = []
        for pos in path:
            blocks.append(Block(position=Point(
                x=pos[0]+offset[0], y=pos[2]+base_pos+offset[2] , z=pos[1]+offset[1]),
                                    type=item))
        CLIENT.spawnBlocks(Blocks(blocks=blocks))
    
    else:
        old_points = []
        for pos in ordered_path:
            points = [pos, (pos[0], pos[1], pos[2] + 1)]
            # points = [Point(x=pos[0]+offset[0], y=pos[2]+base_pos+offset[2], z=pos[1]+offset[1]),
                    #   Point(x=pos[0]+offset[0], y=pos[2]+base_pos+offset[2]+1, z=pos[1]+offset[1])]
            # Erase old blocks first! (Order of list matters.)
            block_lst = \
                [Block(position=Point(x=p[0]+offset[0], y=p[2]+base_pos+offset[2], z=p[1]+offset[1]), type=AIR) for p in old_points] + \
                [Block(position=Point(x=p[0]+offset[0], y=p[2]+base_pos+offset[2], z=p[1]+offset[1]), type=PLAYER_BLOCK) for p in points]
            if old_points and old_points[0] in path:
                op = old_points[0]
                block_lst += [Block(position=Point(
                    x=op[0]+offset[0], y=op[2]+base_pos+offset[2], z=op[1]+offset[1]), type=item)]
            old_points = points
            CLIENT.spawnBlocks(Blocks(blocks=block_lst))
            sleep(0.1)
        CLIENT.spawnBlocks(Blocks(blocks=
            [Block(position=Point(x=points[0][0]+offset[0], y=points[0][2]+base_pos+offset[2], z=points[0][1]+offset[1]), type=item), 
            Block(position=Point(x=points[0][0]+offset[0], y=points[0][2]+base_pos+offset[2]+1, z=points[0][1]+offset[1]), type=AIR), 
            ]
        ))

    return


def erase_3D_path(path, base_pos=5, item=AIR, offset=(0, 0, 0)):
    if len(path) == 0:
        return
    blocks = []
    for pos in path:
        blocks.append(Block(position=Point(
            x=pos[0]+offset[0], y=pos[2]+5+offset[2] , z=pos[1]+offset[1]),
                                type=item))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    return


def get_3D_path_blocks(path, item=LEAVES):
    return {(pos[0], pos[2], pos[1]): item for pos in path}


def get_erased_3D_path_blocks(path, item=AIR):
    return {(pos[0], pos[2], pos[1]): item for pos in path}


def edit_3D_maze(map, i, j, k, base_pos=5):
    '''
    Render function for high-lighting the action

    Parameters:
        map (string[][][]): the current game map
        base_pos (int): the vertical height of the bottom of the maze
        i (int) : the x position that the action take place
        j (int) : the y position that the action take place
        k (int) : the z position that the action take place
    '''
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=i, y=k+base_pos, z=j),
                                            type=RED_GLAZED_TERRACOTTA, orientation=NORTH)]))
    # time.sleep(2)
    item = get_tile(map[k][j][i])
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=i, y=k+base_pos, z=j),
                                            type=item, orientation=NORTH)]))
    # time.sleep(2)
    return

def edit_bordered_3D_maze(map, i, j, k, base_pos=5, offset=(0, 0, 0)):
    '''
    Render function for high-lighting the action

    Parameters:
        map (string[][][]): the current game map
        base_pos (int): the vertical height of the bottom of the maze
        i (int) : the x position that the action take place
        j (int) : the y position that the action take place
        k (int) : the z position that the action take place
    '''
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(
        x=i+1+offset[0], y=k+base_pos+1+offset[2], z=j+1+offset[1]),
                                            type=RED_GLAZED_TERRACOTTA, orientation=NORTH)]))
    # time.sleep(0.5)
    item = get_tile(map[k+1][j+1][i+1])
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(
        x=i+1+offset[0], y=k+base_pos+1+offset[2], z=j+1+offset[1]),
                                            type=item, orientation=NORTH)]))
    return

def spawn_3D_doors(map, entrance, exit, base_pos=5):
    '''
    Spawn the doors in Minecraft

    Parameters:
        map (string[][][]): the current game map
        entrance (tuple): the entrance position
        exit (tuple): the exit position
        base_pos (int): the vertical height of the bottom of the maze
    '''
    # render a border (and inner map) of air
    border_size = (1, 1, 1)
    i, k, j = len(map[0][0]), len(map), len(map[0])
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-border_size[0], y=base_pos+1, z=-border_size[1]),
            max=Point(x=i+border_size[0]-1, y=base_pos + k +
                      border_size[2]-1, z=j+border_size[1]-1)
        ),
        type=AIR
    ))
    
    # spawn colored blocks as supports for the entrance and exit
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=entrance[2], y=entrance[0]+base_pos-1, z=entrance[1]),
                                            type=GOLD_BLOCK, orientation=NORTH),
                                Block(position=Point(x=exit[2], y=exit[0]+base_pos-1, z=exit[1]),
                                      type=DIAMOND_BLOCK, orientation=NORTH)]))

    # # spawn a carpet to step on
    # CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=entrance[2], y=entrance[0]+base_pos, z=entrance[1]),
    #                                         type=CARPET, orientation=NORTH),
    #                             Block(position=Point(x=exit[2], y=exit[0]+base_pos, z=exit[1]),
    #                                   type=CARPET, orientation=NORTH)]))
    return


def init_player_view():
    CLIENT.initDataGen(Point(x=0, y=0, z=0))  # dummy point variable


def set_player_view(x0, y0, z0, i, yaw_deg, pitch_deg, save_dir='./'):
    """Get screenshots viewing a central chunk from 4 cardinal points."""
    radius = 30
    xyz_rots = []
    # for j in range(num_views):
    # rot_y_deg, rot_y_rad = get_view_rot(j, num_views)
    # rot_y_deg = int(rot_y_deg)
    yaw_rad = yaw_deg * np.pi / 180
    pitch_rad = pitch_deg * np.pi / 180
    # double sRadians = Math.toRadians( yaw );
    # double tRadians = Math.toRadians( pitch );
    # double x = radius * Math.cos( sRadians ) * Math.sin( tRadians );
    # double y = radius * Math.cos( tRadians );
    # double z = radius * Math.sin( sRadians ) * Math.sin( tRadians );

    x = radius * math.cos(yaw_rad) * math.sin(pitch_rad)
    y = radius * math.cos(pitch_rad)
    z = radius * math.sin(yaw_rad) * math.sin(pitch_rad)
    # x, z = math.sin(yaw_rad) * radius, math.cos(yaw_rad) * radius
    # y = int(y0)
    x += x0
    y += y0
    z += z0
    x, y, z = int(x), int(y), int(z)
    # y += y0
    theta = math.asin
    # Note that `y` is ignored and player is placed on highest possible block.
    loc = CLIENT.setLocY(Point(x=x, y=y, z=z))
    # loc = CLIENT.setPlayerLocRot(LocRot(loc=Point(x=x, y=y, z=z), rot=Point(x=0, y=rot_y_deg, z=0)))
    # rot_y_deg_ = 180 - yaw_deg

    CLIENT.setRot(Point(x=90-pitch_deg, y=90+yaw_deg, z=0))
    print(f"Set rotation to {yaw_deg}")
    assert loc.x == x and loc.z == z
    y = loc.y
    # foothold_block = read_cube(client, (x, y-1, z), (x, y-1, z))

    top_left_corner_screenshot(f"{i}", bbox=BBOX, save_dir=save_dir)
    xyz_rots.append((x, y, z, int(yaw_deg)))
    print(f"Saved sreenshot {i}, rotation {yaw_deg}, location {x}, {y}, {z}")
    return xyz_rots


def top_left_corner_screenshot(name: str, bbox: tuple, save_dir: str):
    im = ImageGrab.grab(bbox)
    # Save image to file
    if np.array(im).shape != (512, 512, 4):
        raise Exception(f"Screenshot is not 512x512, size is {np.array(im).shape}. Has the display gone to sleep?")
    im.save(os.path.join(save_dir, f"{name}.png"))
    

if __name__ == '__main__':
    CHANNEL = grpc.insecure_channel('localhost:5001')
    CLIENT = minecraft_pb2_grpc.MinecraftServiceStub(CHANNEL)
    # clear(20,20)
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-50, y=3, z=-50),
            max=Point(x=50, y=3, z=50)
        ),
        type=GRASS
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-50, y=4, z=-50),
            max=Point(x=50, y=15, z=50)
        ),
        type=AIR
    ))
