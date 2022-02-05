# import gym
from turtle import position
import grpc
# import minecraft_pb2_grpc
# from minecraft_pb2 import *
import gym_pcgrl.envs.probs.minecraft.minecraft_pb2_grpc as minecraft_pb2_grpc
from gym_pcgrl.envs.probs.minecraft.minecraft_pb2 import *
import time

CHANNEL = grpc.insecure_channel('localhost:5001')
CLIENT = minecraft_pb2_grpc.MinecraftServiceStub(CHANNEL)

b_map = [AIR, STAINED_GLASS]
string_map = ["AIR", "DIRT"]

# map string map entries into Minecraft item type
block_map = dict(zip(string_map, b_map))

# map Minecraft item type into string map entries
inv_block_map = dict(zip(b_map, string_map))

N_BLOCK_TYPE = 3


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
    return the types blocks of each tiles in Minecraft
    '''
    if tile:
        return block_map[tile]
    else:
        return AIR


def spawn_2Dmaze(map, border_tile, border_size=(1,1), base_pos=5, maze_height=3):
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

def spawn_3D_border(map, border_tile, border_size=(1, 1, 1), base_pos=5,\
                    boundary_size=3, backgroud_type=QUARTZ_BLOCK):
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

    # render the base
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-boundary_size-border_size[0] + 2, y=base_pos -
                      1, z=-boundary_size-border_size[2] + 2),
            max=Point(x=i + boundary_size + border_size[0] - 3,
                      y=base_pos - 1, z=j + boundary_size + border_size[2] - 3)
        ),
        type=backgroud_type
    ))

    # render the border
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-border_size[0], y=base_pos, z=-border_size[1]),
            max=Point(x=i+border_size[0]-1, y=base_pos + k +
                      border_size[1]-1, z=j+border_size[2]-1)
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
    return

def spawn_3Dmaze(map, base_pos=5):
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
                # TODO: why base_pos is str
                blocks.append(Block(position=Point(x=i, y=k+5,  z=j),   
                                    type=item, orientation=NORTH))
    CLIENT.spawnBlocks(Blocks(blocks=blocks))
    return

def reps_3D_render(map, i, j, k, base_pos=5):
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
    time.sleep(0.2)
    item = get_tile(map[k][j][i])
    CLIENT.spawnBlocks(Blocks(blocks=[Block(position=Point(x=i, y=k+base_pos, z=j),
                                            type=item, orientation=NORTH)]))
    return


if __name__ == '__main__':
    clear(20,20)
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-5, y=3, z=-5),
            max=Point(x=30, y=3, z=30)
        ),
        type=DIRT
    ))
    CLIENT.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=-5, y=4, z=-5),
            max=Point(x=30, y=10, z=30)
        ),
        type=AIR
    ))
