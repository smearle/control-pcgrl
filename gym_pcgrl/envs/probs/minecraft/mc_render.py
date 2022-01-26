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

b_map = [AIR, DIRT]
string_map = ["AIR", "DIRT"]

# map string map entries into Minecraft item type
block_map = dict(zip(string_map, b_map))  

# map Minecraft item type into string map entries
inv_block_map = dict(zip(b_map, string_map))

N_BLOCK_TYPE = 3


def clear(n, e, boundary_size=3, backgroud_type=QUARTZ_BLOCK):
    '''
    Clear a background of the map whose size is (n e) in position (0 0 0) for rendering in Minecraft
    n stands for length in NORTH direction
    e stands for length in EAST direction
    boundary_size is the border of the background
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
    clear(len(map[0])+border_size[0], len(map)+border_size[1])
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
