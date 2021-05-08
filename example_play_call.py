# from GVGAI_GYM import sketch

# gvgai_path = '/Volumes/Data_01/home/g/hybrid/GVGAI_GYM/'

# level = (
# """wwwwwwwwwwwww
# wwA+g....w..w
# wwww........w
# w...w...w..ww
# www.w2..wwwww
# w.......w...w
# w.2.........w
# w.....2.....w
# wwwwwwwwwwwww"""
# )

# result = sketch.play(level, player, gvgai_path, 1000)

# print(result)

import random
import sys

gvgai_path = '/home/sme/GVGAI_GYM/'
sys.path.insert(0,gvgai_path)
from play import play
# import play.play as play

def random_player(state,action_space=6):
    """
    player takes a current state and returns an action
    state (numpy array dtype uint8 shape (90, 130, 4)): the current state of the game as pixel values
    action_space (int): The number of discrete actions your player can take. 
    returns a randomly selected action. 
    """
    return random.randint(0, action_space-1)



# # Losing level
# level = (
# """wwwwwwwwwwwww
# wwAwg+...w..w
# wwww........w
# w...w...w..ww
# www.w2..wwwww
# w.......w...w
# w.2.........w
# w.....2.....w
# wwwwwwwwwwwww"""
# )

# Winning level
level = (
"""wwwwwwwwwwwww
wwA+g....w..w
wwww........w
w...w...w..ww
www.w2..wwwww
w.......w...w
w.2.........w
w.....2.....w
wwwwwwwwwwwww"""
)

result = play(level, random_player, gvgai_path, 1000)


print(result)
