# package imports
import math
import matplotlib.pyplot as plt

# 3rd party imports
import numpy as np

# local import 
from control_pcgrl.configs.config import Config 
from control_pcgrl.envs.probs.problem import Problem3D


class LegoProblem(Problem3D):
    """ 
        We define information related to 'Lego building construction' in this class.        
    """
    _tile_types = ['empty', '1x1']

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg=cfg)
        self.total_reward = 0 
        self.reward_history = []

        # Max instances per brick type? Make this a dict over brick types?
        self.total_bricks = 100 # ?

        # Default targets (we can overwrite these in the problem config).
        self.static_trgs = {'n_bricks': math.prod(self._map_shape)}

        # Bounds for each metric of interest.
        self.cond_bounds = {'n_bricks': (0, math.prod(self._map_shape))}


        # The probability of placing a tile of a given type when initializing a new (uniform) random map at the
        # beginning of a level-generation episode.
        self._prob = {'empty': 1.0, '1x1': 0.0,}

    def get_stats(self, map):
        # Count the number of blocks not equal to `empty`
        n_bricks = (np.array(map) != 'empty').sum()
        return {'n_bricks': n_bricks}

    def get_reward(self, new_stats, old_stats):
        reward = 0
        y, x, z = new_stats['new_location']
        punish = new_stats['punish']
        old_y = old_stats['old_location'][0]

        # best reward condition so far 
        if (y > old_y or 
            (y == 9)):
            if punish:
                reward = -0.5
            else:
                reward = 2
        else:
            reward = -1

        # Print Reward graph -> Accumulate rewards
        # print("Reward: ", reward)
        self.total_reward += reward

        return reward
    
    def get_episode_over(self, new_stats, old_stats):    
        
        if new_stats['num_of_bricks'] <= 0:
            # print("episode over: ", representation.num_of_bricks)
            # print("episode over: ", np.count_nonzero(representation._map))
            # print("Total reward: ",  self.total_reward)
            self.reward_history.append(self.total_reward)
            self.total_reward = 0 
            return True
        
        return False

    def get_debug_info(self, new_stats, old_stats):
        """Use this function to debug"""
        return new_stats
    
    def plot_reward(self):
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.title('Episodes - Rewards Plot')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig('rewards.png')

    def render(self):
        breakpoint()

    def init_graphics(self):
        pass
