import numpy as np

from gym_pcgrl.envs.helper import (_get_certain_tiles, calc_certain_tile,
                                   calc_num_regions, get_range_reward,
                                   get_tile_locations)
#from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.zelda.zelda_ctrl_prob import ZeldaCtrlProblem

class Player():
    def __init__(self):
        self.health = 100
        self.keys = 0
        # how many doors have we opened?
        self.doors = 0
        # score or reward
        self.rew = 0
        self.won = False
        self.done = False
        self.coords = None
        self.win_time = 0

    def move(self, x_t, y_t):
        self.coords = x_t, y_t

        return True

    def reset(self):
        self.health = 100
        self.keys = 0
        self.doors = 0
        self.rew = 0
        self.won = False
        self.done = False
        self.win_time = 0
        self.coords = None

class ZeldaPlayProblem(ZeldaCtrlProblem):

    ''' A version of zelda in which a player may control Link and play the game.'''
    def __init__(self, max_step=200):
        super().__init__()
        self._width = self.MAP_X = 16
        self._height = self._width
        self.playable = False
        self.active_agent = 0
        # applies only to player turns
        self.player = Player()
        self.win_time = 0
        self.max_step = max_step
        self.min_reward = -1
        self.max_reward = 2
        # one key and one door



    def get_stats(self, map):
        map_stats = super().get_stats(map)
        map_locations = get_tile_locations(map, self.get_tile_types())
        players = _get_certain_tiles(map_locations, ['player'])

        if self.player.coords is None:
            if map_stats["player"] == 1: # and map_stats["key"] > 0 and map_stats["regions"] == 1 and map_stats["key"] >= 1:
                self.player.coords = players[0]
           #else:
           #    self.playable = False

            if len(players) > 1:
                self.player.coords = players[-1]

        return map_stats

    def reset(self, rep_stats):
        super().reset(rep_stats)
        self.player = Player()
       #self.playable = False


    def get_reward(self, new_stats, old_stats):
        if self.active_agent == 0:
           #return 0
            return self.get_designer_reward(new_stats, old_stats)

        return self.player.rew

    def move_player(self, trg_chan):
        ''' Moves the player to map coordinates (x_t, y_t).
            Returns True if player can move to target tile.
        '''
        if not self.player.won:
            self.player.win_time += 1

        # impassable tiles

        passable = True
        tile_types = self.get_tile_types()
        player = tile_types.index('player')
        solid = tile_types.index('solid')
        spider = tile_types.index('spider')
        bat = tile_types.index('bat')
        scorpion = tile_types.index('scorpion')
        key = tile_types.index('key')
        door = tile_types.index('door')

        if trg_chan in [solid, player]:
            passable = False

        if trg_chan in [spider, bat, scorpion]:
            self.player.rew -= 1
            self.player.done = True
        elif trg_chan == key: # and not self.won:
            if self.player.keys == 0:
                self.player.rew += 1
            self.player.keys += 1
           #self._prob.player.rew = self.max_step - self._iteration
           #self.won = True

        if trg_chan == door:
            # door
            if self.player.keys > 0:
                # open door
                self.player.doors += 1
                self.player.keys -= 1
                self.player.rew += 1
                self.player.won = True
                self.player.done = True
#               self.player.rew += self.max_step - self.win_time
            else:
                passable = False

        return passable


    def get_designer_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "key": get_range_reward(new_stats["key"], old_stats["key"], 1, 1),
            "door": get_range_reward(new_stats["door"], old_stats["door"], 1, 1),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], 2, self._max_enemies),
            "regions": get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1),
            "nearest-enemy": get_range_reward(new_stats["nearest-enemy"], old_stats["nearest-enemy"], self._target_enemy_dist, np.inf),
            "path-length": get_range_reward(new_stats["path-length"],old_stats["path-length"], np.inf, np.inf)
        }
        #calculate the total reward

        return rewards["player"] * self._reward_weights["player"] +\
            rewards["key"] * self._reward_weights["key"] +\
            rewards["door"] * self._reward_weights["door"] +\
            rewards["enemies"] * self._reward_weights["enemies"] +\
            rewards["regions"] * self._reward_weights["regions"] +\
            rewards["nearest-enemy"] * self._reward_weights["nearest-enemy"] +\
            rewards["path-length"] * self._reward_weights["path-length"]

    def is_playable(self, stats):
        return stats["player"] == 1  and stats["key"] == 1 and stats["door"] == 1 and stats["regions"] == 1
