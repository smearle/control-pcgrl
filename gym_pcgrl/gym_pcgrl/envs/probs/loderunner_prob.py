from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.loderunner.engine import get_score
from pdb import set_trace as TT


class LoderunnerProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 12
        self._height = 8        
        self._prob = {"solid": 0.03, "brick": 0.23, "ladder": 0.10, "rope": 0.032, "empty": 0.56, "gold":0.02, "enemy":0.05, "player":0.01}
        self._border_size = (0,0)

        self._min_enemies = 1
        self._max_enemies = 3
        self._min_gold = 1
        self._max_gold = 10
        chars_to_tiles = \
            {
                '.': 'empty',
                 'B': 'solid',
                 'b': 'brick',
                 '#': 'ladder',
                 '-': 'rope',
                 'E': 'enemy',
                 'G': 'gold',
                 'M': 'player',
             }
        self.tiles_to_chars = {v: k for k, v in chars_to_tiles.items()}

        self._reward_weights = {
            "player": 1,
#           "enemies": 1,
            "enemies": 0,
#           "gold": 1,
            "gold": 0,
            "win": 1,
#           "path-length": 2,
            "path-length": 0,
        }

    def get_tile_types(self):
        return ["empty", "brick", "ladder", "rope", "solid", "gold", "enemy", "player"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._min_enemies = kwargs.get('min_enemies', self._min_enemies)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._min_gold = kwargs.get('min_gold', self._min_gold)
        self._max_gold = kwargs.get('max_gold', self._max_gold)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._reward_weights:
                    self._reward_weights[t] = rewards[t]

    
    def _run_game(self, map):
   #    string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl= []
        for i in range(len(map)):
            line = []
            for j in range(len(map[i])):
                string = map[i][j]
                line.append(self.tiles_to_chars[string])
            lvl.append(line)
                
        score, dist = get_score(lvl)           

        return score, dist

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "enemies": calc_certain_tile(map_locations, ["enemy"]),
            "gold": calc_certain_tile(map_locations, ["gold"]),
            "win": 0,
            "path-length": 0
        }
#       if map_stats["player"] == 1 and map_stats["gold"] > 0:
        if map_stats["player"] == 1:
                map_stats["win"], map_stats["path-length"] = self._run_game(map)

        return map_stats

    #TODO: calculate reward as below for NCA
    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
#           "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], self._min_enemies, self._max_enemies),
#           "gold": get_range_reward(new_stats["gold"], old_stats["gold"], self._min_gold, self._max_gold),
            "win": get_range_reward(new_stats["win"], old_stats["win"], 0, 1),
#           "path-length": get_range_reward(new_stats["path-length"], old_stats["path-length"], np.inf, np.inf),
        }
        #calculate the total reward
        return rewards["player"] * self._reward_weights["player"] +\
            rewards["enemies"] * self._reward_weights["enemies"] +\
            rewards["gold"] * self._reward_weights["gold"] +\
            rewards["win"] * self._reward_weights["win"] +\
            rewards["path-length"] * self._reward_weights["path-length"] 

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["win"] == 1 and new_stats["path-length"] >= 20

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "enemies": new_stats["enemies"],
            "gold": new_stats["gold"],
            "win": new_stats["win"],
            "path-length": new_stats["path-length"]
        }

    def render(self, map):
        #new_map = self._get_runnable_lvl(map)
        
        if self._graphics == None:
            self._graphics = {
                "solid": Image.open(os.path.dirname(__file__) + "/loderunner/solid.png").convert('RGBA'),
                "brick": Image.open(os.path.dirname(__file__) + "/loderunner/brick.png").convert('RGBA'),
                "ladder": Image.open(os.path.dirname(__file__) + "/loderunner/ladder.png").convert('RGBA'),
                "rope": Image.open(os.path.dirname(__file__) + "/loderunner/rope.png").convert('RGBA'),
                "enemy": Image.open(os.path.dirname(__file__) + "/loderunner/enemy.png").convert('RGBA'),
                "gold": Image.open(os.path.dirname(__file__) + "/loderunner/gold.png").convert('RGBA'),
                "empty": Image.open(os.path.dirname(__file__) + "/loderunner/empty.png").convert('RGBA'),
                "player": Image.open(os.path.dirname(__file__) + "/loderunner/player.png").convert('RGBA')
            }
        #self._border_size = (0, 0)
        img = super().render(map)
        #self._border_size = (3, 0)
        return img
