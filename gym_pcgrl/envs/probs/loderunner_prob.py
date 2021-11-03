from PIL import Image
import os
import numpy as np
from gym_pcgrl.envs.probs.problem import Problem
from gym_pcgrl.envs.helper import get_range_reward, get_tile_locations, calc_certain_tile, get_floor_dist, get_type_grouping, get_changes
from gym_pcgrl.envs.probs.loderunner.engine import get_score


class LRProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 12
        self._height = 8        
        self._prob = {"solid": 0.03, "brick": 0.23, "ladder": 0.10, "rope": 0.032, "empty": 0.56, "gold":0.02, "enemy":0.05, "player":0.01}
        self._border_size = (0,0)

        self._min_enemies = 1
        self._max_enemies = 3
        self._min_golds = 1
        self._max_golds = 10

        self._rewards = {
            "player": 3,
            "enemies": 1,
            "golds": 1,
            "win": 5,
            "path-length": 2
        }

    def get_tile_types(self):
        return ["solid", "brick", "ladder", "rope", "empty", "gold", "enemy", "player"]

    def adjust_param(self, **kwargs):
        super().adjust_param(**kwargs)

        self._min_enemies = kwargs.get('min_enemies', self._min_enemies)
        self._max_enemies = kwargs.get('max_enemies', self._max_enemies)
        self._min_golds = kwargs.get('min_golds', self._min_golds)
        self._max_golds = kwargs.get('max_golds', self._max_golds)

        rewards = kwargs.get('rewards')
        if rewards is not None:
            for t in rewards:
                if t in self._rewards:
                    self._rewards[t] = rewards[t]

    
    def _run_game(self, map):
        gameCharacters="Bb#-.GEM"
        string_to_char = dict((s, gameCharacters[i]) for i, s in enumerate(self.get_tile_types()))
        lvl= []
        for i in range(len(map)):
            line = []
            for j in range(len(map[i])):
                string = map[i][j]
                line.append(string_to_char[string])
            lvl.append(line)
                
        score, dist = get_score(lvl)           

        return score, dist

    def get_stats(self, map):
        map_locations = get_tile_locations(map, self.get_tile_types())
        map_stats = {
            "player": calc_certain_tile(map_locations, ["player"]),
            "enemies": calc_certain_tile(map_locations, ["enemy"]),
            "golds": calc_certain_tile(map_locations, ["gold"]),
            "win": 0,
            "path-length": 0
        }
        if map_stats["player"] == 1 and map_stats["golds"] > 0:
            map_stats["win"], map_stats["length"] = self._run_game(map)
        
        return map_stats

    def get_reward(self, new_stats, old_stats):
        #longer path is rewarded and less number of regions is rewarded
        rewards = {
            "player": get_range_reward(new_stats["player"], old_stats["player"], 1, 1),
            "enemies": get_range_reward(new_stats["enemies"], old_stats["enemies"], self._min_enemies, self._max_enemies),
            "golds": get_range_reward(new_stats["golds"], old_stats["golds"], self._min_golds, self._max_golds),
            "win": get_range_reward(new_stats["win"], old_stats["win"], 0, 1),
            "path-length": get_range_reward(new_stats["path-length"], old_stats["path-length"], np.inf, np.inf),
        }
        #calculate the total reward
        return rewards["player"] * self._rewards["player"] +\
            rewards["enemies"] * self._rewards["enemies"] +\
            rewards["golds"] * self._rewards["golds"] +\
            rewards["win"] * self._rewards["win"] +\
            rewards["path-length"] * self._rewards["path-length"] 

    def get_episode_over(self, new_stats, old_stats):
        return new_stats["win"] == 1 and new_stats["path-length"] >= 20

    def get_debug_info(self, new_stats, old_stats):
        return {
            "player": new_stats["player"],
            "enemies": new_stats["enemies"],
            "golds": new_stats["golds"],
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
