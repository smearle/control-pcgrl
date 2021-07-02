from gym_pcgrl.envs.probs.problem import Problem
from gym_city.envs.env import MicropolisEnv

"""
Generate a top-down city plan leading to maximum population.
"""
class SimCityProblem(Problem):
    def __init__(self):
        super().__init__()
        self._width = 16
        self._height = self._width
        self.tile_types = self.get_tile_types()
        self._border_tile = "Land"
        self._prob = dict((i, s) for i, s in zip(self.tile_types, self._prob))
        self.env = self.init_env()

    def init_env(self):
        size = self._width
        kwargs = {
            'render_gui': True,
            'print_map': False,
                }
        env = MicropolisEnv()
        env.setMapSize(size, **kwargs)
        return env

    def get_tile_types(self):
        if not hasattr(self, 'env'):
            self.env = self.init_env()
        tile_types = self.env.micro.map.zones
        return tile_types

    def get_num_tools(self):
        return self.env.num_tools

    def step(self,a ):
        print('opop')

    def get_stats(self, map):
        map_stats = {
                "res_pop": self.env.micro.engine.resPop,
                "com_pop": self.env.micro.engine.comPop,
                "ind_ind": self.env.micro.engine.indPop,
                }
        self.map_stats = map_stats

    def get_reward(self, new_stats, old_stats):
        r = self.env.getReward()
        return r

    def get_episode_over(self, new_stats, old_stats):
        map_stats = self.map_stats
        o = map_stats["res_pop"] > 10


    def get_debug_info(self, new_stats, old_stats):
        i = {
                }
        return i

    def render(self, mode):
        self.env.render()
