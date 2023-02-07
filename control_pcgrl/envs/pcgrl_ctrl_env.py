import numpy as np
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.pcgrl_env import PcgrlEnv

class PcgrlCtrlEnv(PcgrlEnv):
    # TODO: Remove prob and rep since already contained in config
    def __init__(self, cfg: Config, prob="binary_ctrl", rep="narrow"):
        super(PcgrlCtrlEnv, self).__init__(cfg, prob, rep)
        self.cond_bounds = self._prob.cond_bounds
        self.static_trgs = self._prob.static_trgs
        self.width = self._prob._width
        # self._max_changes = max(int(1 * self._prob._width * self._prob._height), 1)

    def set_map(self, init_map):
        self._rep._random_start = False
        self._rep._old_map = init_map.copy()

    # FIXME: this isn't necessary right? Dictionary is the same yeah? ....yeah?

#   def reset(self):
#       obs = super().reset()
#       self.metrics = self._rep_stats
#       return obs

#   def step(self, actions):
#       ret = super().step(actions)
#       self.metrics = self._rep_stats
#       return ret

    # def adjust_param(self, **kwargs):
        # super().adjust_param(**kwargs)
        # if kwargs.get('change_percentage') == -1:
            # self._max_changes = np.inf

