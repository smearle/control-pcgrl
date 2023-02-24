import numpy as np
from control_pcgrl.configs.config import Config
from control_pcgrl.envs.pcgrl_env import PcgrlEnv

class PcgrlCtrlEnv(PcgrlEnv):
    # TODO: Remove prob and rep since already contained in config
    def __init__(self, cfg: Config, prob="binary_ctrl", rep="narrow"):
        super(PcgrlCtrlEnv, self).__init__(cfg, prob, rep)
        self.cond_bounds = self._prob.cond_bounds
        self.static_trgs = self._prob.static_trgs

    def set_map(self, init_map):
        self._rep._random_start = False
        self._rep._old_map = init_map.copy()
