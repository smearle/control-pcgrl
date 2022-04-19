from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Parkour_prob import Minecraft3DParkourProblem
import numpy as np


class Minecraft3DParkourCtrlProblem(Minecraft3DParkourProblem):
    def __init__(self, *args, **kwargs):
        super(Minecraft3DParkourCtrlProblem, self).__init__(*args, **kwargs)

        self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)
        self._max_sol_length = np.ceil(self._width) * 3  # Suppose he zig-zags back and forth about 3 times
        # like _reward_weights but for use with ParamRew
        self._reward_weights = self._reward_weights

        self.static_trgs = {
                'dist-floor': 0,
                'disjoint-tubes': 0,
                'enemies': (self._min_enemies, self._max_enemies),
                'empty': (self._min_empty, self._width * self._height),
                'noise': 0,
                'jumps': (self._min_jumps, self._width * self._height),
                'jumps-dist': 0,
                'dist-win': 0,
                'sol-length': self._max_sol_length,
                }

        # boundaries for conditional inputs/targets
        self.cond_bounds = {
                'dist-floor': (0, self._width * self._height),
                'disjoint-tubes': (0, self._width * self._height), 
                'enemies': (0, self._width * self._height),
                'empty': (0, self._width), # * self._height),
                'noise': (0, self._width * self._height),
                'jumps': (0, self._width),
                'jumps-dist': (0, self._width * self._height),
                'dist-win': (0, self._width), #* self._height),
                'sol-length': (0, self._max_sol_length),
                }
