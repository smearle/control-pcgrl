from gym_pcgrl.envs.probs.smb_prob import SMBProblem
import numpy as np

class SMBCtrlProblem(SMBProblem):
    def __init__(self, *args, **kwargs):
        super(SMBCtrlProblem, self).__init__(*args, **kwargs)

        self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)
        # like _rewards but for use with ParamRew
        self.weights = self._rewards

        self.static_trgs = {
                'dist-floor': 0,
                'disjoint-tubes': 0,
                'enemies': (self._min_enemies, self._max_enemies),
                'empty': (self._min_empty, self._width * self._height),
                'noise': 0,
                'jumps': (self._min_jumps, self._width * self._height),
                'jumps-dist': 0,
                'dist-win': 0,
                }

        # boundaries for conditional inputs/targets
        self.cond_bounds = {
                'dist-floor': (0, self._width * self._height),
                'disjoint-tubes': (0, self._width * self._height), 
                'enemies': (0, self._width * self._height),
                'empty': (0, self._width * self._height),
                'noise': (0, self._width * self._height),
                'jumps': (0, self._width),
                'jumps-dist': (0, self._width * self._height),
                'dist-win': (0, self._width), #* self._height),
                }
        
