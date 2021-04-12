import numpy as np
from gym_pcgrl.envs.probs.binary_prob import BinaryProblem

class BinaryCtrlProblem(BinaryProblem):
    def __init__(self):
        super(BinaryCtrlProblem, self).__init__()
        self._max_path_length = np.ceil(self._width / 2 + 1) * (self._height)

        self.static_trgs = {
                'regions': 1,
                'path-length': self._max_path_length,
                }

        # conditional inputs/targets
        self.cond_trgs = {
                'regions': 1,
                'path-length': self._max_path_length,
                }
        # boundaries for conditional inputs/targets
        self.cond_bounds = {
                'regions': (0, self._width * np.ceil(self._height / 2)),  # Upper bound: checkerboard

                                                                                          #     10101010
                                                                                          #     01010101
                                                                                          #     10101010
                                                                                          #     01010101
                                                                                          #     10101010
                                                                                            
                #FIXME: we shouldn't assume a square map here! Find out which dimension is bigger
                # and "snake" along that one
                'path-length': (0, self._max_path_length),  # Upper bound: zig-zag

                                                                                            #   11111111
                                                                                            #   00000001
                                                                                            #   11111111
                                                                                            #   10000000
                                                                                            #   11111111
                }

#       self.weights = self._rewards
        self.weights = {
            'regions': 1,
            'path-length': 1,
        }

    # We do these things in the ParamRew wrapper (note that max change and iterations 
    def get_episode_over(self, new_stats, old_stats):
        ''' If the generator has reached its targets. (change percentage and max iterations handled in pcgrl_env)'''
        return False

    def get_reward(self, new_stats, old_stats):
        return None

