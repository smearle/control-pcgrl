import itertools
from pdb import set_trace as TT

import numpy as np
import ray

from gym_pcgrl.envs.probs.problem import Problem


class HoleyProblem(Problem):
    """
    The base class for all the holey problems so that we don't have to repeat the code.
    """
    def __init__(self):
        # super().__init__()
        self._hole_queue = []
        self.fixed_holes = False
        self._border_idxs = self.get_border_idxs()

    def get_border_idxs(self):
        """
        Get the border indices of the map.
        """
        dummy_bordered_map = np.zeros((self._height + 2, self._width + 2), dtype=np.uint8)
        dummy_bordered_map[1:-1, 0] = dummy_bordered_map[1:-1, -1] = 1
        dummy_bordered_map[0, 1:-1] = dummy_bordered_map[-1, 1:-1] = 1

        border_idxs = np.argwhere(dummy_bordered_map == 1)
        return border_idxs


    def gen_holes(self):
        """Generate one entrance and one exit hole into/out of the map randomly. Ensure they will not necessarily result
         in trivial paths in/out of the map. E.g., the below are not valid holes:
        0 0    0 x  
        x      x
        x      0
        0      0

        entrance_coords[0] : y
        entrance_coords[1] : x
        """
        if len(self._hole_queue) > 0:
            (self.entrance_coords, self.exit_coords), self._hole_queue = self._hole_queue[0], self._hole_queue[1:]
        
        elif self.fixed_holes:
            self.entrance_coords = np.array([1, 0])
            self.exit_coords = np.array((self._width, self._height + 1))

        else:
            idxs = np.random.choice(self._border_idxs.shape[0], size=4, replace=False)
            self.entrance_coords = self._border_idxs[idxs[0]]
            for i in range(1, 4):
                xy = self._border_idxs[idxs[i]]
                if self._valid_holes(self.entrance_coords, xy): 
                    self.exit_coords = xy
                    break
        
        return self.entrance_coords, self.exit_coords

    def queue_holes(self, idx_counter):
        """
        Gets a list of holes to be used on reset. `idx_counter` is a global ray actor object. It uses the hash of `self`
        to give a unique id to this environment.
        """
        self._hole_queue = ray.get(idx_counter.get.remote(hash(self)))

    def gen_all_holes(self):
        """
        Generate all the holes in the map for evaluation.
        """
        hole_pairs = list(itertools.product(self._border_idxs, self._border_idxs))
        hole_pairs = [pair for pair in hole_pairs if self._valid_holes(pair[0], pair[1])]
        return hole_pairs

    def _valid_holes(self, entrance_coords, exit_coords):
        """
        Check if the given holes are valid.
        """
        holes = [entrance_coords, exit_coords]
        for i, (x, y) in enumerate(holes):
            if x == 0:
                x = 1
            elif x == self._width-1:
                x = self._width-2
            elif y == 0:
                y = 1
            elif y == self._height-1:
                y = self._height-2
            holes[i] = np.array([x, y])
        return np.max(np.abs(holes[0] - holes[1])) > 1

        # corners = np.array([
        #     [0, 0],
        #     [0, self._width + 1],
        #     [self._height + 1, 0],
        #     [self._height + 1, self._width + 1]
        # ])
        # door_coords = np.array([entrance_coords, exit_coords])
        # # Check if each door is in some corner. If so, then check that they are sufficiently far away from each other.
        # if np.all(np.min(np.max(np.abs(corners - door_coords[:, None, :]), 2), 1) < 3):
        #     return np.max(np.abs(entrance_coords - exit_coords)) > 2
        # return np.max(np.abs(entrance_coords - exit_coords)) > 1

