import itertools
from pdb import set_trace as TT
from gym_pcgrl.envs.probs.holey_prob import HoleyProblem

import numpy as np
import ray

from gym_pcgrl.envs.probs.problem import Problem, Problem3D


class HoleyProblem3D(HoleyProblem, Problem3D):
    """
    The base class for all the holey problems so that we don't have to repeat the code.
    """
    def get_border_idxs(self):
        dummy_bordered_map = np.zeros((self._height + 2, self._width + 2, self._length+ 2), dtype=np.uint8)
        # # Fill in the non-borders with ones
        # dummy_bordered_map[1:-1, 1:-1, 1:-1] = 1
        # # Fill in the corners with ones
        # dummy_bordered_map[:, 0, 0] = 1
        # dummy_bordered_map[:, 0, -1] = 1
        # dummy_bordered_map[:, -1, 0] = 1
        # dummy_bordered_map[:, -1, -1] = 1
        # # fill in the top and bottom with ones
        # dummy_bordered_map[0, ...] = 1
        # dummy_bordered_map[-1, ...] = 1

        dummy_bordered_map[1:-2, 1:-1, 0] = 1
        dummy_bordered_map[1:-2, 1:-1, -1] = 1
        dummy_bordered_map[1:-2, 0, 1:-1] = 1
        dummy_bordered_map[1:-2, -1, 1:-1] = 1
        border_idxs = np.argwhere(dummy_bordered_map == 1)
        return border_idxs

    def gen_all_holes(self):
        hole_pairs = list(itertools.product(self._border_idxs, self._border_idxs))
        hole_pairs = [pair for pair in hole_pairs if self._valid_holes((pair[0], pair[0] + np.array([1, 0, 0])), pair[1])]
        hole_pairs = [((pair[0], pair[0] + np.array([1, 0, 0])), (pair[1], pair[1] + np.array([1,0,0]))) for pair in hole_pairs]
        return hole_pairs

    def gen_holes(self):
        """Generate one entrance and one exit hole into/out of the map randomly. Ensure they will not necessarily result
         in trivial paths in/out of the map. E.g., the below are not valid holes:
        0 0    0 x  
        x      x
        x      0
        0      0

        entrance_coords[0]: the foot room of the entrance
        entrance_coords[0][0]: z  (height)
        entrance_coords[0][1]: y  (width)
        entrance_coords[0][2]: x  (length)
        """
        # assert the map is not too small
        assert self._height > 2 

        if len(self._hole_queue) > 0:
            (self.entrance_coords, self.exit_coords), self._hole_queue = self._hole_queue[0], self._hole_queue[1:]

        elif self.fixed_holes:
            # Fix the holes diagonally across the cube
            # self.entrance_coords = np.array(([1, 1, 0], [2, 1, 0]))
            # self.exit_coords = np.array(((1, self._width, self._length + 1),
            #                          (2, self._width, self._length + 1)))

            # Fix the holes to be stacked together
            # self.entrance_coords = np.array(((1, 0, self._length),
            #                          (2, 0, self._length)))
            # self.exit_coords = np.array(((5, 0, self._length),
            #                          (6, 0, self._length)))

            # Fix the holes at diagonal corners
            self.entrance_coords = np.array(((1, 0, self._length),
                                     (2, 0, self._length)))
            self.exit_coords = np.array(((2, self._width + 1, 1),
                                     (3, self._width + 1, 1)))

        else:
            self.entrance_coords = np.ones((2, 3), dtype=np.uint8)  
            potential = 26
            idxs = np.random.choice(self._border_idxs.shape[0], size=potential, replace=False)

            # randomly select a hole as the foot room of the entrance
            self.entrance_coords[0] = self._border_idxs[idxs[0]]
            # select a coresonding hole as the head room of the entrance
            self.entrance_coords[1] = self.entrance_coords[0] + np.array([1, 0, 0])

            self.exit_coords = np.ones((2, 3), dtype=np.uint8)
            # select the exit
            # NOTE: Some valid cases are excluded, e.g.:
            #
            #     0
            #     0 0
            #       0
            #
            for i in range(1, potential):
                xyz = self._border_idxs[idxs[i]]
                if self._valid_holes(self.entrance_coords, xyz):
                    self.exit_coords[0] = xyz
                    self.exit_coords[1] = xyz + np.array([1, 0, 0])
                    break
        
        # print(f"Setting holes: {self.entrance_coords, self.exit_coords}")
        return self.entrance_coords, self.exit_coords

    def _valid_holes(self, entrance_coords, exit_coords) -> bool:      
        """
        Args:
            entrance_coords: a tuple with the foot/head tiles of the entrance
            exit_coords: the foot tile of the exit
        """
        return np.max((np.abs(entrance_coords[0] - exit_coords), np.abs(entrance_coords[1] - exit_coords))) > 1
