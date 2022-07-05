from pdb import set_trace as TT

from numba import njit
import numpy as np
from qdpy import containers
from ribs.archives import GridArchive
from ribs.archives._add_status import AddStatus

class InitStatesArchive():
    def __init__(self, bin_sizes, bin_bounds, n_init_states, map_dims, **kwargs):
#       if CONTINUOUS:
#           self.init_states_archive = np.empty(
#               shape=(*bin_sizes, n_init_states, 3, map_w, map_h)
#           )
#       else:
        self.init_states_archive = np.empty(
            shape=(*bin_sizes, n_init_states, *map_dims)
        )
        self.door_coords_archive = np.empty(
            shape=(*bin_sizes, n_init_states, 2, 2, len(map_dims))
        )

    def set_init_states(self, init_states, door_coords):
        self.init_states = init_states
        self.door_coords = door_coords


class CMAInitStatesGrid(InitStatesArchive, GridArchive):
    """Save (some of) the initial states upon which the elites were evaluated when added to the archive, so that we can
    reproduce their behavior at evaluation time (and compare it to evaluation to other seeds)."""
    def __init__(self, bin_sizes, bin_bounds, n_init_states, map_dims, **kwargs):
        InitStatesArchive.__init__(self, bin_sizes, bin_bounds, n_init_states, map_dims, **kwargs)
        GridArchive.__init__(self, bin_sizes, bin_bounds, **kwargs)


    def add(self, solution, objective_value, behavior_values, meta, index=None):
        status, dtype_improvement = GridArchive.add(self,
            solution, objective_value, behavior_values
        )

        # NOTE: for now we won't delete these when popping an elite for re-evaluation

        if status != AddStatus.NOT_ADDED:
            if index is None:
                index = self.get_index(behavior_values)
            archive_init_states(self.init_states_archive, self.door_coords_archive, self.init_states, self.door_coords, index)

        return status, dtype_improvement


class MEGrid(containers.Grid):
    def __init__(self, bin_sizes, bin_bounds, **kwargs):
        max_items_per_bin = int(200) if np.all(np.array(bin_sizes) == 1) else 1
        super(MEGrid, self).__init__(shape=bin_sizes, max_items_per_bin=max_items_per_bin,
                                     features_domain=bin_bounds,
                                     fitness_domain=((-np.inf, np.inf),),
                                     )

        # pyribs compatibility
        def get_index(self, bcs):
            return self.index_grid(features=bcs)

    def add(self, item):
        # We'll clip the feature calues at the extremes
        # TODO: what's happening in this case using pyribs?
        item.features.setValues([np.clip(item.features.values[i], *self.features_domain[i])
                                 for i in range(len(item.features.values))])

        return super(MEGrid, self).add(item)


class MEInitStatesArchive(InitStatesArchive, MEGrid):
    """Save (some of) the initial states upon which the elites were evaluated when added to the archive, so that we can
    reproduce their behavior at evaluation time (and compare it to evaluation to other seeds)."""

    def __init__(self, bin_sizes, bin_bounds, n_init_states, map_dims, **kwargs):
        InitStatesArchive.__init__(self, bin_sizes, bin_bounds, n_init_states, map_dims, **kwargs)
        MEGrid.__init__(self, bin_sizes, bin_bounds, **kwargs)


    def add(self, item):
        index = MEGrid.add(self, item)

        if index is not None:
            idx = self.index_grid(item.features)
            archive_init_states(self.init_states_archive, self.door_coords_archive, self.init_states, self.door_coords, idx)

        return index


class FlexArchive(InitStatesArchive):
    """ Subclassing a pyribs archive class to do some funky stuff."""

    def __init__(self, *args, **kwargs):
        self.n_evals = {}
        #       self.obj_hist = {}
        #       self.bc_hist = {}
        super().__init__(*args, **kwargs)
        #       # "index of indices", so we can remove them from _occupied_indices when removing
        #       self._index_ranks = {}
        self._occupied_indices = set()

    def _add_occupied_index(self, index):
        #       rank = len(self._occupied_indices)
        #       self._index_ranks[index] = rank  # the index of the index in _occupied_indices

        return super()._add_occupied_index(index)

    def _remove_occupied_index(self, index):
        self._occupied_indices.remove(index)
        self._occupied_indices_cols = tuple(
            [self._occupied_indices[i][j] for i in range(len(self._occupied_indices))]
            for j in range(len(self._storage_dims))
        )

    def pop_elite(self, obj, bcs, old_bcs):
        """
        Need to call update_elite after this!
        """
        # Remove it, update it
        old_idx = self.get_index(np.array(old_bcs))
        self._remove_occupied_index(old_idx)

        #       rank = self._index_ranks.pop(old_idx)
        #       self._occupied_indices.pop(rank)
        #       [self._occupied_indices_cols[i].pop(rank) for i in range(len(self._storage_dims))]
        n_evals = self.n_evals.pop(old_idx)
        old_obj = self._objective_values[old_idx]
        mean_obj = (old_obj * n_evals + obj) / (n_evals + 1)
        mean_bcs = np.array(
            [
                (old_bcs[i] * n_evals + bcs[i]) / (n_evals + 1)
                for i in range(len(old_bcs))
            ]
        )
        #       obj_hist = self.obj_hist.pop(old_idx)
        #       obj_hist.append(obj)
        #       mean_obj = np.mean(obj_hist)
        #       bc_hist = self.bc_hist.pop(old_idx)
        #       bc_hist.append(bcs)
        #       bc_hist_np = np.asarray(bc_hist)
        #       mean_bcs = bc_hist_np.mean(axis=0)
        self._objective_values[old_idx] = np.nan
        self._behavior_values[old_idx] = np.nan
        self._occupied[old_idx] = False
        solution = self._solutions[old_idx].copy()
        self._solutions[old_idx] = np.nan
        self._metadata[old_idx] = np.nan
        #       while len(obj_hist) > 100:
        #           obj_hist = obj_hist[-100:]
        #       while len(bc_hist) > 100:
        #           bc_hist = bc_hist[-100:]

        return solution, mean_obj, mean_bcs, n_evals

    def update_elite(self, solution, mean_obj, mean_bcs, n_evals):
        """
        obj: objective score from new evaluations
        bcs: behavior characteristics from new evaluations
        old_bcs: previous behavior characteristics, for getting the individuals index in the archive
        """

        # Add it back

        self.add(solution, mean_obj, mean_bcs, None, n_evals=n_evals)


    def add(self, solution, objective_value, behavior_values, meta, n_evals=0):

        index = self.get_index(behavior_values)

        status, dtype_improvement = super().add(
            solution, objective_value, behavior_values, meta, index
        )

        if not status == AddStatus.NOT_ADDED:
            if n_evals == 0:
                self.n_evals[index] = 1
            else:
                self.n_evals[index] = min(n_evals + 1, 100)

        return status, dtype_improvement


@njit
def archive_init_states(init_states_archive, door_coord_archive, init_states, door_coords, index):
    """Store the initial state/seed a generator used during training in a parallel archive, when the generator is added 
    to the generator archive.
    TODO: refactor storing initial states into the environment or an environment wrapped.
    """
    init_states_archive[index] = init_states
    if door_coords is not None:
        door_coord_archive[index] = door_coords


def get_qd_score(archive, args):
    """Get the QD score of the archive.
    
    Archive can be either a GridArchive (pyribs, CMA-ME) or a MEArchive (qdpy, MAP-Elites). Translate scores in the grid
    to all be > 0, so that the QD score is always positive, and never decreases when new individuals are added."""
    if args.algo == 'ME':
        # TODO: work out max diversity bonus to make this possible ?? Would this bias scores between n. latent seeds
        #   though?
        # qd_score = archive.qd_score()  # we need to specify lower *and upper* bounds for this
        qd_score = np.nansum(archive.quality_array + args.max_loss)
    else:
        df = archive.as_pandas(include_solutions=False)
        qd_score = (df['objective'] + args.max_loss).sum()
    return qd_score

