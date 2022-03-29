#    This file is part of qdpy.
#
#    qdpy is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    qdpy is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with qdpy. If not, see <http://www.gnu.org/licenses/>.

"""TODO"""

import numpy as np
#from scipy.spatial.distance import euclidean
#from itertools import starmap
from typing import Sequence, Callable, Tuple

from qdpy.phenotype import *
from qdpy.base import *


########### METRICS ########### {{{1

#@jit(nopython=True)
def features_distances(individual: IndividualLike, container: Sequence, dist: Union[str, Callable] = "euclidean") -> Sequence:
    distances = np.zeros(len(container))
    ind_features = individual.features
    if isinstance(dist, str):
        if dist == "euclidean":
            for i, other in enumerate(container):
                other_features = other.features
                for j in range(len(ind_features.values)):
                    distances[i] += pow(ind_features.values[j] - other_features.values[j], 2.)
            distances = np.power(distances, 1./2.)
        else:
            raise ValueError(f"Unknown `dist` type: '{dist}'.")
    else:
        for i, ind in enumerate(container):
            distances[i] = dist(ind_features, ind.features)
    return distances

def novelty(individual: IndividualLike, container: Sequence, k: int = 1, dist: Union[str, Callable] = "euclidean", ignore_first: bool = False, default_novelty: float = 0.1) -> float:
    """Returns the novelty score of ``individual`` in ``container``.
    Novelty is defined as the average distance to the ``k``-nearest neighbours of ``individual``. If ``container`` is empty, return ``default_novelty``."""
    if len(container) == 0:
        return default_novelty
    n_k = min(len(container), k)
    distances: Sequence = features_distances(individual, container, dist)
    if ignore_first:
        nearest_neighbours_dists: Sequence = sorted(distances)[1:n_k+1]
    else:
        nearest_neighbours_dists = sorted(distances)[:n_k]
    return np.mean(nearest_neighbours_dists)


def novelty_nn(individual: IndividualLike, container: Sequence, k: int = 1, nn_size: int = 1, dist: Union[str, Callable] = "euclidean", ignore_first: bool = False, default_novelty: float = 0.1) -> Tuple[float, Sequence]:
    """Returns the novelty score of ``individual`` in ``container`` and the indexes of its ``nn_size`` nearest neighbours.
    Novelty is defined as the average distance to the ``k``-nearest neighbours of ``individual``.  If ``container`` is empty, return ``default_novelty``."""
    if len(container) == 0:
        return default_novelty, []
    n_k = min(len(container), k)
    n_nn_size = min(len(container), nn_size)
    distances: Sequence = features_distances(individual, container, dist)
    idx_container = list(range(len(container)))
    if ignore_first:
        nearest_neighbours_dists: Sequence = sorted(distances)[1:n_k+1]
        nn: Sequence = sorted(zip(distances, idx_container))[1:n_nn_size+1]
    else:
        nearest_neighbours_dists = sorted(distances)[:n_k]
        nn = sorted(zip(distances, idx_container))[:n_nn_size]
    novelty: float = np.mean(nearest_neighbours_dists)
    nearest_neighbours_idx: Sequence
    _, nearest_neighbours_idx = tuple(zip(*nn))
    return novelty, nearest_neighbours_idx


def novelty_local_competition(individual: IndividualLike, container: Sequence, k: int = 1, dist: Union[str, Callable] = "euclidean", ignore_first: bool = False, default_novelty: float = 0.1, default_local_competition: float = 1.0) -> Tuple[float, float]:
    """Returns the novelty and normalised local competition scores of ``individual`` in ``container``.
    Novelty is defined as the average distance to the ``k``-nearest neighbours of ``individual``.
    Local competition is defined as the number of ``k``-nearest neighbours of ``individual`` that are outperformed by ``individual``. This value is normalised by ``k`` to be in domain [0., 1.].
    If ``container`` is empty, return ``default_novelty`` and ``default_local_competition``."""
    if len(container) == 0:
        return default_novelty, default_local_competition
    distances: Sequence = features_distances(individual, container, dist)
    nearest_neighbours_dists: Sequence
    nearest_neighbours: Sequence
    if ignore_first:
        nn: Sequence = sorted(zip(distances, container))[1:k+1]
    else:
        nn = sorted(zip(distances, container))[:k]
    nearest_neighbours_dists, nearest_neighbours = tuple(zip(*nn))
    novelty: float = np.mean(nearest_neighbours_dists)
    local_competition: float = sum((individual.fitness.dominates(ind.fitness) for ind in nearest_neighbours)) / float(k)
    return novelty, local_competition




# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
