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
#from __future__ import annotations

__all__ = ["OrderedSet", "Container", "Grid", "CVTGrid", "NoveltyArchive"]

########### IMPORTS ########### {{{1
from pdb import set_trace as TT
import sys
import warnings
import abc
import collections.abc
import math
from functools import reduce
import operator
import numpy as np
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, MutableMapping, Mapping, overload
from typing_extensions import runtime, Protocol
import traceback
import random

from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.metrics import novelty, novelty_local_competition, novelty_nn



########### BACKEND CLASSES ########### {{{1

@registry.register
class OrderedSet(MutableSet[T], Sequence[T]):
    """A MutableSet variant that conserves entries order, and can be accessed like a Sequence.
    This implementation is not optimised, but does not requires the type ``T`` of items to be hashable.
    It also does not implement indexing by slices.
    
    Parameters
    ----------
    iterable: Optional[Iterable[T]]
        items to add to the OrderedSet
    """

    _items: List[T]    # Internal storage

    def __init__(self, iterable: Optional[Iterable] = None) -> None:
        self._items = []
        if iterable is not None:
            self.update(iterable)

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[T]: ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError
        return self._items[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._items)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def __delitem__(self, idx) -> None:
        del self._items[idx]

    def count(self, key: T) -> int:
        return 1 if key in self else 0

    def index(self, key: T, start: int = 0, stop: int = sys.maxsize) -> int:
        return self._items.index(key, start, stop)

    def add(self, key: T) -> None:
        """Add ``key`` to this OrderedSet, if it is not already present. """
        try:
            self._items.index(key)
        except ValueError:
            self._items.append(key)

    def discard(self, key: T) -> None:
        """Discard ``key`` in this OrderedSet. Does not raise an exception if absent."""
        try:
            self._items.remove(key)
        except ValueError:
            return

    def update(self, iterable: Iterable) -> None:
        """Add the items in ``iterable``, if they are not already present in the OrderedSet.  """
        try:
            for item in iterable:
                self.add(item)
        except TypeError:
            raise ValueError(f"Argument needs to be an iterable, got {type(iterable)}")


BackendLike = Union[MutableSequence[T], OrderedSet[T]]

########### CONTAINER CLASSES ########### {{{1

# TODO verify that containers are thread-safe
@registry.register
class Container(Sequence, Summarisable, Copyable, CreatableFromConfig):
    """TODO
    
    Parameters
    ----------
    iterable: Iterable[IndividualLike] or None
        TODO
    storage_type: Backend (MutableSet or MutableSequence)
        TODO
    depot_type: bool or Backend (MutableSet or MutableSequence)
        TODO
    fitness_domain: Sequence[DomainLike] (sequence of 2-tuple of numbers)
        TODO
    features_domain: Sequence[DomainLike] (sequence of 2-tuple of numbers)
        TODO
    """ # TODO

    name: Optional[str]
    items: BackendLike[IndividualLike]
    depot: Optional[BackendLike[IndividualLike]]
    fitness_domain: Optional[Sequence[DomainLike]]
    features_domain: Optional[Sequence[DomainLike]]
    recentness: MutableSequence[int]
    _capacity: float
    _size: int
    _best: Optional[IndividualLike]
    _best_fitness: Optional[FitnessLike]
    _best_features: Optional[FeaturesLike]
    _nb_discarded: int
    _nb_added: int
    _nb_rejected: int

    def __init__(self, iterable: Optional[Iterable] = None,
            storage_type: Type[BackendLike] = list, depot_type: Union[bool, Type[BackendLike]] = False,
            fitness_domain: Optional[Sequence[DomainLike]] = None,
            features_domain: Optional[Sequence[DomainLike]] = None,
            capacity: Optional[float] = None, name: Optional[str] = None,
            **kwargs: Any) -> None:
        self.items = storage_type()
        if depot_type is True:
            self.depot = storage_type()
        elif not isinstance(depot_type, bool):
            self.depot = depot_type()
        else:
            self.depot = None
        self.fitness_domain = fitness_domain
        if self.fitness_domain is not None:
            for f in self.fitness_domain:
                if not is_iterable(f) or len(f) != 2:
                    raise ValueError("``fitness_domain`` must be a sequence of 2-tuples.")
        self.features_domain = features_domain
        if self.features_domain is not None:
            for f in self.features_domain:
                if not is_iterable(f) or len(f) != 2:
                    raise ValueError("``features_domain`` must be a sequence of 2-tuples.")
        self.recentness = []
        self.name = name if name is not None else f"{self.__class__.__name__}-{id(self)}"
        self._capacity = math.inf if capacity is None else capacity
        self._size = 0
        self._best = None
        self._best_fitness = None
        self._best_features = None
        self._nb_discarded = 0
        self._nb_rejected = 0
        self._nb_added = 0
        if iterable is not None:
            self.update(iterable)

    @property
    def capacity(self) -> float:
        """Return the capacity of the container (i.e. maximal number of items/spots/bins/etc). Can be math.inf."""
        return self._capacity

    @property
    def free(self) -> float:
        """Return the number of free spots in the container. Can be math.inf."""
        return self._capacity - self._size

    @property
    def size(self) -> int:
        """Return the size of the container (i.e. number of items, spots, bins, etc)."""
        return self._size

    @property
    def best(self) -> Any:
        """Return the best individual. """
        return self._best

    @property
    def best_fitness(self) -> Optional[FitnessLike]:
        """Return the fitness values of the individual with the best quality, or None. """
        return self._best_fitness

    @property
    def best_features(self) -> Optional[FeaturesLike]:
        """Return the features values of the individual with the best quality, or None. """
        return self._best_features

    @property
    def nb_discarded(self) -> int:
        """Return the number of individuals discarded by the container since its creation. """
        return self._nb_discarded

    @property
    def nb_added(self) -> int:
        """Return the number of individuals added into the container since its creation. """
        return self._nb_added

    @property
    def nb_rejected(self) -> int:
        """Return the number of individuals rejected (when added) by the container since its creation. """
        return self._nb_rejected

    @property
    def nb_operations(self) -> int:
        """Return the number of adds, modifications and discards since the creation of this container. """
        return self._nb_added + self._nb_discarded


    @property
    def fitness_extrema(self) -> Optional[Sequence[DomainLike]]:
        """Return the extrema values of the fitnesses of the stored individuals."""
        if len(self) == 0:
            return None
        maxima = np.array(self[0].fitness.values) # type: ignore
        minima = np.array(self[0].fitness.values) # type: ignore
        for ind in self:
            ind_f = ind.fitness.values
            for i in range(len(maxima)):
                if ind_f[i] > maxima[i]:
                    maxima[i] = ind_f[i]
                elif ind_f[i] < minima[i]:
                    minima[i] = ind_f[i]
        return tuple(zip(minima, maxima))


    @property
    def features_extrema(self) -> Optional[Sequence[DomainLike]]:
        """Return the extrema values of the features of the stored individuals."""
        if len(self) == 0:
            return None
        maxima = np.array(self[0].features) # type: ignore
        minima = np.array(self[0].features) # type: ignore
        for ind in self:
            ind_f = ind.features
            for i in range(len(maxima)):
                if ind_f[i] > maxima[i]:
                    maxima[i] = ind_f[i]
                elif ind_f[i] < minima[i]:
                    minima[i] = ind_f[i]
        return tuple(zip(minima, maxima))



    def size_str(self) -> str:
        """Return a string describing the fullness of the container."""
        if math.isinf(self.capacity):
            return str(self.size)
        else:
            return "%i/%i" % (self.size, self.capacity)

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, s: slice) -> Sequence[IndividualLike]: ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError
        return self.items[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.items

    def __iter__(self) -> Iterator[IndividualLike]:
        return iter(self.items)

#    def __reversed__(self) -> Iterator[IndividualLike]:
#        return reversed(self.items)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))


    def _add_to_collection(self, collection: BackendLike[IndividualLike], individual: IndividualLike) -> Tuple[bool,int]:
        """Add ``individual`` to ``collection``.
        Return a tuple containing (added, index), with ``added`` a bool saying whether ``individual`` was added or not to ``collection``, and ``index`` the index in the ``collection``.
        ``collection`` can be a MutableSequence or an ordered set implementing Sequence and MutableSet."""
        old_len: int = len(collection)
        if isinstance(collection, MutableSet):
            collection.add(individual)
            if len(collection) == old_len:
                return False, collection.index(individual)
            else:
                return True, len(collection) - 1
        elif isinstance(collection, MutableSequence):
            added: bool = False
            try:
                index: int = collection.index(individual)
            except ValueError:
                collection.append(individual)
                index = len(collection) - 1
                added = True
            return added, index
        else:
            raise ValueError("collection must be an ordered set implementing MutableSet or a Sequence")


    def in_bounds(self, val: Any, domain: Any) -> bool:
        """TODO"""
        if domain is None or len(domain) == 0:
            return True
        return in_bounds(val, domain)


    def _check_if_can_be_added(self, individual: IndividualLike) -> None:
        """TODO"""
        # Retrieve features and fitness from individual
        #ind_fitness: FitnessValuesLike = self._get_fitness_from_ind(individual)
        #ind_features: FeaturesLike = self._get_features_from_ind(individual)
        exc = None
        if not individual.fitness.valid:
            exc = ValueError(f"Fitness is not valid.")
        # Check if fitness and features are out of bounds
        if not self.in_bounds(individual.fitness.values, self.fitness_domain):
            exc = ValueError(f"fitness ({str(individual.fitness.values)}) out of bounds ({str(self.fitness_domain)}).")
        if not self.in_bounds(individual.features, self.features_domain):
            exc = ValueError(f"features ({str(individual.features)}) out of bounds ({str(self.features_domain)}).")
        if exc:
            self._nb_rejected += 1
            raise exc

    #def _add_internal(self, individual: T, raise_if_not_added_to_depot: bool, only_to_depot: bool, ind_fitness: FitnessLike, ind_features: FeaturesLike) -> Optional[int]:
    def _add_internal(self, individual: IndividualLike, raise_if_not_added_to_depot: bool, only_to_depot: bool) -> Optional[int]:
        """TODO"""
        # Verify if we do not exceed capacity
        if self.free < 1:
            self._nb_rejected += 1
            raise IndexError(f"No free slot available in this container.")
        if raise_if_not_added_to_depot and self.depot is None:
            self._nb_rejected += 1
            raise ValueError(f"`raise_if_not_added_to_depot` can only be set to True if a depot exists.")
        if only_to_depot and self.depot is None:
            self._nb_rejected += 1
            raise ValueError(f"`only_to_depot` can only be set to True if a depot exists.")
        # Add to depot, if needed
        added_depot, index_depot = self._add_to_collection(self.depot, individual) if self.depot is not None else False, 0
        if raise_if_not_added_to_depot and not added_depot:
            self._nb_rejected += 1
            raise ValueError(f"Individual could not be added to the depot.")
        if only_to_depot:
            return None
        else:
            # Add to storage
            added, index = self._add_to_collection(self.items, individual)
            # Update best_fitness
            if added:
                #if self._best is None or self._dominates(individual, self._best):
                if self._best is None or individual.fitness.dominates(self._best_fitness):
                    self._best = individual
                    self._best_fitness = individual.fitness
                    self._best_features = individual.features
                self.recentness.append(self._nb_added)
                self._size += 1
                self._nb_added += 1
            return index

    def add(self, individual: IndividualLike, raise_if_not_added_to_depot: bool = False) -> Optional[int]:
        """Add ``individual`` to the container, and returns its index, if successful, None elsewise. If ``raise_if_not_added_to_depot`` is True, it will raise and exception if it was not possible to add it also to the depot."""
        # Retrieve features and fitness from individual and check if they are not out-of-bounds
        self._check_if_can_be_added(individual)
        # Add
        return self._add_internal(individual, raise_if_not_added_to_depot, False)

    def _discard_by_index(self, individual: IndividualLike, idx: Optional[int] = None, idx_depot: Optional[int] = None, also_from_depot: bool = False) -> None:
        """Remove ``individual`` of the container. If ``also_from_depot`` is True, discard it also from the depot, if it exists. Use the indexes ``idx`` and ``idx_depot`` if they are provided."""
        # Remove from depot
        if also_from_depot and self.depot is not None:
            if idx_depot is None:
                try:
                    self.depot.remove(individual)
                except KeyError:
                    pass
            else:
                del self.depot[idx_depot]
        # Remove from container
        if idx is None:
            try:
                idx = self.items.index(individual)
            except KeyError:
                return
        del self.items[idx]
        del self.recentness[idx]
        self._size -= 1
        self._nb_discarded += 1
        if self._size < 0:
            raise RuntimeError("`self.size` < 0 !")


    def discard(self, individual: IndividualLike, also_from_depot: bool = False) -> None:
        """Remove ``individual`` of the container. If ``also_from_depot`` is True, discard it also from the depot, if it exists."""
        idx_depot = None
        idx = None
        if also_from_depot and self.depot is not None:
            try:
                idx_depot = self.depot.index(individual)
            except KeyError:
                pass
        # Remove from container
        try:
            idx = self.items.index(individual)
        except KeyError:
            pass
        self._discard_by_index(individual, idx, idx_depot, also_from_depot)


    def update(self, iterable: Iterable, ignore_exceptions: bool = True, issue_warning: bool = False) -> int:
        """Add the individuals in ``iterable``, if they are not already present in the container.
        If ``ignore_exceptions`` is True, it will ignore exceptions raised when adding each individual, but issue warnings instead.
        Returns the number of elements inserted or updated."""
        nb_inserted: int = 0
        item_index: Optional[int] = None
        try:
            if ignore_exceptions and issue_warning:
                for item in iterable:
                    try:
                        item_index = self.add(item)
                        if item_index is not None:
                            nb_inserted += 1
                    except IndexError as e:
                        warnings.warn(f"Adding individual failed (index out of bounds): {str(e)}")
                    except ValueError as e:
                        warnings.warn(f"Adding individual failed (attribute out of bounds): {str(e)}")
                    except Exception as e:
                        warnings.warn(f"Adding individual failed: {str(e)}")
                        traceback.print_exc()
            elif ignore_exceptions:
                for item in iterable:
                    try:
                        item_index = self.add(item)
                        if item_index is not None:
                            nb_inserted += 1
                    except Exception:
                        pass
            else:
                for item in iterable:
                    item_index = self.add(item)
                    if item_index is not None:
                        nb_inserted += 1
        except TypeError as e:
            raise e
#           raise ValueError(f"Argument needs to be an Iterable, got {type(iterable)}")
        return nb_inserted


    def clear(self) -> None:
        """Clear all individual in the collection (but not those in the depot, see method ``clear_all``)."""
        items = list(self)
        for e in items:
            self.discard(e)

    def clear_all(self) -> None:
        """Clear all individual in the collection AND in the depot."""
        self.clear()
        if self.depot is not None:
            self.depot.clear()


    def novelty(self, individual: IndividualLike, **kwargs):
        """Returns the novelty score of `individual`, using the depot as archive. TODO""" # TODO
        if self.depot is None:
            raise RuntimeError(f"A depot is necessary to assess novelty.")
        return novelty(individual, self.depot, **kwargs)

    def novelty_local_competition(self, individual: IndividualLike, **kwargs):
        """Returns the novelty and local competition scores of `individual`, using the depot as archive. TODO""" # TODO
        if self.depot is None:
            raise RuntimeError(f"A depot is necessary to assess novelty.")
        return novelty_local_competition(individual, self.depot, **kwargs)


    def to_grid(self, shape: Union[ShapeLike, int],
            max_items_per_bin: int = 1,
            capacity: Optional[float] = None, **kwargs: Any) -> Any:
        """Return a grid representation of this container, with `shape` the shape of the grid.
        
        Parameters
        ----------
        :param shape: Union[ShapeLike, int]
            The shape of the grid.
        :param max_items_per_bin: int
            The maximal number of entries stored in each bin of the grid. Defaults to 1.
        :param fitness_domain: Optional[Sequence[DomainLike]]
            The domain of the fitness of the individual of this grid. Default to `self.fitness_extrema`.
        :param features_domain: Optional[Sequence[DomainLike]]
            The domain of the features of the individual of this grid. Default to `self.features_extrema`.
        :param capacity: Optional[float] = None
            The capacity (i.e. maximal number of entries) of the returned grid representation. Default to None (i.e. no limit).
        :param storage_type: Type[BackendLike]
            How individuals are stored internally. Defaults to list.

        Return
        ------
        grid: Grid
            Grid representation of this container.
        """
        if not 'fitness_domain' in kwargs:
            kwargs['fitness_domain'] = self.fitness_domain if self.fitness_domain is not None else self.fitness_extrema
        if not 'features_domain' in kwargs:
            kwargs['features_domain'] = self.features_domain if self.features_domain is not None else self.features_extrema
        return Grid(self, shape=shape, max_items_per_bin=max_items_per_bin, capacity=capacity, **kwargs)


    def qd_score(self, normalized: bool = True) -> float:
        """Return the QD score of this container. It corresponds to the sum of the fitness values of all individuals in the container.

        Parameters
        ----------
        :param normalized: bool = True
            Normalize fitness values. If False, the returned QD score is computed by just summing all fitness values.
            If True, all fitness values are normalized depending on the domain of the fitness values, and on their weight (i.e., minimization vs maximization). Each fitness value is normalized as follow:
                if weight < 0.:
                    normalized_fitness_value = (bounds[1] - fitness_value) / (bounds[1] - bounds[0])
                else:
                    normalized_fitness_value = (fitness_value - bounds[0]) / (bounds[1] - bounds[0])
            Then, the returned QD is computed by summing all normalized fitness values.

        Return
        ------
        qd_score: float
            QD score of this container.
        """
        score: float = 0.
        if normalized:
            if self.fitness_domain is None:
                raise RuntimeError(f"'fitness_domain' must be set to compute normalized QD scores.")
            else:
                for ind in self:
                    for v, w, bounds in zip(ind.fitness.values, ind.fitness.weights, self.fitness_domain):
                        if w < 0.:
                            score += (bounds[1] - v) / (bounds[1] - bounds[0])
                        else:
                            score += (v - bounds[0]) / (bounds[1] - bounds[0])
        else:
            for ind in self:
                score += np.sum(ind.fitness.values)
        return score





# TODO Container is already a population ?
@registry.register
class Population(Container):
    """TODO""" # TODO
    pass



#def _nparray_from_dict(d: dict, shape: ShapeLike, default_val: float = np.nan, dtype: Type = float) -> np.array:
#    res = np.full(shape, default_val, dtype = dtype)
#    for k,v in d:
#        res[k] = v
#    return res



########### GRID-BASED CLASSES ########### {{{2

# Custom types
GridSolutionsLike = MutableMapping[GridIndexLike, MutableSequence]
GridItemsPerBinLike = MutableMapping[GridIndexLike, int]
GridFitnessLike = MutableMapping[GridIndexLike, MutableSequence[FitnessLike]]
GridFeaturesLike = MutableMapping[GridIndexLike, MutableSequence[FeaturesLike]]
GridQualityLike = MutableMapping[GridIndexLike, Optional[FitnessLike]]
GridRecentnessPerBinLike = MutableMapping[GridIndexLike, MutableSequence[int]]


@registry.register
class Grid(Container):
    """TODO""" # TODO

    fitness_domain: Sequence[DomainLike]
    features_domain: Sequence[DomainLike]

    _shape: ShapeLike
    _max_items_per_bin: int
    _filled_bins: int
    _solutions: GridSolutionsLike
    _nb_items_per_bin: GridItemsPerBinLike
    _fitness: GridFitnessLike
    _features: GridFeaturesLike
    _quality: GridQualityLike
    _quality_array: np.array
    _bins_size: Sequence[float]
    _nb_bins: int
    recentness_per_bin: GridRecentnessPerBinLike
    history_recentness_per_bin: GridRecentnessPerBinLike
    activity_per_bin: np.array
    discard_random_on_bin_overload: bool


    def __init__(self, iterable: Optional[Iterable] = None,
            shape: Union[ShapeLike, int] = (1,), max_items_per_bin: int = 1,
            fitness_domain: Optional[Sequence[DomainLike]] = ((0., np.inf),),
            discard_random_on_bin_overload = False,
            **kwargs: Any) -> None:
        self._shape = tuplify(shape)
        self._max_items_per_bin = max_items_per_bin
        self.discard_random_on_bin_overload = discard_random_on_bin_overload
        super().__init__([], fitness_domain=fitness_domain, **kwargs)
        if self.features_domain is None or len(self.features_domain) == 0:
            raise ValueError("`features_domain` must be specified and have a length > 0.")
        if self.fitness_domain is None or len(self.fitness_domain) == 0:
            raise ValueError("`fitness_domain` must be specified and have a length > 0.")
        #if len(self.features_domain) != len(self.shape):
        #    raise ValueError("`features_domain` must have the same shape as `shape`.")
        self._init_grid()
        if not "capacity" in kwargs:
            self._capacity = self._nb_bins * self._max_items_per_bin
        if iterable is not None:
            self.update(iterable)


    def _init_grid(self) -> None:
        """Initialise the grid to correspond to the shape `self.shape`."""
        self._solutions = {x: [] for x in self._index_grid_iterator()}
        self._nb_items_per_bin = np.zeros(self._shape, dtype=int) #{x: 0 for x in self._index_grid_iterator()}
        self._fitness = {x: [] for x in self._index_grid_iterator()}
        self._features = {x: [] for x in self._index_grid_iterator()}
        self._quality = {x: None for x in self._index_grid_iterator()}
        self._quality_array = np.full(self._shape + (len(self.fitness_domain),), np.nan)
        self._bins_size = [(self.features_domain[i][1] - self.features_domain[i][0]) / float(self.shape[i]) for i in range(len(self.shape))]
        self._filled_bins = 0
        self._nb_bins = reduce(operator.mul, self._shape)
        self.recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.history_recentness_per_bin = {x: [] for x in self._index_grid_iterator()}
        self.activity_per_bin = np.zeros(self._shape, dtype=float)


    @property
    def shape(self) -> ShapeLike:
        """Return the shape of the grid."""
        return self._shape

    @property
    def max_items_per_bin(self) -> int:
        """Return the maximal number of items stored in a bin of the grid."""
        return self._max_items_per_bin

    @property
    def filled_bins(self) -> int:
        """Return the number of filled bins of the container."""
        return self._filled_bins

    @property
    def solutions(self) -> GridSolutionsLike:
        """Return the solutions in the grid."""
        return self._solutions

    @property
    def nb_items_per_bin(self) -> GridItemsPerBinLike:
        """Return the number of items stored in each bin of the grid."""
        return self._nb_items_per_bin

    @property
    def fitness(self) -> GridFitnessLike:
        """Return the fitness values in the grid."""
        return self._fitness

    @property
    def features(self) -> GridFeaturesLike:
        """Return the features values in the grid."""
        return self._features

    @property
    def quality(self) -> GridQualityLike:
        """Return the best fitness values in the grid."""
        return self._quality

    @property
    def quality_array(self) -> np.array:
        """Return the best fitness values in the grid, as a numpy array."""
        return self._quality_array

    @property
    def best_index(self) -> Optional[GridIndexLike]:
        """Return the index of the individual with the best quality, or None. """
        if self._best_features is None:
            return None
        else:
            return self.index_grid(self._best_features)


    def filled_str(self) -> str:
        """Return a string describing the fullness of the grid (not the container itself, which is handled by ``Container.size_str``)."""
        return "%i/%i" % (self.filled_bins, self._nb_bins)


    def index_grid(self, features: FeaturesLike) -> GridIndexLike:
        """Get the index in the grid of a given individual with features ``features``, raising an IndexError if it is outside the grid. """
        index: List[int] = []
        if len(features) != len(self.shape):
            raise IndexError(f"Length of parameter ``features`` ({len(features)}) does not corresponds to the number of dimensions of the grid ({len(self.shape)}).")
        for i in range(len(features)):
            normalised_feature: float = features[i] - self.features_domain[i][0]
            if normalised_feature == self.features_domain[i][1] - self.features_domain[i][0]:
                partial: int = self.shape[i] - 1
            elif normalised_feature > self.features_domain[i][1] - self.features_domain[i][0]:
                raise IndexError(f"``features`` ({str(features)}) out of bounds ({str(self.features_domain)})")
            else:
                partial = int(normalised_feature / self._bins_size[i])
            index.append(partial)
        return tuple(index)


    def _index_grid_iterator(self) -> Generator[GridIndexLike, None, None]:
        """Return an iterator of the index of the grid, based on ``self.shape``."""
        val: List[int] = [0] * len(self._shape)
        yield tuple(val)
        while True:
            for i in reversed(range(len(self._shape))):
                val[i] += 1
                if val[i] >= self._shape[i]:
                    if i == 0:
                        return
                    val[i] = 0
                else:
                    yield tuple(val)
                    break

    def _update_quality(self, ig: GridIndexLike) -> None:
        """Update quality in bin ``ig`` of the grid."""
        if self._nb_items_per_bin[ig] == 0:
            val: Optional[FitnessLike] = None
        elif self._nb_items_per_bin[ig] == 1:
            val = self.fitness[ig][0]
        else:
            best: IndividualLike = self.solutions[ig][0]
            for s in self.solutions[ig][1:]:
                if s.fitness.dominates(best.fitness):
                    best = s
            val = best.fitness
        self.quality[ig] = val
        if val is None:
            self.quality_array[ig] = math.nan
        else:
            self.quality_array[ig] = val.values


    def add(self, individual: IndividualLike, raise_if_not_added_to_depot: bool = False) -> Optional[int]:
        """Add ``individual`` to the grid, and returns its index, if successful, None elsewise. If ``raise_if_not_added_to_depot`` is True, it will raise and exception if it was not possible to add it also to the depot."""
        # Retrieve features and fitness from individual and check if they are not out-of-bounds
        self._check_if_can_be_added(individual)
        # Find corresponding index in the grid
        ig = self.index_grid(individual.features) # Raise exception if features are out of bounds

        # Check if individual can be added in grid, if there are enough empty spots
        can_be_added: bool = False
        if self._nb_items_per_bin[ig] < self.max_items_per_bin:
            can_be_added = True
        else:
            if self.discard_random_on_bin_overload:
                idx_to_discard = random.randint(0, len(self.solutions[ig])-1)
                Container.discard(self, self.solutions[ig][idx_to_discard])
                self._discard_from_grid(ig, idx_to_discard)
                can_be_added = True
            else:
                worst_idx = 0
                worst: IndividualLike = self.solutions[ig][worst_idx]
                if self._nb_items_per_bin[ig] > 1:
                    for i in range(1, self._nb_items_per_bin[ig]):
                        s = self.solutions[ig][i]
                        if worst.fitness.dominates(s.fitness):
                            worst = s
                            worst_idx = i
                if individual.fitness.dominates(worst.fitness):
                    Container.discard(self, self.solutions[ig][worst_idx])
                    self._discard_from_grid(ig, worst_idx)
                    can_be_added = True

        # Add individual in grid, if there are enough empty spots
        if can_be_added:
            if self._nb_items_per_bin[ig] == 0:
                self._filled_bins += 1
            # Add individual in container
            old_len: int = self._size
            index: Optional[int] = self._add_internal(individual, raise_if_not_added_to_depot, False)
            if index == old_len: # Individual was not already present in container
                self._solutions[ig].append(individual)
                self._fitness[ig].append(individual.fitness)
                self._features[ig].append(individual.features)
                self.recentness_per_bin[ig].append(self._nb_added)
                self.history_recentness_per_bin[ig].append(self._nb_added)
                self._nb_items_per_bin[ig] += 1
                self.activity_per_bin[ig] += 1
            # Update quality
            self._update_quality(ig)
            return index
        else:
            # Only add to depot
            if self.depot is not None:
                self._add_internal(individual, raise_if_not_added_to_depot, True)
            return None

    def _discard_from_grid(self, ig: GridIndexLike, index_in_bin: int) -> None:
        # Remove individual from grid
        del self._solutions[ig][index_in_bin]
        self._nb_items_per_bin[ig] -= 1
        del self._fitness[ig][index_in_bin]
        del self._features[ig][index_in_bin]
        del self.recentness_per_bin[ig][index_in_bin]
        # Update quality
        self._update_quality(ig)
        # Update number of filled bins
        if self._nb_items_per_bin[ig] == 0:
            self._filled_bins -= 1


    def discard(self, individual: IndividualLike, also_from_depot: bool = False) -> None:
        """Remove ``individual`` of the container. If ``also_from_depot`` is True, discard it also from the depot, if it exists."""
        old_len: int = self._size
        # Remove individual from container
        Container.discard(self, individual, also_from_depot)
        if self._size == old_len:
            return
        # Remove individual from grid
        ig = self.index_grid(individual.features) # Raise exception if features are out of bounds
        index_in_bin = self.solutions[ig].index(individual)
        self._discard_from_grid(ig, index_in_bin)

    def _get_best_inds(self):
        best_inds = []
        for idx, inds in self.solutions.items():
            if len(inds) == 0:
                continue
            best = inds[0]
            for ind in inds[1:]:
                if ind.fitness.dominates(best.fitness):
                    best = ind
            best_inds.append(best)
        return best_inds


    def qd_score(self, normalized: bool = True) -> float:
        """Return the QD score of this grid. It corresponds to the sum of the fitness values of the best individuals of each bin of the grid.

        Parameters
        ----------
        :param normalized: bool = True
            Normalize fitness values. If False, the returned QD score is computed by just summing all fitness values.
            If True, all fitness values are normalized depending on the domain of the fitness values, and on their weight (i.e., minimization vs maximization). Each fitness value is normalized as follow:
                if weight < 0.:
                    normalized_fitness_value = (bounds[1] - fitness_value) / (bounds[1] - bounds[0])
                else:
                    normalized_fitness_value = (fitness_value - bounds[0]) / (bounds[1] - bounds[0])
            Then, the returned QD is computed by summing all normalized fitness values.

        Return
        ------
        qd_score: float
            QD score of this container.
        """
        score: float = 0.
        if normalized:
            if self.fitness_domain is None:
                raise RuntimeError(f"'fitness_domain' must be set to compute normalized QD scores.")
            else:
                best_inds = self._get_best_inds()
                for ind in best_inds:
                    for v, w, bounds in zip(ind.fitness.values, ind.fitness.weights, self.fitness_domain):
                        if w < 0.:
                            score += (bounds[1] - v) / (bounds[1] - bounds[0])
                        else:
                            score += (v - bounds[0]) / (bounds[1] - bounds[0])
        else:
            for ind in self:
                score += np.sum(ind.fitness.values)
        return score





from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor

@registry.register
class CVTGrid(Grid):
    """TODO""" # TODO

    _grid_shape: ShapeLike
    _nb_sampled_points: int
    cluster_centers: np.array

    def __init__(self, iterable: Optional[Iterable] = None,
            shape: Union[ShapeLike, int] = (1,), max_items_per_bin: int = 1,
            grid_shape: Union[ShapeLike, int] = (1,), nb_sampled_points: int = 50000, **kwargs: Any) -> None:
        self._grid_shape = tuplify(grid_shape)
        self._nb_sampled_points = nb_sampled_points
        super().__init__(iterable, shape, max_items_per_bin, **kwargs)
        if len(self.shape) != 1:
            raise ValueError("Using CVTGrid, `shape` must be a scalar or a sequence of length 1.")
        if nb_sampled_points <= 0:
            raise ValueError("`nb_sampled_points` must be positive and greatly superior to `shape` and `grid_shape`.")
        self._init_clusters()

    def _init_clusters(self) -> None:
        """Initialise the clusters and tessellate the grid."""
        sample = np.random.uniform(0.0, 1.0, (self.nb_sampled_points, len(self.grid_shape)))
        for i, d in enumerate(self.features_domain):
            sample[:, i] = d[0] + (d[1] - d[0]) * sample[:, i]
        kmeans = KMeans(init="k-means++", n_clusters=self.shape[0], n_init=1, n_jobs=1, verbose=0)
        kmeans.fit(sample)
        self.cluster_centers = kmeans.cluster_centers_

    @property
    def grid_shape(self) -> ShapeLike:
        """Return the shape of the grid."""
        return self._grid_shape

    @property
    def nb_sampled_points(self) -> int:
        """Return the number of sampled points to identify cluster centers."""
        return self._nb_sampled_points

    def index_grid(self, features: FeaturesLike) -> GridIndexLike:
        """Get the index in the cvt of a given individual with features ``features``, raising an IndexError if it is outside the cvt. """
        dists = np.empty(self._shape[0])
        for i in range(len(dists)):
            dists[i] = math.sqrt(np.sum(np.square(self.cluster_centers[i] - features)))
        return (np.argmin(dists),)



########### ARCHIVE-BASED CLASSES ########### {{{2

@registry.register
class NoveltyArchive(Container):
    """TODO""" # TODO

    depot: BackendLike[IndividualLike]

    k: int
    threshold_novelty: float
    novelty_distance: Union[str, Callable]

    def __init__(self, iterable: Optional[Iterable] = None,
            k: int = 15, threshold_novelty: float = 0.01, novelty_distance: Union[str, Callable] = "euclidean",
            depot_type: Union[bool, Type[BackendLike]] = True, **kwargs: Any) -> None:
        self.k = k
        self.threshold_novelty = threshold_novelty
        self.novelty_distance = novelty_distance
        if depot_type is None:
            raise ValueError("``depot_type`` must be specified for an archive container (either True or a BackendLike class).")
        super().__init__(iterable, depot_type=depot_type, **kwargs)


    def add(self, individual: IndividualLike, raise_if_not_added_to_depot: bool = True) -> Optional[int]:
        """Add ``individual`` to the archive, and returns its index, if successful, None elsewise. If ``raise_if_not_added_to_depot`` is True, it will raise and exception if it was not possible to add it also to the depot."""
        # Retrieve features and fitness from individual and check if they are not out-of-bounds
        self._check_if_can_be_added(individual)
        # Find novelty of this individual, and its nearest neighbour
        novelty, nn = novelty_nn(individual, self.depot, k=self.k, nn_size=1, dist=self.novelty_distance, ignore_first=False)
        if novelty > self.threshold_novelty:
            # Add individual
            return self._add_internal(individual, raise_if_not_added_to_depot, False)
        else:
            ind_nn = self.depot[nn[0]]
            if len(nn) > 0 and individual.fitness.dominates(ind_nn.fitness):
                #self.discard(nn[0])
                self._discard_by_index(ind_nn, idx_depot=nn[0])
                return self._add_internal(individual, raise_if_not_added_to_depot, False)
            else:
                return None



## MODELINE	"{{{1
## vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
## vim:foldmethod=marker
