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


__all__ = ["SampleAveragingStrategy", "ExplicitSampleAveragingStrategy", "SampledFitness", "SampledFeatures", "SampledIndividual", "GenSampledIndividuals", "SamplingDecorator", "FixedSamplingDecorator", "TimeBasedSamplingDecorator", "AdaptiveSamplingDecorator"]

from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
import numpy as np
import random

from .base import *
from qdpy.base import *
from qdpy.phenotype import *
from qdpy.utils import *
from qdpy.containers import *
from qdpy import tools


@runtime
class SampleAveragingStrategy(Protocol):
    """TODO"""
    def sample(self, collection: Sequence[Any]) -> Tuple: ...

class ExplicitSampleAveragingStrategy(SampleAveragingStrategy):
    """TODO"""
    def sample(self, collection: Sequence[Any]) -> Tuple:
        if len(collection) > 0:
            return tuple(np.mean(np.array(collection), axis=0))
        else:
            return ()



class SampledFitness(Fitness):
    """TODO"""
    sampling_strategy: SampleAveragingStrategy
    ##nb_samples: int
    _samples: List[FitnessValuesLike]

#    def __new__(cls, values: FitnessValuesLike=(), weights: Optional[FitnessValuesLike]=None):
#        return super(Fitness, cls).__new__(cls)

    def __init__(self, sampling_strategy: SampleAveragingStrategy = ExplicitSampleAveragingStrategy(), values: FitnessValuesLike=(), weights: Optional[FitnessValuesLike]=None) -> None:
        self.sampling_strategy = sampling_strategy
        self._samples = []
        super().__init__(values, weights)
        self._compute_values()

    def reset(self) -> None:
        self._samples = []
        super().reset()

    def _compute_values(self) -> None:
        values = tuple(self.sampling_strategy.sample(self._samples))
        #self.wvalues = values
        self.wvalues = tuple(map(mul, values, self.weights))

    # XXX slow
    @property
    def values(self) -> FitnessValuesLike:
        #print("values:", self.wvalues, self.weights, tuple(map(truediv, self.wvalues, self.weights)))
        return tuple(map(truediv, self.wvalues, self.weights))

    # XXX slow
    @values.setter
    def values(self, values: FitnessValuesLike) -> None:
        #self.add_sample(tuple(map(mul, values, self.weights)))
        self.add_sample(values)
        #try:
        #    self.wvalues = tuple(map(mul, values, self.weights))
        #except TypeError:
        #    raise ValueError("Invalid ``value`` parameter.")

#    @values.deleter
#    def values(self) -> None:
#        self.wvalues = ()


    @property
    def samples(self) -> Sequence[FitnessValuesLike]:
        return list(self._samples)

    @property
    def nb_samples(self) -> int:
        return len(self._samples)

    def add_sample(self, sample) -> None:
        if sample != ():
            self._samples.append(sample)
            self._compute_values()
        #print("DEBUG add_sample:", sample, self._samples)

    def merge(self, other) -> None:
        self._samples += [x for x in other._samples if x != ()]
        self._compute_values()
        #print("DEBUG merge:", self._samples)



class SampledFeatures(Features):
    """TODO"""
    sampling_strategy: SampleAveragingStrategy
    #_nb_samples: int
    _samples: List[FeaturesValuesLike]

    def __init__(self, sampling_strategy: SampleAveragingStrategy = ExplicitSampleAveragingStrategy(), *args, **kwargs) -> None:
        self.sampling_strategy = sampling_strategy
        self._samples = []
        super().__init__(*args, **kwargs)
        self._compute_values()

    def reset(self) -> None:
        super().reset()
        self._samples = []

    def _compute_values(self) -> None:
        ##if self._samples_changed:
        ##print("DEBUG _compute_values:", self._samples, self.sampling_strategy.sample(self._samples))
        self._values = tuple(self.sampling_strategy.sample(self._samples))
        #self._values = np.mean(np.array(self._samples), axis=0)

    # XXX slow
    @property
    def values(self) -> FeaturesValuesLike:
        #self._compute_values()
        return self._values

    # XXX slow
    @values.setter
    def values(self, values: FeaturesValuesLike) -> None:
        self.add_sample(values)

    @property
    def samples(self) -> List[FeaturesValuesLike]:
        return list(self._samples)

    @property
    def nb_samples(self) -> int:
        return len(self._samples)

    def add_sample(self, sample) -> None:
        #self._samples_changed = True
        self._samples.append(sample)
        self._compute_values()

    def merge(self, other) -> None:
        self._samples += other._samples
        self._compute_values()



class SampledIndividual(Individual):
    def __init__(self, iterable: Optional[Iterable] = None,
            name: Optional[str] = None,
            fitness: FitnessLike = SampledFitness(), features: FeaturesLike = SampledFeatures()) -> None:
        super().__init__(iterable, name, fitness, features)
        self.expected_max_samples = 1
        assert(isinstance(fitness, SampledFitness))
        assert(isinstance(features, SampledFeatures))

    def reset(self):
        super().reset()
        #self.expected_max_samples = 1

    def merge(self, other):
        self.fitness.merge(other.fitness)
        self.features.merge(other.features)
        self.elapsed += other.elapsed

    @property
    def nb_samples(self) -> int:
        return self.fitness.nb_samples or 1 # type: ignore # XXX hack ?



@registry.register
class GenSampledIndividuals(GenIndividuals):
    def __next__(self):
        return SampledIndividual()




class SamplingDecorator(QDAlgorithm):
    """TODO"""
    inds_to_ask: List[List[Any]]
    staging_inds: List[SampledIndividual]

    def __init__(self, base_alg: QDAlgorithm, **kwargs):
        self.base_alg = base_alg
        #self.base_alg_batch_start = 0.
        self.inds_to_ask = []
        self.staging_inds = []

        kwargs['batch_size'] = kwargs.get('batch_size') or base_alg.batch_size
        kwargs['dimension'] = kwargs.get('dimension') or base_alg.dimension
        kwargs['optimisation_task'] = kwargs.get('optimisation_task') or base_alg.optimisation_task
        #kwargs['ind_domain'] = kwargs.get('ind_domain') or base_alg.ind_domain
        kwargs['name'] = kwargs.get('name') or base_alg.name
        kwargs['base_ind_gen'] = kwargs.get('base_ind_gen') or base_alg._base_ind_gen
        if kwargs['base_ind_gen'] is None:
            kwargs['base_ind_gen'] = GenSampledIndividuals()
        kwargs['budget'] = kwargs.get('budget') or base_alg.budget
        kwargs['container'] = base_alg.container
        super().__init__(**kwargs)

    @property # type: ignore
    def dimension(self) -> Optional[int]: # type: ignore
        return self.base_alg.dimension
    @dimension.setter
    def dimension(self, dimension: Optional[int]): # type: ignore
        self.base_alg.dimension = dimension

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        if len(self.inds_to_ask) > 0:
            ind, nb_samples = self.inds_to_ask[-1]
            if nb_samples == 0:
                print("WARNING: encoutered 'nb_samples' = 0")
                self.inds_to_ask.pop()
                return self._internal_ask(base_ind)
            elif nb_samples == 1:
                self.inds_to_ask.pop()
            else:
                self.inds_to_ask[-1][1] -= 1
            ind.reset()
            return ind
        else:
            ind = self._ask_sampling()
            #if ind.expected_max_samples > 1:
            #    self.inds_to_ask.append([ind, ind.expected_max_samples - 1])
            return ind

    def _ask_sampling(self) -> IndividualLike:
        ind = self.base_alg.ask()
        assert(isinstance(ind, SampledIndividual))
        self._ensure_enough_samples(ind, 1)
        return ind

    def _tell_sampling_before_container_update(self, individual: IndividualLike, xattr: MutableMapping[str, Any], add: bool) -> Tuple[MutableMapping[str, Any], bool]:
        return xattr, add

    def _tell_sampling_after_container_update(self, individual: IndividualLike, added: bool, xattr: Mapping[str, Any] = {}) -> None:
        pass


    def _tell_before_container_update(self, individual: IndividualLike) -> Tuple[MutableMapping[str, Any], bool]:
        assert(isinstance(individual, SampledIndividual))
        if individual in self.container:
            index = self.container.index(individual)
            ind_in_grid: SampledIndividual = self.container[index]
            #print(f"DEBUG _tell_before_container_update: {individual.features._samples} {ind_in_grid.features._samples}")
            self.container.discard(ind_in_grid)
            individual.merge(ind_in_grid)

        if individual in self.staging_inds:
            idx_staging = self.staging_inds.index(individual)
            staging = self.staging_inds.pop(idx_staging)
            individual.merge(staging)

        xattr, allowed_to_tell = super()._tell_before_container_update(individual)
        if individual.nb_samples >= individual.expected_max_samples:
            base_alg_xattr, allowed_to_tell2 = self.base_alg._tell_before_container_update(individual)
            base_alg_xattr['base_alg_reached_max_nb_samples'] = True
            xattr2, allowed_to_tell3 = self._tell_sampling_before_container_update(
                    individual, {**base_alg_xattr, **xattr}, allowed_to_tell and allowed_to_tell2)
            return xattr2, allowed_to_tell3
        else:
            xattr['base_alg_reached_max_nb_samples'] = False
            self.staging_inds.append(individual)
            return xattr, False

    def _tell_after_container_update(self, individual: IndividualLike, added: bool, xattr: Mapping[str, Any] = {}) -> None:
        if xattr.get("base_alg_reached_max_nb_samples", False):
            self.base_alg._tell_after_container_update(individual, added, xattr)
        super()._tell_after_container_update(individual, added, xattr)
        self._tell_sampling_after_container_update(individual, added, xattr)


    def _schedule_sampling(self, ind: SampledIndividual, nb_samples: int) -> None:
        if nb_samples > 0:
            ind.expected_max_samples += nb_samples
            self.inds_to_ask.append([ind, nb_samples])

    def _ensure_enough_samples(self, ind: SampledIndividual, nb_samples: int) -> None:
        self._schedule_sampling(ind, nb_samples - ind.expected_max_samples)



@registry.register
class FixedSamplingDecorator(SamplingDecorator):
    """TODO"""
    nb_samples: int

    def __init__(self, base_alg: QDAlgorithm, nb_samples = 1, **kwargs):
        self.nb_samples = nb_samples
        assert(self.nb_samples >= 1)
        super().__init__(base_alg, **kwargs)

    def _ask_sampling(self) -> IndividualLike:
        ind = self.base_alg.ask()
        assert(isinstance(ind, SampledIndividual))
        self._ensure_enough_samples(ind, self.nb_samples)
        return ind



@registry.register
class TimeBasedSamplingDecorator(SamplingDecorator):
    """TODO"""
    min_samples: int
    max_samples: int

    def __init__(self, base_alg: QDAlgorithm, min_samples = 5, max_samples = 100, **kwargs):
        self.min_samples = min_samples
        self.max_samples = max_samples
        assert(self.min_samples >= 1)
        assert(self.max_samples >= 1)
        assert(self.max_samples >= self.min_samples)
        super().__init__(base_alg, **kwargs)

    def _ask_sampling(self) -> IndividualLike:
        ind = self.base_alg.ask()
        assert(isinstance(ind, SampledIndividual))
        # XXX based on self.budget or self.base_alg.budget ??? Option ?
        nb_samples = int(self.min_samples +
                (self.max_samples - self.min_samples) * (self.nb_evaluations / self.budget))
        self._ensure_enough_samples(ind, nb_samples)
        return ind



@registry.register
class AdaptiveSamplingDecorator(TimeBasedSamplingDecorator):
    """TODO"""

    def __init__(self, base_alg: QDAlgorithm, min_samples = 5, max_samples = 100, **kwargs):
        super().__init__(base_alg, min_samples, max_samples, **kwargs)

    def _ask_sampling(self) -> IndividualLike:
        ind = self.base_alg.ask()
        assert(isinstance(ind, SampledIndividual))
        self._ensure_enough_samples(ind, self.min_samples)
        return ind

    def _tell_sampling_before_container_update(self, individual: IndividualLike, xattr: MutableMapping[str, Any], add: bool) -> Tuple[MutableMapping[str, Any], bool]:
        assert(isinstance(individual, SampledIndividual))
        # If the container is a grid, find if there is already an elite in the grid with the same index
        prev_elites: MutableSequence[Any] = []
        if isinstance(self.base_alg.container, Grid):
            try:
                index_g = self.base_alg.container.index_grid(individual.features)
                prev_elites = self.base_alg.container.solutions[index_g]
            except Exception as e:
                pass

        # Check if ``individual`` would be stored into an empty bin of the container
        if len(prev_elites) == 0:
            return xattr, add

        # Check if ``individual`` dominates all previous elites
        elite = prev_elites[0] # XXX hack ! Always selects the first elite of the bin
        assert(isinstance(elite, SampledIndividual))
        #dominates = np.all([individual.dominates(e) for e in prev_elites])
        if individual.dominates(elite):
            if individual.nb_samples >= elite.nb_samples or individual.nb_samples >= self.max_samples:
                return xattr, add
            else:
                return xattr, False
        else:
            elite_samples = min(elite.nb_samples + 1, self.max_samples)
            self._ensure_enough_samples(elite, elite_samples)
            return xattr, False



# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
