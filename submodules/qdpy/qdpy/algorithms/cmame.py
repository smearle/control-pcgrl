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


"""This file contains the CMA-ME emitters, as defined in paper "Covariance Matrix Adaptation for the Rapid Illumination of Behavior Space"(https://arxiv.org/abs/1912.02400). Additional algorithms are also included, notably the ME-MAP-Elites algorithm (Multi-Emitter MAP-Elites), as presented in the paper "Multi-Emitter MAP-Elites: Improving quality, diversity and convergence speed with heterogeneous sets of emitters".  All emitters are implemented as subclasses of the CMA-ES class, so the library "cma" needs to be installed to use them. """

__all__ = ["CMAMEOptimizingEmitter", "CMAMERandomDirectionEmitter", "CMAMEImprovementEmitter", "MEMAPElitesUCB1"]

from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
import numpy as np
import random


from .base import *
from .evolution import *
from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.containers import *
from qdpy import tools


try:
    import cma
except ImportError:
    pass



@registry.register
class CMAMEOptimizingEmitter(CMAES):
    """TODO"""

#    def __init__(self, container: Container, budget: int,
#            dimension: int, **kwargs):
#        super().__init__(container, budget, dimension=dimension, **kwargs)

    def reinit(self) -> None:
        super().reinit()
        self.restart()

    def restart(self):
        if len(self.container):
            init_ind = random.choice(self.container)
        else:
            init_ind = [0.] * self.dimension
        self.es = cma.CMAEvolutionStrategy(init_ind, self.sigma0, self._opts)
        self._pop_inds.clear()
        self._pop_fitness_vals.clear()

    def _stop(self) -> bool:
        term_cond = self.es.stop(check=False)
        #if len(term_cond) > 0:
        #    print(f"RESTART!!!: {term_cond}")
        return len(term_cond) > 0

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        if self._stop():
            self.restart()
        return super()._internal_ask(base_ind)


@registry.register
class CMAMERandomDirectionEmitter(CMAMEOptimizingEmitter):
    """TODO"""

    def restart(self):
        super().restart()
        features_domain = np.array(self.container.features_domain)
        self._direction = np.random.normal(0., 1., features_domain.shape[0])
        self._direction = features_domain[:,0] + self._direction * (features_domain[:,1] - features_domain[:,0])
        self._features_mean = np.zeros(features_domain.shape[0])

    def _internal_tell(self, individual: IndividualLike, added_to_container: bool, xattr: Mapping[str, Any] = {}) -> None:
        if self.ignore_if_not_added_to_container and not added_to_container:
            return

        self._features_mean += np.array(individual.features)
        if added_to_container:
            self._pop_inds += [individual]

        if len(self._pop_inds) >= self.es.popsize:
            self._features_mean /= self.es.popsize
            _pop_delta = []
            for ind in self._pop_inds:
                dv = ind.features - self._features_mean
                _pop_delta.append(self._direction.dot(dv))
            try:
                self.es.tell(self._pop_inds, _pop_delta)
            except RuntimeError:
                pass
            else:
                self._pop_inds.clear()
                self._pop_fitness_vals.clear()


@registry.register
class CMAMEImprovementEmitter(CMAMEOptimizingEmitter):
    """TODO"""

    _novel_pop_inds: MutableSequence[IndividualLike]
    _novel_pop_fitness_vals: MutableSequence[Any]

    def restart(self):
        super().restart()
        self._novel_pop_inds = []
        self._novel_pop_fitness_vals = []

    def _tell_before_container_update(self, individual: IndividualLike) -> Tuple[MutableMapping[str, Any], bool]:
        novel = None
        delta = 0.
        prev = None
        if hasattr(self.container, "index_grid") and hasattr(self.container, "solutions"):
            try:
                idx = self.container.index_grid(individual.features) # type: ignore
                novel = len(self.container.solutions[idx]) == 0 # type: ignore
                if not novel:
                    prev = self.container.solutions[0] # type: ignore
            except:
                pass
        delta = sum(individual.fitness.values)
        if prev is not None:
            delta -= sum(prev.fitness.values)
        return {"novel": novel, "delta": delta}, True

    def _internal_tell(self, individual: IndividualLike, added_to_container: bool, xattr: Mapping[str, Any] = {}) -> None:
        if self.ignore_if_not_added_to_container and not added_to_container:
            return
        novel = xattr.get("novel", False)
        delta = xattr.get("delta", 0.)

        if added_to_container:
            if novel:
                self._novel_pop_inds += [individual]
                self._novel_pop_fitness_vals += [-1. * delta]
            else:
                self._pop_inds += [individual]
                self._pop_fitness_vals += [-1. * delta]

        if len(self._pop_inds) + len(self._novel_pop_inds) >= self.es.popsize:
            tot_pop = []
            tot_pop_fitness_vals = []

            # Sort individuals
            sorted_pop_inds = [self._pop_inds[k] for k in argsort(self._pop_fitness_vals)]
            #sorted_pop_fitness_vals = sorted(self._pop_fitness_vals)
            sorted_novel_pop_inds = [self._novel_pop_inds[k] for k in argsort(self._novel_pop_fitness_vals)]
            #sorted_novel_pop_fitness_vals = sorted(self._novel_pop_fitness_vals)
            for i, ind in enumerate(sorted_novel_pop_inds):
                tot_pop.append(ind)
                tot_pop_fitness_vals.append(i)
            for i, ind in enumerate(sorted_pop_inds):
                tot_pop.append(ind)
                tot_pop_fitness_vals.append(len(sorted_novel_pop_inds) + i)

            try:
                self.es.tell(tot_pop, tot_pop_fitness_vals)
            except RuntimeError:
                pass
            else:
                self._pop_inds.clear()
                self._pop_fitness_vals.clear()
                self._novel_pop_inds.clear()
                self._novel_pop_fitness_vals.clear()


@registry.register
class MEMAPElitesUCB1(AlgWrapper):
    """Implementation of the ME-MAP-Elites algorithm (Multi-Emitter MAP-Elites), as presented in the paper "Multi-Emitter MAP-Elites: Improving quality, diversity and convergence speed with heterogeneous sets of emitters". UCB1 is used to select the active emitters. """
    zeta: float
    nb_active_emitters: int
    _active_emitters_idx: Sequence[int]
    current_active_idx: int
    initial_expected_rwds: float
    shuffle_emitters: bool

    def __init__(self, algorithms: Any, zeta: float = 0.0005, nb_active_emitters: int = 1, initial_expected_rwds: float = 1.,
            shuffle_emitters: bool = True, **kwargs):
        self.zeta = zeta
        self.nb_active_emitters = nb_active_emitters
        self.initial_expected_rwds = initial_expected_rwds
        self.shuffle_emitters = shuffle_emitters
        super().__init__(algorithms, **kwargs)

    def reinit(self) -> None:
        super().reinit()
        self._up_active_emitters()
        self.current_active_idx = 0

    def _up_active_emitters(self) -> None:
        """Update the list of active emitters."""
        #expected_rwds = [
        #        self.initial_expected_rwds if alg.nb_evaluations == 0 else
        #        float(alg.nb_updated) / float(alg.nb_evaluations) + self.zeta * math.sqrt(math.log(self.nb_evaluations) / float(alg.nb_evaluations))
        #        for alg in self.algorithms]
        #self._active_emitters_idx = argsort(expected_rwds, reverse=True)[:self.nb_active_emitters]

        algos_idx = list(range(len(self.algorithms)))
        if self.shuffle_emitters:
            # Shuffle algorithm list
            random.shuffle(algos_idx)
        # Compute expected rewards
        expected_rwds = [
                self.initial_expected_rwds if self.algorithms[a].nb_evaluations == 0 else
                    float(self.algorithms[a].nb_updated) / float(self.algorithms[a].nb_evaluations) +
                    self.zeta * math.sqrt(math.log(self.nb_evaluations) / float(self.algorithms[a].nb_evaluations))
                for a in algos_idx]
        # Update active emitters indexes list
        self._active_emitters_idx = [algos_idx[a] for a in argsort(expected_rwds, reverse=True)][:self.nb_active_emitters]
        #print(f"DEBUG _up_active_emitters: {argsort(expected_rwds, reverse=True)}")
        #print(f"DEBUG _up_active_emitters: {expected_rwds}")
        #print(f"DEBUG _up_active_emitters: {self._active_emitters_idx}")

    def next(self) -> None:
        """Switch to the next algorithm in Sequence `self.algorithms`, if there is one."""
        if self.current_active_idx < len(self._active_emitters_idx) - 1:
            self.current_active_idx += 1
        else:
            self._up_active_emitters()
            self.current_active_idx = 0
        next_idx = self._active_emitters_idx[self.current_active_idx]
        self.switch_to(next_idx)
        #print(f"NEXT: {next_idx}")






# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
