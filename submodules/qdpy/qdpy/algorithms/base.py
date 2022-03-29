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

"""All the base classes and protocols describing QD algorithms (optimisers). All of these algorithms perform optimisation on a defined objective functions, and use a provided container to store the evaluated solutions. The base abstract class of algorithms is `QDAlgorithm`, itself implementing the `QDAlgorithmLike` Protocol. Most QD algorithms will subclass QDAlgorithm directly, but in some special cases, algorithms might need to subclass QDAlgorithmLike instead (e.g. sequences `Sq` of QD algorithms).

All algorithms must be registered to the default qdpy.base.Registry `qdpy.base.registry` if they are to be easily accessed or created by factories. It can be done by decorating QD algorithms classes with `@registry.register`. """

__all__ = ["partial", "QDAlgorithmLike", "QDAlgorithm", "Lambda", "AlgWithBatchSuggestions", "Sq", "AlgWrapper", "AlternatingAlgWrapper", "SqAlgWrapper", "FromArchive"]

import abc
import math
import time
from timeit import default_timer as timer
from inspect import signature
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
from typing_extensions import runtime, Protocol
import warnings
import numpy as np
import copy
import traceback
import os


from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.containers import *
from qdpy import tools

import functools
partial = functools.partial # type: ignore

default_algorithm_logger: Any

def _evalWrapper(eval_id: int, fn: Callable, *args, **kwargs) -> Tuple[int, float, Any, Any]:
    """Wrapper around an evaluation function. It catches any exceptions raised by the evaluation function (as exceptions might be lost depending on which kind of parallelism scheme is applied to launch evaluations). It also measures the time elapsed by the evaluation function computation."""
    start_time = timer()
    exc = None
    fitness = None
    features = None
    res = None
    try:
        res = fn(*args, **kwargs)
    except Exception as e:
        print(f"Exception during evaluation: {e}")
        traceback.print_exc()
        exc = e
    elapsed = timer() - start_time
    return eval_id, elapsed, res, exc


def _severalEvalsWrapper(eval_id: int, fn: Callable, *args, **kwargs) -> Tuple[int, float, Any, Any]:
    """Wrapper around several evaluations of an objective function. It catches any exceptions raised by the evaluation function (as exceptions might be lost depending on which kind of parallelism scheme is applied to launch evaluations). It also measures the time elapsed by the evaluation function computation."""
    start_time = timer()
    exc = None
    fitness = None
    features = None
    res = None
    try:
        res = fn(*args, **kwargs)
    except Exception as e:
        print(f"Exception during evaluation: {e}")
        traceback.print_exc()
        exc = e
    elapsed = timer() - start_time
    return eval_id, elapsed, res, exc


class _CallbacksManager(Copyable):
    """Manager of callback functions that are called when specific events of the `QDAlgorithm` class are triggered.
    It is possible to register a callback function using the `self.add_callback` method. The callback functions
    attached to a specific event are retrieved through the `self.get` method.

    We regroup callback function handling in this private class to ease the serialisation of QDAlgorithm and its sub-classes.
    Namely, the callback functions provided by the user may not be pickleable.  To prevent them to be taken into account during serialisation, we declare a __getstate__ method returning the __dict__ dictionary of object attributes, minus the registered callback function objects. This __getstate__ is not inherited by sub-classes, which make it unadapted to be specified directly in the QDAlgorithm class, as it would force all sub-classes of QDAlgorithm to redefine the __getstate__ method to be pickleable."""
    callbacks: MutableMapping[str, MutableSequence[Callable]]

    def __init__(self):
        self.callbacks = {}
        pass

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['callbacks']
        return odict

    def get(self, event: str) -> Sequence[Callable]:
        """Return the Sequence of functions that can be triggered when `event` happens. 

        Parameters
        ----------
        :param event: str
            The event triggering the returned callback functions.
        """
        return self.callbacks.get(event, [])

    def add_callback(self, event: str, fn: Callable) -> None:
        """Attach a callable `fn` to this CallbacksManager. `fn` will be called each time the event specified in `event` occurs.
        
        Parameters
        ----------
        :param event: str
            The event triggering the callback.
        :param fn: Callable
            The callback function called when the event specified in `event` occurs.
        """
        self.callbacks.setdefault(event, []).append(fn)


@runtime
class QDAlgorithmLike(Protocol):
    """Protocol (Python interfaces) describing QD algorithms. All classes describing qdpy algorithms must implement this protocol -- or alternatively the `qdpy.algorithms.QDAlgorithm` abstract class, which already implements `qdpy.algorithms.QDAlgorithmLike`.

    Qdpy algorithms are based around the ask-and-tell paradigm, introduced by [Collete2010].
    TODO description ask-and-tell

    TODO description functions

    References
    ----------
    [Collete2010] TODO
    """
    name: str
    container: Container
    dimension: Optional[int]
    budget: int

    def reinit(self) -> None: ...

    def ask(self) -> IndividualLike: ...

    def tell(self, individual: IndividualLike, fitness: Optional[Any] = None,
            features: Optional[FeaturesLike] = None, elapsed: Optional[float] = None) -> bool: ...

    def tell_container_entries(self, cont: Optional[Container] = None) -> None: ...

    def best(self) -> IndividualLike: ...

    def optimise(self, evaluate: Callable, budget: Optional[int] = None, 
            batch_mode: bool = True, executor: Optional[ExecutorLike] = None,
            send_several_suggestions_to_fn: bool = False) -> IndividualLike: ...

    def add_callback(self, event: str, fn: Callable) -> None: ...

    @property
    def nb_suggestions(self) -> int: ...

    @property
    def nb_suggestions_in_iteration(self) -> int: ...

    @property
    def nb_evaluations(self) -> int: ...

    @property
    def nb_evaluations_in_iteration(self) -> int: ...

    @property
    def nb_updated(self) -> int: ...

    @property
    def nb_updated_in_iteration(self) -> int: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def nb_max_iter(self) -> Union[int,float]: ...

    @property
    def current_iter(self) -> int: ...





# TODO say inspiration from nevergrad
class QDAlgorithm(abc.ABC, QDAlgorithmLike, Summarisable, Saveable, Copyable, CreatableFromConfig):
    """TODO"""

    #ind_domain: Optional[DomainLike]
    #executor: Optional[ExecutorLike]

    name: str
    container: Container
    dimension: Optional[int]
    budget: int
    _base_ind_gen: Optional[Generator[IndividualLike, None, None]]
    _nb_objectives: int
    _optimisation_task: Union[str, Sequence[float]]
    #_elapsed: float = 0.
    _fitness_weights: Sequence
    _nb_suggestions: int
    _nb_suggestions_in_iteration: int
    _nb_evaluations: int
    _nb_evaluations_in_iteration: int
    _nb_updated: int
    _nb_updated_in_iteration: int
    _batch_size: Optional[int]
    _callbacks: _CallbacksManager


    def __init__(self, container: Container, budget: int, batch_size: Optional[int] = None,
            dimension: Optional[int] = None, nb_objectives: Optional[int] = None,
            optimisation_task: Union[str, Sequence[float]] = "minimisation",
            base_ind_gen: Optional[Generator[IndividualLike, None, None]] = None,
            tell_container_at_init: Union[bool, str] = False, # XXX
            add_default_logger: bool = False, name: Optional[str] = None,
            **kwargs):
        self.container = container
        self.dimension = dimension
        self.budget = budget
        self._batch_size = batch_size
        self._base_ind_gen = base_ind_gen
        self.name = name if name is not None else f"{self.__class__.__name__}-{id(self)}"

        if nb_objectives is not None:
            self._nb_objectives = nb_objectives
        else:
            if self.container.fitness_domain is None or len(self.container.fitness_domain) == 0:
                #raise ValueError("You must either specify `nb_objectives` or providing as `container` a Container with a declared fitness domain.")
                warnings.warn(f"Unspecified number of objectives ! When creating a QDAlgorithm class, you must either specify `nb_objectives` or providing as `container` a Container with a declared fitness domain. Defaulting to only 1 objective.")
                self._nb_objectives = 1
            else:
                self._nb_objectives = len(self.container.fitness_domain)

        self.reinit()
        self._callbacks = _CallbacksManager()
        self.optimisation_task = optimisation_task
        if add_default_logger:
            default_algorithm_logger.monitor(self)

        # If needed, tell container entries
        tell_container_depot: bool = False
        tell_container: bool = False
        if isinstance(tell_container_at_init, bool):
            tell_container = tell_container_at_init
        else:
            if tell_container_at_init.lower() == "yes":
                tell_container = True
            elif tell_container_at_init.lower() == "depot":
                tell_container_depot = True
        if tell_container_depot:
            if self.container.depot is None:
                raise ValueError("`tell_container_depot` can only be set to 'depot' if `container` possesses a depot.")
            for e in self.container.depot:
                is_in_container = e in self.container
                self._internal_tell(e, is_in_container)
        elif tell_container:
            for e in self.container:
                self._internal_tell(e, True)


    def reinit(self) -> None:
        self._nb_suggestions = 0
        self._nb_suggestions_in_iteration = 0
        self._nb_evaluations = 0
        self._nb_evaluations_in_iteration = 0
        self._nb_updated = 0
        self._nb_updated_in_iteration = 0


    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['_base_ind_gen']
        return odict

    # Do not copy the container when deep copying
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "container":
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


    @property
    def optimisation_task(self) -> Union[str, Sequence[float]]:
        return self._optimisation_task

    @optimisation_task.setter
    def optimisation_task(self, optimisation_task: Union[str, Sequence[float]]) -> None:
        """Update the kind of optimisation task performed by this `QDAlgorithm`.
        
        Parameters
        ----------
        :param optimisation_task: Union[str, Sequence[float]]
            The kind of optimisation task to perform for all objectives of the fitness functions.
            If `optimisation_task` is a string, it can take the following values:
                - for minimising all fitness objectives: 'minimisation', 'minimization', 'minimise', 'minimize', 'min'
                - for maximising all fitness objectives: 'maximisation', 'maximization', 'maximise', 'maximize', 'max'
            If `optimisation_task` is a Sequence of floats, its length corresponds to the number of objectives, and each
            entry is a weight to apply to each objective score.
            Maximised objectives have a corresponding weight > 0 while minimised objectives have weights < 0.
            The length of this Sequence must be same as the length of the `self.container.fitness_domain` attribute.
        """
        if isinstance(optimisation_task, str) and \
                any((optimisation_task == a for a in ["minimisation", "minimization", "minimise", "minimize", "min"])):
            self._optimisation_task = "minimisation"
            self._fitness_weights = (-1.,) * self._nb_objectives
        elif isinstance(optimisation_task, str) and \
                any((optimisation_task == a for a in ["maximisation", "maximization", "maximise", "maximize", "max"])):
            self._optimisation_task = "maximisation"
            self._fitness_weights = (1.,) * self._nb_objectives
        elif isinstance(optimisation_task, Sequence) and not isinstance(optimisation_task, str) and len(optimisation_task) > 0:
            self._optimisation_task = optimisation_task
            self._fitness_weights = optimisation_task
        else:
            raise ValueError("`optimisation_task` must be either a tuple of fitness weights, or a string equals to either 'minimisation' or 'maximisation'.")


    @property
    def nb_suggestions(self) -> int:
        return self._nb_suggestions

    @property
    def nb_suggestions_in_iteration(self) -> int:
        return self._nb_suggestions_in_iteration

    @property
    def nb_evaluations(self) -> int:
        return self._nb_evaluations

    @property
    def nb_evaluations_in_iteration(self) -> int:
        return self._nb_evaluations_in_iteration

    @property
    def nb_updated(self) -> int:
        return self._nb_updated

    @property
    def nb_updated_in_iteration(self) -> int:
        return self._nb_updated_in_iteration

    @property
    def batch_size(self) -> int:
        return self._batch_size if self._batch_size is not None else self.budget

    @property
    def nb_max_iter(self) -> Union[int,float]:
        return math.ceil(self.budget / self.batch_size) if not math.isinf(self.budget) else math.inf

    @property
    def current_iter(self) -> int:
        return math.floor(self._nb_evaluations / self.batch_size)


    def ask(self) -> IndividualLike:
        """Ask for a suggestion of an individual to evaluate.

        Returns
        -------
        out: IndividualLike
            The suggested individual.
        """
        # Create base individual
        if self._base_ind_gen is None:
            base_ind: IndividualLike = Individual()
        else:
            base_ind = next(self._base_ind_gen)
        # Call _internal_ask
        suggestion = self._internal_ask(base_ind)
        # Update individual name and fitness weights
        if suggestion.name == "":
            suggestion.name = f"{int(round(time.time() * 1000))}-{self._nb_suggestions}"
        suggestion.fitness.weights = self._fitness_weights
        self._nb_suggestions += 1
        self._nb_suggestions_in_iteration += 1
        # Call callback functions
        for fn in self._callbacks.get("ask"):
            fn(self, suggestion)
        return suggestion


    @abc.abstractmethod
    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        """Internal ask method. To be implemented by subclasses.

        Parameters
        ----------
        :param base_ind: IndividualLike
            The base (empty) individual object. The `_internal_ask` method must change
            this object, by filling out the genome and providing other information.
        
        Returns
        -------
        out: IndividualLike
            The suggested individual.
        """
        raise NotImplementedError("Method `_internal_ask` must be implemented by QDAlgorithm subclasses.")


    def tell(self, individual: IndividualLike, fitness: Optional[Any] = None,
            features: Optional[FeaturesLike] = None, elapsed: Optional[float] = None,
            tell_container = True) -> bool:
        """Give the algorithm an evaluated `individual` (with a valid fitness function)
        and store it into the container.

        Parameters
        ----------
        :param individual: IndividualLike
            The `individual` provided to the algorithm.
        :param fitness: Optional[FitnessLike]
            If provided, set `individual.fitness` to `fitness`.
        :param features: Optional[FeaturesLike]
            If provided, set `individual.features` to `features`.
        :param elapsed: Optional[float]
            If provided, set `individual.elapsed` to `elapsed`.
            It corresponds to the time (in seconds) elapsed during evaluation.
        """
        ind = copy.deepcopy(individual)
        if fitness is not None:
            if isinstance(fitness, FitnessLike):
                ind.fitness = fitness
            else:
                ind.fitness.values = fitness
        if len(ind.fitness.values) != self._nb_objectives:
            raise ValueError(f"`individual` must possess a `fitness` attribute of dimension {self._nb_objectives} (specified through parameter `nb_objectives` when creating a QDAlgorithm sub-class), instead of {len(individual.fitness.values)}.")
        if features is not None:
            if isinstance(features, FeaturesLike):
                ind.features = features
            else:
                ind.features.values = features
        if elapsed is not None:
            ind.elapsed = elapsed
        xattr: Mapping[str, Any]
        allowed_to_tell: bool
        xattr, allowed_to_tell = self._tell_before_container_update(ind)
        added: bool = self.container.update([ind]) == 1 if tell_container and allowed_to_tell else False
        self._tell_after_container_update(ind, added, xattr)
        return added


    def _tell_before_container_update(self, individual: IndividualLike) -> Tuple[MutableMapping[str, Any], bool]:
        """Called by the ``tell`` method before adding the individual to the container. Can be used to compile
        a dictionary of information about the individual or container before the update operation occurs.
        Return a dictionary of extended attributes and a boolean flag indicating whether the algorithm is allowed to tell the
        individual or not.

        Parameters
        ----------
        :param individual: IndividualLike
            The `individual` provided to the algorithm.
        """
        return {}, True


    def _tell_after_container_update(self, individual: IndividualLike, added: bool, xattr: Mapping[str, Any] = {}) -> None:
        if added:
            self._nb_updated += 1
            self._nb_updated_in_iteration += 1
        self._internal_tell(individual, added, xattr)
        self._nb_evaluations += 1
        self._nb_evaluations_in_iteration += 1
        # Call callback functions
        for fn in self._callbacks.get("tell"):
            fn(self, individual)


    def _internal_tell(self, individual: IndividualLike, added_to_container: bool, xattr: Mapping[str, Any] = {}) -> None:
        """Internal tell method. To be overridden by subclasses, or left empty.
        
        Parameters
        ----------
        :param individual: IndividualLike
            The `individual` provided to the algorithm.
        :param added_to_container: bool
            True if `individual` was successfully added to the container.
        """
        pass


    def tell_container_entries(self, cont: Optional[Container] = None) -> None:
        """Tell this algorithm all entries from a container. By default, use
        `self.container` as container of interest, but if `cont` is specified,
        it will it instead.

        Parameters
        ----------
        :param cont: Optional[Container] = None
            If None, this method will tell all entries from the local container `self.container`.
            If set, this method will tell all entries from container `cont`.
        """
        if cont is None:
            cont = self.container
        for ind in cont:
            self._internal_tell(ind, True)


    def best(self) -> IndividualLike:
        """Return the individual with the best quality.
        Raise a RuntimeError if there are no individual in the container."""
        if len(self.container) == 0:
            raise RuntimeError("`self.container` contains no individual !")
        return self._internal_best()


    def _internal_best(self) -> IndividualLike:
        """Return the individual with the best quality. To be overridden by subclasses.
        By default, returns the best individual of `self.container`."""
        return self.container.best # Assume `self.container` is not empty



    def _verify_if_finished_iteration(self, batch_start_time: float) -> bool:
        """TODO"""
        # Verify if we finished an iteration
        if self._nb_evaluations % self.batch_size != 0:
            return False
        batch_elapsed: float = timer() - batch_start_time
        # Call callback functions
        for fn in self._callbacks.get("iteration"):
            fn(self, batch_elapsed)
        self._nb_suggestions_in_iteration = 0
        self._nb_evaluations_in_iteration = 0
        self._nb_updated_in_iteration = 0
        return True


    def optimise(self, evaluate: Callable, budget: Optional[int] = None, 
            batch_mode: bool = True, executor: Optional[ExecutorLike] = None,
            send_several_suggestions_to_fn: bool = False) -> IndividualLike:
        """TODO"""
        optimisation_start_time: float = timer()
        # Init budget
        if budget is not None:
            self.budget = budget
        if self.budget is None:
            raise ValueError("`budget` must be provided.")

        _executor: ExecutorLike = executor if executor is not None else SequentialExecutor()

        # Call callback functions
        for fn in self._callbacks.get("started_optimisation"):
            fn(self)

        def optimisation_loop(budget_fn: Callable):
            budget = budget_fn()
            remaining_evals = budget
            batch_start_time: float = timer()
            futures: MutableMapping[int, FutureLike] = {}
            individuals: MutableMapping[int, Union[IndividualLike, Sequence[IndividualLike]]] = {}
            while remaining_evals > 0 or len(futures) > 0:
                # Update budget
                new_budget = budget_fn()
                if new_budget > budget:
                    remaining_evals += new_budget - budget
                elif new_budget < budget:
                    remaining_evals = max(0, remaining_evals - (budget - new_budget))
                budget = new_budget

                nb_suggestions = min(remaining_evals, self.batch_size - len(futures))

                if send_several_suggestions_to_fn:
                    # Launch evals on suggestions
                    eval_id: int = remaining_evals
                    inds: Sequence[IndividualLike] = [self.ask() for _ in range(nb_suggestions)]
                    individuals[eval_id] = inds
                    futures[eval_id] = _executor.submit(_severalEvalsWrapper, eval_id, evaluate, inds)
                    remaining_evals -= len(inds)

                    # Wait for next completed future
                    f = generic_as_completed(list(futures.values()))
                    ind_id, ind_elapsed, ind_res, ind_exc = f.result()
                    if ind_exc is None:
                        inds = individuals[ind_id] # type: ignore
                        for i in range(len(inds)):
                            self.tell(inds[i], fitness=ind_res[i][0], features=ind_res[i][1], elapsed=ind_elapsed)
                    else:
                        warnings.warn(f"Individual evaluation raised the following exception: {ind_exc} !")
                        self._nb_evaluations += len(ind_res)

                else:
                    # Launch evals on suggestions
                    for _ in range(nb_suggestions):
                        eval_id = remaining_evals
                        ind: IndividualLike = self.ask()
                        individuals[eval_id] = ind
                        futures[eval_id] = _executor.submit(_evalWrapper, eval_id, evaluate, ind)
                        remaining_evals -= 1

                    # Wait for next completed future
                    f = generic_as_completed(list(futures.values()))
                    #ind_id, ind_elapsed, ind_fitness, ind_features, ind_exc = f.result()
                    ind_id, ind_elapsed, ind_res, ind_exc = f.result()
                    if ind_exc is None:
                        if isinstance(ind_res, IndividualLike):
                            self.tell(ind_res, elapsed=ind_elapsed)
                        else:
                            ind_fitness, ind_features = ind_res
                            ind = individuals[ind_id] # type: ignore
                            self.tell(ind, fitness=ind_fitness, features=ind_features, elapsed=ind_elapsed)
                    else:
                        warnings.warn(f"Individual evaluation raised the following exception: {ind_exc} !")
                        self._nb_evaluations += 1

                # Clean up
                del futures[ind_id]
                del individuals[ind_id]
                # Verify if we finished an iteration
                if self._verify_if_finished_iteration(batch_start_time):
                    batch_start_time = timer()


        if batch_mode:
            budget = self.budget
            remaining_evals: int = budget
            new_budget: int
            while remaining_evals > 0:
                budget_iteration: int = min(remaining_evals, self.batch_size)
                optimisation_loop(lambda: budget_iteration)
                remaining_evals -= budget_iteration

                # Update budget
                new_budget = self.budget
                if new_budget > budget:
                    remaining_evals += new_budget - budget
                elif budget > new_budget:
                    remaining_evals = max(0, remaining_evals - (budget - new_budget))
                budget = new_budget

        else:
            def ret_budget():
                return self.budget
            optimisation_loop(ret_budget)

        # Call callback functions
        optimisation_elapsed: float = timer() - optimisation_start_time
        for fn in self._callbacks.get("finished_optimisation"):
            fn(self, optimisation_elapsed)

        return self.best()


    def add_callback(self, event: str, fn: Callable) -> None:
        """Attach a callable `fn` to the optimiser. `fn` will be called each time the event specified in `event` occurs.
        
        Parameters
        ----------
        :param event: str
            The event triggering the callback. Can be either 'ask', 'tell', 'iteration' (when an iteration finishes)
            'started_optimisation', or 'finished_optimisation'.
        :param fn: Callable
            The callback function called when the event specified in `event` occurs.
        """
        _lst_events: Sequence[str] = ['ask', 'tell', 'iteration', 'started_optimisation', 'finished_optimisation']
        if not event in _lst_events:
            raise ValueError(f"`event` can be either {','.join(_lst_events)}.")
        #self._callbacks.setdefault(event, []).append(fn)
        self._callbacks.add_callback(event, fn)



@registry.register
class Lambda(QDAlgorithm):
    """TODO"""
    _ask_fn: Union[Callable[[], IndividualLike], Callable[[IndividualLike], IndividualLike]]
    _tell_fn: Optional[Callable]

    def __getstate__(self):
        odict = super().__getstate__()
        del odict['_ask_fn']
        del odict['_tell_fn']
        return odict

    def __init__(self, container: Container, budget: int,
            ask: Union[Callable[[], IndividualLike], Callable[[IndividualLike], IndividualLike]],
            tell: Optional[Callable] = None, **kwargs):
        super().__init__(container, budget, **kwargs)
        self._ask_fn = ask
        self._tell_fn = tell

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        """Internal ask method. Call the function specified by the `ask` argument of the constructor.
        This function can either take no argument (and must return a Sequence corresponding to the genome)
        or take a base (empty) `qdpy.base.IndividualLike` as sole argument
        (and must return a `qdpy.base.IndividualLike` with a filled out genome)."""
        sig = signature(self._ask_fn)
        if len(sig.parameters) == 0:
            base_ind[:] = self._ask_fn() # type: ignore
            return base_ind
        else:
            return self._ask_fn(base_ind) # type: ignore

    def _internal_tell(self, individual: IndividualLike, added_to_container: bool, xattr: Mapping[str, Any] = {}) -> None:
        if self._tell_fn is not None:
            self._tell_fn(individual, added_to_container)




class AlgWithBatchSuggestions(QDAlgorithm):
    """An abstract QDAlgorithm class that generates all of its suggestions batch-by-batch instead of at each call of `self.ask`.
    The `self._generate_batch_suggestions` method is called at the beginning of each batch to compute all suggestions
    for the current batch.
    """

    _batch_inds: List[IndividualLike]

    def __init__(self, container: Container, budget: int, **kwargs):
        super().__init__(container, budget, **kwargs)
        self._batch_inds = []

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        if len(self._batch_inds) == 0:
            self._generate_batch_suggestions()
            #assert len(self._batch_inds) == self._batch_size, f"`self._generate_batch_suggestions` must populate `self._batch_inds` with `self.batch_size` IndividualLike objects."
        base_ind[:] = self._batch_inds.pop()
        return base_ind

    @abc.abstractmethod
    def _generate_batch_suggestions(self) -> None:
        """Compute all suggestions for the current batch and store them in `self._batch_inds`."""
        raise NotImplementedError("Method `_generate_batch_suggestions` must be implemented by AlgWithBatchSuggestions subclasses.")



#class EmitterManager(QDAlgorithmLike, Summarisable, Saveable, Copyable, CreatableFromConfig):
class AlgWrapper(QDAlgorithm):
    """Algorithm that contains and wraps other algorithms."""
    algorithms: Sequence[QDAlgorithm]
    current_idx: int = 0
    current: QDAlgorithm
    tell_container_when_switching: Union[bool, str]
    batch_mode: bool

    def __init__(self, algorithms: Any, container: Optional[Container] = None, budget: Optional[int] = None,
            tell_container_when_switching: Union[bool, str] = False,
            name: Optional[str] = None, batch_mode: bool = False, **kwargs):
        if len(algorithms) == 0:
            raise ValueError("`algorithms` must not be empty.")
#        if isinstance(algorithms, Mapping):
#            self.algorithms = []
#            for k,v in algorithms.items():
#                for i in range(v):
#                    self.algorithms.append(k)
#        else:
#            self.algorithms = list(algorithms)
        self.algorithms = list(algorithms)
        self.current_idx = 0
        self.current = self.algorithms[self.current_idx]
        self.batch_mode = batch_mode

        container = algorithms[0].container if container is None else container
        budget_: int = sum((algo.budget for algo in self.algorithms)) if budget is None else budget
        super().__init__(container, budget_, **kwargs)
        delattr(self, 'container')

        if isinstance(tell_container_when_switching, str):
            if tell_container_when_switching not in ["True", "False", "only_best"]:
                raise ValueError("`tell_container_when_switching` must be a bool or one of these strings: 'True', 'False', 'only_best'.")
        self.tell_container_when_switching = tell_container_when_switching
        self.name = name if name is not None else f"{self.__class__.__name__}-{id(self)}"
        self.switch_to(0)


    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        if "current" not in self.__dict__:
            raise AttributeError
        return getattr(self.current, attr)


#    def __get_saved_state__(self) -> Mapping[str, Any]:
#        """Return a dictionary containing the relevant information to save. Only include information from `self.algorithms`."""
#        res = copy.deepcopy(self.current.__get_saved_state__())
#        entries = {k:v for k,v in inspect.getmembers(self) if not k.startswith('_') and not inspect.ismethod(v)}
#        return {**entries, **res}


    @abc.abstractmethod
    def next(self) -> None:
        """Switch to the next algorithm in Sequence `self.algorithms`, if there is one."""
        raise NotImplementedError("Method `_internal_ask` must be implemented by AlgWrapper subclasses.")
#        if self.current_idx < len(self.algorithms) - 1:
#            self.switch_to(self.current_idx + 1)
#        else:
#            self.switch_to(0)

    def switch_to(self, idx: int):
        """Switch to another algorithm of index `idx` in `self.algorithms`."""
        assert idx < len(self.algorithms), f"`idx` must be < len(self.algorithms)."
        #print(f"SWITCH to {idx}")
        previous = self.current
        self.current_idx = idx
        self.current = self.algorithms[idx]
        if (isinstance(self.tell_container_when_switching, bool) and self.tell_container_when_switching) or \
                self.tell_container_when_switching == "True":
            self.current.tell_container_entries()
        elif self.tell_container_when_switching == "only_best":
            try:
                self.current.tell(previous.best()) # Exception when no individual is present
            except Exception:
                pass

    def best(self) -> IndividualLike:
        return self.current.best()

#    def ask(self) -> IndividualLike:
#        ind = self.current.ask()
#        self.next()
#        return ind

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        if self.batch_mode:
            batch_size = self.current.batch_size
            if np.isinf(batch_size):
                batch_size = self.batch_size
            if np.isinf(batch_size):
                batch_size = 1
            if self.current.nb_suggestions % self.current.batch_size == 0:
                self.next()
        else:
            self.next()
        ind = self.current.ask()
        return ind

    def tell(self, individual: IndividualLike, fitness: Optional[Any] = None,
            features: Optional[FeaturesLike] = None, elapsed: Optional[float] = None,
            tell_container = False) -> bool:
        added: bool = self.current.tell(individual, fitness, features, elapsed)
        self._tell_after_container_update(individual, added)
        return added

#    def optimise(self, evaluate: Callable, budget: Optional[int] = None, 
#            batch_mode: bool = True, executor: Optional[ExecutorLike] = None,
#            send_several_suggestions_to_fn: bool = False) -> IndividualLike:
#        for _ in range(self.current_idx, len(self.algorithms)):
#            try:
#                res = self.current.optimise(evaluate, budget, batch_mode, executor, send_several_suggestions_to_fn)
#            except Exception as e:
#                warnings.warn(f"Optimisation failed with algorithm '{self.current.name}': {e}")
#                traceback.print_exc()
#            self.next()
#        return res

#    def add_callback(self, event: str, fn: Callable) -> None:
#        for a in self.algorithms:
#            a.add_callback(event, fn)




@registry.register
class AlternatingAlgWrapper(AlgWrapper):
    def next(self) -> None:
        """Switch to the next algorithm in Sequence `self.algorithms`, if there is one."""
        if self.current_idx < len(self.algorithms) - 1:
            self.switch_to(self.current_idx + 1)
        else:
            self.switch_to(0)


@registry.register
class SqAlgWrapper(AlgWrapper):
    def next(self) -> None:
        """Switch to the next algorithm in Sequence `self.algorithms`, if there is one."""
        if self.current.nb_suggestions >= self.current.budget:
            if self.current_idx < len(self.algorithms) - 1:
                self.switch_to(self.current_idx + 1)




@registry.register
class Sq(QDAlgorithmLike, Summarisable, Saveable, Copyable, CreatableFromConfig):
    """Sequence of algorithms. Useful to easily optimise several algorithms sequentially.  """
    algorithms: Sequence[QDAlgorithm]
    current_idx: int = 0
    current: QDAlgorithm
    tell_container_when_switching: Union[bool, str]

    def __init__(self, algorithms: Any, tell_container_when_switching: Union[bool, str] = True,
            name: Optional[str] = None, **kwargs):
        if len(algorithms) == 0:
            raise ValueError("`algorithms` must not be empty.")
        self.algorithms = list(algorithms)
        self.current_idx = 0
        self.current = self.algorithms[self.current_idx]
        if isinstance(tell_container_when_switching, str):
            if tell_container_when_switching not in ["True", "False", "only_best"]:
                raise ValueError("`tell_container_when_switching` must be a bool or one of these strings: 'True', 'False', 'only_best'.")
        self.tell_container_when_switching = tell_container_when_switching
        self.name = name if name is not None else f"{self.__class__.__name__}-{id(self)}"
        #self.budget = sum((algo.budget for algo in self.algorithms))
        self.switch_to(0)


    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        if "current" not in self.__dict__:
            raise AttributeError
        return getattr(self.current, attr)


    def __get_saved_state__(self) -> Mapping[str, Any]:
        """Return a dictionary containing the relevant information to save. Only include information from `self.algorithms`."""
        res = copy.deepcopy(self.current.__get_saved_state__())
        entries = {k:v for k,v in inspect.getmembers(self) if not k.startswith('_') and not inspect.ismethod(v)}
        return {**entries, **res}


    def next(self) -> None:
        """Switch to the next algorithm in Sequence `self.algorithms`, if there is one."""
        if self.current_idx < len(self.algorithms) - 1:
            self.switch_to(self.current_idx + 1)

    def switch_to(self, idx: int):
        """Switch to another algorithm of index `idx` in `self.algorithms`."""
        assert idx < len(self.algorithms), f"`idx` must be < len(self.algorithms)."
        previous = self.current
        self.current_idx = idx
        self.current = self.algorithms[idx]
        if (isinstance(self.tell_container_when_switching, bool) and self.tell_container_when_switching) or \
                self.tell_container_when_switching == "True":
            self.current.tell_container_entries()
        elif self.tell_container_when_switching == "only_best":
            try:
                self.current.tell(previous.best()) # Exception when no individual is present
            except Exception:
                pass

    def best(self) -> IndividualLike:
        return self.current.best()

    def ask(self) -> IndividualLike:
        return self.current.ask()

    def tell(self, individual: IndividualLike, fitness: Optional[Any] = None,
            features: Optional[FeaturesLike] = None, elapsed: Optional[float] = None) -> bool:
        return self.current.tell(individual, fitness, features, elapsed)

    def optimise(self, evaluate: Callable, budget: Optional[int] = None, 
            batch_mode: bool = True, executor: Optional[ExecutorLike] = None,
            send_several_suggestions_to_fn: bool = False) -> IndividualLike:
        for _ in range(self.current_idx, len(self.algorithms)):
            try:
                res = self.current.optimise(evaluate, budget, batch_mode, executor, send_several_suggestions_to_fn)
            except Exception as e:
                warnings.warn(f"Optimisation failed with algorithm '{self.current.name}': {e}")
                traceback.print_exc()
            self.next()
        return res

    def add_callback(self, event: str, fn: Callable) -> None:
        for a in self.algorithms:
            a.add_callback(event, fn)





@registry.register
class FromArchive(QDAlgorithm):
    """TODO"""
    reference_data_files: List[str]
    reference_inds_idx: int = 0

    def __init__(self, container: Container, reference_data_files: List[str], **kwargs):
        self.reference_data_files = [os.path.expanduser(x) for x in reference_data_files]
        self.reference_inds: List[Any] = []
        for reference_data_file in self.reference_data_files:
            with open(reference_data_file, "rb") as f:
                reference_data = pickle.load(f)
            self.reference_inds += list(reference_data['container'])
        budget = len(self.reference_inds)
        super().__init__(container, budget, **kwargs)

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        base_ind[:] = self.reference_inds[self.reference_inds_idx]
        self.reference_inds_idx += 1
        return base_ind





# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
