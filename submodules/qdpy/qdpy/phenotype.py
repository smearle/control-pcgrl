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

"""Some base classes, stubs and types."""
#from __future__ import annotations

#__all__ = ["jit"]


########### IMPORTS ########### {{{1
from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, MutableMapping, overload
from typing_extensions import runtime, Protocol
from operator import mul, truediv
import math
import sys
import importlib
import pickle
import textwrap
import copy
from inspect import signature
from functools import partial
import warnings
import numpy as np

import os
import psutil
import signal

from qdpy.utils import *
from qdpy.base import *


########### INTERFACES AND STUBS ########### {{{1
FitnessValuesLike = Union[Sequence[T], np.ndarray]
#FeaturesLike = Union[Sequence[Any], np.ndarray]
FeaturesValuesLike = Union[Sequence[Any], np.ndarray]

@runtime
class FitnessLike(Protocol):
    """Fitness protocol inspired from (and compatible with) DEAP Fitness class."""
    weights: FitnessValuesLike
    def dominates(self, other: Any, obj: Any = slice(None)) -> bool: ...
    def getValues(self) -> FitnessValuesLike: ...
    def setValues(self, values: FitnessValuesLike) -> None: ...
    def delValues(self) -> None: ...
    @property
    def values(self) -> FitnessValuesLike: ...
    @values.setter
    def values(self, values: FitnessValuesLike) -> None: ...
    @values.deleter
    def values(self) -> None: ...
    @property
    def valid(self) -> bool: ...
    def reset(self) -> None: ...
#    @overload
#    def __getitem__(self, i: int) -> Any: ...
#    @overload
#    def __getitem__(self, s: slice) -> Sequence[Any]: ...
#    def __len__(self) -> int: ...


@runtime
class FeaturesLike(Protocol):
    """Features protocol similar to the ``FitnessLike`` protocol."""
    def getValues(self) -> FeaturesValuesLike: ...
    def setValues(self, values: FeaturesValuesLike) -> None: ...
    def delValues(self) -> None: ...
    @property
    def values(self) -> FeaturesValuesLike: ...
    @values.setter
    def values(self, values: FeaturesValuesLike) -> None: ...
    @values.deleter
    def values(self) -> None: ...
    @property
    def valid(self) -> bool: ...
    def reset(self) -> None: ...
    def __getitem__(self, key) -> Any: ...
    def __len__(self) -> int: ...



@runtime
class IndividualLike(Protocol):
    name: str
    fitness: FitnessLike
    features: FeaturesLike
    elapsed: float
    def dominates(self, other: Any) -> bool: ...
    def reset(self) -> None: ...
    def __setitem__(self, key, values) -> None: ...



#FitnessLike = Sequence
FitnessGetter = Callable[[T], FitnessLike]
FeaturesGetter = Callable[[T], FeaturesLike]
GridIndexLike = ShapeLike



########### BASE OPTIMISATION CLASSES ########### {{{1

class Fitness(FitnessLike, Sequence[Any]):
    """Fitness implementation inspired from DEAP Fitness class. It can be used without problem with most (propably all) DEAP methods."""

    weights: FitnessValuesLike = ()
    wvalues: FitnessValuesLike = ()

    def __new__(cls, values: FitnessValuesLike=(), weights: Optional[FitnessValuesLike]=None):
        return super(Fitness, cls).__new__(cls)

    def __init__(self, values: FitnessValuesLike=(), weights: Optional[FitnessValuesLike]=None) -> None:
        if weights is None:
            self.weights = tuple([-1.0 for _ in range(len(values))]) # Defaults to minimisation
        else:
            self.weights = weights
        if len(self.weights) != len(values):
            raise ValueError("``values`` and ``weights`` must have the same length.")
        self.values = values

    @property
    def values(self) -> FitnessValuesLike:
        return tuple(map(truediv, self.wvalues, self.weights))

    @values.setter
    def values(self, values: FitnessValuesLike) -> None:
        try:
            self.wvalues = tuple(map(mul, values, self.weights))
        except TypeError:
            raise ValueError("Invalid ``value`` parameter.")

    @values.deleter
    def values(self) -> None:
        self.wvalues = ()

    def getValues(self) -> FitnessValuesLike:
        return self.values

    def setValues(self, values: FitnessValuesLike) -> None:
        self.values = values

    def delValues(self) -> None:
        del self.values

    # FROM DEAP
    def dominates(self, other: Any, obj: Any = slice(None)) -> bool:
        """Return true if each objective of ``self`` is not strictly worse than
        the corresponding objective of ``other`` and at least one objective is
        strictly better.
        """
#    :param obj: Slice indicating on which objectives the domination is
#                tested. The default value is `slice(None)`, representing
#                every objectives.  """
        not_equal: bool = False
        for self_wvalue, other_wvalue in zip(self.wvalues[obj], other.wvalues[obj]):
            if self_wvalue > other_wvalue:
                not_equal = True
            elif self_wvalue < other_wvalue:
                return False
        return not_equal

    def reset(self) -> None:
        self.wvalues = (np.nan,) * len(self.wvalues)

    @property
    def valid(self) -> bool:
        """Assess if a fitness is valid or not."""
        return len(self.wvalues) != 0

    def __len__(self) -> int:
        return len(self.wvalues)

#    @overload
#    def __getitem__(self, index: int) -> T: ...
#
#    @overload
#    def __getitem__(self, s: slice) -> Sequence: ...

    def __getitem__(self, key):
        return self.values[key]

    def __contains__(self, key: Any) -> bool:
        return key in self.values

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __reversed__(self) -> Iterator:
        return reversed(self.values)


    def __hash__(self) -> int:
        return hash(self.wvalues)

    def __gt__(self, other: Any) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Any) -> bool:
        return not self.__lt__(other)

    def __le__(self, other: Any) -> bool:
        return self.wvalues <= other.wvalues

    def __lt__(self, other: Any) -> bool:
        return self.wvalues < other.wvalues

    def __eq__(self, other: Any) -> bool:
        return self.wvalues == other.wvalues

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return str(self.values if self.valid else tuple())

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))


class Features(FeaturesLike, Sequence[Any]):
    _values: FeaturesValuesLike

    def __init__(self, values: FeaturesValuesLike=(), *args, **kwargs) -> None:
        self._values = values

    @property
    def values(self) -> FeaturesValuesLike:
        return self._values

    @values.setter
    def values(self, values: FeaturesValuesLike) -> None:
        self._values = values

    @values.deleter
    def values(self) -> None:
        self._values = ()

    def getValues(self) -> FeaturesValuesLike:
        return self.values

    def setValues(self, values: FeaturesValuesLike) -> None:
        self.values = values

    def delValues(self) -> None:
        del self.values

    def reset(self) -> None:
        self._values = (np.nan,) * len(self._values)

    def __len__(self) -> int:
        return len(self.values)

#    @overload
#    def __getitem__(self, index: int) -> T: ...
#
#    @overload
#    def __getitem__(self, s: slice) -> Sequence: ...

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, idx, value):
        self._values[idx] = value
        #if idx == slice(None, None, None):
        #    self.add_sample(value)
        #else:
        #    self._values[idx] = value

    def __contains__(self, key: Any) -> bool:
        return key in self.values

    def __iter__(self) -> Iterator:
        return iter(self.values)

    def __reversed__(self) -> Iterator:
        return reversed(self.values)


    def __hash__(self) -> int:
        return hash(self.values)

    def __gt__(self, other: Any) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Any) -> bool:
        return not self.__lt__(other)

    def __le__(self, other: Any) -> bool:
        return self.values <= other.values

    def __lt__(self, other: Any) -> bool:
        return self.values < other.values

    def __eq__(self, other: Any) -> bool:
        return self.values == other.values

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return str(self.values)

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))


class Individual(list, IndividualLike):
    """Qdpy Individual class. Note that containers and algorithms all use internally either the QDPYIndividualLike Protocol or the IndividualWrapper class, so you can easily declare an alternative class to Individual. TODO""" # TODO

    name: str
    fitness: FitnessLike
    features: FeaturesLike
    elapsed: float = math.nan

    def __init__(self, iterable: Optional[Iterable] = None,
            name: Optional[str] = None,
            fitness: Optional[FitnessLike] = None, features: Optional[FeaturesLike] = None) -> None:
        if iterable is not None:
            self.extend(iterable)
        self.name = name if name else ""
        self.fitness = fitness if fitness is not None else Fitness()
        self.features = features if features is not None else Features([])

    def __repr__(self) -> str:
        if not self:
            return "%s()" % (self.__class__.__name__,)
        return "%s(%r)" % (self.__class__.__name__, list(self))

    def dominates(self, other: Any) -> bool:
        """Return true if ``self`` dominates ``other``. """
        return self.fitness.dominates(other.fitness)

    def reset(self) -> None:
        self.fitness.reset()
        self.features.reset()
        self.elapsed = math.nan

    # TODO : improve performance ! (quick and dirty solution !)
    def __hash__(self):
        return hash(tuple(self))

    def __eq__(self, other):
        return (self.__class__ == other.__class__ and tuple(self) == tuple(other))


@registry.register # type: ignore
class GenIndividuals(CreatableFromConfig):
    def __init__(self, *args, **kwargs):
        pass
    def __iter__(self):
        return self
    def __next__(self):
        return Individual()
    def __call__(self):
        while(True):
            yield self.__next__()



# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
