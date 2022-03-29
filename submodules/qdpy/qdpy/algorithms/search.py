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

"""TODO""" # TODO

__all__ = ["RandomUniform", "Sobol"]

from typing import Optional, Tuple, List, Iterable, Iterator, Any, TypeVar, Generic, Union, Sequence, MutableSet, MutableSequence, Type, Callable, Generator, Mapping, MutableMapping, overload
import numpy as np
import random


from .base import *
from qdpy.utils import *
from qdpy.phenotype import *
from qdpy.base import *
from qdpy.containers import *
from qdpy import tools

import qdpy.hdsobol as hdsobol


@registry.register
class RandomUniform(QDAlgorithm):
    """TODO"""
    ind_domain: DomainLike
    dimension: int

    def __init__(self, container: Container, budget: int, ind_domain: DomainLike, **kwargs):
        super().__init__(container, budget, **kwargs)
        self.ind_domain = ind_domain
        if self.dimension is None:
            raise ValueError("`dimension` must be provided.")

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        #base_ind[:] = np.random.uniform(self.ind_domain[0], self.ind_domain[1], size=self.dimension)
        base_ind[:] = [random.uniform(self.ind_domain[0], self.ind_domain[1]) for _ in range(self.dimension)]
        return base_ind



@registry.register
class Sobol(RandomUniform):
    """TODO"""

    def __init__(self, container: Container, budget: int, ind_domain: DomainLike, **kwargs):
        super().__init__(container, budget, ind_domain=ind_domain, **kwargs)
        self.sobol_vect = np.array(hdsobol.gen_sobol_vectors(self.budget + 1, self.dimension), dtype=float)
        self.sobol_vect = self.sobol_vect * (ind_domain[1] - ind_domain[0]) + ind_domain[0]

    def _internal_ask(self, base_ind: IndividualLike) -> IndividualLike:
        base_ind[:] = self.sobol_vect[self._nb_suggestions]
        return base_ind



# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
