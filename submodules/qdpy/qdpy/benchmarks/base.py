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


__all__ = ["Benchmark"]

from qdpy.base import *


class Benchmark(object):
    fn: Callable
    ind_domain: DomainLike
    fitness_domain: Sequence[DomainLike]
    features_domain: Sequence[DomainLike]
    default_task: str
    def __init__(self, fn: Callable, ind_domain: DomainLike, fitness_domain: Sequence[DomainLike], features_domain: Sequence[DomainLike], default_task: str = "minimisation"):
        self.fn = fn # type: ignore
        self.ind_domain = ind_domain
        self.fitness_domain = fitness_domain
        self.features_domain = features_domain
        self.default_task = default_task


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
