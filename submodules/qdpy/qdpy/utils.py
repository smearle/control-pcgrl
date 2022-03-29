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

"""The :mod:`~qdpy.utils` module is a collection of small functions and classes handling common patterns."""

#__all__ = ["jit"]

from collections.abc import Iterable
from typing import Optional, Tuple, TypeVar, Union, Any, MutableSet, Mapping, MutableMapping, Sequence, MutableSequence, Callable, Tuple
from typing_extensions import runtime, Protocol
import inspect
import numpy as np


########### UTILS ########### {{{1

def is_iterable(obj: Any) -> bool:
    """Return if ``obj`` is iterable or not."""
    return isinstance(obj, Iterable)

#def in_bounds(val: Union[float, Sequence[float]], domain: Union[DomainLike, Sequence[DomainLike]]) -> bool:
def in_bounds(val: Any, domain: Any) -> bool:
    """Return if ``val`` (if a value) or all values in ``val`` (if an iterable) are in ``domain``."""
    #assert len(domain) >= 2, f"``domain`` must be a 2-tuple of numbers or a sequence of 2-tuples of numbers."
    if isinstance(val, Sequence) or isinstance(val, np.ndarray):
        if isinstance(domain[0], Sequence) or isinstance(domain[0], np.ndarray):
            if len(val) == len(domain):
                return all((v >= d[0] and v <= d[1] for v, d in zip(val, domain)))
            else:
                raise ValueError(f"if ``val`` is a Sequence, ``domain`` must have the same length as ``val``.")
        else:
            return all((v >= domain[0] and v <= domain[1] for v in val))
    else:
        if isinstance(domain[0], Sequence) or isinstance(domain[0], np.ndarray):
            raise ValueError(f"if ``val`` is not a Sequence, ``domain`` must be a 2-tuple of numbers.")
        else:
            return val >= domain[0] and val <= domain[1]

#def _hashify(item):
#    """Verify if *item* is hashable, if not, try and return it as a tuple."""
#    if isinstance(item, collections.abc.Hashable):
#        return item
#    else:
#        return tuple(item)

def tuplify(item: Union[Any, Sequence[Any]]) -> Tuple[Any, ...]:
    if isinstance(item, Sequence):
        return tuple(item)
    else:
        return (item,)


def argsort(a, **kwargs):
    return sorted(range(len(a)), key=a.__getitem__, **kwargs)



########### NUMBA ########### {{{1
def _dummyJit(*args, **kwargs):
    """
    Dummy version of jit decorator, does nothing
    """
    if len(args) == 1 and callable(args[0]):
        return args[0]
    else:
        def wrap(func):
            return func
        return wrap
try:
    import numba
    from numba import jit
except ImportError:
    jit = _dummyJit



# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
