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

"""Basic operators and tools used to design QD or evoluationary algorithms."""

import numpy as np
import random
from typing import Sequence, Callable, Tuple
from itertools import repeat
import copy

from qdpy.phenotype import *
from qdpy.base import *
from qdpy import containers


########### SELECTION ########### {{{1

def sel_random(collection: Sequence[Any]) -> Sequence[Any]:
    """Select and return one individual at random among `collection`.

    Parameters
    ----------
    :param collection: Sequence[Any]
        The individuals to select from.
    """
    return random.choice(collection)


def non_trivial_sel_random(container: containers.Container) -> Any:
    """Select and return one individual at random among `container`.
    Ignore individuals with a trivial fitness (i.e. equal to the minimum fitness value),
    except if there are only individuals with trivial fitness.

    Parameters
    ----------
    :param container: Container
        The individuals to select from.
    """
    fst_ind: Any = container[0]
    min_fitness: Any = copy.deepcopy(fst_ind.fitness)
    assert container.fitness_domain is not None and len(min_fitness.values) == len(container.fitness_domain), f"You must specify `fitness_domain` in Container, and use individuals with fitness values of the same length as `fitness_domain`."
    min_fitness.values = tuple([x[0] for x in container.fitness_domain])
    candidates: MutableSequence[Any] = [ind for ind in container if ind.fitness.dominates(min_fitness)]
    if len(candidates):
        return random.choice(candidates)
    else:
        return random.choice(container)


def sel_grid_roulette(collection: Sequence[Any]) -> Sequence[Any]:
    """Select and return one individual at random (using a roulette selection) from a random bin of a grid.

    Parameters
    ----------
    :param collection: Grid
        The grid containing individuals.
    """
    assert(isinstance(collection, containers.Grid))
    assert(len(collection))
    # Select randomly an occupied bin of the grid
    tmp_idx = random.randint(0, len(collection)-1)
    tmp_ind: IndividualLike = collection[tmp_idx]
    bin_coord = collection.index_grid(tmp_ind.features)
    bin_pop = collection.solutions[bin_coord]
    # Roulette selection of ind within this bin
    sum_fit_val = [sum(i.fitness.values) for i in bin_pop]
    sum_all_fit = sum(sum_fit_val)
    probs = [f / sum_all_fit for f in sum_fit_val]
    return random.choices(bin_pop, weights=probs)[0]


########### MUTATIONS ########### {{{1

def mut_gaussian(individual: MutableSequence[Any], mu: float, sigma: float, mut_pb: float) -> MutableSequence[Any]:
    """Return a gaussian mutation of mean `mu` and standard deviation `sigma`
    on selected items of `individual`. `mut_pb` is the probability for each
    item of `individual` to be mutated.
    Mutations are applied directly on `individual`, which is then returned.

    Parameters
    ----------
    :param individual
        The individual to mutate.
    :param mu: float
        The mean of the gaussian mutation.
    :param sigma: float
        The standard deviation of the gaussian mutation.
    :param mut_pb: float
        The probability for each item of `individual` to be mutated.
    """
    for i in range(len(individual)):
        if random.random() < mut_pb:
            individual[i] += random.gauss(mu, sigma)
    return individual



def mut_polynomial_bounded(individual: MutableSequence[Any], eta: float, low: float, up: float, mut_pb: float) -> MutableSequence[Any]:
    """Return a polynomial bounded mutation, as defined in the original NSGA-II paper by Deb et al.
    Mutations are applied directly on `individual`, which is then returned.
    Inspired from code from the DEAP library (https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py).

    Parameters
    ----------
    :param individual
        The individual to mutate.
    :param eta: float
        Crowding degree of the mutation.
        A high ETA will produce mutants close to its parent,
        a small ETA will produce offspring with more differences.
    :param low: float
        Lower bound of the search domain.
    :param up: float
        Upper bound of the search domain.
    :param mut_pb: float
        The probability for each item of `individual` to be mutated.
    """
    for i in range(len(individual)):
        if random.random() < mut_pb:
            x = individual[i]
            delta_1 = (x - low) / (up - low)
            delta_2 = (up - x) / (up - low)
            rand = random.random()
            mut_pow = 1. / (eta + 1.)

            if rand < 0.5:
                xy = 1. - delta_1
                val = 2. * rand + (1. - 2. * rand) * xy**(eta + 1.)
                delta_q = val**mut_pow - 1.
            else:
                xy = 1. - delta_2
                val = 2. * (1. - rand) + 2. * (rand - 0.5) * xy**(eta + 1.)
                delta_q = 1. - val**mut_pow

            x += delta_q * (up - low)
            x = min(max(x, low), up)
            individual[i] = x
    return individual




########### COMBINED OPERATORS ########### {{{1

def sel_or_init(collection: Sequence[IndividualLike], base_ind: IndividualLike,
        sel_fn: Callable, sel_pb: float,
        init_fn: Callable, init_pb: float = 0., return_flag: bool = True):
    """Either select an individual from `collection` by using function `sel_pb`,
    or initialise a new individual by using function `init_pb`.
    If `collection` is empty, it will always initialise a new individual, not perform selection.

    Parameters
    ----------
    :param collection: Sequence[IndividualLike]
        The individuals to select from.
    :param base_ind: IndividualLike
        The base individual to initialise.
    :param sel_fn: Callable
        The selection function.
    :param sel_pb: float
        The probability of performing selection.
    :param init_fn: Callable
        The initialisation function.
    :param init_pb: float
        The probability of performing initialisation.
    :param return_flag: bool
        If set to True, the function will return a Tuple[IndividualLike, bool] with a first item corresponding
        to the selected or initialised individual, and the second item a flag set to True if the first item
        was selected, and set to False if it was initialised.
        If set to False, the function will return the selected or initialised IndividualLike.
    """
    def ret(res, f):
        return (res, f) if return_flag else res
    if len(collection) == 0:
        return ret(init_fn(base_ind), False)
    operation = np.random.choice(range(2), p=[sel_pb, init_pb])
    if operation == 0: # Selection
        return ret(sel_fn(collection), True)
    else: # Initialisation
        return ret(init_fn(base_ind), False)


def mut_or_cx(individuals: Union[IndividualLike, Sequence[IndividualLike]],
        mut_fn: Callable, cx_fn: Callable) -> Sequence[IndividualLike]:
    """Either perform a mutation (using function `mut_fn`) or a crossover (using function `cx_fn`)
    depending on the nature and length of `individuals`.
    If `individuals` is an IndividualLike or a Sequence of one IndividualLike, a mutation will be performed.
    If `individuals` is a Sequence of two IndividualLike, a crossover will be performed.

    Parameters
    ----------
    :param individuals: Union[IndividualLike, Sequence[IndividualLike]]
        The individual(s) to mutate or crossover.
    :param mut_fn: Callable
        The mutation function.
    :param cx_fn: Callable
        The crossover function.

    Return
    ------
    The resulting individual(s).
    """
    if isinstance(individuals, IndividualLike):
        return [mut_fn(individuals)]
    elif isinstance(individuals, Sequence) and len(individuals) == 1:
        return [mut_fn(individuals[0])]
    elif isinstance(individuals, Sequence) and len(individuals) > 1:
        return cx_fn(individuals)
    else:
        raise ValueError(f'`individuals` can be an Individual or a Sequence.')


# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
