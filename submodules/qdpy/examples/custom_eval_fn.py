#!/usr/bin/env python3
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


"""A simple example to illuminate a given evaluation function, returning a one dimensional fitness score and two feature descriptors."""

from qdpy import algorithms, containers, plots
from qdpy.base import ParallelismManager
import math


def eval_fn(ind):
    """An example evaluation function. It takes an individual as input, and returns the pair ``(fitness, features)``, where ``fitness`` and ``features`` are sequences of scores."""
    normalization = sum((x for x in ind))
    k = 10.
    score = 1. - sum(( math.cos(k * ind[i]) * math.exp(-(ind[i]*ind[i])/2.) for i in range(len(ind)))) / float(len(ind))
    fit0 = sum((x * math.sin(abs(x) * 2. * math.pi) for x in ind)) / normalization
    fit1 = sum((x * math.cos(abs(x) * 2. * math.pi) for x in ind)) / normalization
    features = (fit0, fit1)
    return (score,), features


if __name__ == "__main__":
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evo.
    grid = containers.Grid(shape=(16,16), max_items_per_bin=1, fitness_domain=((-math.pi, math.pi),), features_domain=((0., 1.), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=3000, batch_size=500,
            dimension=3, optimisation_task="minimisation")

    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.TQDMAlgorithmLogger(algo)

    # Run illumination process !
    with ParallelismManager("none") as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch

    # Print results info
    print("\n" + algo.summary())

    # Plot the results
    plots.default_plots_grid(logger)

    print("\nAll results are available in the '%s' pickle file." % logger.final_filename)
    print(f"""
To open it, you can use the following python code:
    import pickle
    # You may want to import your own packages if the pickle file contains custom objects

    with open("{logger.final_filename}", "rb") as f:
        data = pickle.load(f)
    # ``data`` is now a dictionary containing all results, including the final container, all solutions, the algorithm parameters, etc.

    grid = data['container']
    print(grid.best)
    print(grid.best.fitness)
    print(grid.best.features)
    """)

# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
