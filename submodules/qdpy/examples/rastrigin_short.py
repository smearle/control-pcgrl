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


"""A simple example to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3."""

from qdpy import algorithms, containers, benchmarks, plots

# Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evo.
grid = containers.Grid(shape=(64,64), max_items_per_bin=1, fitness_domain=((0., 1.),), features_domain=((0., 1.), (0., 1.)))
algo = algorithms.RandomSearchMutPolyBounded(grid, budget=60000, batch_size=500,
        dimension=3, optimisation_task="minimisation")

# Create a logger to pretty-print everything and generate output data files
logger = algorithms.TQDMAlgorithmLogger(algo)

# Define evaluation function
eval_fn = algorithms.partial(benchmarks.illumination_rastrigin_normalised, nb_features = len(grid.shape))

# Run illumination process !
best = algo.optimise(eval_fn)

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
