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


"""A simple example of NSLC to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools

import os
import numpy as np
import random
from functools import partial


# Iteration callback to modify the archive parameter each iteration
def iteration_callback(algo, batch_elapsed):
    if algo.container.threshold_novelty > 0.02:
        algo.container.threshold_novelty -= 0.003


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxTotalBins', type=int, default=1000, help="Maximum number of bins in the grid")
    parser.add_argument('--dimension', type=int, default=4, help="Problem dimension")
    parser.add_argument('--nbFeatures', type=int, default=2, help="Number of features")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, concurrent, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")
    args = parser.parse_args()

    if args.seed != None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

    # Algorithm parameters
    dimension = args.dimension                  # The dimension of the target problem (i.e. genomes size)
    assert(dimension >= 2)
    nb_features = args.nbFeatures               # The number of features to take into account in the container
    assert(nb_features >= 1)
    bins_per_dim = int(pow(args.maxTotalBins, 1./nb_features))
    nb_bins = (bins_per_dim,) * nb_features     # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $maxTotalBins^(1/nb_features)$ bins each
    ind_domain = (0., 1.)                       # The domain (min/max values) of the individual genomes
    features_domain = [(0., 1.),] * nb_features # The domain (min/max values) of the features
    fitness_domain = [(0., 1.),]                # The domain (min/max values) of the fitness
    #fitness_domain = [(-np.inf, np.inf),]      # The domain (min/max values) of the fitness
    init_batch_size = 400#0                     # The number of evaluations of the initial batch ('batch' = population)
    batch_size = 400#0                          # The number of evaluations in each subsequent batch
    nb_iterations = 10                          # The number of iterations (i.e. times where a new batch is evaluated)
    mutation_pb = 0.4                           # The probability of mutating each value of a genome
    eta = 20.0                                  # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    k = 15                                      # The number of nearest neighbours used to compute novelty
    threshold_novelty = 0.06                    # The threshold of novelty score used to assess whether an individual can be added to the archive
    max_items_per_bin = 1                       # The number of items in each bin of the grid
    verbose = True
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    # Create container
    container = NoveltyArchive(k=k, threshold_novelty=threshold_novelty, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list, depot_type=OrderedSet)

    # Define evaluation function
    eval_fn = partial(illumination_rastrigin_normalised, nb_features = nb_features)

    # Create algo
    algo = RandomSearchMutPolyBounded(container, budget=batch_size*nb_iterations, batch_size=batch_size,
            optimisation_task="minimisation", dimension=dimension,
            ind_domain=ind_domain, sel_pb=0.5, init_pb=0.5, mut_pb=mutation_pb, eta=eta, name="algo")

    algo.add_callback("iteration", iteration_callback)
    logger = TQDMAlgorithmLogger(algo, log_base_path = log_base_path)

    # Run illumination process !
    with ParallelismManager(args.parallelismType) as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False)


    # Print results info
    print("\n------------------------\n")
    print(algo.summary())
    #print(f"Total elapsed: {algo.total_elapsed}\n")
    #print(container.summary())
    ##print("Best ever fitness: ", container.best_fitness)
    ##print("Best ever ind: ", container.best)
    ##print("Performances container: ", container.fitness)
    ##print("Features container: ", container.features)

    novelty_best, local_competition_best = container.novelty_local_competition(container.best, k=15, ignore_first=True)
    print(f"\nNovelty best: {novelty_best}  local competition best: {local_competition_best}")

    novelty_first, local_competition_first = container.novelty_local_competition(container[0], k=15, ignore_first=True)
    print(f"Novelty first: {novelty_first}  local competition first: {local_competition_first}")

    # Transform the container into a grid, if needed
    if isinstance(container, containers.Grid):
        grid = container
    else:
        print("\n{:70s}".format("Transforming the container into a grid, for visualisation..."), end="", flush=True)
        grid = Grid(container.depot, shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list)
        print("\tDone !")
    print(grid.summary())

    # Create plot of the performance grid
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[... ,0], plot_path, plt.get_cmap("nipy_spectral_r"), grid.features_domain, grid.fitness_domain[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))

    plot_path = os.path.join(log_base_path, "activityGrid.pdf")
    max_activity = np.max(grid.activity_per_bin)
    plotGridSubplots(grid.activity_per_bin, plot_path, plt.get_cmap("Reds", max_activity), grid.features_domain, [0, max_activity], nbTicks=None)
    print("\nA plot of the activity grid was saved in '%s'." % os.path.abspath(plot_path))

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
