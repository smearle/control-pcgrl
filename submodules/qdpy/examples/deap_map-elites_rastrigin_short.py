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


"""A simple example of MAP-elites to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3. This code uses the library DEAP to implement the evolutionary part."""

import matplotlib as mpl

import evo.models

mpl.use('Agg')
import matplotlib.pyplot as plt

from qdpy.algorithms.deap import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *
from qdpy.base import *

from deap import base
from deap import creator
from deap import tools

import os
import numpy as np
import random


# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(fitness_weight,))
creator.create("Individual", list, fitness=creator.FitnessMin, features=list)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxTotalBins', type=int, default=1000, help="Maximum number of bins in the grid")
    parser.add_argument('--dimension', type=int, default=4, help="Problem dimension")
    parser.add_argument('--nbFeatures', type=int, default=2, help="Number of features")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")
    args = parser.parse_args()

    if args.seed != None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

    # Algorithm parameters
    dimension = args.dimension                 # The dimension of the target problem (i.e. genomes size)
    assert(dimension >= 2)
    nb_features = args.nbFeatures              # The number of features to take into account in the container
    assert(nb_features >= 1)
    bins_per_dim = int(pow(args.maxTotalBins, 1./nb_features))
    nb_bins = (bins_per_dim,) * nb_features    # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $maxTotalBins^(1/nb_features)$ bins each
    ind_domain = (0., 1.)                      # The domain (min/max values) of the individual genomes
    features_domain = [(0., 1.)] * nb_features # The domain (min/max values) of the features
    fitness_domain = [(0., 1.)]                # The domain (min/max values) of the fitness
    init_batch_size = 12000                    # The number of evaluations of the initial batch ('batch' = population)
    batch_size = 4000                          # The number of evaluations in each subsequent batch
    nb_iterations = 20                         # The number of iterations (i.e. times where a new batch is evaluated)
    mutation_pb = 0.4                          # The probability of mutating each value of a genome
    eta = 20.0                                 # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    max_items_per_bin = 1                      # The number of items in each bin of the grid
    verbose = True
    show_warnings = False                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    # Create Toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, ind_domain[0], ind_domain[1])
    toolbox.register("individual", tools.initRepeat, evo.models.Individual, toolbox.attr_float, dimension)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", illumination_rastrigin_normalised, nb_features = nb_features)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta, indpb=mutation_pb)
    toolbox.register("select", tools.selRandom) # MAP-Elites = random selection on a grid container
    #toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm

    # Create a dict storing all relevant infos
    results_infos = {}
    results_infos['dimension'] = dimension
    results_infos['ind_domain'] = ind_domain
    results_infos['features_domain'] = features_domain
    results_infos['fitness_domain'] = fitness_domain
    results_infos['nb_bins'] = nb_bins
    results_infos['init_batch_size'] = init_batch_size
    results_infos['nb_iterations'] = nb_iterations
    results_infos['batch_size'] = batch_size
    results_infos['mutation_pb'] = mutation_pb
    results_infos['eta'] = eta

    # Create container
    grid = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, fitness_weight=fitness_weight, features_domain=features_domain, storage_type=list)

    with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_size = init_batch_size,
                batch_size = batch_size, niter = nb_iterations,
                verbose = verbose, show_warnings = show_warnings,
                results_infos = results_infos, log_base_path = log_base_path)
        # Run the illumination process !
        algo.run()

    # Print results info
    print(f"Total elapsed: {algo.total_elapsed}\n")
    print(grid.summary())
    #print("Best ever fitness: ", container.best_fitness)
    #print("Best ever ind: ", container.best)
    #print("%s filled bins in the grid" % (grid.size_str()))
    ##print("Solutions found for bins: ", grid.solutions)
    #print("Performances grid: ", grid.fitness)
    #print("Features grid: ", grid.features)


    # It is possible to access the results (including the genomes of the solutions, their performance, etc) stored in the pickle file by using the following code:
    #----8<----8<----8<----8<----8<----8<
    #from deap import base, creator
    #import pickle
    #creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    #creator.create("Individual", list, fitness=creator.FitnessMin, features=list)
    #with open("final.p", "rb") as f:
    #    data = pickle.load(f)
    #print(data)
    #----8<----8<----8<----8<----8<----8<
    # --> data is a dictionary containing the results.

    # Create plot of the performance grid
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[..., 0], plot_path, plt.get_cmap("nipy_spectral_r"), features_domain, fitness_domain[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    print("All results are available in the '%s' pickle file." % algo.final_filename)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
