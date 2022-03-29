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

from qdpy.phenotype import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.plots import *

import os
import multiprocessing
import numpy as np
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from timeit import default_timer as timer
import pickle



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
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, multiprocessing, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")
    args = parser.parse_args()

    if args.seed != None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

    # Algorithm parameters
    dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
    assert(dimension >= 2)
    nbFeatures = args.nbFeatures              # The number of features to take into account in the container
    assert(nbFeatures >= 1)
    binsPerDim = int(pow(args.maxTotalBins, 1./nbFeatures))
    nbBins = (binsPerDim,) * nbFeatures       # The number of bins of the grid of elites. Here, we consider only $nbFeatures$ features with $maxTotalBins^(1/nbFeatures)$ bins each
    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
    features_domain = [(0., 1.)] * nbFeatures # The domain (min/max values) of the features
    fitness_domain = ((0., 1.),)              # The domain (min/max values) of the fitness
    initBatchSize = 12000                     # The number of evaluations of the initial batch ('batch' = population)
    batchSize = 4000                          # The number of evaluations in each subsequent batch
    nbIterations = 20                         # The number of iterations (i.e. times where a new batch is evaluated)
    mutationPb = 0.4                          # The probability of mutating each value of a genome
    eta = 20.0                                # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    max_items_per_bin = 1                     # The number of items in each bin of the grid
    verbose = True                            # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
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
    toolbox.register("evaluate", illumination_rastrigin_normalised)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta, indpb=mutationPb)
    toolbox.register("select", tools.selRandom) # MAP-Elites = random selection on a grid container
    #toolbox.register("select", tools.selBest) # But you can also use all DEAP selection functions instead to create your own QD-algorithm
    cxpb = 0.0 # No crossover in this example

    # Init parallelism
    if args.parallelismType == "none":
        pass
    elif args.parallelismType == "multiprocessing":
        import multiprocessing
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    elif args.parallelismType == "scoop":
        import scoop
        toolbox.register("map", scoop.futures.map)
    else:
        raise Exception("Unknown parallelismType: '%s'" % args.parallelismType)

    # Create a dict storing all relevant infos
    results_dict = {}
    results_dict['dimension'] = dimension
    results_dict['nbBins'] = nbBins
    results_dict['ind_domain'] = ind_domain
    results_dict['features_domain'] = features_domain
    results_dict['fitness_domain'] = fitness_domain
    results_dict['initBatchSize'] = initBatchSize
    results_dict['nbIterations'] = nbIterations
    results_dict['batchSize'] = batchSize
    results_dict['mutationPb'] = mutationPb
    results_dict['eta'] = eta

    # Generate initial batch
    startTime = timer()
    init_batch = toolbox.population(n=initBatchSize)
    results_dict['init_batch'] = init_batch
    halloffame = tools.HallOfFame(1)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in init_batch if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0]
        ind.features = fit[1]
    halloffame.update(init_batch)

    # Store batch in container
    grid = Grid(shape=nbBins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list)
    nb_updated = grid.update(init_batch, issue_warning=verbose)

    # Init stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("containerSize", lambda c: grid.size_str())
    stats.register("nbUpdated", lambda _: nb_updated)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    stats.register("elapsed", lambda _: timer() - startTime)
    logbook = tools.Logbook()
    logbook.header = "iteration", "containerSize", "evals", "nbUpdated", "avg", "std", "min", "max", "elapsed"

    # Compile stats and update logs
    record = stats.compile(grid) if stats else {}
    logbook.record(iteration=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for iteration in range(1, nbIterations + 1):
        startTime = timer()
        # Select the next batch individuals
        batch = toolbox.select(grid, batchSize)

        # Vary the pool of individuals
        offspring = []
        for o in batch:
            newO = toolbox.clone(o)
            ind, = toolbox.mutate(newO)
            del ind.fitness.values
            offspring.append(ind)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0]
            ind.features = fit[1]

        # Replace the current population by the offspring
        nb_updated = grid.update(offspring, issue_warning=verbose)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(grid)

        # Append the current generation statistics to the logbook
        record = stats.compile(grid) if stats else {}
        logbook.record(iteration=iteration, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    # Stop parallelism
    if args.parallelismType == "none":
        pass
    elif args.parallelismType == "multiprocessing":
        pool.close()
    elif args.parallelismType == "scoop":
        pass

    # Save results
    results_dict['solutions'] = grid.solutions
    results_dict['fitness'] = grid.fitness
    results_dict['features'] = grid.features
    results_dict['container'] = grid
    outputFile = os.path.abspath(os.path.join(log_base_path, "final.p"))
    with open(outputFile, "wb") as f:
        pickle.dump(results_dict, f)

    # Print results info
    print(grid.summary())
    print("Best ever fitness: ", halloffame[0].fitness.values)
    print("Best ever ind: ", halloffame[0])
    print("%s filled bins in the grid" % (grid.size_str()))
    #print("Solutions found for bins: ", grid.solutions)
    #print("Performances grid: ", grid.fitness)
    #print("Features grid: ", grid.features)
    print("All results are available in the '%s' pickle file." % outputFile)


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
    plotPath = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[...,0], plotPath, plt.get_cmap("nipy_spectral_r"), features_domain, fitness_domain[0])
    print("A plot of the performance grid was saved in '%s'." % os.path.abspath(plotPath))



# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
