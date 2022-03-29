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
from deap import algorithms
from deap import gp
import operator

import os
import numpy as np
import random
import warnings
import scipy


# Create fitness classes (must NOT be initialised in __main__ if you want to use scoop)
fitness_weight = -1.0
creator.create("FitnessMin", base.Fitness, weights=(fitness_weight,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, features=list)

def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    #print(individual)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Compute tested function
            func_vals = np.array([func(x) for x in points])
            # Evaluate the mean squared error between the expression
            # and the target function
            sqerrors = (func_vals - ref_vals) ** 2.
            fitness = [np.real(np.mean(sqerrors))]

            # Compute slopes
            #dfunc = np.diff(func_vals)
            #slopes_func = dfunc / dpoints
            #slopes_sqerrors = (slopes_func - slopes_ref) ** 2.
            #slopes_score = np.mean(slopes_sqerrors)

            # Compute derivatives by using scipy B-splines
            spline_func = scipy.interpolate.splrep(points, func_vals, k=3)
            spline_dfunc = scipy.interpolate.splev(points, spline_func, der=1)
            spline_dfunc2 = scipy.interpolate.splev(points, spline_func, der=2)
            spline_score = np.real(np.mean((spline_dfunc - spline_dref) ** 2.))
            spline_score2 = np.real(np.mean((spline_dfunc2 - spline_dref2) ** 2.))

        except Exception:
            fitness = [100.]
            #slopes_score = 100.
            spline_score = 100.
            spline_score2 = 100.

    length = len(individual)
    height = individual.height
    #features = [len(individual), slopes_score]
    features = [length, height, spline_score, spline_score2]
    #print(fitness, features)
    return [fitness, features]



# Compute reference function and stats
points = np.array(np.linspace(-1., 1., 1000), dtype=float)
dpoints = np.diff(points)
ref_vals = np.array([(x**4 + x**3 + x**2 + x) for x in points])
dref = np.diff(ref_vals)
slopes_ref = dref / dpoints
spline_ref = scipy.interpolate.splrep(points, ref_vals, k=3)
spline_dref = scipy.interpolate.splev(points, spline_ref, der=1)
spline_dref2 = scipy.interpolate.splev(points, spline_ref, der=2)


# Create primitives
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(operator.pow, 2)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)
#pset.addEphemeralConstant("rand101", lambda: random.uniform(-1.,1.))
pset.addEphemeralConstant("rand101", lambda: random.randint(-4.,4.))
pset.renameArguments(ARG0='x')


# Create Toolbox
max_size = 25
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, evo.models.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
#toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("evaluate", evalSymbReg, points=points)
#toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("select", tools.selRandom)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_size))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_size))
#toolbox.register("mutate", tools.mutPolynomialBounded, low=ind_domain[0], up=ind_domain[1], eta=eta, indpb=mutation_pb)
#toolbox.register("select", tools.selRandom) # MAP-Elites = random selection on a grid container



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-p', '--parallelismType', type=str, default='multiprocessing', help = "Type of parallelism to use (none, multiprocessing, concurrent, multithreading, scoop)")
    parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")
    args = parser.parse_args()

    if args.seed != None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

#    # Algorithm parameters
#    dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
    nb_features = 2                            # The number of features to take into account in the container
    nb_bins = [max_size // 5, 5, 10, 10]
#    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
    features_domain = [(1, max_size), (1, 5), (0., 20.), (0., 100.)]      # The domain (min/max values) of the features
    fitness_domain = [(0., np.inf)]               # The domain (min/max values) of the fitness
    init_batch_size = 3000                     # The number of evaluations of the initial batch ('batch' = population)
    batch_size = 400                           # The number of evaluations in each subsequent batch
    nb_iterations = 10                         # The number of iterations (i.e. times where a new batch is evaluated)
    cxpb = 0.5
    mutation_pb = 1.0                          # The probability of mutating each value of a genome
    max_items_per_bin = 1                      # The number of items in each bin of the grid
    verbose = True                             
    show_warnings = True                      # Display warning and error messages. Set to True if you want to check if some individuals were out-of-bounds
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)



    # Create a dict storing all relevant infos
    results_infos = {}
#    results_infos['dimension'] = dimension
#    results_infos['ind_domain'] = ind_domain
    results_infos['features_domain'] = features_domain
    results_infos['fitness_domain'] = fitness_domain
    results_infos['nb_bins'] = nb_bins
    results_infos['init_batch_size'] = init_batch_size
    results_infos['nb_iterations'] = nb_iterations
    results_infos['batch_size'] = batch_size
#    results_infos['mutation_pb'] = mutation_pb
#    results_infos['eta'] = eta

    # Create container
    grid = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list)

    with ParallelismManager(args.parallelismType, toolbox=toolbox) as pMgr:
        # Create a QD algorithm
        algo = DEAPQDAlgorithm(pMgr.toolbox, grid, init_batch_size = init_batch_size, batch_size = batch_size, niter = nb_iterations,
                cxpb = cxpb, mutpb = mutation_pb,
                verbose = verbose, show_warnings = show_warnings, results_infos = results_infos, log_base_path = log_base_path)
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


    # Search for the smallest best in the grid:
    smallest_best = grid.best
    smallest_best_fitness = grid.best_fitness
    smallest_best_length = grid.best_features[0]
    interval_match = 1e-10
    for ind in grid:
        if abs(ind.fitness.values[0] - smallest_best_fitness.values[0]) < interval_match:
            if ind.features[0] < smallest_best_length:
                smallest_best_length = ind.features[0]
                smallest_best = ind
    print("Smallest best:", smallest_best)
    print("Smallest best fitness:", smallest_best.fitness)
    print("Smallest best features:", smallest_best.features)

    # It is possible to access the results (including the genomes of the solutions, their performance, etc) stored in the pickle file by using the following code:
    #----8<----8<----8<----8<----8<----8<
    #from deap import base, creator, gp
    #import pickle
    #fitness_weight = -1.0
    #creator.create("FitnessMin", base.Fitness, weights=(fitness_weight,))
    #creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, features=list)
    #pset = gp.PrimitiveSet("MAIN", 1)
    #pset.addEphemeralConstant("rand101", lambda: random.randint(-4.,4.))
    #with open("final.p", "rb") as f:
    #    data = pickle.load(f)
    #print(data)
    #----8<----8<----8<----8<----8<----8<
    # --> data is a dictionary containing the results.

    # Create plots
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[... ,0], plot_path, plt.get_cmap("nipy_spectral"), grid.features_domain, grid.fitness_extrema[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))

    plot_path = os.path.join(log_base_path, "activityGrid.pdf")
    plotGridSubplots(grid.activity_per_bin, plot_path, plt.get_cmap("nipy_spectral"), grid.features_domain, [0, np.max(grid.activity_per_bin)], nbTicks=None)
    print("\nA plot of the activity grid was saved in '%s'." % os.path.abspath(plot_path))

    print("All results are available in the '%s' pickle file." % algo.final_filename)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
