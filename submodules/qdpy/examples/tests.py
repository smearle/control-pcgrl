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



# XXX
import yaml
import threading
def _eval(ind):
    #print("starting: ", threading.get_ident(), ind[0])
    #print(type(ind[0]))
#    x = float(ind[0])
#    for i in range(1000000):
#        x*x
    #tmp = np.zeros(100000)
    #for i in range(len(tmp)):
    #    tmp[i] = i * 42.
    #return illumination_rastrigin_normalised(ind, nb_features=nb_features)
    #res = illumination_rastrigin_normalised(ind, nb_features=nb_features)
    #res = [[np.random.random()], list(np.random.random(2))]
    res = illumination_rastrigin_normalised(ind, nb_features=2)
    fitness, features = res
    fitness[0] = 0.0 if fitness[0] < 0.90 else fitness[0]
    #print("finishing: ", threading.get_ident(), ind[0])
    #return res
    return fitness, features

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
    dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
    assert(dimension >= 2)
    nb_features = args.nbFeatures              # The number of features to take into account in the container
    assert(nb_features >= 1)
    bins_per_dim = int(pow(args.maxTotalBins, 1./nb_features))
    nb_bins = (bins_per_dim,) * nb_features       # The number of bins of the grid of elites. Here, we consider only $nb_features$ features with $maxTotalBins^(1/nb_features)$ bins each
    ind_domain = (0., 1.)                     # The domain (min/max values) of the individual genomes
    features_domain = ((0., 1.),) * nb_features # The domain (min/max values) of the features
    fitness_domain = ((0., 1.),)               # The domain (min/max values) of the fitness
    #fitness_domain = ((-np.inf, np.inf),)               # The domain (min/max values) of the fitness
    init_batch_size = 120#0#0                     # The number of evaluations of the initial batch ('batch' = population)
    batch_size = 400#0                          # The number of evaluations in each subsequent batch
    nb_iterations = 20                         # The number of iterations (i.e. times where a new batch is evaluated)
    mut_pb = 0.4                          # The probability of mutating each value of a genome
    eta = 20.0                                # The ETA parameter of the polynomial mutation (as defined in the origin NSGA-II paper by Deb.). It corresponds to the crowding degree of the mutation. A high ETA will produce mutants close to its parent, a small ETA will produce offspring with more changes.
    k = 15                                    # The number of nearest neighbours used to compute novelty
    threshold_novelty = 0.06                  # The threshold of novelty score used to assess whether an individual can be added to the archive
    max_items_per_bin = 1                     # The number of items in each bin of the grid
    verbose = True
    log_base_path = args.outputDir if args.outputDir is not None else "."

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)

    # Create container
    #container = CVTGrid(shape=100, max_items_per_bin=max_items_per_bin, grid_shape=nb_bins, nb_sampled_points=1000, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=OrderedSet, depot_type=OrderedSet)
    #container = NoveltyArchive(k=k, threshold_novelty=threshold_novelty, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=OrderedSet, depot_type=OrderedSet)
    #container = NoveltyArchive(k=k, threshold_novelty=threshold_novelty, fitness_domain=fitness_domain, storage_type=OrderedSet, depot_type=OrderedSet)
    #container = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list, depot_type=False)
    #container2 = Grid(shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list, depot_type=False)
    #container = Container(fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list, depot_type=False)
    #container = Container.from_config({}, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list, depot_type=False)

    config_str = f"""
    cont0:
        type: Grid
        name: cont0
        shape: {list(nb_bins)}
        max_items_per_bin: 1
        fitness_domain: [[0., 1.]]
        features_domain: [[0., 1.], [0., 1.]]

    cont2:
        type: Grid
        name: cont2
        shape: {list(nb_bins)}
        max_items_per_bin: 1
        fitness_domain: [[0., 1.]]
        features_domain: [[0., 1.], [0., 1.]]

    algo1:
        type: RandomSearchMutPolyBounded
        budget: {batch_size*nb_iterations}
        batch_size: {batch_size}
        sel_pb: 0.5
        init_pb: 0.5
        mut_pb: {mut_pb}
        eta: {eta}
        container:
            type: Grid
            shape: {list(nb_bins)}
            max_items_per_bin: 1
            fitness_domain: [[0., 1.]]
            features_domain: [[0., 1.], [0., 1.]]


    algorithms:
        optimisation_task: maximisation
        dimension: {dimension}
        ind_domain: {list(ind_domain)}
        container: cont0

        algo2:
            type: CMAES
            budget: {batch_size*nb_iterations}
            sigma0: 0.1
            ignore_if_not_added_to_container: False

        algo3:
            type: RandomSearchMutPolyBounded
            budget: {batch_size*nb_iterations}
            batch_size: {batch_size}
            sel_pb: 0.5
            init_pb: 0.5
            mut_pb: {mut_pb}
            eta: {eta}

        algoSurrogate:
            type: RandomSearchMutPolyBounded
            container: cont2
            budget: {batch_size*nb_iterations}
            batch_size: {batch_size}
            sel_pb: 0.5
            init_pb: 0.5
            mut_pb: {mut_pb}
            eta: {eta}

        algoInit:
            type: RandomUniform
            budget: 2000

            
        algoSAIL:
            type: SAIL
            illumination_algo: algo3
            acquisition_algo: algoSurrogate


        algoTot:
            type: Sq
            algorithms: ['algoInit', 'algoSAIL', 'algo2']
            tell_container_when_switching: only_best
    """
    config = yaml.load(config_str)
    #print(config)
    factory = Factory()
    container = factory.build(config["cont0"], storage_type=list, depot_type=False)
    container2 = factory.build(config["cont2"], storage_type=list, depot_type=False)
    #algo = factory.build(config["algorithms"]["algo1"])
    #algo = factory.build(config["algorithms"]["algo2"])
    #algo = factory.build(config["algorithms"]["algoTot"])
    factory.build(config["algorithms"])
    algo = factory["algoTot"]
    print(factory)

    # Define evaluation function
    #eval_fn = partial(illumination_rastrigin_normalised, nb_features = nb_features)
    eval_fn = _eval


    # Create algos
    init_fn = partial(lambda dim, base_ind: [random.uniform(ind_domain[0], ind_domain[1]) for _ in range(dim)], dimension)
    select_or_initialise = partial(tools.sel_or_init,
            sel_fn = tools.non_trivial_sel_random,
            sel_pb = 1.0,
            init_fn = init_fn,
            init_pb = 0.0)
    vary = partial(tools.mut_polynomial_bounded, low=ind_domain[0], up=ind_domain[1], eta=eta, mut_pb=mut_pb)
    algo = Evolution(container, budget=batch_size*nb_iterations, batch_size=batch_size, optimisation_task="maximisation", 
            select_or_initialise=select_or_initialise, vary=vary)
#    algo.add_callback("iteration", iteration_callback)
    #logger = AlgorithmLogger(algo)
    logger = TQDMAlgorithmLogger(algo)

    # Create algos
    #algo = RandomUniform(container, budget=batch_size*nb_iterations,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain)
    #algo = Sobol(container, budget=batch_size*nb_iterations,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain)
    #algo = CMAES(container, budget=batch_size*nb_iterations,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain, sigma0 = 0.1, ignore_if_not_added_to_container = True)
    #algo = RandomSearchMutPolyBounded(container, budget=batch_size*nb_iterations, batch_size=batch_size,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain, sel_pb=0.5, init_pb=0.5, mut_pb=mut_pb, eta=eta)
    #algo = RandomSearchMutGaussian(container, budget=batch_size*nb_iterations, batch_size=batch_size,
    #        optimisation_task="maximisation", dimension=dimension,
    #        #nb_objectives = 1,
    #        sel_pb=0.5, init_pb=0.5, mut_pb=mut_pb, mu=1.0, sigma=0.5)

    #algo1 = RandomSearchMutPolyBounded(container, budget=batch_size*nb_iterations, batch_size=batch_size,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain, sel_pb=0.5, init_pb=0.5, mut_pb=mut_pb, eta=eta)
    #algo2 = RandomSearchMutPolyBounded(container2, budget=1000, batch_size=10,
    #        optimisation_task="maximisation", dimension=dimension,
    #        ind_domain=ind_domain, sel_pb=0.5, init_pb=0.5, mut_pb=mut_pb, eta=eta)
    #algo = SAIL(illumination_algo=algo1, acquisition_algo=algo2, max_selected_acquisition_suggestions=np.inf, budget_not_suggested_by_surrogate=np.inf)

    #algo.add_callback("iteration", iteration_callback)
    #logger = TQDMAlgorithmLogger(algo, final_hdf_filename="final.h5", log_base_path=log_base_path)
    logger = TQDMAlgorithmLogger(algo, log_base_path=log_base_path)

    # Run illumination process !
    with ParallelismManager(args.parallelismType) as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False)


    # Print results info
    print("\n------------------------\n")
    print(algo.summary())

    #print(container.summary())
    #print(f"Total elapsed: {algo.total_elapsed}\n")
    #print(container.summary())
    ##print("Best ever fitness: ", container.best_fitness)
    ##print("Best ever ind: ", container.best)
    ##print("Performances container: ", container.fitness)
    ##print("Features container: ", container.features)

#    novelty_best, local_competition_best = container.novelty_local_competition(container.best, k=15, ignore_first=True)
#    print(f"\nNovelty best: {novelty_best}  local competition best: {local_competition_best}")
#
#    novelty_first, local_competition_first = container.novelty_local_competition(container[0], k=15, ignore_first=True)
#    print(f"Novelty first: {novelty_first}  local competition first: {local_competition_first}")

    # It is possible to access the results (including the genomes of the solutions, their performance, etc) stored in the pickle file by using the following code:
    #----8<----8<----8<----8<----8<----8<
    #from deap import base, creator
    #import pickle
    #creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #creator.create("Individual", list, fitness=creator.FitnessMax, features=list)
    #with open("final.p", "rb") as f:
    #    data = pickle.load(f)
    #print(data)
    #----8<----8<----8<----8<----8<----8<
    # --> data is a dictionary containing the results.

    # Transform the container into a grid
    print("\n{:70s}".format("Transforming the container into a grid, for visualisation..."), end="", flush=True)
    #grid = Grid(container.depot, shape=nb_bins, max_items_per_bin=max_items_per_bin, fitness_domain=fitness_domain, features_domain=features_domain, storage_type=list)
    grid = container.to_grid(nb_bins, features_domain=features_domain)
    #grid = container.to_grid(nb_bins)
    print("\tDone !")
    print(grid.summary())

    # Create plot of the performance grid
    plot_path = os.path.join(log_base_path, "performancesGrid.pdf")
    plotGridSubplots(grid.quality_array[... ,0], plot_path, plt.get_cmap("nipy_spectral"), grid.features_domain, grid.fitness_domain[0], nbTicks=None)
    print("\nA plot of the performance grid was saved in '%s'." % os.path.abspath(plot_path))
    print("All results are available in the '%s' pickle file." % logger.final_filename)

    plot_path = os.path.join(log_base_path, "activityGrid.pdf")
    plotGridSubplots(container.activity_per_bin, plot_path, plt.get_cmap("nipy_spectral"), container.features_domain, [0, np.max(container.activity_per_bin)], nbTicks=None)


# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
