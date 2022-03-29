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


"""A simple example of MAP-elites to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3."""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from qdpy.legacy import *



# Rastrigin with inputs 'nx' and outputs normalised between 0. and 1.
def normalised_rastrigin(nx, a=10.):
    x = -5.12 + nx * (5.12 * 2.)
    n = float(len(x))
    res = a * n + np.sum(x * x - a * np.cos(2. * np.pi * x))
    nres = res / (a * n + n * (5.12 * 5.12 + a))
    return nres


def _evalFn(ind, expeId):
    dimensionInd = len(ind)
    dimensionFeatures = 2 #len(ind) - 2
    fitness = 1. - normalised_rastrigin(np.array(ind))
    features = ind[:dimensionFeatures]
    res = [fitness] + list(features)
    #print("DEBUGeval: ", expeId, ind, res)
    return res




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=3, help="Problem dimension")
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-c', '--configFile', type=str, default=None, help="Path of the configuration file")  # An example of config file is provided in examples/conf/legacy_rastrigin.py
    parser.add_argument('--algorithm', type=str, default="MAP-Elites", help="Type of QD algorithm to use (MAP-Elites, CVT-MAP-Elites)")
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, multiprocessing, scoop)")
    args = parser.parse_args()

    if args.seed != None:
        seed = args.seed
    else:
        seed = np.random.randint(1000000)

    if args.configFile == None or args.configFile == "":
        # Algorithm parameters
        dimension = args.dimension                # The dimension of the target problem (i.e. genomes size)
        assert(dimension >= 2)
        nbBins = (64, 64)                         # The number of bins of the grid of elites. Here, we consider only 2 features with 64 bins each
        indBounds = (0., 1.)                      # The domain (min/max values) of the individual genomes
        featuresBounds = [(0., 1.), (0., 1.)]     # The domain (min/max values) of the features
        fitnessBounds = (0., 1.)                  # The domain (min/max values) of the fitness
        initBatchSize = 12000#0                   # The number of evaluations of the initial batch ('batch' = population)
        batchSize = 4000#0                        # The number of evaluations in each subsequent batch
        nbIterations = 20                         # The number of iterations (i.e. times where a new batch is evaluated)
        initiateFn = generateUniform              # The function to call to initiate a genome in the initial batch. It can accept any function with arguments (dimension, indBounds, nb)
        mutateFn = mutateUniform                  # The function to call to mutate a genome. It can accept any function with arguments (ind, mutationPb, indBounds)
        mutationPb = 0.2                          # The probability of mutating each value of a genome
        savePeriod = 0                            # Save a pickle result file every 'savePeriod' iterations. If set to 0, only save a result file at the end of the illumination process

        # CVT-MAP-Elites parameters
        nbClusters = 100                          # Number of clusters in the feature space to take into account; corresponds to the number of entries in the solutions container
        nbSampledPoints = 50000                   # Number of points to sample with k-means to determine the centroids in the feature space

        # Create QD algorithm
        if args.algorithm == "MAP-Elites":
            algo = MapElites(dimension, _evalFn, nbBins, featuresBounds=featuresBounds, fitnessBounds=fitnessBounds, initBatchSize=initBatchSize, batchSize=batchSize, nbIterations=nbIterations, indBounds=indBounds, mutationPb=mutationPb, mutate=mutateFn, initiate=initiateFn, savePeriod=savePeriod, parallelismType = args.parallelismType)
        elif args.algorithm == "CVT-MAP-Elites":
            algo = CVTMapElites(dimension, _evalFn, nbBins, nbClusters, nbSampledPoints=nbSampledPoints, featuresBounds=featuresBounds, fitnessBounds=fitnessBounds, initBatchSize=initBatchSize, batchSize=batchSize, nbIterations=nbIterations, indBounds=indBounds, mutationPb=mutationPb, mutate=mutateFn, initiate=initiateFn, savePeriod=savePeriod, parallelismType = args.parallelismType)
        else:
            print("Unknown algorithm: '%s'" % args.algorithm)

    else:
        # Load configuration
        import yaml
        config = yaml.safe_load(open(args.configFile))
        featuresBounds = config['algorithm']['featuresBounds']
        fitnessBounds = config['algorithm']['fitnessBounds']
        if 'randomSeed' in config and args.seed == None:
            seed = config['randomSeed']
        # Create QD algorithm
        factory = AlgorithmFactory()
        factory.fromConfig(config['algorithm'])
        factory.update(evalFn=_evalFn, parallelismType=args.parallelismType, mutate=mutateUniform, initiate=generateUniform)
        algo = factory.build()
        algo.addSavingInfo('config', config)

    # Update and print seed
    np.random.seed(seed)
    print("Seed: %i" % seed)

    # Launch the selected QD algorithm !
    algo.run()

    # Print results info
    print("Best ever fitness: ", algo.bestEverFitness)
    print("Best ever ind: ", algo.bestEver)
    resultFullFilename = os.path.abspath(os.path.join(algo.logBasePath, algo.finalFilename))
    print("%i/%i empty bins in the grid" % (algo.nbEmptyBins, algo.containerSize))
    #print("Solutions found for bins: ", algo.solutions.keys())
    #print("Performances grid: ", algo.performances)
    #print("Features grid: ", algo.features)
    print("All results are available in the '%s' pickle file." % resultFullFilename)

    # It is possible to access the results (including the genomes of the solutions, their performance, etc) stored in the pickle file by using the following code:
    #----8<----8<----8<----8<----8<----8<
    #import pickle
    #with open("final.p", "rb") as f:
    #    data = pickle.load(f)
    #print(data)
    #----8<----8<----8<----8<----8<----8<
    # --> data is a dictionary containing the results.
    #     it contains the following keys:
    #     ['performances', 'features', 'solutions', 'dimension', 'nbBins', 'featuresBounds', 'initBatchSize', 'nbIterations', 'batchSize', 'indBounds', 'mutationPb', 'currentIteration', 'initPop', 'bestEver', 'bestEverFitness']

    # Create plot of the performance grid
    plotPath = os.path.join(algo.logBasePath, "performancesGrid.pdf")
    plotMAP(algo.performances, plotPath, plt.get_cmap("nipy_spectral"), featuresBounds, fitnessBounds)
    print("A plot of the performance grid was saved in '%s'." % os.path.abspath(plotPath))



# MODELINE "{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
