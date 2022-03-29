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

"""Old preliminary code. It is not documented and will be removed in the future."""


########### IMPORTS ########### {{{1
import math
import numpy as np
import sys
import copy
import os
import pickle
import time
import gc
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
import itertools
from timeit import default_timer as timer

#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functools import reduce
import operator
import random
import warnings

import qdpy.hdsobol as hdsobol
from qdpy.utils import *


########### INITIALISATION AND MUTATIONS ########### {{{1
@jit(nopython=True)
def mutateUniformInt(ind, mutationPb, indBounds):
    for i in range(len(ind)):
        if np.random.random() < mutationPb:
            ind[i] = np.random.randint(indBounds[0], indBounds[1])

@jit(nopython=True)
def mutateUniform(ind, mutationPb, indBounds):
    for i in range(len(ind)):
        if np.random.random() < mutationPb:
            ind[i] = np.random.uniform(indBounds[0], indBounds[1])

@jit(nopython=True)
def generateUniformInt(dimension, indBounds, nb):
    res = []
    for i in range(nb):
        res.append(np.random.randint(indBounds[0], indBounds[1] + 1, dimension))
    return res

@jit(nopython=True)
def generateUniform(dimension, indBounds, nb):
    res = []
    for i in range(nb):
        res.append(np.random.uniform(indBounds[0], indBounds[1], dimension))
    return res

@jit(nopython=True)
def generateSparseUniform(dimension, indBounds, nb, sparsity):
    res = []
    for i in range(nb):
        base = np.random.uniform(indBounds[0], indBounds[1], dimension)
        mask = np.random.uniform(0., 1., dimension)
        base[mask < sparsity] = 0.
        res.append(base)
    return res

@jit(nopython=True)
def generateSobol(dimension, indBounds, nb):
    res = hdsobol.gen_sobol_vectors(nb+1, dimension)
    res = res * (indBounds[1] - indBounds[0]) + indBounds[0]
    return res

@jit(nopython=True)
def generateBinarySobol(dimension, indBounds, nb, cutoff = 0.50):
    res = hdsobol.gen_sobol_vectors(nb+1, dimension)
    res = (res > cutoff).astype(int)
    return res






########### MAP-Elites ########### {{{1
class MapElites(object):
    def __init__(self, dimension, evalFn, nbBins, featuresBounds = [(0., 1.)], initBatchSize = 120, batchSize=40, nbIterations = 10, indBounds = (0, 100), mutationPb = 0.2, savePeriod = 0, logBasePath = ".", reevalTimeout = None, mutate = None, initiate = None, iterationFilenames = "iteration-%i.p", finalFilename = "final.p", fitnessBounds = (0., 1.), parallelismType = "multiprocessing", completelyNewGenomePb = 0.0):
        self.dimension = dimension
        self.evalFn = evalFn
        self.nbBins = nbBins
        self.featuresBounds = featuresBounds
        self.initBatchSize = initBatchSize
        self.nbIterations = nbIterations
        self.batchSize = batchSize
        self.indBounds = indBounds
        self.mutationPb = mutationPb
        self.savePeriod = savePeriod
        self.logBasePath = logBasePath
        self.reevalTimeout = reevalTimeout
        self.mutate = mutate
        self.initiate = initiate
        self.iterationFilenames = iterationFilenames
        self.finalFilename = finalFilename
        self.fitnessBounds = fitnessBounds
        self.callEvalFnOnEntierBatch = False
        self.parallelismType = parallelismType
        self.completelyNewGenomePb = completelyNewGenomePb
        self.reinit()

    def __del__(self):
        self._closePool()

    def __getstate__(self):
        odict = self.__dict__.copy()
        if 'pool' in odict:
            del odict['pool']
        del odict['map']
        return odict

    def _closePool(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.terminate()

    def _defaultMutate(self, ind):
        #return mutateUniformInt(ind, self.mutationPb, self.indBounds)
        return mutateUniformInt

    def _initParallelism(self):
        if self.parallelismType == "none":
            #import itertools
            self.map = map #itertools.starmap
        elif self.parallelismType == "multiprocessing":
            import multiprocessing
            self._closePool()
            self.pool = multiprocessing.Pool()
            self.map = self.pool.map #self.pool.starmap
        elif self.parallelismType == "scoop":
            import scoop
            self.map = scoop.futures.map
        else:
            raise Exception("Unknown parallelismType: '%s'" % self.parallelismType)


    def reinit(self):
        if not hasattr(self, '_infoToSave'):
            self._infoToSave = {}
        self._initParallelism()
        self._unfinishedEvals = 0
        self._evalId = itertools.count()
        if not self.mutate:
            self.mutate = self._defaultMutate
        if not self.initiate:
            self.initiate = self._defaultInitiation
        self.totalElapsed = 0.

        self.initPop = []
        assert(self.nbIterations > 0)
        self.currentIteration = 0
        self.currentEvaluation = 0
        if self.nbBins == None:
            self.nbBins = np.repeat(10, len(self.featuresBounds))
        assert(len(self.featuresBounds) == len(self.nbBins))
        #self._binsSize = [(self.featuresBounds[1] - self.featuresBounds[0]) / float(b) for b in self.nbBins] 
        self._binsSize = [(self.featuresBounds[bi][1] - self.featuresBounds[bi][0]) / float(self.nbBins[bi]) for bi in range(len(self.nbBins))]
        assert(len(self.nbBins))
        self.performances = np.full(self.nbBins, self.fitnessBounds[0]) #np.zeros(shape=self.nbBins)
        self.features = np.zeros(shape=list(self.nbBins) + [len(self.nbBins)])
        self.solutions = {}
        self.bestInIteration = None
        self.bestInIterationFitness = self.fitnessBounds[0]
        self.bestEver = None
        self.bestEverFitness = self.fitnessBounds[0]
        self.containerSize = reduce(operator.mul, self.nbBins)
        self.nbEmptyBins = self.containerSize

    def _valsToElitesIndex(self, vals):
        index = []
        assert(len(vals) == len(self.nbBins))
        for i in range(len(vals)):
            normalisedVal = vals[i] - self.featuresBounds[i][0]
            if normalisedVal == self.featuresBounds[i][1] - self.featuresBounds[i][0]:
                partial = self.nbBins[i] - 1
            elif normalisedVal > self.featuresBounds[i][1] - self.featuresBounds[i][0]:
                print("WARNING: feature %i out of bounds: %s" % (i, vals))
                return None
            else:
                partial = int(normalisedVal / self._binsSize[i])
            index.append(partial)
        return tuple(index)

    def _asyncEvalFnWrapper(self, ind, expeId):
        def startEval(newInd):
            startTime = timer()
            res = self.evalFn(ind, expeId)
            endTime = timer()
            elapsed = endTime - startTime
            return elapsed, res
        totalElapsed = 0.
        currentInd = ind
        while(True): # XXX bad !
            elapsed, res = startEval(currentInd)
            totalElapsed += elapsed
            if self.reevalTimeout and self.reevalTimeout > totalElapsed:
                fitness = res[0]
                if fitness <= 0.000001:
                    # Perform a reevaluation
                    print("Performed a Reevaluation after:%f sec" % totalElapsed)
                    self.mutate(currentInd, self.mutationPb, self.indBounds)
                    sys.stdout.flush()
                    continue
                else:
                    return currentInd, res
            else:
                return currentInd, res

    def _evalFnWrapper(self, vals):
        ind, expeId = vals
        return self.evalFn(ind, expeId)

    def _evaluatePop(self, pop):
        startTime = timer()
        pop = np.unique(pop, axis=0)
        listEvalIds = list(next(self._evalId) for _ in pop)
        # Evaluate the individuals
        #popFitnesses = np.array(self.map(self.evalFn, pop)) #self.toolbox.map(self.toolbox.evaluate, pop)
        #popFitnesses = np.array(self.map(self.evalFn, list(zip(pop, listEvalIds)))) #self.toolbox.map(self.toolbox.evaluate, pop)
        #resFromEvals = self.map(self._evalFnWrapper, list(zip(pop, listEvalIds)))
        #popFinalInds, popFitnesses = zip(*resFromEvals)

        if self.callEvalFnOnEntierBatch:
            popFitnesses = np.array(list(self.evalFn(pop, listEvalIds)))
        else:
            #popFitnesses = self.map(self.evalFn, list(zip(pop, listEvalIds)))
            popFitnesses = self.map(self._evalFnWrapper, list(zip(pop, listEvalIds)))
        #nbMaxEvalsPerRound = 1000
        #if len(listEvalIds) > nbMaxEvalsPerRound:
        #    nbRounds = int(len(listEvalIds) / nbMaxEvalsPerRound)
        #    popFitnesses = []
        #    for i in range(nbRounds):
        #        fitList = self.map(self.evalFn, list(zip(pop[i * nbMaxEvalsPerRound: (i+1) * nbMaxEvalsPerRound], listEvalIds[i * nbMaxEvalsPerRound: (i+1) * nbMaxEvalsPerRound])))
        #        popFitnesses += fitList
        #        # Perform garbage collection
        #        while gc.collect() > 0:
        #            pass
        #else:
        #    popFitnesses = self.map(self.evalFn, list(zip(pop, listEvalIds)))
        while gc.collect() > 0:
            pass

        popFitnesses = np.array(list(popFitnesses))
        popPerformances = popFitnesses[:,0]
        popFeatures = popFitnesses[:,1:]
        # Record individuals
        for indIndex in range(len(pop)):
            ind = pop[indIndex]
            elitesIndex = self._valsToElitesIndex(popFeatures[indIndex])
            if elitesIndex == None:
                continue
            if self.performances[elitesIndex] < popPerformances[indIndex]:
                self.solutions[elitesIndex] = ind
                self.performances[elitesIndex] = popPerformances[indIndex]
                self.features[elitesIndex] = popFeatures[indIndex]
            if popPerformances[indIndex] > self.bestInIterationFitness:
                self.bestInIterationFitness = popPerformances[indIndex]
                self.bestInIteration = ind
        if self.bestInIterationFitness > self.bestEverFitness:
            self.bestEverFitness = self.bestInIterationFitness
            self.bestEver = self.bestInIteration
        self.nbEmptyBins = self.containerSize - len(self.solutions.keys())
        endTime = timer()
        elapsed = endTime - startTime
        #return popFinalInds, self.bestInIterationFitness, elapsed
        return self.bestInIterationFitness, elapsed


    def _evaluateIndAsync(self, ind = None):
        if ind is None:
            currentElites = list(self.solutions.values())
            if not len(currentElites):
                raise ValueError("No elites were found in initial batch !")
            ind = copy.deepcopy(currentElites[np.random.randint(0, len(currentElites))])
            self.mutate(ind, self.mutationPb, self.indBounds)
        #print("DEBUGasync: ", ind)
        evalId = next(self._evalId)
        self._unfinishedEvals += 1
        asyncResFromEval = self.pool.apply_async(self._asyncEvalFnWrapper, [ind, evalId], callback = self._evaluateIndAsyncCallback)
        return asyncResFromEval

    def _evaluateIndAsyncCallback(self, param):
        ind, res = param
        #print("DEBUGasyncC1: ", res)
        performance = res[0]
        features = res[1:]
        elitesIndex = self._valsToElitesIndex(features)
        if self.performances[elitesIndex] < performance:
            self.solutions[elitesIndex] = ind
            self.performances[elitesIndex] = performance
            self.features[elitesIndex] = features
        if performance > self.bestInIterationFitness:
                self.bestInIterationFitness = performance
                self.bestInIteration = ind
        # Update bestever
        if self.bestInIterationFitness > self.bestEverFitness:
            self.bestEverFitness = self.bestInIterationFitness
            self.bestEver = self.bestInIteration
        self.nbEmptyBins = self.containerSize - len(self.solutions.keys())

        self._unfinishedEvals -= 1
        self.currentEvaluation += 1
        #print("DEBUGasyncC2: ", self._unfinishedEvals, self.currentEvaluation, (self.currentEvaluation - self.initBatchSize) % self.batchSize)
        sys.stdout.flush()
        if self.currentEvaluation == self.initBatchSize:
            # Verify if the initial batch is finished
            print("#0 bestInBatch:%f bestEver:%f" % (self.bestInIterationFitness, self.bestInIterationFitness))
            sys.stdout.flush()
        elif self.currentEvaluation > self.initBatchSize and (self.currentEvaluation - self.initBatchSize) % self.batchSize == 0:
            # Verify if this iteration is finished
            self.currentIteration += 1
            print("#%i bestInBatch:%f bestEver:%f" % (self.currentIteration, self.bestInIterationFitness, self.bestEverFitness))
            sys.stdout.flush()
            if self.savePeriod and np.mod(self.currentIteration, self.savePeriod) == 0:
                #print("DEBUGasyncC3: ", self.iterationFilenames, self.currentIteration)
                self.save(os.path.join(self.logBasePath, self.iterationFilenames % self.currentIteration))

        # Verify if a new eval must be launched
        if self.currentEvaluation + self._unfinishedEvals >= self.initBatchSize + self.nbIterations * self.batchSize:
            if self._unfinishedEvals == 0:
                # All evaluation are finished
                self.save(os.path.join(self.logBasePath, self.finalFilename))
        else:
            # Launch new eval
            self._evaluateIndAsync()



    def _defaultInitiation(self, dimension, indBounds, nb):
        return generateUniformInt(dimension, self.indBounds, nb)

    def _generateInitBatch(self):
        self.initPop = np.unique(self.initiate(self.dimension, self.indBounds, self.initBatchSize), axis=0)
        #self.initPop = []
        #for i in range(self.initBatchSize):
        #    #self.initPop.append(np.random.randint(self.indBounds[0], self.indBounds[1], self.dimension))
        #    #self.initPop.append(np.random.uniform(self.indBounds[0], self.indBounds[1], self.dimension))
        #    self.initPop.append(self.initiate(self.dimension, self.indBounds))
        return self.initPop

    def _iterationMessage(self, prefixLogs, iteration, batch, bestInIterationFitness, bestEverFitness, elapsed):
        #print("%s%i batchSize:%i bestInBatch:%f bestEver:%f elapsed:%f" % (prefixLogs, iteration, len(batch), bestInIterationFitness, bestEverFitness, elapsed))
        print("%s%i batchSize:%i bestEver:%f emptyBins:%i/%i elapsed:%f" % (prefixLogs, iteration, len(batch), bestEverFitness, self.nbEmptyBins, self.containerSize, elapsed))

    def _genNewIndFromElites(self):
        if self.completelyNewGenomePb > 0.0 and np.random.random() < self.completelyNewGenomePb:
            return self.initiate(self.dimension, self.indBounds, 1)[0]
        else:
            currentElites = list(self.solutions.values())
            if not len(currentElites):
                raise ValueError("No elites were found !")
            ind = copy.deepcopy(currentElites[np.random.randint(0, len(currentElites))])
            self.mutate(ind, self.mutationPb, self.indBounds)
            return ind

    def run(self, initBatch = None, nbIterations = None, disableLogs = False, prefixLogs = "#"):
        startTime = timer()
        if initBatch:
            self.initPop = initBatch
        else:
            self._generateInitBatch()
        if nbIterations == None:
            nbIterations = self.nbIterations
        #self.initPop, self.bestInIterationFitness, elapsed = self._evaluatePop(self.initPop)
        self.bestInIterationFitness, elapsed = self._evaluatePop(self.initPop)
        self.bestEverFitness = self.bestInIterationFitness
        self.currentEvaluation = len(self.initPop)
        if not disableLogs:
            self._iterationMessage(prefixLogs, 0, self.initPop, self.bestInIterationFitness, self.bestEverFitness, elapsed)
        sys.stdout.flush()
        for iteration in range(1, nbIterations):
            self.currentIteration = iteration
            self.currentEvaluation += self.batchSize
            currentElites = list(self.solutions.values())
            if not len(currentElites):
                raise ValueError("No elites were found in initial batch !")
            newPop = []
            for i in range(self.batchSize):
                ind = self._genNewIndFromElites()
                newPop.append(ind)
            #newPop, self.bestInIterationFitness, elapsed = self._evaluatePop(newPop)
            self.bestInIterationFitness, elapsed = self._evaluatePop(newPop)
            #print("%s%i bestInBatch:%f bestEver:%f elapsed:%f elapsedPerInd:%f" % (prefixLogs, iteration, self.bestInIterationFitness, self.bestEverFitness, elapsed, elapsed / len(newPop)))
            if not disableLogs:
                self._iterationMessage(prefixLogs, iteration, newPop, self.bestInIterationFitness, self.bestEverFitness, elapsed)
            sys.stdout.flush()
            if not disableLogs and self.savePeriod and np.mod(self.currentIteration, self.savePeriod) == 0:
                self.save(os.path.join(self.logBasePath, self.iterationFilenames % iteration))
            # Perform garbage collection
            while gc.collect() > 0:
                pass
        if not disableLogs and self.finalFilename:
            self.save(os.path.join(self.logBasePath, self.finalFilename))
        endTime = timer()
        self.totalElapsed += endTime - startTime
        if not disableLogs and nbIterations > 1:
            print("Total elapsed:%f" % (self.totalElapsed))
        return (self.bestEver, self.bestEverFitness)


    def runAsync(self, initBatch = None):
        if initBatch:
            self.initPop = initBatch
        else:
            self._generateInitBatch()
        evalsFuturesList = []
        for i in range(len(self.initPop)):
            evalsFuturesList.append(self._evaluateIndAsync(self.initPop[i]))
            #self._evaluateIndAsync(self.initPop[i])
        # Main loop
        while(True):
            #print("DEBUG1:", self.currentEvaluation, self.initBatchSize + self.nbIterations * self.batchSize)
            #print("DEBUG1b: ", [x.ready() for x in evalsFuturesList])
            if self.currentEvaluation >= self.initBatchSize + self.nbIterations * self.batchSize:
                break
            time.sleep(1)
        return (self.bestEver, self.bestEverFitness)


    def generateOutputDict(self):
        outputDict = {}
        outputDict['performances'] = self.performances
        outputDict['features'] = self.features
        outputDict['solutions'] = self.solutions
        outputDict['dimension'] = self.dimension
        outputDict['nbBins'] = self.nbBins
        outputDict['featuresBounds'] = self.featuresBounds
        outputDict['initBatchSize'] = self.initBatchSize
        outputDict['nbIterations'] = self.nbIterations
        outputDict['batchSize'] = self.batchSize
        outputDict['indBounds'] = self.indBounds
        outputDict['mutationPb'] = self.mutationPb
        outputDict['currentIteration'] = self.currentIteration
        outputDict['initPop'] = self.initPop
        outputDict['bestEver'] = self.bestEver
        outputDict['bestEverFitness'] = self.bestEverFitness
        outputDict = {**outputDict, **self._infoToSave}
        return outputDict

    def save(self, outputFile):
        outputDict = self.generateOutputDict()
        with open(outputFile, "wb") as f:
            pickle.dump(outputDict, f)

    def addSavingInfo(self, key, value):
        self._infoToSave[key] = value



########### CVT-MAP-Elites ########### {{{1

#@jit(nopython=True)
@jit(nopython=True, parallel=True)
def _cvtMapElites_valsToElitesIndex(clusterCenters, vals):
    dists = np.empty((clusterCenters.shape[0]))
    for i in range(len(dists)):
        dists[i] = math.sqrt(np.sum(np.square(clusterCenters[i] - vals)))
    return (np.argmin(dists),)


class CVTMapElites(MapElites):
    def __init__(self, dimension, evalFn, nbBins, nbClusters, featuresBounds = [(0.0, 1.0)], initBatchSize = 120, batchSize=40, nbIterations = 10, indBounds = (0, 100), mutationPb = 0.2, savePeriod = 0, logBasePath = ".", nbSampledPoints = 50000, reevalTimeout = 5., mutate = None, initiate = None, iterationFilenames = "iteration-%i.p", finalFilename = "final.p", fitnessBounds = (0., 1.), parallelismType = "multiprocessing", completelyNewGenomePb = 0.0):
        self.nbClusters = nbClusters
        self.nbSampledPoints = nbSampledPoints
        super().__init__(dimension, evalFn, nbBins, featuresBounds, initBatchSize, batchSize, nbIterations, indBounds, mutationPb, savePeriod, logBasePath, reevalTimeout, mutate, initiate, iterationFilenames, finalFilename, fitnessBounds, parallelismType, completelyNewGenomePb=completelyNewGenomePb)

    def reinit(self):
        super().reinit()
        self.performances = np.zeros(shape=self.nbClusters)
        self.features = np.zeros(shape=[self.nbClusters, len(self.nbBins)])
        # Init clusters
        sample = np.random.uniform(self.featuresBounds[0][0], self.featuresBounds[0][1], (self.nbSampledPoints, len(self.nbBins)))
        kmeans = KMeans(init="k-means++", n_clusters=self.nbClusters, n_init=1, n_jobs=1, verbose=0)
        kmeans.fit(sample)
        self.clusterCenters = kmeans.cluster_centers_
        self.containerSize = self.nbClusters
        self.nbEmptyBins = self.containerSize

    def _valsToElitesIndex(self, vals):
        return _cvtMapElites_valsToElitesIndex(self.clusterCenters, vals)
        #closestDist = float("inf")
        #closestCluster = 0
        #for i in range(len(self.clusterCenters)):
        #    dist = euclidean(self.clusterCenters[i], vals)
        #    if dist < closestDist:
        #        closestDist = dist
        #        closestCluster = i
        #return (closestCluster,)

    def generateOutputDict(self):
        outputDict = super().generateOutputDict()
        outputDict['nbClusters'] = self.nbClusters
        outputDict['nbSampledPoints'] = self.nbSampledPoints
        outputDict['clusterCenters'] = self.clusterCenters
        outputDict = {**outputDict, **self._infoToSave}
        return outputDict



########### SAIL ########### {{{1
class SAIL(object):
    def __init__(self, dimension, evalFn, nbBins, illuminationAlgo, acquisitionMapAlgo, predictionMapAlgo, featuresBounds = [(0.0, 1.0)], initBatchSize = 120, batchSize=40, indBounds = (0, 100), mutationPb = 0.2, savePeriod = 0, logBasePath = ".", mutate = None, initiate = None, nbAcquisitionRounds = 10, ucbStdDevFactor = 1.0, nbPreciseEvalsPerAcquisitionRound = 10, instanceName = "", fitnessBounds = (0., 1.), finalFilename = "final.p", parallelismType = "multiprocessing", maintainBatchSizeWithMutatedInd = False, preciseEvalsSelectionType = "sobol"):
        self.dimension = dimension
        self.evalFn = evalFn
        self.nbBins = nbBins
        self.illuminationAlgo = illuminationAlgo
        self.acquisitionMapAlgo = acquisitionMapAlgo
        self.predictionMapAlgo = predictionMapAlgo
        self.featuresBounds = featuresBounds
        self.initBatchSize = initBatchSize
        self.batchSize = batchSize
        self.indBounds = indBounds
        self.mutationPb = mutationPb
        self.savePeriod = savePeriod
        self.logBasePath = logBasePath
        self.mutate = mutate
        self.initiate = initiate
        self.nbAcquisitionRounds = nbAcquisitionRounds
        self.ucbStdDevFactor = ucbStdDevFactor
        self.nbPreciseEvalsPerAcquisitionRound = nbPreciseEvalsPerAcquisitionRound
        self.instanceName = instanceName
        self.fitnessBounds = fitnessBounds
        self.finalFilename = finalFilename
        self.parallelismType = parallelismType
        self.maintainBatchSizeWithMutatedInd = maintainBatchSizeWithMutatedInd
        self.preciseEvalsSelectionType = preciseEvalsSelectionType
        self.reinit()

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['gp']
        return odict


    def reinit(self):
        self._infoToSave = {}
        self.acquisitionMapAlgo.evalFn = self._acquisitionEvalFn
        self.predictionMapAlgo.evalFn = self._predictionEvalFn
        self.acquisitionMapAlgo.parallelismType = "none" # Disable multiprocessing
        self.predictionMapAlgo.parallelismType = "none" # Disable multiprocessing
        self.currentRound = 0


    def _acquisitionEvalFn(self, inds, expeIds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_mean, y_std = self.gp.predict(np.array(inds), return_std = True)
        ucbVal = y_mean + self.ucbStdDevFactor * y_std.reshape((-1, 1))
        # Normalise ucbVal
        ucbVal[:,0][ucbVal[:,0] < self.fitnessBounds[0]] = self.fitnessBounds[0]
        ucbVal[:,0][ucbVal[:,0] > self.fitnessBounds[1]] = self.fitnessBounds[1]
        for i in range(len(self.featuresBounds)):
            ucbVal[:,i+1][ucbVal[:,i+1] < self.featuresBounds[i][0]] = self.featuresBounds[i][0]
            ucbVal[:,i+1][ucbVal[:,i+1] > self.featuresBounds[i][1]] = self.featuresBounds[i][1]
        return copy.deepcopy(ucbVal)

    def _predictionEvalFn(self, inds, expeIds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_mean, y_std = self.gp.predict(np.array(inds), return_std = True)
        # Normalise y_mean
        y_mean[:,0][y_mean[:,0] < self.fitnessBounds[0]] = self.fitnessBounds[0]
        y_mean[:,0][y_mean[:,0] > self.fitnessBounds[1]] = self.fitnessBounds[1]
        for i in range(len(self.featuresBounds)):
            y_mean[:,i+1][y_mean[:,i+1] < self.featuresBounds[i][0]] = self.featuresBounds[i][0]
            y_mean[:,i+1][y_mean[:,i+1] > self.featuresBounds[i][1]] = self.featuresBounds[i][1]
        return copy.deepcopy(y_mean)

    def _createGaussianRegressor(self):
        # Create dataset
        elitesIndexes = list(self.illuminationAlgo.solutions.keys())
        assert(len(elitesIndexes))
        dataX = []
        dataY = []
        for index in elitesIndexes:
            dataX.append(self.illuminationAlgo.solutions[index])
            perfAndFeatures = [self.illuminationAlgo.performances[index]] + list(self.illuminationAlgo.features[index])
            dataY.append(perfAndFeatures)
        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # Fit gaussian process on the dataset
        #print("DEBUGii", dataX, dataY, self.fitnessBounds, np.min(self.illuminationAlgo.performances), elitesIndexes)
        self.gp = GaussianProcessRegressor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(dataX, dataY)
        return self.gp

    def _genNewIndFromElites(self):
        return self.illuminationAlgo._genNewIndFromElites()

    def _produceAcquisitionMap(self, disableLogs = True, prefixLogs = "#"):
        for roundId in range(1, self.nbAcquisitionRounds):
            startTimeAcquisition = timer()
            self.currentRound += 1
            # Reset acquisitionMapAlgo
            self.acquisitionMapAlgo.reinit()

            self._createGaussianRegressor()
            pop = list(self.illuminationAlgo.solutions.values())
            #print("DEBUGAC:", len(pop))
            # Launch map-elites using the gp as surrogate model
            if self.savePeriod and np.mod(roundId, self.savePeriod) == 0:
                self.acquisitionMapAlgo.iterationFilenames = "iterationAcquisition-%i-%s_%s.p" % (roundId, "%i", self.instanceName)
                self.acquisitionMapAlgo.finalFilename = "finalAcquisition-%i_%s.p" % (roundId, self.instanceName)
            else:
                self.acquisitionMapAlgo.iterationFilenames = None
                self.acquisitionMapAlgo.finalFilename = None
            self.acquisitionMapAlgo.initBatchSize = len(pop)
            bestEver, bestEverFitness = self.acquisitionMapAlgo.run(pop, disableLogs = disableLogs, prefixLogs = "Acquisition%i-%s" % (roundId, prefixLogs))

            # Select bins from acquisitionMapAlgo map
            #acquisitionSolutions = list(self.acquisitionMapAlgo.solutions.values())
            acquisitionSolutions = set([tuple(x) for x in self.acquisitionMapAlgo.solutions.values()])
            acquisitionSolutions -= set([tuple(x) for x in pop])
            if len(acquisitionSolutions):
                acquisitionSolutions = np.unique([np.array(x) for x in list(acquisitionSolutions)], axis=0)
                if len(acquisitionSolutions) <= self.nbPreciseEvalsPerAcquisitionRound:
                    selectedSolutions = acquisitionSolutions
                else:
                    # Select how the index is generated (random uniform sampling vs sobol sampling)
                    if self.preciseEvalsSelectionType == "sobol":
                        indexList = np.unique((hdsobol.gen_sobol_vectors(self.nbPreciseEvalsPerAcquisitionRound+1, 1) * len(acquisitionSolutions)).astype(int))
                    elif self.preciseEvalsSelectionType == "uniform":
                        indexList = random.sample(range(len(acquisitionSolutions)), self.nbPreciseEvalsPerAcquisitionRound)
                    else:
                        raise ValueError("Unknown value '%s' for entry 'preciseEvalsSelectionType'" % self.preciseEvalsSelectionType)
                    selectedSolutions = [acquisitionSolutions[x] for x in indexList]

                pop = list(selectedSolutions)
                if self.maintainBatchSizeWithMutatedInd:
                    for i in range(self.batchSize - len(selectedSolutions)):
                        pop.append(self._genNewIndFromElites())

                bestInIterationFitness, elapsedIllumination = self.illuminationAlgo._evaluatePop(pop)
                endTimeAcquisition = timer()
                elapsed = elapsedIllumination + endTimeAcquisition - startTimeAcquisition
                self.illuminationAlgo._iterationMessage(prefixLogs, roundId, pop, self.illuminationAlgo.bestInIterationFitness, self.illuminationAlgo.bestEverFitness, elapsed)
                if self.savePeriod and np.mod(roundId, self.savePeriod) == 0:
                    self.illuminationAlgo.iterationFilenames = "iteration-%i-%s_%s.p" % (roundId, "%i", self.instanceName)
                    self.illuminationAlgo.save(os.path.join(self.logBasePath, self.illuminationAlgo.iterationFilenames % roundId))

            self.bestEverFitness = self.illuminationAlgo.bestEverFitness
            self.bestEver = self.illuminationAlgo.bestEver
            self.containerSize = self.illuminationAlgo.containerSize
            self.nbEmptyBins = self.illuminationAlgo.nbEmptyBins
            self.performances = self.illuminationAlgo.performances
            self.solutions = self.illuminationAlgo.solutions
            self.features = self.illuminationAlgo.features

            # Perform garbage collection
            while gc.collect() > 0:
                pass



    def _producePredictionMap(self, disableLogs = True, prefixLogs = "#"):
        self.predictionMapAlgo.reinit()
        self._createGaussianRegressor()
        pop = list(self.illuminationAlgo.solutions.values())
        # Launch map-elites using the gp as surrogate model
        self.predictionMapAlgo.iterationFilenames = "iterationPrediction-%s_%s.p" % ("%i", self.instanceName)
        self.predictionMapAlgo.finalFilename = "finalPrediction_%s.p" % self.instanceName
        self.predictionMapAlgo.initBatchSize = len(pop)
        bestEver, bestEverFitness = self.predictionMapAlgo.run(pop, disableLogs = disableLogs, prefixLogs = "Prediction-" + prefixLogs)

    def _updateConfig(self):
        def updateEntry(entry):
            self.illuminationAlgo.__dict__[entry] = self.__dict__[entry]
            self.acquisitionMapAlgo.__dict__[entry] = self.__dict__[entry]
            self.predictionMapAlgo.__dict__[entry] = self.__dict__[entry]
        list(map(updateEntry, ["initiate", "mutate", "logBasePath", "nbBins", "_infoToSave", "dimension", "featuresBounds", "indBounds", "fitnessBounds"]))
        self.illuminationAlgo.evalFn = self.evalFn
        self.illuminationAlgo.iterationFilenames = None
        self.illuminationAlgo.finalFilename = None
        self.illuminationAlgo.initBatchSize = self.initBatchSize
        self.acquisitionMapAlgo.callEvalFnOnEntierBatch = True
        self.predictionMapAlgo.callEvalFnOnEntierBatch = True
        self.illuminationAlgo.reinit()

    def run(self, initBatch = None, nbIterations = None, disableLogs = False, prefixLogs = "#"):
        self._updateConfig()
        # Generate initial population
        self.illuminationAlgo.run(initBatch = initBatch, nbIterations = 0, disableLogs = disableLogs, prefixLogs = prefixLogs) #, prefixLogs="Init-#")
        # Create maps
        if len(list(self.illuminationAlgo.solutions.keys())):
            self._produceAcquisitionMap(disableLogs = True, prefixLogs = prefixLogs)
            self._producePredictionMap(disableLogs = True, prefixLogs = prefixLogs)
        # Save
        #self.illuminationAlgo.finalFilename = "final_%s.p" % (self.instanceName)
        #self.illuminationAlgo.save(os.path.join(self.logBasePath, self.illuminationAlgo.finalFilename))
        #self.finalFilename = "final_%s.p" % (self.instanceName)
        if self.finalFilename:
            self.save(os.path.join(self.logBasePath, self.finalFilename))
        return (self.illuminationAlgo.bestEver, self.illuminationAlgo.bestEverFitness)


    def generateOutputDict(self):
        outputDict = {}
        outputDict['performances'] = self.performances
        outputDict['features'] = self.features
        outputDict['solutions'] = self.solutions
        outputDict['dimension'] = self.dimension
        outputDict['nbBins'] = self.nbBins
        outputDict['featuresBounds'] = self.featuresBounds
        outputDict['initBatchSize'] = self.initBatchSize
        outputDict['batchSize'] = self.batchSize
        outputDict['indBounds'] = self.indBounds
        outputDict['mutationPb'] = self.mutationPb
        outputDict['currentRound'] = self.currentRound
        outputDict['fitnessBounds'] = self.fitnessBounds
        outputDict['illuminationAlgo'] = self.illuminationAlgo.generateOutputDict()
        outputDict['acquisitionMapAlgo'] = self.acquisitionMapAlgo.generateOutputDict()
        outputDict['predictionMapAlgo'] = self.predictionMapAlgo.generateOutputDict()
        outputDict['bestEver'] = self.bestEver
        outputDict['bestEverFitness'] = self.bestEverFitness
        outputDict = {**outputDict, **self._infoToSave}
        return outputDict

    def save(self, outputFile):
        outputDict = self.generateOutputDict()
        with open(outputFile, "wb") as f:
            pickle.dump(outputDict, f)

    def addSavingInfo(self, key, value):
        self._infoToSave[key] = value





########### Factory ########### {{{1

class AlgorithmFactory(object):
    def __init__(self, algoType = "map-elites", nbDim = 50, evalFn = None, nbBins = None, nbClusters = 30, nbSampledPoints = 50000, featuresBounds = [(0., 1.)], indBounds = (0, 100), initBatchSize = 10000, batchSize = 4000, nbIterations = 500, mutationPb = 0.2, nbCenters = 20, savePeriod = 0, logBasePath = ".", reevalTimeout = None, mutate = None, initiate = None, iterationFilenames = "iteration-%i.p", finalFilename = "final.p",  instanceName = "", fitnessBounds = (0., 1.), parallelismType = "multiprocessing", maintainBatchSizeWithMutatedInd = False, completelyNewGenomePb = 0.0, preciseEvalsSelectionType = "sobol"):
        self.algoType = algoType
        self.nbDim = nbDim
        self.evalFn = evalFn
        self.nbBins = nbBins
        self.nbClusters = nbClusters
        self.nbSampledPoints = nbSampledPoints
        self.featuresBounds = featuresBounds
        self.indBounds = indBounds
        self.initBatchSize = initBatchSize
        self.batchSize = batchSize
        self.nbIterations = nbIterations
        self.mutationPb = mutationPb
        self.nbCenters = nbCenters
        self.savePeriod = savePeriod
        self.logBasePath = logBasePath
        self.reevalTimeout = reevalTimeout
        self.mutate = mutate
        self.initiate = initiate
        self.iterationFilenames = iterationFilenames
        self.finalFilename = finalFilename
        self.instanceName = instanceName
        self.fitnessBounds = fitnessBounds
        self.parallelismType = parallelismType
        self.maintainBatchSizeWithMutatedInd = maintainBatchSizeWithMutatedInd
        self.completelyNewGenomePb = completelyNewGenomePb
        self.preciseEvalsSelectionType = preciseEvalsSelectionType
        self.algo = None

    def update(self, **kwargs):
        for k,v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
            else:
                raise ValueError("Unknown entry: '%s'" % k)

    def build(self):
        if self.algoType == "map-elites":
            self.algo = MapElites(self.nbDim, self.evalFn, self.nbBins, self.featuresBounds, self.initBatchSize, self.batchSize, self.nbIterations, self.indBounds, self.mutationPb, self.savePeriod, self.logBasePath, self.reevalTimeout, self.mutate, self.initiate, self.iterationFilenames, self.finalFilename, fitnessBounds = self.fitnessBounds, parallelismType = self.parallelismType, completelyNewGenomePb = self.completelyNewGenomePb)
        elif self.algoType == "cvt-map-elites":
            self.algo = CVTMapElites(self.nbDim, self.evalFn, self.nbBins, self.nbClusters, self.featuresBounds, self.initBatchSize, self.batchSize, self.nbIterations, self.indBounds, self.mutationPb, self.savePeriod, self.logBasePath, self.nbSampledPoints, self.reevalTimeout, self.mutate, self.initiate, self.iterationFilenames, self.finalFilename, fitnessBounds = self.fitnessBounds, parallelismType = self.parallelismType, completelyNewGenomePb = self.completelyNewGenomePb)
        elif self.algoType == "sail":
            self.algo = SAIL(self.nbDim, self.evalFn, self.nbBins, self.illuminationAlgo, self.acquisitionMapAlgo, self.predictionMapAlgo, self.featuresBounds, self.initBatchSize, self.batchSize, self.indBounds, self.mutationPb, self.savePeriod, self.logBasePath, self.mutate, self.initiate, self.nbAcquisitionRounds, self.ucbStdDevFactor, self.nbPreciseEvalsPerAcquisitionRound, instanceName=self.instanceName, fitnessBounds = self.fitnessBounds, finalFilename = self.finalFilename, parallelismType = self.parallelismType, maintainBatchSizeWithMutatedInd = self.maintainBatchSizeWithMutatedInd, preciseEvalsSelectionType = self.preciseEvalsSelectionType)
        else:
            raise ValueError("Unknown algoType: '%s'" % self.algoType)
        return self.algo


    def fromConfig(self, config):
        def setIfExists(key):
            o = config.get(key)
            if o:
                self.__dict__[key] = o
        def buildIfExists(key):
            o = config.get(key)
            if o:
                fact = AlgorithmFactory()
                fact.fromConfig(o)
                fact.update(evalFn = self.evalFn, featuresBounds = self.featuresBounds, \
                        fitnessBounds = self.fitnessBounds, indBounds = self.indBounds, \
                        nbDim = self.nbDim, nbClusters = self.nbClusters, mutationPb = self.mutationPb, \
                        parallelismType = self.parallelismType, completelyNewGenomePb = self.completelyNewGenomePb)
                self.__dict__[key] = fact.build()
        setIfExists('parallelismType')
        setIfExists('algoType')
        setIfExists('nbDim')
        setIfExists('nbBins')
        setIfExists('nbClusters')
        setIfExists('nbSampledPoints')
        setIfExists('featuresBounds')
        setIfExists('indBounds')
        setIfExists('fitnessBounds')
        setIfExists('initBatchSize')
        setIfExists('batchSize')
        setIfExists('nbIterations')
        setIfExists('mutationPb')
        setIfExists('nbCenters')
        setIfExists('savePeriod')
        setIfExists('reevalTimeout')
        setIfExists('completelyNewGenomePb')
        # SAIL config
        setIfExists('nbAcquisitionRounds')
        setIfExists('ucbStdDevFactor')
        setIfExists('nbPreciseEvalsPerAcquisitionRound')
        setIfExists('maintainBatchSizeWithMutatedInd')
        setIfExists('preciseEvalsSelectionType')
        buildIfExists('illuminationAlgo')
        buildIfExists('acquisitionMapAlgo')
        buildIfExists('predictionMapAlgo')






########### Plots ########### {{{1

def plotMAP(performances, outputFilename, cmap, featuresBounds=[(0., 1.), (0., 1.)], fitnessBounds=(0., 1.), drawCbar = True, xlabel = "", ylabel = "", cBarLabel = "", nbTicks = 10):
    data = performances
    if len(data.shape) == 1:
        data = data.reshape((1, data.shape[0]))
    nbBins = data.shape

    figsize = [2.1 + 10. * nbBins[0] / (nbBins[0] + nbBins[1]), 1. + 10. * nbBins[1] / (nbBins[0] + nbBins[1])]
    aspect = "equal"
    if figsize[1] < 2:
        figsize[1] = 2.
        aspect = "auto"

    fig, ax  = plt.subplots(figsize=figsize)
    cax = ax.imshow(data.T, interpolation="none", cmap=cmap, vmin=fitnessBounds[0], vmax=fitnessBounds[1], aspect=aspect)
    #ax.set_aspect('equal')
    ax.invert_yaxis()

    if nbBins[0] > nbBins[1]:
        nbTicksX = nbTicks
        nbTicksY = int(nbTicksX * nbBins[1] / nbBins[0])
    elif nbBins[1] > nbBins[0]:
        nbTicksY = nbTicks
        nbTicksX = int(nbTicksY * nbBins[0] / nbBins[1])
    else:
        nbTicksX = nbTicksY = nbTicks
    if nbTicksX > nbBins[0] or nbTicksX < 1:
        nbTicksX = nbBins[0]
    if nbTicksY > nbBins[1] or nbTicksY < 1:
        nbTicksY = nbBins[1]

    # Set ticks
    ax.xaxis.set_tick_params(which='major', left=False, bottom=False, top=False, right=False)
    ax.yaxis.set_tick_params(which='major', left=False, bottom=False, top=False, right=False)
    xticks = list(np.arange(0, data.shape[0] + 1, data.shape[0] / nbTicksX))
    yticks = list(np.arange(0, data.shape[1] + 1, data.shape[1] / nbTicksY))
    plt.xticks(xticks, rotation='vertical')
    plt.yticks(yticks)
    deltaFeature0 = featuresBounds[0][1] - featuresBounds[0][0]
    deltaFeature1 = featuresBounds[1][1] - featuresBounds[1][0]
    ax.set_xticklabels([round(float(x / float(data.shape[0]) * deltaFeature0 + featuresBounds[0][0]), 2) for x in xticks], fontsize=20)
    ax.set_yticklabels([round(float(y / float(data.shape[1]) * deltaFeature1 + featuresBounds[1][0]), 2) for y in yticks], fontsize=20)

    # Draw grid
    ax.xaxis.set_tick_params(which='minor', direction="in", left=True, bottom=True, top=True, right=True)
    ax.yaxis.set_tick_params(which='minor', direction="in", left=True, bottom=True, top=True, right=True)
    ax.set_xticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[1], 1), minor=True)
    ax.grid(which='minor', color=(0.8,0.8,0.8,0.5), linestyle='-', linewidth=0.1)

    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    if drawCbar:
        divider = make_axes_locatable(ax)
        #cax2 = divider.append_axes("right", size="5%", pad=0.15)
        cax2 = divider.append_axes("right", size=0.5, pad=0.15)
        cbar = fig.colorbar(cax, cax=cax2, format="%.2f")
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.set_ylabel(cBarLabel, fontsize=22)

    ax.autoscale_view()
    plt.tight_layout()
    fig.savefig(outputFilename)





# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
