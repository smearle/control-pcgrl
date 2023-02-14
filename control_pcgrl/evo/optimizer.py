import copy
from timeit import default_timer as timer

import deap
from deap.base import Toolbox
import numpy as np
from qdpy import tools


def mate_individuals(ind_0, ind_1):
    return ind_0.mate(ind_1)

def mutate_individual(ind):
    ind.mutate()
    return (ind,)

class MEOptimizer():
    def __init__(self, grid, ind_cls, batch_size, ind_cls_args, start_time=None, stats=None):
        self.batch_size = batch_size
        self.grid = grid
        self.inds = []
        self.stats=stats
        for _ in range(batch_size):
            self.inds.append(ind_cls(**ind_cls_args))
        toolbox = Toolbox()
        toolbox.register("clone", copy.deepcopy)
        toolbox.register("mutate", mutate_individual)
        toolbox.register("mate", mate_individuals)
        toolbox.register("select", tools.sel_random)

        self.cxpb = 0
        self.mutpb = 1.0
        self.toolbox = toolbox
        if start_time == None:
            self.start_time = timer()
        self.logbook = deap.tools.Logbook()
        self.logbook.header = ["iteration", "containerSize", "evals", "nbUpdated"] + (stats.fields if stats else []) + \
            ["meanFitness", "maxFitness", "elapsed"]
        self.i = 0


    def tell(self, objective_values, behavior_values):
        """Tell MAP-Elites about the performance (and diversity measures) of new offspring / candidate individuals, 
        after evaluation on the task."""
        # Update individuals' stats with results of last batch of simulations
#       [(ind.fitness.setValues(obj), ind.fitness.features.setValues(bc)) for
#        (ind, obj, bc) in zip(self.inds, objective_values, behavior_values)]
        for (ind, obj, bc) in zip(self.inds, objective_values, behavior_values):
            ind.fitness.setValues([obj])
            ind.features.setValues(bc)
        # Replace the current population by the offspring
        nb_updated = self.grid.update(self.inds, issue_warning=True, ignore_exceptions=False)
        # Compile stats and update logs
        record = self.stats.compile(self.grid) if self.stats else {}

        assert len(self.grid._best_fitness.values) == 1, "Multi-objective evolution is not supported."

        # FIXME: something is wrong here, this is the min, not max.
        # maxFitness = self.grid._best_fitness[0]

        fits = [ind.fitness.values[0] for ind in self.grid]
        maxFitness = np.max(fits)
        meanFitness = np.mean(fits)
        self.logbook.record(iteration=self.i, containerSize=self.grid.size_str(), evals=len(self.inds), 
                            nbUpdated=nb_updated, elapsed=timer()-self.start_time, meanFitness=meanFitness, maxFitness=maxFitness,
                            **record)
        self.i += 1
        print(self.logbook.stream)

    def ask(self):

        if len(self.grid) == 0:
            # Return the initial batch
            return self.inds

        elif len(self.grid) < self.batch_size:
            # If few elites, supplement the population with individuals from the last generation
            np.random.shuffle(self.inds)
            breedable = self.grid.items + self.inds[:-len(self.grid)]

        else:
            breedable = self.grid

        # Select the next batch individuals
        batch = [self.toolbox.select(breedable) for i in range(self.batch_size)]

        ## Vary the pool of individuals
        self.inds = deap.algorithms.varAnd(batch, self.toolbox, self.cxpb, self.mutpb)

        return self.inds

