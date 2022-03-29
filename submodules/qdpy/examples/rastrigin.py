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


"""A simple example to illuminate a fitness function based on a normalised rastrigin function. The illumination process is ran with 2 features corresponding to the first 2 values of the genomes. It is possible to increase the difficulty of the illumination process by using problem dimension above 3. The containers and algorithms must be described in a configuration file (default: 'examples/conf/rastrigin.yaml')"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.benchmarks import *
from qdpy.base import *
from qdpy.plots import *
from qdpy import tools

import os
import numpy as np
import random
from functools import partial
import yaml



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help="Numpy random seed")
    parser.add_argument('-p', '--parallelismType', type=str, default='none', help = "Type of parallelism to use (none, concurrent, scoop)")
    parser.add_argument('-c', '--configFile', type=str, default='examples/conf/rastrigin.yaml', help = "Path of the configuration file")
    parser.add_argument('-o', '--outputDir', type=str, default=None, help = "Path of the output log files")
    args = parser.parse_args()


    # Retrieve configuration from configFile
    config = yaml.safe_load(open(args.configFile))
    print("Retrieved configuration:")
    print(config)
    print("\n------------------------\n")

    # Find where to put logs
    log_base_path = config.get("log_base_path", ".") if args.outputDir is None else args.outputDir

    # Find random seed
    if args.seed is not None:
        seed = args.seed
    elif "seed" in config:
        seed = config["seed"]
    else:
        seed = np.random.randint(1000000)

    # Update and print seed
    np.random.seed(seed)
    random.seed(seed)
    print("Seed: %i" % seed)


    # Create containers and algorithms from configuration 
    factory = Factory()
    assert "containers" in config, f"Please specify configuration entry 'containers' containing the description of all containers."
    factory.build(config["containers"])
    assert "algorithms" in config, f"Please specify configuration entry 'algorithms' containing the description of all algorithms."
    factory.build(config["algorithms"])
    assert "main_algorithm_name" in config, f"Please specify configuration entry 'main_algorithm' containing the name of the main algorithm."
    algo = factory[config["main_algorithm_name"]]
    container = algo.container

    # Define evaluation function
    eval_fn = partial(illumination_rastrigin_normalised, nb_features = len(container.shape))

    # Create a logger to pretty-print everything and generate output data files
    logger = TQDMAlgorithmLogger(algo, log_base_path=log_base_path, config=config)

    # Run illumination process !
    with ParallelismManager(args.parallelismType) as pMgr:
        best = algo.optimise(eval_fn, executor = pMgr.executor, batch_mode=False) # Disable batch_mode (steady-state mode) to ask/tell new individuals without waiting the completion of each batch


    # Print results info
    print("\n------------------------\n")
    print(algo.summary())

    # Transform the container into a grid, if needed
    if isinstance(container, containers.Grid):
        grid = container
    else:
        print("\n{:70s}".format("Transforming the container into a grid, for visualisation..."), end="", flush=True)
        grid = container.to_grid(container.shape, features_domain=container.features_domain)
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

    print("All results are available in the '%s' pickle file." % logger.final_filename)
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
