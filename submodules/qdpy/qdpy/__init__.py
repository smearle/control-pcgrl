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

"""Package `qdpy` implements recent Quality-Diversity algorithms: Map-Elites, CVT-Map-Elites, NSLC, SAIL, etc. 
QD algorithms can be accessed directly, but `qdpy` also includes building blocks that can be easily assembled together to build your own QD algorithms. It can be used with parallelism mechanisms and in distributed environments.

This package requires Python 3.6+.

`qdpy` includes the following features:
 * Generic support for diverse Containers: Grids, Novelty-Archives, Populations, etc
 * Optimisation algorithms for QD: random search methods, quasi-random methods, evolutionary algorithms
 * Support for multi-objective optimisation methods
 * Possible to use optimisation methods not designed for QD, such as [CMA-ES](https://arxiv.org/pdf/1604.00772.pdf)
 * Parallelisation of evaluations, using parallelism libraries, such as multiprocessing, concurrent.futures or [SCOOP](https://github.com/soravux/scoop)
 * Easy integration with the popular [DEAP](https://github.com/DEAP/deap) evolutionary computation framework 

Install
=======
`qdpy` requires Python 3.6+. It can be installed with:
    pip3 install qdpy

`qdpy` includes optional features that need extra packages to be installed:
 * `cma` for CMA-ES support
 * `deap` to integrate with the DEAP library
 * `tables` to output results files in the HDF5 format
 * `tqdm` to display a progress bar showing optimisation progress
 * `colorama` to add colours to pretty-printed outputs

You can install `qdpy` and all of these optional dependencies with:
    pip3 install qdpy[all]

The latest version can be installed from the GitLab repository:
    pip3 install git+https://gitlab.com/leo.cazenille/qdpy.git@master

Example
=======
From a python shell::

    from qdpy import algorithms, containers, benchmarks, plots
    
    # Create container and algorithm. Here we use MAP-Elites, by illuminating a Grid container by evo.
    grid = containers.Grid(shape=(64,64), max_items_per_bin=1, fitness_domain=((0., 1.),), features_domain=((0., 1.), (0., 1.)))
    algo = algorithms.RandomSearchMutPolyBounded(grid, budget=60000, batch_size=500,
            dimension=3, optimisation_task="maximisation")
    
    # Create a logger to pretty-print everything and generate output data files
    logger = algorithms.AlgorithmLogger(algo)
    
    # Define evaluation function
    eval_fn = algorithms.partial(benchmarks.illumination_rastrigin_normalised,
            nb_features = len(grid.shape))
    
    # Run illumination process !
    best = algo.optimise(eval_fn)
    
    # Print results info
    print(algo.summary())
    
    # Plot the results
    plots.default_plots_grid(logger)
    
    print("All results are available in the '%s' pickle file." % logger.final_filename)


Usage, Documentation
====================
Please to go the GitLab repository main page (https://gitlab.com/leo.cazenille/qdpy) and the documentation main page (https://leo.cazenille.gitlab.io/qdpy/).


:Author: Leo Cazenille, 2018-*

:License: LGPLv3, see LICENSE file.

"""


__author__ = "qdpy Team"
__license__ = "LGPLv3"
__version__ = "0.1.2.1"
__revision__ = "0.1.2.1"

# MODELINE	"{{{1
# vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
# vim:foldmethod=marker
